import torch
import numpy as np
import faiss
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer as HF_AutoTokenizer
from doctr.io import DocumentFile
from docling.document_converter import DocumentConverter, WordFormatOption
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import PictureItem, TableItem
from tqdm import tqdm
import openai

# ========= 1. טען מודלים =========
def setup_models():
    model_name = "intfloat/multilingual-e5-large"
    tokenizer_hf = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = HuggingFaceTokenizer(tokenizer=tokenizer_hf, max_tokens=512)
    chunker_small = HybridChunker(tokenizer=tokenizer, merge_peers=False)
    chunker_large = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    return tokenizer_hf, model, device, chunker_small, chunker_large

# ========= 2. טען מודל Re-ranker =========
def load_reranker(device):
    rerank_model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
    rerank_tokenizer = HF_AutoTokenizer.from_pretrained(rerank_model_name)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        rerank_model_name, trust_remote_code=True
    )
    rerank_model.to(device)
    rerank_model.eval()
    return rerank_tokenizer, rerank_model

# ========= 3. המרת קובץ וחלוקה לצ'אנקים =========
def chunk_docx_file(docx_path, chunker_small, chunker_large):
    pipeline_options = PaginatedPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(Path(docx_path))
    doc = result.document

    items = list(doc.iterate_items())
    pos_map = {e.self_ref: idx for idx, (e, _) in enumerate(items)}

    small_chunks = list(chunker_small.chunk(dl_doc=doc))
    large_chunks = list(chunker_large.chunk(dl_doc=doc))
    all_chunks = small_chunks + large_chunks
    n_small = len(small_chunks)

    chunk_ranges_small = []
    for i, chunk in enumerate(small_chunks):
        refs = [it.self_ref for it in chunk.meta.doc_items]
        poses = [pos_map[r] for r in refs if r in pos_map]
        if poses:
            chunk_ranges_small.append({"chunk_index": i, "max_pos": max(poses)})

    chunk_ranges_large = []
    for j, chunk in enumerate(large_chunks):
        idx = n_small + j
        refs = [it.self_ref for it in chunk.meta.doc_items]
        poses = [pos_map[r] for r in refs if r in pos_map]
        if poses:
            chunk_ranges_large.append({"chunk_index": idx, "min_pos": min(poses), "max_pos": max(poses)})

    image_paths = {}
    output_dir = Path("scratch_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    doc_name = Path(docx_path).stem
    table_counter = 0
    picture_counter = 0

    for element, _ in doc.iterate_items():
        img = element.get_image(doc)
        if not img:
            continue
        if isinstance(element, TableItem):
            table_counter += 1
            img_path = output_dir / f"{doc_name}-table-{table_counter}.png"
        else:
            picture_counter += 1
            img_path = output_dir / f"{doc_name}-picture-{picture_counter}.png"
        img.save(img_path)
        image_paths[element.self_ref] = str(img_path)

    related_imgs = {i: [] for i in range(len(all_chunks))}

    for e, _ in items:
        if not isinstance(e, PictureItem):
            continue
        ref = e.self_ref
        if ref not in image_paths:
            continue
        pos = pos_map[ref]
        path = image_paths[ref]

        closest_chunk_idx = None
        closest_dist = float("inf")

        for cr in chunk_ranges_small:
            dist = abs(pos - cr["max_pos"])
            if dist < closest_dist:
                closest_dist = dist
                closest_chunk_idx = cr["chunk_index"]

        for cr in chunk_ranges_large:
            if cr["min_pos"] <= pos <= cr["max_pos"]:
                closest_chunk_idx = cr["chunk_index"]
                break

        if closest_chunk_idx is not None:
            related_imgs[closest_chunk_idx].append(path)

    MAX_IMAGES_PER_CHUNK = 5
    for k in related_imgs:
        related_imgs[k] = related_imgs[k][:MAX_IMAGES_PER_CHUNK]

    metadata = []
    texts = []
    for i, chunk in enumerate(all_chunks):
        text = chunker_small.contextualize(chunk) if i < n_small else chunker_large.contextualize(chunk)
        metadata.append({
            "chunk_id": f"{doc_name}_chunk_{i}",
            "text": text,
            "image_paths": related_imgs[i]
        })
        texts.append(text)

    return metadata

# ========= 4. צור אמבדינגס =========
def create_embeddings_for_chunks(metadata, tokenizer_hf, model, device):
    texts = [m["text"] for m in metadata]
    model.eval()
    all_embeddings = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer_hf(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            cls = out.hidden_states[-1][:, 0].cpu().numpy().astype("float32")
        all_embeddings.append(cls)
    return np.vstack(all_embeddings), metadata

# ========= 5. שמור אינדקס =========
def save_index(index, metadata, embs, index_path, metadata_path, embs_path):
    faiss.write_index(index, index_path)
    np.save(embs_path, embs)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# ========= 6. Re-Ranking =========
def rerank(question, candidates, rerank_tokenizer, rerank_model, device, top_n=5):
    pairs = [(question, c["text"]) for c in candidates]
    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze(-1)
    top_indices = torch.topk(scores, k=min(top_n, len(candidates))).indices.tolist()
    reranked = [candidates[i] | {"score": float(scores[i])} for i in top_indices]
    return reranked

# ========= 7. חיפוש =========
def search(question, tokenizer_hf, model, index, chunk_ids, texts, image_paths_list, device, rerank_tokenizer, rerank_model, k=30, rerank_top_n=5):
    inputs = tokenizer_hf([question], return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        q_emb = out.hidden_states[-1][:, 0].cpu().numpy().astype("float32")

    D, I = index.search(q_emb, k)
    candidates = []
    for dist, idx in zip(D[0], I[0]):
        candidates.append({
            "chunk_id": chunk_ids[idx],
            "text": texts[idx],
            "image_paths": image_paths_list[idx],
            "distance": float(dist)
        })

    reranked_results = rerank(question, candidates, rerank_tokenizer, rerank_model, device, top_n=rerank_top_n)
    return reranked_results

# ========= 8. בניית פרומפט RAG =========
def build_rag_prompt(query, matched_texts):
    context = "\n\n---\n\n".join(matched_texts)
    prompt = (
        "להלן טקסטים רלוונטיים למענה על השאלה. השב אך ורק על סמך הטקסטים המצורפים.\n"
        "אם לא ניתן למצוא תשובה בטקסטים או שהשאלה לא מובנת מהם, תוכל לנסות למצוא את התשובה ברשת,\n"
        "אך רק אם אין כלל התייחסות או תשובה לשאלה בטקסטים שסיפקתי.\n"
        "שים לב: עליך להשיב **בשפה שבה נשאלה השאלה**.\n\n"
        f"שאלה: {query}\n\n"
        f"טקסטים:\n{context}\n\n"
        "תשובה:"
    )
    return prompt

# ========= 9. שאילתה ל-OpenAI =========

def query_openai_rag(prompt, openai_api_key, model="gpt-4o", max_tokens=800):
    import openai
    openai.api_key = openai_api_key
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "אתה עוזר חכם לענות על שאלות על סמך טקסטים שהתקבלו."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

