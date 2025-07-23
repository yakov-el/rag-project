from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import json
import numpy as np
import faiss
# from rag_pipeline_module import *
from rag_pipeline_module import build_rag_prompt, query_openai_rag, chunk_docx_file, create_embeddings_for_chunks, search, setup_models, load_reranker, save_index
import uvicorn
import threading
import webbrowser
import time
import os
import openai

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# מונטינג לתיקיות סטטיות
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

if not os.path.exists("scratch_images"):
    os.makedirs("scratch_images")
app.mount("/scratch_images", StaticFiles(directory="scratch_images"), name="scratch_images")

os.makedirs("data", exist_ok=True)
os.makedirs("outputs/faiss", exist_ok=True)
os.makedirs("outputs/embeddings", exist_ok=True)

log_messages = []
def log(msg: str):
    print(msg)
    log_messages.append(msg)

tokenizer_hf, model, device, chunker_small, chunker_large = setup_models()
rerank_tokenizer, rerank_model = load_reranker(device)

faiss_path = "outputs/faiss/faiss.index"
embedding_path = "outputs/embeddings/embeddings.npy"
metadata_path = "chunks_metadata.json"

if os.path.exists(faiss_path) and os.path.exists(embedding_path) and os.path.exists(metadata_path):
    index = faiss.read_index(faiss_path)
    embs = np.load(embedding_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    chunk_ids = [m["chunk_id"] for m in metadata]
    texts = [m["text"] for m in metadata]
    image_paths_list = [m["image_paths"] for m in metadata]
else:
    index = None
    embs = None
    chunk_ids, texts, image_paths_list = [], [], []
    metadata = []

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_content = """
    <!DOCTYPE html>
    <html lang="he">
    <head>
        <meta charset="UTF-8" />
        <title>RAG Project</title>
        <style>
            body { font-family: Arial, sans-serif; direction: rtl; padding: 20px; }
            #logs { background: #f0f0f0; border: 1px solid #ccc; height: 200px; overflow-y: scroll; padding: 10px; margin-bottom: 20px; white-space: pre-wrap; }
            input, button, textarea { font-size: 1rem; margin: 5px 0; }
            #results { margin-top: 20px; }
            .result-item { margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .result-text { margin-bottom: 8px; }
            .result-images img { max-height: 150px; margin: 5px; border: 1px solid #ccc; }
            #apiKeyInput { width: 300px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>מערכת RAG - העלאת קבצים וחיפוש</h1>
        <h2>1. העלאת קבצים (docx, xlsx)</h2>
        <form id="uploadForm">
            <input type="file" id="files" name="files" multiple />
            <br />
            <button type="submit">העלה קבצים ואונדקס</button>
        </form>
        
        <h2>2. הזן מפתח API ל-OpenAI</h2>
        <input type="password" id="apiKey" placeholder="הזן כאן את מפתח ה-API שלך" />

        <h2>3. לוגים</h2>
        <pre id="logs"></pre>
        
        <h2>4. חיפוש במסמכים שהועלו</h2>
        <form id="searchForm">
            <input type="text" id="query" placeholder="הקלד את השאלה כאן..." size="50"/>
            <button type="submit">חפש</button>
        </form>
        
        <div id="results"></div>
        
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return html_content

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global index, embs, chunk_ids, texts, image_paths_list, metadata
    all_new_embs = []
    all_new_metadata = []

    for file in files:
        upload_path = f"data/{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        log(f"הקובץ {file.filename} נשמר.")
        
        chunks = chunk_docx_file(upload_path, chunker_small, chunker_large)
        log(f"נוצרו {len(chunks)} צ'אנקים עבור הקובץ {file.filename}.")
        
        new_embs, new_metadata = create_embeddings_for_chunks(chunks, tokenizer_hf, model, device)
        log(f"נוצרו אמבדינגים ל־{len(new_metadata)} צ'אנקים.")
        
        all_new_embs.append(new_embs)
        all_new_metadata.extend(new_metadata)
    
    if len(all_new_embs) > 0:
        new_embs_concat = np.concatenate(all_new_embs, axis=0)
        if index is None:
            index = faiss.IndexFlatL2(new_embs_concat.shape[1])
            index.add(new_embs_concat)
            embs = new_embs_concat
            metadata = all_new_metadata
        else:
            index.add(new_embs_concat)
            embs = np.concatenate([embs, new_embs_concat], axis=0)
            metadata.extend(all_new_metadata)

        chunk_ids[:] = [m["chunk_id"] for m in metadata]
        texts[:] = [m["text"] for m in metadata]
        image_paths_list[:] = [m["image_paths"] for m in metadata]

        save_index(index, metadata, embs, faiss_path, metadata_path, embedding_path)
        log("כל הקבצים אונדקסו ונשמרו בהצלחה.")
    else:
        log("לא הועלו קבצים.")

    return {"message": "כל הקבצים הועלו ונשמרו בהצלחה"}

@app.get("/search")
def search_chunks(q: str, apikey: str = Query(None)):
    log(f"--- New search request received ---")
    log(f"Query: {q}")

    if index is None:
        log("Error: No active index found.")
        return JSONResponse(status_code=400, content={"error": "אין אינדקס פעיל. העלה קובץ קודם."})

    if not apikey:
        log("Error: Missing API key.")
        return JSONResponse(status_code=400, content={"error": "חסר מפתח API"})

    log("Starting search in FAISS index...")
    results = search(
        question=q,
        tokenizer_hf=tokenizer_hf,
        model=model,
        index=index,
        chunk_ids=chunk_ids,
        texts=texts,
        image_paths_list=image_paths_list,
        device=device,
        rerank_tokenizer=rerank_tokenizer,
        rerank_model=rerank_model,
        k=30,
        rerank_top_n=5
    )
    log(f"Search returned {len(results)} results")

    matched_texts = [r["text"] for r in results]
    prompt = build_rag_prompt(q, matched_texts)
    log(f"Built prompt for OpenAI with length {len(prompt)} characters")

    # קריאה ל-OpenAI עם המפתח שסיפק המשתמש
    answer = query_openai_rag(prompt, apikey)
    log("Received answer from OpenAI:\n" + answer)


    for res in results:
        if res.get("image_paths"):
            res["image_paths"] = ["/scratch_images/" + os.path.basename(p) for p in res["image_paths"]]

    log("Returning results to client")
    return {"results": results, "answer": answer}



def open_browser():
    time.sleep(1)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
