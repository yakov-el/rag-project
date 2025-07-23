# RAG System for Excel and Word Documents

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to load and process two types of files:

- Excel (.xlsx) files with structured data  
- Word (.docx) files with unstructured text

The system ingests these files, chunks their content, converts the chunks into vector embeddings using a multilingual transformer model, and stores them in a FAISS vector index. It supports searching for relevant chunks based on a user query, applies a reranking model to improve result relevance, and finally generates an answer using an OpenAI large language model (LLM).

The application is built with Python and FastAPI, exposing endpoints for file upload and search queries.

---

## Technical Highlights

- **Embedding model:** `intfloat/multilingual-e5-large` transformer for multilingual text embeddings  
- **Vector index:** FAISS for efficient nearest neighbor search  
- **Chunking:** HybridChunker from docling to segment documents into semantically coherent chunks  
- **Re-ranking:** Alibaba-NLP/gte-multilingual-reranker-base model for improving search result relevance  
- **Answer generation:** OpenAI GPT-4o (chat completions) used to generate final answers from retrieved contexts  
- **API framework:** FastAPI for serving upload and search endpoints  
- **Image support:** Extracts images from Word documents and links them to related text chunks  

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yakov-el/rag-project.git
   cd rag-project
Create and activate a Python virtual environment:

bash
Copy
Edit
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/Mac:
source env/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set your OpenAI API key as an environment variable or pass it directly when querying.

Running the Application
Run the FastAPI server:

bash
Copy
Edit
uvicorn main:app --reload
The API will be available at http://localhost:8000.

Usage Example
Upload files:
Send a POST request to /upload with your .xlsx or .docx files.
The system will process, chunk, embed, and index them.

Search query:
Send a GET request to /search?q=your_question&apikey=your_openai_api_key.

Example:

http
Copy
Edit
http://localhost:8000/search?q=What%20is%20the%20purpose%20of%20the%20home%20test?&apikey=sk-...
The system will return the most relevant chunks, reranked and combined, plus a generated answer by the LLM.

Architecture Summary
Ingestion: DocumentConverter and HybridChunker from docling handle loading and chunking of Excel and Word files.

Embedding: Chunks are converted into embeddings with a transformer model and stored in a FAISS index.

Retrieval: Queries are embedded, and the nearest neighbors are retrieved from FAISS.

Re-ranking: Candidate chunks are reranked with a cross-encoder model to improve answer relevance.

Generation: The final prompt is constructed with the top chunks and sent to OpenAI's GPT-4o to generate a natural language answer.

Notes
Keep your OpenAI API key private.

The system currently supports both English and Hebrew content but can be extended.

Images extracted from Word documents are linked to text chunks but are not directly used in embeddings or generation.

