# Week 7 — RAG and LLMs
## Project: Document Question Answering System (RAG)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Pipeline-1C3C3C?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-blue?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-Frontend-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

</div>

---

## Overview

This project builds a **Retrieval-Augmented Generation (RAG)** system that answers questions based on custom documents — PDFs and text files.

Instead of relying only on a language model's built-in knowledge, the system retrieves the most relevant sections from an uploaded document and gives those to the LLM as context. This means answers are grounded in the actual document, not hallucinated from training data.

**Real-world use cases:** Chatbots over private documents, enterprise knowledge search, research paper Q&A, resume analysis, legal document review.

---

## Folder Structure

```
Week_07_RAG_and_LLMs/
│
├── backend.py          ← FastAPI server — all RAG logic
├── index.html          ← Frontend UI — HTML + Tailwind + Vanilla JS
├── requirements.txt    ← All dependencies
├── run.sh              ← One-command start script
└── README.md           ← You are here
```

---

## Architecture Decision — Why not Streamlit

The initial approach used **Streamlit** for the UI. After multiple blocking issues:

- `st.chat_input()` cannot be used inside `st.columns()` — hard constraint
- Empty label errors on `st.radio()` and `st.text_input()` in newer versions
- `langchain.chains`, `langchain.text_splitter`, `langchain.schema` all moved to separate packages in LangChain 0.3+
- HuggingFace Inference API failures (`StopIteration`, deprecated `HuggingFaceHub` class)
- Groq model deprecations (`llama3-8b-8192` retired May 2025)

**Decision:** Replaced Streamlit entirely with a **FastAPI backend + plain HTML frontend**. This gives:
- No framework restrictions on UI elements
- Full control over layout and rendering
- Markdown-rendered responses via `marked.js`
- REST API that can be extended or tested independently
- Zero build step — just open `index.html`

---

## System Architecture

```
User uploads PDF / TXT
         |
         v
  [ FastAPI /upload ]
         |
         |-- PyPDFLoader / TextLoader  →  raw text
         |-- RecursiveCharacterTextSplitter  →  chunks
         |-- FastEmbedEmbeddings (BAAI/bge-small-en-v1.5)  →  vectors
         |-- FAISS.from_documents()  →  vector index stored in memory
         |
         v
  [ FastAPI /ask ]
         |
         |-- User question → embedded by FastEmbed
         |-- FAISS similarity search → Top-K most relevant chunks
         |-- Chunks + question → formatted prompt
         |-- ChatGroq (llama-3.1-8b-instant) → generates answer
         |
         v
  JSON response → marked.js renders markdown in browser
```

---

## Pipeline — 7 Stages (as per assignment)

| Stage | Implementation | Why |
|-------|---------------|-----|
| **1. Document Ingestion** | `PyPDFLoader` for PDF, `TextLoader` for TXT | LangChain loaders handle encoding, page metadata automatically |
| **2. Text Chunking** | `RecursiveCharacterTextSplitter` (size=500, overlap=50) | Recursive splitting tries paragraphs → sentences → words, preserving semantic units |
| **3. Embedding Creation** | `FastEmbedEmbeddings` — `BAAI/bge-small-en-v1.5` | ONNX-based, 25MB, loads in seconds vs 90MB PyTorch models. BGE models rank top on MTEB benchmark |
| **4. Vector Database** | `FAISS` (Facebook AI Similarity Search) | In-memory, no server needed, exact and approximate nearest-neighbor search, free |
| **5. Query Processing** | Same `FastEmbedEmbeddings` model on user question | Embedding must match — same model for documents and queries |
| **6. Context Retrieval** | `store.as_retriever(search_kwargs={"k": top_k})` | Cosine similarity search in FAISS returns top-K most relevant chunks |
| **7. Answer Generation** | `ChatGroq` — `llama-3.1-8b-instant` | Groq runs LLaMA on custom LPU hardware — sub-second latency, free API |

---

## Tech Stack

### Backend

| Library | Version | Why used |
|---------|---------|----------|
| `fastapi` | latest | Modern async Python web framework, auto docs at `/docs`, clean REST API design |
| `uvicorn` | latest | ASGI server for FastAPI, handles async requests |
| `langchain-community` | 0.2+ | Document loaders (`PyPDFLoader`, `TextLoader`), FAISS vectorstore integration |
| `langchain-text-splitters` | latest | `RecursiveCharacterTextSplitter` — moved from `langchain` core in v0.3 |
| `langchain-groq` | latest | Official LangChain integration for Groq API |
| `langchain-core` | latest | Base types: `Document`, `PromptTemplate` |
| `fastembed` | 0.8.0 | ONNX-based embeddings, 5x faster load time than `sentence-transformers` |
| `faiss-cpu` | 1.8+ | Vector similarity search library by Meta AI |
| `pypdf` | 4.0+ | PDF text extraction |
| `python-multipart` | latest | Required for FastAPI file upload (`multipart/form-data`) |
| `pydantic` | v2 | Request/response validation in FastAPI |

### Frontend

| Technology | Why used |
|-----------|----------|
| Vanilla HTML + JavaScript | No build step, no npm, runs directly in browser |
| Tailwind CSS (CDN) | Utility-first CSS, professional UI without writing custom CSS |
| `marked.js` (CDN) | Parses LLM markdown responses into rendered HTML — numbered lists, bold, bullets |
| Fetch API | Native browser HTTP client for calling FastAPI endpoints |

### LLM

| Component | Choice | Why |
|-----------|--------|-----|
| LLM Provider | Groq | Free tier, sub-second response, runs LLaMA on custom LPU hardware |
| LLM Model | `llama-3.1-8b-instant` | Fast, free, good quality for RAG. `llama-3.3-70b-versatile` available for better quality |
| Embedding Model | `BAAI/bge-small-en-v1.5` via FastEmbed | Top MTEB scores for retrieval tasks, ONNX format, 25MB, loads in ~5 seconds |

---

## Issues Faced and How They Were Resolved

### 1. Streamlit `st.chat_input()` cannot be inside `st.columns()`
**Error:** `StreamlitAPIException: st.chat_input() can't be used inside st.columns()`
**Fix:** Moved `st.chat_input()` to top level, then switched to `st.form()`. Eventually dropped Streamlit entirely.

### 2. Empty labels on `st.radio()` and `st.text_input()`
**Error:** `label got an empty value` warning becoming an error in newer Streamlit
**Fix:** Added `label_visibility="collapsed"` with non-empty label strings.

### 3. `langchain.chains`, `langchain.text_splitter`, `langchain.schema` not found
**Error:** `ModuleNotFoundError: No module named 'langchain.chains'`
**Cause:** LangChain 0.3+ split into `langchain-core`, `langchain-text-splitters`, `langchain-community`
**Fix:** Updated all imports to new package paths. Removed `RetrievalQA` entirely — replaced with manual retriever + `llm.invoke()`.

### 4. Groq model deprecated
**Error:** `model_decommissioned: llama3-8b-8192`
**Fix:** Updated to `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`.

### 5. `HuggingFaceHub` / `HuggingFaceEndpoint` broken
**Error:** `StopIteration` from `huggingface_hub` inference client
**Cause:** New HF library changed provider routing for `text2text-generation` models
**Fix:** Dropped HuggingFace Hub LLM entirely. Groq is free and works reliably.

### 6. Sentence-transformers taking 10+ minutes to load
**Cause:** `all-MiniLM-L6-v2` is ~90MB PyTorch model, slow download and load on first run
**Fix:** Switched to `fastembed` with `BAAI/bge-small-en-v1.5` — ONNX format, 25MB, loads in seconds.

### 7. `OMP: Error #15: Initializing libomp.dylib already initialized` (Mac)
**Cause:** macOS links OpenMP twice — once from Python, once from ONNX runtime
**Fix:** Added `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the very top of `backend.py`.

### 8. Responses displaying as plain text
**Problem:** LLM returns markdown but it rendered as raw text `1. item 2. item`
**Fix:** Added `marked.js` CDN to frontend. Updated prompts to instruct LLM to format responses using markdown. Added `.prose-chat` CSS class for list and bold styling inside chat bubbles.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | System status — ready, chunks, docs, questions asked |
| `POST` | `/config` | Set Groq API key and model name |
| `POST` | `/upload` | Upload and process PDF/TXT files — returns chunk count |
| `POST` | `/ask` | Ask a question — returns answer + source chunks |
| `GET` | `/history` | Get full Q&A history |
| `DELETE` | `/history` | Clear chat history |
| `GET` | `/export` | Export conversation as plain text |

---

## How to Run

```bash
# 1. Create virtual environment with Python 3.11
python3.11 -m venv rag_env
source rag_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python backend.py
```

Open **http://localhost:8000** in your browser.

### Getting a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up free with Google or email
3. Click **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key starting with `gsk_...`
6. Paste it in the API Key field in the app

---

## How to Use

1. Paste your Groq API key in the sidebar
2. Select a model (`llama-3.1-8b-instant` is fastest)
3. Drag and drop a PDF or TXT file into the upload area
4. Adjust chunk size, overlap, and Top-K if needed
5. Click **Process Documents** — wait ~5 seconds
6. Type a specific question in the chat box and press Enter or click **Ask**
7. The answer appears with markdown formatting, source chunks shown on the right

**Tip:** Ask specific questions rather than "summarize this PDF". RAG retrieves only the top-K relevant chunks, so targeted questions work better.

---

## Key Takeaways

- RAG combines the strengths of retrieval (finding relevant info) and generation (writing a good answer) — neither alone is sufficient.
- The embedding model must be the same for both document indexing and query encoding — mismatched models produce meaningless similarity scores.
- `RecursiveCharacterTextSplitter` is preferred over fixed-size splitting because it respects natural language boundaries (paragraphs → sentences → words).
- FAISS stores vectors in RAM — no database server needed, making it ideal for single-user local demos and prototypes.
- Groq's LPU (Language Processing Unit) delivers sub-second LLM responses that would take 5–10 seconds on standard GPU API providers.
- Chunk size and Top-K are the two most impactful settings — smaller chunks give more precise retrieval, higher K gives more context to the LLM.
- Prompt engineering matters even in RAG — explicitly instructing the LLM to format responses in markdown dramatically improves readability.

---

<div align="center">

![Celebal](https://img.shields.io/badge/Celebal_Technologies-ML_Internship-blue?style=flat-square)
&nbsp;&nbsp;
![Week](https://img.shields.io/badge/Week-7_of_8-orange?style=flat-square)

</div>
