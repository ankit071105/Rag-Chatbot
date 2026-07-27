import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile, uuid, datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="DocRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "store"    : None,
    "llm"      : None,
    "doc_meta" : [],
    "n_chunks" : 0,
    "history"  : [],
    "api_key"  : "",
    "model"    : "llama-3.1-8b-instant",
}

def get_embedder():
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_llm(key, model):
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=key, model_name=model, temperature=0.2)

def chunk_docs(docs, csz=500, cov=50):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    spl = RecursiveCharacterTextSplitter(chunk_size=csz, chunk_overlap=cov)
    return spl.split_documents(docs)

def build_store(chunks):
    from langchain_community.vectorstores import FAISS
    emb = get_embedder()
    return FAISS.from_documents(chunks, emb)

def load_file(path, name):
    suf = Path(name).suffix.lower()
    if suf == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(path).load()
    from langchain_community.document_loaders import TextLoader
    return TextLoader(path, encoding="utf-8").load()

def rag_answer(question, k, mode):
    prompts = {
        "precise":
            "Answer the question using only the context below. "
            "Format your answer using markdown: use numbered lists, bullet points, and bold for key terms where appropriate. "
            "If the answer is not in the context, say: Not found in the document. "
            "Context: {ctx} Question: {q} Answer:",
        "detailed":
            "Give a thorough, well-structured answer using the context below. "
            "Use markdown formatting: headings, numbered lists, bullet points, and bold key terms. "
            "Context: {ctx} Question: {q} Detailed answer:",
        "summary":
            "Summarize the key points from the context to answer the question. "
            "Use markdown bullet points for each key point. "
            "Context: {ctx} Question: {q} Summary:",
    }
    src_docs = state["store"].as_retriever(search_kwargs={"k": k}).invoke(question)
    ctx      = " | ".join(d.page_content[:400] for d in src_docs)[:3000]
    prompt   = prompts[mode.lower()].format(ctx=ctx, q=question)
    response = state["llm"].invoke(prompt)
    text     = response.content if hasattr(response, "content") else str(response)
    return text.strip(), src_docs

@app.get("/status")
def status():
    return {
        "ready"    : state["store"] is not None,
        "n_chunks" : state["n_chunks"],
        "n_docs"   : len(state["doc_meta"]),
        "n_qa"     : len(state["history"]),
        "docs"     : state["doc_meta"],
        "model"    : state["model"],
    }

class ConfigBody(BaseModel):
    api_key : str
    model   : str = "llama-3.1-8b-instant"

@app.post("/config")
def set_config(body: ConfigBody):
    state["api_key"] = body.api_key
    state["model"]   = body.model
    state["llm"]     = get_llm(body.api_key, body.model)
    return {"ok": True}

@app.post("/upload")
async def upload(
    files        : list[UploadFile] = File(...),
    chunk_size   : int = Form(500),
    chunk_overlap: int = Form(50),
):
    if not state["api_key"]:
        raise HTTPException(400, "Set API key first via /config")

    all_docs, meta = [], []
    for f in files:
        suf = Path(f.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
            tmp.write(await f.read())
            path = tmp.name
        try:
            docs = load_file(path, f.filename)
            all_docs.extend(docs)
            meta.append({
                "name"  : f.filename,
                "pages" : len(docs),
                "size"  : f"{os.path.getsize(path)/1024:.1f} KB"
            })
        finally:
            os.unlink(path)

    chunks            = chunk_docs(all_docs, chunk_size, chunk_overlap)
    state["store"]    = build_store(chunks)
    state["n_chunks"] = len(chunks)
    state["doc_meta"] = meta
    state["history"]  = []
    return {"ok": True, "chunks": len(chunks), "docs": meta}

class AskBody(BaseModel):
    question : str
    top_k    : int = 3
    mode     : str = "precise"

@app.post("/ask")
def ask(body: AskBody):
    if not state["store"]:
        raise HTTPException(400, "Upload and process documents first")
    if not state["llm"]:
        raise HTTPException(400, "Set API key first")
    try:
        answer, src_docs = rag_answer(body.question, body.top_k, body.mode)
    except Exception as e:
        raise HTTPException(500, str(e))

    sources = [
        {
            "text"  : d.page_content[:300],
            "source": Path(d.metadata.get("source", "doc")).name,
            "page"  : d.metadata.get("page", ""),
        }
        for d in src_docs
    ]

    entry = {
        "id"     : str(uuid.uuid4())[:8],
        "q"      : body.question,
        "a"      : answer,
        "sources": sources,
        "time"   : datetime.datetime.now().strftime("%H:%M"),
    }
    state["history"].append(entry)
    return entry

@app.get("/history")
def history():
    return state["history"]

@app.delete("/history")
def clear_history():
    state["history"] = []
    return {"ok": True}

@app.get("/export")
def export():
    lines = [f"DocRAG Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", "="*50]
    for i, h in enumerate(state["history"], 1):
        lines += [f"\nQ{i} [{h['time']}]: {h['q']}", f"A{i}: {h['a']}", "-"*40]
    return {"text": "\n".join(lines)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)