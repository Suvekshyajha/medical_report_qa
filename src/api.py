"""
api.py — FastAPI Backend
Matches exact signatures from rag_pipeline.py, llm_answer.py, utils.py.
Drop this into src/ — do NOT modify any other file.
"""

import sys
import os
# Makes Python find rag_pipeline, llm_answer, utils inside src/
sys.path.insert(0, os.path.dirname(__file__))

import uuid
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import (
    initialize_chromadb,   # returns (vectorstore_bg, vectorstore_pdf, embeddings)
    index_csv_data,        # (vectorstore_bg, embeddings)
    index_pdf_document,    # (pdf_source, vectorstore_pdf, filename)
    retrieve_chunks,       # (query, vectorstore_pdf, vectorstore_bg, k)
)
from llm_answer import (
    load_llm,              # () → ChatGroq
    get_answer,            # (question, vectorstore_bg, vectorstore_pdf, llm)
)


# ── In-memory store ────────────────────────────────────────────────────────────
sessions: dict[str, dict] = {}

# Singletons — initialised once at startup
vectorstore_bg  = None
vectorstore_pdf = None
embeddings      = None
llm             = None


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore_bg, vectorstore_pdf, embeddings, llm

    print("[Startup] Loading embedding model and connecting to Milvus...")
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()

    print("[Startup] Indexing CSV (skipped if already indexed)...")
    index_csv_data(vectorstore_bg, embeddings)

    print("[Startup] Loading LLM (Groq / LLaMA 3.3-70B)...")
    llm = load_llm()

    print("[Startup] ✅ Ready!")
    _create_session("Initial Session")

    yield

    print("[Shutdown] Goodbye.")


app = FastAPI(title="Medical Report Q&A API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _create_session(name: str = "New Session") -> dict:
    session_id = str(uuid.uuid4())
    session = {
        "id":            session_id,
        "name":          name,
        "created":       datetime.utcnow().strftime("%m/%d/%Y"),
        "messages":      [],
        "indexed_files": [],
        "last_result":   None,
    }
    sessions[session_id] = session
    return session


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return sessions[session_id]


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


class NewSessionRequest(BaseModel):
    name: Optional[str] = "New Session"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Sessions ----------------------------------------------------------

@app.get("/sessions")
def list_sessions():
    return [
        {
            "id":            s["id"],
            "name":          s["name"],
            "created":       s["created"],
            "message_count": len(s["messages"]),
            "indexed_files": s["indexed_files"],
        }
        for s in sessions.values()
    ]


@app.post("/sessions")
def create_session(body: NewSessionRequest):
    session = _create_session(body.name)
    return {
        "id":            session["id"],
        "name":          session["name"],
        "created":       session["created"],
        "message_count": 0,
        "indexed_files": [],
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return _get_session(session_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    _get_session(session_id)
    del sessions[session_id]
    return {"deleted": session_id}


# ---------- PDF Upload --------------------------------------------------------

@app.post("/sessions/{session_id}/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    session = _get_session(session_id)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if file.filename in session["indexed_files"]:
        return {"status": "already_indexed", "filename": file.filename}

    contents = await file.read()

    try:
        # Exact signature: index_pdf_document(pdf_source, vectorstore_pdf, filename)
        chunk_count = index_pdf_document(
            pdf_source=contents,
            vectorstore_pdf=vectorstore_pdf,
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    session["indexed_files"].append(file.filename)
    session["name"] = file.filename

    return {
        "status":   "indexed",
        "filename": file.filename,
        "chunks":   chunk_count,
    }


# ---------- Chat --------------------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    session = _get_session(req.session_id)

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Add user message
    session["messages"].append({"role": "user", "content": req.message})

    # Exact signature: get_answer(question, vectorstore_bg, vectorstore_pdf, llm)
    result = get_answer(
        question=req.message,
        vectorstore_bg=vectorstore_bg,
        vectorstore_pdf=vectorstore_pdf,
        llm=llm,
    )

    # Add assistant reply
    session["messages"].append({"role": "assistant", "content": result["answer"]})
    session["last_result"] = result

    # Bump message count
    return {
        "answer":  result["answer"],
        "sources": result["sources"],
        "scores":  result["scores"],
        "labels":  result["labels"],
    }


# ---------- Last Result -------------------------------------------------------

@app.get("/sessions/{session_id}/results")
def get_last_result(session_id: str):
    session = _get_session(session_id)
    if not session["last_result"]:
        raise HTTPException(status_code=404, detail="No results yet for this session.")
    return session["last_result"]
