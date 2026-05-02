# ============================================================
# rag_pipeline.py
# PURPOSE: Core RAG engine — Milvus backend
#
# UPGRADES vs original:
#   - Embedding model → MedCPT (biomedical-specific, free)
#   - Reranker        → MedCPT cross-encoder (free, HuggingFace)
#   - retrieve_chunks → reranks results before returning
#
# REQUIRES:
#   pip install langchain-milvus pymilvus
#   pip install sentence-transformers
#   Milvus Docker container running on localhost:19530
# ============================================================

import os
from utils import load_csv_data, extract_text_from_pdf, clean_text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()


os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")
# ============================================================
# GLOBAL CONFIG
# ============================================================
MILVUS_HOST     = "localhost"
MILVUS_PORT     = "19530"
MILVUS_URI      = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_BG   = "medical_background"
COLLECTION_PDF  = "medical_pdf_uploads"
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH        = os.path.join(BASE_DIR, "data", "mtsamples.csv")

# ── UPGRADED: MedCPT query encoder (biomedical-specific) ──
# Original used BAAI/bge-base-en-v1.5 (general purpose)
# MedCPT is trained on PubMed search logs — far better for medical text
EMBED_MODEL    = "ncbi/MedCPT-Query-Encoder"
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
RERANKER_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type":  "IVF_FLAT",
    "params":      {"nlist": 128}
}

# Global reranker — loaded once, reused across calls
_reranker = None


# ============================================================
# HELPER: Load reranker (lazy, singleton)
# ============================================================
def _get_reranker() -> CrossEncoder:
    """
    Loads the cross-encoder reranker once and caches it.
    Tries MedCPT first, falls back to ms-marco if unavailable.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        print(f"🔁 Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        print("✅ MedCPT cross-encoder loaded!\n")
    except Exception as e:
        print(f"⚠️  MedCPT cross-encoder failed ({e}), using fallback...")
        _reranker = CrossEncoder(RERANKER_FALLBACK)
        print(f"✅ Fallback reranker loaded: {RERANKER_FALLBACK}\n")

    return _reranker


# ============================================================
# FUNCTION 1: Initialize TWO Milvus collections
# ============================================================
def initialize_chromadb():
    """
    Loads MedCPT query encoder and connects to both Milvus collections.

    UPGRADE NOTE:
        Original used BAAI/bge-base-en-v1.5 (general English).
        Now uses ncats/MedCPT-Query-Encoder, trained on PubMed search
        logs — significantly better for medical/clinical text retrieval.

    Returns:
        (vectorstore_bg, vectorstore_pdf, embeddings)
    """
    print("🤖 Loading MedCPT embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ MedCPT query encoder loaded!")

    connection_args = {"uri": MILVUS_URI}

    vectorstore_bg = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_BG,
        connection_args=connection_args,
        drop_old=True,
        auto_id=True,
        index_params=INDEX_PARAMS,
    )
    print("✅ Background collection connected!")

    vectorstore_pdf = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_PDF,
        connection_args=connection_args,
        drop_old=False,
        auto_id=True,
        index_params=INDEX_PARAMS,
    )
    print("✅ PDF collection connected!")
    print("✅ Both Milvus collections ready!\n")

    return vectorstore_bg, vectorstore_pdf, embeddings


# ============================================================
# FUNCTION 2: Index CSV Data into Background Collection
# ============================================================
def index_csv_data(vectorstore_bg, embeddings):
    """
    Loads MTSamples CSV and indexes into the background Milvus collection.
    Skips if data is already indexed.

    NOTE: If you previously indexed with bge-base-en-v1.5, you must
    drop the old collection and re-index with MedCPT embeddings.
    They use different vector spaces and cannot be mixed.
    To re-index: set drop_old=True in initialize_chromadb() for
    vectorstore_bg, run once, then set it back to False.
    """

    try:
        existing_count = vectorstore_bg.col.num_entities
    except Exception:
        existing_count = 0

    if existing_count > 0:
        print(f"✅ Background collection already has {existing_count} chunks.")
        print("   Skipping CSV indexing.\n")
        return

    print("📂 Loading CSV transcripts...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"❌ CSV file not found at '{CSV_PATH}'.\n"
            "Download mtsamples.csv from Kaggle and place it in data/."
        )

    texts = load_csv_data(CSV_PATH)

    if not texts:
        raise ValueError("❌ No texts loaded from CSV.")

    print("✂️  Splitting transcripts into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    documents = []
    for i, text in enumerate(texts):
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": "mtsamples_csv", "row": i, "type": "background"}]
        )
        documents.extend(chunks)

    print(f"   Created {len(documents)} chunks from {len(texts)} transcripts")
    print("💾 Embedding and inserting into Milvus (MedCPT)...")
    print("   ⏳ First run takes 3-7 minutes. Please wait...")

    batch_size = 500
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i: i + batch_size]
        vectorstore_bg.add_documents(batch)
        print(f"   Indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"\n✅ Done! {total} chunks saved to '{COLLECTION_BG}'\n")


# ============================================================
# FUNCTION 3: Index PDF Document into PDF Collection
# ============================================================
def index_pdf_document(pdf_source, vectorstore_pdf, filename: str = "uploaded_pdf"):
    """
    Extracts text from a PDF and adds it to the PDF Milvus collection.

    Args:
        pdf_source:      file path, bytes, or file-like object
        vectorstore_pdf: Milvus vectorstore for PDFs
        filename:        original filename (stored in metadata)

    Returns:
        Number of chunks added
    """

    print(f"📄 Processing PDF: {filename}")

    if hasattr(pdf_source, "read"):
        pdf_source = pdf_source.read()

    text = extract_text_from_pdf(pdf_source)

    if not text:
        print("❌ Could not extract text from PDF")
        return 0

    print(f"   Extracted {len(text)} characters from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    documents = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source":   "pdf_upload",
            "filename": filename,
            "type":     "user_document"
        }]
    )

    print(f"   Split into {len(documents)} chunks")
    vectorstore_pdf.add_documents(documents)
    print(f"✅ PDF indexed! {len(documents)} chunks added to '{COLLECTION_PDF}'\n")
    return len(documents)


# ============================================================
# FUNCTION 4: Rerank chunks using cross-encoder
# ============================================================
def rerank_chunks(query: str, candidates: list, top_k: int = 5) -> list:
    """
    Reranks retrieved (doc, score) pairs using a cross-encoder.

    A bi-encoder (like MedCPT-Query-Encoder) embeds query and documents
    independently — fast but less accurate. A cross-encoder sees the
    query AND document together, giving much better relevance scores.

    Args:
        query:      the user's question
        candidates: list of (Document, float) from similarity search
        top_k:      how many to return after reranking

    Returns:
        List of (Document, reranker_score) sorted by score desc
    """

    if not candidates:
        return []

    reranker = _get_reranker()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc, _ in candidates]

    # Cross-encoder returns raw logits — higher = more relevant
    scores = reranker.predict(pairs)

    # Zip docs back with their new scores
    reranked = [
        (candidates[i][0], float(scores[i]))
        for i in range(len(candidates))
    ]

    # Sort descending by cross-encoder score
    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:top_k]


# ============================================================
# FUNCTION 5: Retrieve + Rerank chunks from BOTH collections
# ============================================================
def retrieve_chunks(query, vectorstore_pdf, vectorstore_bg, k=5):
    """
    Two-stage retrieval:
      Stage 1 — Bi-encoder similarity search (fast, approximate)
      Stage 2 — Cross-encoder reranking (slow, precise)

    PDF collection is searched first (user's own docs take priority).
    Falls back to background collection if PDF results are sparse.

    UPGRADE vs original:
        Original returned similarity scores directly.
        Now adds a reranking step so the LLM sees the most relevant
        chunks, not just the most similar embeddings.

    Args:
        query:           user's question string
        vectorstore_pdf: Milvus collection for PDFs
        vectorstore_bg:  Milvus collection for background knowledge
        k:               max chunks to return

    Returns:
        List of (Document, reranker_score) tuples, sorted by score desc
    """

    RELEVANCE_THRESHOLD = 0.3
    # Fetch more candidates than needed so reranker has room to reorder
    FETCH_K = k * 3

    # ── Stage 1: Bi-encoder retrieval ──
    pdf_results_raw = vectorstore_pdf.similarity_search_with_score(query, k=FETCH_K)
    pdf_results = [
        (doc, float(1.0 / (1.0 + score)))
        for doc, score in pdf_results_raw
        if (1.0 / (1.0 + score)) >= RELEVANCE_THRESHOLD
    ]

    bg_results_raw = vectorstore_bg.similarity_search_with_score(query, k=FETCH_K)
    bg_results = [
        (doc, float(1.0 / (1.0 + score)))
        for doc, score in bg_results_raw
        if (1.0 / (1.0 + score)) >= RELEVANCE_THRESHOLD
    ]

    # Merge and deduplicate
    combined = pdf_results + bg_results
    seen, unique = set(), []
    for doc, score in combined:
        h = hash(doc.page_content[:100])
        if h not in seen:
            seen.add(h)
            unique.append((doc, score))

    if not unique:
        return []

    # ── Stage 2: Cross-encoder reranking ──
    print(f"   🔁 Reranking {len(unique)} candidates with cross-encoder...")
    reranked = rerank_chunks(query, unique, top_k=k)
    print(f"   ✅ Reranking done. Returning top {len(reranked)} chunks.")

    return reranked


# ============================================================
# FUNCTION 6: Get Retriever Objects
# ============================================================
def get_retriever_bg(vectorstore_bg):
    """Returns MMR retriever for background knowledge collection."""
    return vectorstore_bg.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )


def get_retriever_pdf(vectorstore_pdf):
    """Returns MMR retriever for PDF uploads collection."""
    return vectorstore_pdf.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )


# ============================================================
# TEST — python rag_pipeline.py
# Make sure Milvus container is running first!
# ============================================================
if __name__ == "__main__":

    print("=" * 55)
    print("TESTING rag_pipeline.py  (MedCPT + Reranker)")
    print("=" * 55)

    print("\n🔧 Step 1: Initialize Milvus collections")
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()

    print("\n📚 Step 2: Index CSV Data")
    index_csv_data(vectorstore_bg, embeddings)

    print("\n🔍 Step 3: Test Retrieval + Reranking")
    test_query = "What are the symptoms of chest pain?"
    results = retrieve_chunks(test_query, vectorstore_pdf, vectorstore_bg, k=5)

    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results after reranking:\n")

    for i, (doc, score) in enumerate(results):
        print(f"--- Chunk {i+1}  (reranker score: {score:.3f}) ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:200])
        print()

    print("=" * 55)
    print("✅ rag_pipeline.py ALL TESTS DONE!")
    print("=" * 55)