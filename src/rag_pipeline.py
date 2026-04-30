# ============================================================
# rag_pipeline.py
# PURPOSE: The core RAG engine — using Milvus instead of ChromaDB
#
# REQUIRES: pip install langchain-milvus pymilvus
# MILVUS:   Docker container running on localhost:19530
# ============================================================

import os
from utils import load_csv_data, extract_text_from_pdf, clean_text

# LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Milvus via LangChain (replaces Chroma) ──
from langchain_milvus import Milvus

# HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# PyMilvus direct connection
from pymilvus import connections


# ============================================================
# GLOBAL CONFIG
# ============================================================
MILVUS_HOST         = "localhost"
MILVUS_PORT         = "19530"
MILVUS_URI          = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_BG       = "medical_background"
COLLECTION_PDF      = "medical_pdf_uploads"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "mtsamples.csv")
EMBED_MODEL         = "BAAI/bge-base-en-v1.5"

# ── Index params (IP metric works best with normalized BGE embeddings) ──
INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type":  "IVF_FLAT",
    "params":      {"nlist": 128}
}


# ============================================================
# FUNCTION 1: Initialize TWO Milvus collections
# ============================================================
def initialize_chromadb():
    print("🤖 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded!")

    # Pass connection directly in connection_args — do NOT use connections.connect()
    connection_args = {"uri": "http://localhost:19530"}

    vectorstore_bg = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_BG,
        connection_args=connection_args,
        drop_old=False,
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

    print("✅ Both Milvus collections connected!\n")
    return vectorstore_bg, vectorstore_pdf, embeddings



# ============================================================
# FUNCTION 2: Index CSV Data into Background Collection
# ============================================================
def index_csv_data(vectorstore_bg, embeddings):
    """
    Loads MTSamples CSV and indexes it into the background Milvus collection.
    Skips if data is already indexed.
    """

    # Check if collection already has data
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
            "Download mtsamples.csv from Kaggle and place it in the data/ folder."
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
    print("💾 Embedding and inserting into Milvus...")
    print("   ⏳ This takes 3-7 minutes on first run. Please wait...")

    # Insert in batches to avoid memory spikes
    batch_size = 500
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i: i + batch_size]
        vectorstore_bg.add_documents(batch)
        print(f"   Indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"\n✅ Done! {total} chunks saved to Milvus collection '{COLLECTION_BG}'\n")


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

    # Handle file-like objects from Streamlit uploader
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

    print(f"✅ PDF indexed! {len(documents)} chunks added to Milvus collection '{COLLECTION_PDF}'\n")
    return len(documents)


# ============================================================
# FUNCTION 4: Retrieve Relevant Chunks from BOTH collections
# ============================================================
def retrieve_chunks(query, vectorstore_pdf, vectorstore_bg, k=5):
    """
    Search PDF collection first (user's own documents take priority).
    Falls back to background collection if PDF results are sparse.

    Args:
        query:           user's question string
        vectorstore_pdf: Milvus collection for PDFs
        vectorstore_bg:  Milvus collection for background knowledge
        k:               max chunks to return

    Returns:
        List of (Document, relevance_score) tuples, sorted by score desc
    """

    RELEVANCE_THRESHOLD = 0.3

    # ── Search PDF collection ──
    pdf_results_raw = vectorstore_pdf.similarity_search_with_score(query, k=k)

    pdf_results = [
        (doc, float(1.0 / (1.0 + score)))
        for doc, score in pdf_results_raw
        if (1.0 / (1.0 + score)) >= RELEVANCE_THRESHOLD
    ]

    if len(pdf_results) >= 3:
        return sorted(pdf_results, key=lambda x: x[1], reverse=True)[:k]

    # ── Fall back to background collection ──
    bg_results_raw = vectorstore_bg.similarity_search_with_score(query, k=k)

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

    return sorted(unique, key=lambda x: x[1], reverse=True)[:k]


# ============================================================
# FUNCTION 5: Get Retriever Objects (used by llm_answer.py)
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
# Make sure Milvus container is running before testing!
# ============================================================
if __name__ == "__main__":

    print("=" * 55)
    print("TESTING rag_pipeline.py  (Milvus backend)")
    print("=" * 55)

    print("\n🔧 Step 1: Initialize Milvus collections")
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()

    print("\n📚 Step 2: Index CSV Data")
    index_csv_data(vectorstore_bg, embeddings)

    print("\n🔍 Step 3: Test Retrieval")
    test_query = "What are the symptoms of chest pain?"
    results    = retrieve_chunks(test_query, vectorstore_pdf, vectorstore_bg, k=5)

    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:\n")

    for i, (doc, score) in enumerate(results):
        print(f"--- Chunk {i+1}  (relevance: {score:.3f}) ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:200])
        print()

    print("=" * 55)
    print("✅ rag_pipeline.py ALL TESTS DONE!")
    print("=" * 55)