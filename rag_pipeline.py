# ============================================================
# rag_pipeline.py
# PURPOSE: The core RAG engine of the project
# ============================================================

import os
from utils import load_csv_data, extract_text_from_pdf, clean_text

# LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB vector store via LangChain
from langchain_community.vectorstores import Chroma

# HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================================
# GLOBAL PATHS
# ============================================================
CHROMA_DIR_BG = "vectorstore/background"  # Background knowledge (CSV)
CHROMA_DIR_PDF = "vectorstore/pdf"  # Uploaded PDFs only
CSV_PATH = "data/mtsamples.csv"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


# ============================================================
# FUNCTION 1: Initialize TWO ChromaDB databases
# ============================================================
def initialize_chromadb():
    """
    Initialize TWO separate ChromaDB databases:
    1. Background knowledge (CSV dataset)
    2. PDF uploads (user documents)

    Returns:
        vectorstore_bg: ChromaDB for background knowledge
        vectorstore_pdf: ChromaDB for user PDFs
        embeddings: The embedding model
    """
    print("🤖 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("✅ Embedding model loaded!")

    # Create directories if they don't exist
    os.makedirs(CHROMA_DIR_BG, exist_ok=True)
    os.makedirs(CHROMA_DIR_PDF, exist_ok=True)

    # Initialize TWO separate vector stores
    vectorstore_bg = Chroma(
        persist_directory=CHROMA_DIR_BG,
        embedding_function=embeddings
    )

    vectorstore_pdf = Chroma(
        persist_directory=CHROMA_DIR_PDF,
        embedding_function=embeddings
    )

    print("✅ Both ChromaDB databases connected!\n")
    return vectorstore_bg, vectorstore_pdf, embeddings


# ============================================================
# FUNCTION 2: Index CSV Data into Background DB
# ============================================================
def index_csv_data(vectorstore_bg, embeddings):
    """
    Loads MTSamples CSV and indexes it into background ChromaDB.
    """
    existing_count = vectorstore_bg._collection.count()

    if existing_count > 0:
        print(f"✅ Background DB already has {existing_count} chunks.")
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

    print("💾 Embedding and saving to ChromaDB...")
    print("   ⏳ This takes 3-7 minutes on first run. Please wait...")

    batch_size = 500
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i: i + batch_size]
        vectorstore_bg.add_documents(batch)
        print(f"   Indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"\n✅ Done! {total} chunks saved to Background DB")
    print(f"   Location: {CHROMA_DIR_BG}\n")


# ============================================================
# FUNCTION 3: Index PDF Document into PDF DB
# ============================================================
def index_pdf_document(pdf_source, vectorstore_pdf, filename: str = "uploaded_pdf"):
    """
    Extracts text from PDF and adds it to PDF ChromaDB.
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
        metadatas=[{"source": "pdf_upload", "filename": filename, "type": "user_document"}]
    )

    print(f"   Split into {len(documents)} chunks")

    vectorstore_pdf.add_documents(documents)

    print(f"✅ PDF indexed! {len(documents)} chunks added to PDF DB\n")
    return len(documents)


# ============================================================
# FUNCTION 4: Retrieve Relevant Chunks from BOTH databases
# rag_pipeline.py - fix score conversion + add threshold
def retrieve_chunks(query, vectorstore_pdf, vectorstore_bg, k=5):
    RELEVANCE_THRESHOLD = 0.3  # discard chunks below this

    pdf_results_raw = vectorstore_pdf.similarity_search_with_score(query, k=k)

    # Correct: ChromaDB L2 distance → similarity (lower distance = more similar)
    # Use exponential decay for better score distribution
    pdf_results = [
        (doc, float(1.0 / (1.0 + score)))  # better than 1 - score/2
        for doc, score in pdf_results_raw
        if (1.0 / (1.0 + score)) >= RELEVANCE_THRESHOLD
    ]

    if len(pdf_results) >= 3:
        return sorted(pdf_results, key=lambda x: x[1], reverse=True)[:k]

    bg_results_raw = vectorstore_bg.similarity_search_with_score(query, k=k)
    bg_results = [
        (doc, float(1.0 / (1.0 + score)))
        for doc, score in bg_results_raw
        if (1.0 / (1.0 + score)) >= RELEVANCE_THRESHOLD
    ]

    combined = pdf_results + bg_results
    seen, unique = set(), []
    for doc, score in combined:
        h = hash(doc.page_content[:100])
        if h not in seen:
            seen.add(h)
            unique.append((doc, score))

    return sorted(unique, key=lambda x: x[1], reverse=True)[:k]


# ============================================================
# FUNCTION 5: Get Retriever Objects
# ============================================================
def get_retriever_bg(vectorstore_bg):
    """Returns retriever for background knowledge"""
    return vectorstore_bg.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )


def get_retriever_pdf(vectorstore_pdf):
    """Returns retriever for PDF documents"""
    return vectorstore_pdf.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TESTING rag_pipeline.py")
    print("=" * 50)

    print("\n🔧 Step 1: Initialize ChromaDB")
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()

    print("\n📚 Step 2: Index CSV Data")
    index_csv_data(vectorstore_bg, embeddings)

    print("\n🔍 Step 3: Test Retrieval")
    test_query = "What are the symptoms of chest pain?"
    results = retrieve_chunks(test_query, vectorstore_pdf, vectorstore_bg, k=5)

    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:\n")

    for i, (doc, score) in enumerate(results):
        print(f"--- Chunk {i + 1} (relevance: {score:.3f}) ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:200])
        print()

    print("=" * 50)
    print("✅ rag_pipeline.py ALL TESTS DONE!")
    print("=" * 50)