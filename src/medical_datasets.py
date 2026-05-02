# ============================================================
# datasets.py  (NEW FILE)
# PURPOSE: Download and index free medical datasets into your
#          existing Milvus background collection.
#
# DATASETS:
#   1. PubMedQA   — 500k biomedical Q&A from PubMed abstracts
#   2. MedQA      — USMLE-style medical exam questions
#   3. MedMCQA    — 194k Indian medical entrance questions
#
# USAGE:
#   python datasets.py                    # index all datasets
#   python datasets.py --dataset pubmedqa # index one dataset
#
# REQUIRES:
#   pip install datasets                  # HuggingFace datasets library
#   Milvus running + rag_pipeline.py ready
# ============================================================

import os
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── These come from your existing files ──
from rag_pipeline import initialize_chromadb


# ============================================================
# CONFIG
# ============================================================

# How many records to load from each dataset.
# Start small (500-1000) to test, then increase.
# PubMedQA has 500k rows — loading all takes a long time.
LIMITS = {
    "pubmedqa": 2000,   # increase to 10000+ for production
    "medqa":    2000,
    "medmcqa":  2000,
}


# ============================================================
# HELPER: Check if enough data already indexed
# ============================================================
def _already_indexed(vectorstore_bg, min_expected: int = 1000) -> bool:
    """
    Returns True if the background collection already has enough data
    to skip re-indexing. Uses a conservative minimum threshold.
    """
    try:
        count = vectorstore_bg.col.num_entities
        return count >= min_expected
    except Exception:
        return False


# ============================================================
# DATASET 1: PubMedQA
# ============================================================
def index_pubmedqa(vectorstore_bg, limit: int = None) -> int:
    """
    Indexes PubMedQA dataset into the background Milvus collection.

    PubMedQA contains biomedical questions paired with PubMed abstracts
    and yes/no/maybe answers. The context (abstract) is what we index —
    it gives the system real biomedical research content.

    HuggingFace: https://huggingface.co/datasets/pubmed_qa

    Args:
        vectorstore_bg: Milvus vectorstore (background collection)
        limit:          max records to load (None = use LIMITS config)

    Returns:
        Number of chunks indexed
    """
    from datasets import load_dataset

    limit = limit or LIMITS["pubmedqa"]
    print(f"\n📚 Loading PubMedQA (up to {limit} records)...")

    # pqa_labeled is the human-annotated subset (~1000 samples)
    # pqa_unlabeled is the larger silver set (~61k samples)
    dataset = load_dataset(
        "pubmed_qa",
        "pqa_labeled",
        split="train",
        trust_remote_code=True
    )

    # Limit records
    dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"   Loaded {len(dataset)} records")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    documents = []
    for i, row in enumerate(dataset):
        # Each row has: question, context (dict of abstracts), long_answer, final_decision
        question = row.get("question", "")
        contexts = row.get("context", {})
        long_answer = row.get("long_answer", "")
        decision = row.get("final_decision", "")

        # Combine all context abstracts into one text block
        context_texts = []
        if isinstance(contexts, dict):
            for key, val in contexts.items():
                if isinstance(val, list):
                    context_texts.extend([str(v) for v in val if v])
                elif val:
                    context_texts.append(str(val))

        combined = (
            f"Question: {question}\n"
            f"Context: {' '.join(context_texts)}\n"
            f"Answer: {long_answer}\n"
            f"Decision: {decision}"
        )

        if not combined.strip():
            continue

        chunks = splitter.create_documents(
            texts=[combined],
            metadatas=[{
                "source":  "pubmedqa",
                "row":     i,
                "type":    "background",
                "dataset": "pubmedqa"
            }]
        )
        documents.extend(chunks)

    print(f"   Created {len(documents)} chunks")
    _batch_index(vectorstore_bg, documents, label="PubMedQA")
    return len(documents)


# ============================================================
# DATASET 2: MedQA (USMLE)
# ============================================================
def index_medqa(vectorstore_bg, limit: int = None) -> int:
    """
    Indexes MedQA (USMLE) dataset into the background Milvus collection.

    MedQA contains US Medical Licensing Exam questions with answer
    options and explanations. Great for clinical reasoning content.

    HuggingFace: https://huggingface.co/datasets/bigbio/med_qa

    Args:
        vectorstore_bg: Milvus vectorstore
        limit:          max records to load

    Returns:
        Number of chunks indexed
    """
    from datasets import load_dataset

    limit = limit or LIMITS["medqa"]
    print(f"\n📚 Loading MedQA USMLE (up to {limit} records)...")

    dataset = load_dataset(
        "bigbio/med_qa",
        "med_qa_en_source",
        split="train",
        trust_remote_code=True
    )

    dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"   Loaded {len(dataset)} records")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    documents = []
    for i, row in enumerate(dataset):
        # Each row: question, answer, options (list of dicts)
        question = row.get("question", "")
        answer = row.get("answer", "")

        # Format answer options as readable text
        options = row.get("options", [])
        options_text = ""
        if options:
            options_text = "\n".join(
                f"  {opt.get('key', '')}: {opt.get('value', '')}"
                for opt in options
                if isinstance(opt, dict)
            )

        combined = (
            f"USMLE Question: {question}\n"
            f"Options:\n{options_text}\n"
            f"Correct Answer: {answer}"
        )

        if not combined.strip():
            continue

        chunks = splitter.create_documents(
            texts=[combined],
            metadatas=[{
                "source":  "medqa_usmle",
                "row":     i,
                "type":    "background",
                "dataset": "medqa"
            }]
        )
        documents.extend(chunks)

    print(f"   Created {len(documents)} chunks")
    _batch_index(vectorstore_bg, documents, label="MedQA")
    return len(documents)


# ============================================================
# DATASET 3: MedMCQA
# ============================================================
def index_medmcqa(vectorstore_bg, limit: int = None) -> int:
    """
    Indexes MedMCQA dataset into the background Milvus collection.

    MedMCQA has 194k multiple-choice questions from Indian medical
    entrance exams (AIIMS, NEET PG), with explanations and subjects.

    HuggingFace: https://huggingface.co/datasets/medmcqa

    Args:
        vectorstore_bg: Milvus vectorstore
        limit:          max records to load

    Returns:
        Number of chunks indexed
    """
    from datasets import load_dataset

    limit = limit or LIMITS["medmcqa"]
    print(f"\n📚 Loading MedMCQA (up to {limit} records)...")

    dataset = load_dataset(
        "medmcqa",
        split="train",
        trust_remote_code=True
    )

    dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"   Loaded {len(dataset)} records")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    # Answer index → letter mapping
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    documents = []
    for i, row in enumerate(dataset):
        question  = row.get("question", "")
        opa       = row.get("opa", "")   # option A
        opb       = row.get("opb", "")   # option B
        opc       = row.get("opc", "")   # option C
        opd       = row.get("opd", "")   # option D
        cop       = row.get("cop", 0)    # correct option index (0-3)
        exp       = row.get("exp", "")   # explanation
        subject   = row.get("subject_name", "")
        topic     = row.get("topic_name", "")

        correct_letter = answer_map.get(cop, "A")
        correct_text   = [opa, opb, opc, opd][cop] if cop < 4 else ""

        combined = (
            f"Subject: {subject} | Topic: {topic}\n"
            f"Question: {question}\n"
            f"Options: A) {opa}  B) {opb}  C) {opc}  D) {opd}\n"
            f"Correct Answer: {correct_letter}) {correct_text}\n"
            f"Explanation: {exp}"
        )

        if not combined.strip():
            continue

        chunks = splitter.create_documents(
            texts=[combined],
            metadatas=[{
                "source":  "medmcqa",
                "row":     i,
                "type":    "background",
                "dataset": "medmcqa",
                "subject": subject,
                "topic":   topic
            }]
        )
        documents.extend(chunks)

    print(f"   Created {len(documents)} chunks")
    _batch_index(vectorstore_bg, documents, label="MedMCQA")
    return len(documents)


# ============================================================
# HELPER: Batch index documents
# ============================================================
def _batch_index(vectorstore_bg, documents: list, label: str = "dataset"):
    """Inserts documents into Milvus in batches to avoid memory spikes."""

    if not documents:
        print(f"   ⚠️  No documents to index for {label}")
        return

    batch_size = 500
    total = len(documents)
    print(f"💾 Inserting {total} chunks into Milvus ({label})...")
    print("   ⏳ This may take a few minutes...")

    for i in range(0, total, batch_size):
        batch = documents[i: i + batch_size]
        vectorstore_bg.add_documents(batch)
        print(f"   Indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"✅ {label} done! {total} chunks added.\n")


# ============================================================
# MAIN: Index all or one dataset
# ============================================================
def index_all_datasets(vectorstore_bg, force: bool = False):
    """
    Indexes all three datasets into the background collection.

    Args:
        vectorstore_bg: Milvus vectorstore
        force:          if True, re-index even if data already exists

    Returns:
        Total number of chunks added
    """
    if not force and _already_indexed(vectorstore_bg, min_expected=5000):
        print("✅ Background collection appears already populated.")
        print("   Use force=True or --force flag to re-index.\n")
        return 0

    total = 0
    total += index_pubmedqa(vectorstore_bg)
    total += index_medqa(vectorstore_bg)
    total += index_medmcqa(vectorstore_bg)

    print(f"\n🎉 All datasets indexed! Total new chunks: {total}")
    return total


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index free medical datasets into Milvus background collection"
    )
    parser.add_argument(
        "--dataset",
        choices=["pubmedqa", "medqa", "medmcqa", "all"],
        default="all",
        help="Which dataset to index (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records per dataset (default: see LIMITS config)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if data already exists"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("datasets.py — Medical Dataset Indexer")
    print("=" * 55)

    print("\n🔧 Connecting to Milvus...")
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()

    if args.dataset == "all":
        index_all_datasets(vectorstore_bg, force=args.force)

    elif args.dataset == "pubmedqa":
        index_pubmedqa(vectorstore_bg, limit=args.limit)

    elif args.dataset == "medqa":
        index_medqa(vectorstore_bg, limit=args.limit)

    elif args.dataset == "medmcqa":
        index_medmcqa(vectorstore_bg, limit=args.limit)

    print("\n" + "=" * 55)
    print("✅ datasets.py DONE!")
    print("=" * 55)
