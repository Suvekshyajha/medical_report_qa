# ============================================================
# llm_answer.py
# PURPOSE: Send retrieved chunks + question to Groq API
#
# UPGRADES vs original:
#   - HyDE (Hypothetical Document Embeddings) query expansion
#     Before retrieving, asks the LLM to write a hypothetical answer,
#     then uses THAT as the search query. Dramatically improves
#     retrieval because the hypothetical answer looks more like
#     the indexed documents than a short question does.
# ============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from rag_pipeline import (
    initialize_chromadb,
    index_csv_data,
    retrieve_chunks,
)

load_dotenv()


# ============================================================
# FUNCTION 1: Load LLM
# ============================================================
def load_llm():
    """
    Connects to Groq API and loads LLaMA model.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "❌ GROQ_API_KEY not found!\n"
            "Please add it to your .env file:\n"
            "GROQ_API_KEY=your_key_here\n"
        )

    print("✅ Groq API key loaded!")

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024
    )

    print("✅ LLaMA 3.3-70B loaded via Groq!\n")
    return llm


# ============================================================
# FUNCTION 2: HyDE — Hypothetical Document Expansion
# ============================================================
def expand_query_with_hyde(question: str, llm) -> str:
    """
    Generates a hypothetical answer to use as the retrieval query.

    WHY THIS WORKS:
        A short question like "What are symptoms of chest pain?"
        is very different in embedding space from the clinical notes
        you indexed. But a hypothetical clinical paragraph about
        chest pain symptoms is much closer to those notes.

        This closes the gap between question and document embeddings,
        dramatically improving retrieval quality — especially for
        medical text where precise terminology matters.

    Args:
        question: the user's original question
        llm:      the loaded ChatGroq LLM

    Returns:
        A hypothetical passage string to use as the search query.
        Falls back to the original question if LLM call fails.
    """

    hyde_prompt = f"""You are a medical expert. Write a short clinical paragraph (3-5 sentences) 
that would be found in a medical record or textbook and directly answers this question.
Write ONLY the paragraph — no preamble, no explanation.

Question: {question}

Clinical paragraph:"""

    try:
        print("🔮 HyDE: generating hypothetical passage for better retrieval...")
        response = llm.invoke(hyde_prompt)
        hypothetical = response.content.strip()
        print(f"   Hypothetical passage: {hypothetical[:120]}...\n")
        return hypothetical
    except Exception as e:
        print(f"⚠️  HyDE failed ({e}), falling back to original query.\n")
        return question


# ============================================================
# FUNCTION 3: Build Answer Prompt
# ============================================================
def build_prompt():
    template = """You are a precise medical assistant helping patients understand their records.

STRICT RULES:
1. Answer ONLY from the provided context — never from general knowledge
2. If information is absent, say exactly: "This information is not in your records."
3. Cite which source supports each key claim (e.g., "According to Source 2...")
4. Flag anything that needs urgent medical attention with ⚠️
5. Use plain language — avoid jargon unless you define it

--- CONTEXT ---
{context}
--- END CONTEXT ---

Patient Question: {question}

Answer (be specific, cite sources, use plain language):"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


# ============================================================
# FUNCTION 4: Get Answer  (with HyDE)
# ============================================================
def get_answer(question: str, vectorstore_bg, vectorstore_pdf, llm):
    """
    Gets answer from LLaMA for a given question.

    UPGRADE vs original:
        Now uses HyDE to expand the query before retrieval.
        The hypothetical passage is used for vector search,
        but the original question is shown to the LLM.

    Args:
        question:        user's question
        vectorstore_bg:  background knowledge DB
        vectorstore_pdf: user PDF DB
        llm:             ChatGroq LLM object
    """

    print(f"❓ Question: '{question}'")

    # ── UPGRADE: HyDE query expansion ──
    # Use the hypothetical passage for retrieval, keep original for the prompt
    search_query = expand_query_with_hyde(question, llm)

    print(f"🔍 Retrieving chunks (using HyDE-expanded query)...")
    results = retrieve_chunks(search_query, vectorstore_pdf, vectorstore_bg, k=5)

    if not results:
        return {
            "answer": "No relevant information found in the database.",
            "sources": [],
            "scores": [],
            "labels": []
        }

    # Extract chunks and scores
    chunks = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]
    labels = [f"Chunk {i + 1}" for i in range(len(chunks))]

    # Build context string
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", chunk.metadata.get("filename", "unknown"))
        doc_type = chunk.metadata.get("type", "unknown")
        context_parts.append(
            f"[Source {i + 1} - {doc_type}/{source}]:\n{chunk.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Build and send prompt — use ORIGINAL question, not the HyDE expansion
    prompt = build_prompt()
    formatted_prompt = prompt.format(context=context, question=question)

    print("🤖 Sending to LLaMA via Groq...")
    response = llm.invoke(formatted_prompt)
    answer = response.content

    print("✅ Answer received!\n")

    return {
        "answer": answer,
        "sources": [chunk.page_content for chunk in chunks],
        "scores": scores,
        "labels": labels
    }


# ============================================================
# FUNCTION 5: Format Sources
# ============================================================
def format_sources(sources: list) -> str:
    """Formats source chunks for display."""
    if not sources:
        return "No sources found."

    formatted = []
    for i, source in enumerate(sources):
        preview = source[:300] + "..." if len(source) > 300 else source
        formatted.append(f"📄 Source {i + 1}:\n{preview}")

    return "\n\n".join(formatted)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TESTING llm_answer.py  (with HyDE)")
    print("=" * 50)

    print("\n🔧 Step 1: Initialize ChromaDB")
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()
    index_csv_data(vectorstore_bg, embeddings)

    print("\n🤖 Step 2: Load LLM")
    llm = load_llm()

    print("\n❓ Step 3: Ask a question")
    question = "What are the symptoms of chest pain?"

    result = get_answer(question, vectorstore_bg, vectorstore_pdf, llm)

    print(f"\nQuestion: {question}")
    print(f"\n{'=' * 50}")
    print(f"Answer:\n{result['answer']}")
    print(f"\n{'=' * 50}")
    print(f"Reranker Scores: {result['scores']}")

    print("\n" + "=" * 50)
    print("✅ llm_answer.py ALL TESTS DONE!")
    print("=" * 50)
