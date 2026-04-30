# 🏗️ Architecture — Medical Report Q&A System

---

## Data Flow

```
User
 │
 │  PDF upload + question
 ▼
┌─────────────────────────────────┐
│         Streamlit UI            │
│  Session history · Chat · Upload│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│         Ingestion               │
│  PyMuPDF → Text splitter        │
│  500 chars / 100 char overlap   │
└────────────┬────────────────────┘
             │  clean chunks
             ▼
┌─────────────────────────────────┐
│         Embedding               │
│  BAAI/bge-base-en-v1.5          │
│  HuggingFace · CPU · normalised │
└────────────┬────────────────────┘
             │  vectors
             ▼
┌───────────────────┐   ┌──────────────────────┐
│    PDF Store      │   │    Background KB      │
│    ChromaDB       │   │    ChromaDB           │
│  User documents   │   │  MTSamples ~50k chunks│
└────────┬──────────┘   └──────────┬────────────┘
         │                         │
         └────────────┬────────────┘
                      │  priority cascade
                      ▼
┌─────────────────────────────────┐
│       retrieve_chunks()         │
│  Threshold 0.3 · top-k = 5     │
└────────────┬────────────────────┘
             │  context string
             ▼
┌─────────────────────────────────┐
│       Prompt Builder            │
│  PromptTemplate · strict rules  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│     LLaMA 3.3-70B via Groq      │
│  temperature 0.2 · 1024 tokens  │
└────────────┬────────────────────┘
             │  answer + sources + scores
             ▼
┌─────────────────────────────────┐
│     Chat + Plotly Chart         │
│  Streamed answer · relevance bar│
└─────────────────────────────────┘
```

---

## Component Breakdown

| Component | File | Responsibility |
|---|---|---|
| UI Dashboard | `app.py` | Session management, file upload, chat, results tabs |
| RAG Engine | `rag_pipeline.py` | Embedding model, ChromaDB init, indexing, retrieval |
| LLM Layer | `llm_answer.py` | Groq connection, prompt building, answer generation |
| Utilities | `utils.py` | CSV/PDF loading, text cleaning, Plotly chart |
| Styling | `style.css` | Custom Streamlit theme (DM Sans, blue accent) |

---

## Dual Vector Store Design

Two separate ChromaDB collections run throughout the app lifetime:

```
vectorstore/
├── background/     ← MTSamples CSV (~4,000 transcripts → ~50,000 chunks)
│                     Indexed once on first run. Reused on every subsequent start.
│
└── pdf/            ← User-uploaded PDFs
                      Chunks appended each time a new PDF is uploaded.
```

**Why two stores?** Keeping them separate lets retrieval always prioritise the user's own document. The background KB only supplements when the PDF doesn't contain enough relevant content — it never overrides it.

---

## Retrieval — Priority Cascade

```
Query
  │
  ▼
Search PDF Store
  │
  ▼
Convert L2 distance → similarity
  score = 1 / (1 + L2_distance)
  │
  ▼
Apply threshold: score >= 0.3
  │
  ├─ 3 or more chunks pass?
  │     └─► Return top-k PDF chunks immediately
  │
  └─ Fewer than 3?
        └─► Also search Background KB
              │
              ▼
            Combine + Deduplicate (hash first 100 chars)
              │
              ▼
            Sort descending by score → Return top-5
```

---

## LLM Prompt Structure

```
You are a precise medical assistant helping patients understand their records.

STRICT RULES:
1. Answer ONLY from the provided context — never from general knowledge
2. If information is absent: "This information is not in your records."
3. Cite which source supports each claim (e.g. "According to Source 2...")
4. Flag urgent medical items with ⚠️
5. Use plain language — define any jargon used

--- CONTEXT ---
[Source 1 - user_document/report.pdf]:
<chunk text>

[Source 2 - background/mtsamples_csv]:
<chunk text>
--- END CONTEXT ---

Patient Question: <user question>
```

---

## Session State

| Key | Type | Description |
|---|---|---|
| `sessions` | `List[dict]` | One dict per session — id, name, messages, indexed_files, last_result |
| `active_session` | `int` | Index of the currently displayed session |
| `vectorstore_bg` | `Chroma` | Background knowledge vector store (shared) |
| `vectorstore_pdf` | `Chroma` | PDF upload vector store (shared) |
| `llm` | `ChatGroq` | Loaded LLaMA model (shared) |
| `initialized` | `bool` | Guards the one-time startup sequence |

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| UI | Streamlit | 1.45.0 |
| LLM | LLaMA 3.3-70B via Groq | — |
| LLM client | langchain-groq / ChatGroq | 0.3.2 |
| Embeddings | BAAI/bge-base-en-v1.5 | — |
| Embedding runtime | sentence-transformers | 3.4.1 |
| Vector DB | ChromaDB | 0.6.3 |
| Orchestration | LangChain | 0.3.25 |
| PDF parsing | PyMuPDF (fitz) | 1.24.5 |
| Data loading | pandas | 2.2.3 |
| Charts | Plotly | 5.22.0 |
| Env management | python-dotenv | 1.1.0 |

---

## Relevance Score Colour Coding

| Score Range | Colour | Meaning |
|---|---|---|
| score > 0.8 | 🟢 Green | Highly relevant |
| 0.6 – 0.8 | 🟡 Amber | Somewhat relevant |
| score < 0.6 | 🔴 Red | Low relevance |

---

## Known Limitations

| Limitation | Impact | Suggested Fix |
|---|---|---|
| Shared PDF store across sessions | Users mix documents in multi-user deployments | Assign a UUID namespace per session |
| Synchronous LLM call | Full response must arrive before streaming begins | Use Groq native `stream=True` |
| No auth / access control | Any user can query any indexed PDF | Session-scoped collection namespacing |
| Simulated streaming | 15 ms word delay is cosmetic only | True async streaming |
