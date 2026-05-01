# 🩺 Medical Report Q&A System

A RAG-based web application that lets patients and clinicians upload PDF medical reports and ask natural-language questions about them. Built as a capstone project covering the full pipeline — from document ingestion to a live Streamlit deployment.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-1C3C3C?style=flat)
![Milvus](https://img.shields.io/badge/Milvus-Standalone-00A1EA?style=flat&logo=milvus&logoColor=white)
![Groq](https://img.shields.io/badge/LLaMA_3.3_70B-Groq-F55036?style=flat)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=flat)

---

## 📌 Project Scope

| Property | Detail |
|---|---|
| Dataset | MTSamples — ~4,000 medical transcripts (Kaggle) |
| Problem Type | Retrieval-Augmented Generation (RAG) |
| Input | User-uploaded PDF medical report + natural language question |
| Output | Cited, plain-language answer grounded in the document |
| Deployment | Streamlit app on localhost |

---

## ✨ Features

- **PDF upload & indexing** — drag-and-drop any medical PDF; chunked and embedded instantly
- **Dual knowledge base** — uploaded PDF takes priority; MTSamples background KB fills any gaps
- **LLaMA 3.3-70B answers** — fast, precise responses via the Groq API
- **Source citations** — every answer references which chunk it drew from
- **Relevance scoring** — interactive Plotly bar chart shows how relevant each retrieved chunk was
- **Multi-session history** — left-panel session cards to switch between past conversations
- **Strict medical prompt** — LLM is instructed never to answer from general knowledge, reducing hallucination risk

---

## 📁 Project Structure

```
medical-qa/
├── src/
│   ├── app.py               # Streamlit UI — entry point
│   ├── rag_pipeline.py      # Embedding, indexing, retrieval (Milvus)
│   ├── llm_answer.py        # Prompt building, Groq API call
│   ├── utils.py             # CSV/PDF loading, text cleaning, chart
│   └── style.css            # Custom Streamlit theme
├── notebook/                # Experiments and exploration notebooks
├── main.py                  # Project entry point
├── pyproject.toml           # Project metadata and dependencies (uv)
├── uv.lock                  # Locked dependency versions
├── .env                     # Your API key (not committed)
├── .env.example.txt         # API key template (committed)
├── .gitignore
├── README.md
└── architecture.md          # System design documentation
```

> **Note:** `data/` (MTSamples CSV) and `vectorstore/` (Milvus collections) are created locally and not committed to the repo.

---

## 🚀 Setup

### 1. Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [Docker](https://www.docker.com/) — for running Milvus Standalone
- A free Groq API key — https://console.groq.com
- `mtsamples.csv` from Kaggle — search "medicaltranscriptions"

### 2. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Start Milvus Standalone (Docker)

```bash
# Download and start Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

> ✅ Milvus will be available at `localhost:19530` by default.

### 5. Add your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Add the dataset

Place the downloaded CSV at:

```
data/mtsamples.csv
```

### 7. Launch the Streamlit app

```bash
uv run streamlit run src/app.py
```

> ⏳ **First run only:** embedding the full MTSamples dataset (~4,000 transcripts) takes **3–7 minutes** on CPU. Every subsequent start is instant — vectors are cached in Milvus collections.

---

## 🖥️ Usage

1. Click **＋ New Session** in the left panel.
2. Drag and drop a medical PDF (lab report, discharge summary, etc.) into the upload zone.
3. Wait for the indexing confirmation, then type a question in the chat input.
4. Review the answer, then expand the **Relevance Scores** or **Source Chunks** tabs to verify what the model used.

---

## 🔍 How Retrieval Works

| Step | Description |
|---|---|
| 1 | Query the **PDF collection** first (the uploaded document) in Milvus |
| 2 | Convert Milvus L2 distances to `[0,1]` similarity: `score = 1 / (1 + distance)` |
| 3 | If 3+ PDF chunks score above `0.3`, return those immediately |
| 4 | Otherwise also query the **background KB collection** and merge, deduplicated by content hash |
| 5 | Return top-5 chunks sorted by descending score |

---

## ✅ Project Checklist

- [x] **Data ingestion** — MTSamples CSV loading, PDF text extraction via PyMuPDF
- [x] **Text cleaning** — unicode normalisation, whitespace collapse, control-character removal
- [x] **Chunking** — RecursiveCharacterTextSplitter (500 chars / 100 overlap)
- [x] **Embedding** — BAAI/bge-base-en-v1.5 via HuggingFace (CPU, normalised)
- [x] **Vector storage** — two persistent Milvus collections (background + PDF) via Standalone Docker
- [x] **Retrieval** — priority cascade with relevance threshold and deduplication
- [x] **LLM integration** — LLaMA 3.3-70B via Groq with structured medical prompt
- [x] **Deployment** — Streamlit app in `src/app.py`
- [x] **Visualisation** — Plotly relevance score bar chart + source chunk expanders
- [x] **Documentation** — README, inline comments, and docstrings throughout
- [x] **Version control** — organised as a GitHub repository with `uv` for dependency management

---

## ⚙️ Configuration

| Variable | Location | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` | Groq API key |
| `EMBED_MODEL` | `src/rag_pipeline.py` | HuggingFace embedding model name |
| `chunk_size` | `src/rag_pipeline.py` | Characters per chunk (default: 500) |
| `chunk_overlap` | `src/rag_pipeline.py` | Overlap between chunks (default: 100) |
| `RELEVANCE_THRESHOLD` | `src/rag_pipeline.py` | Minimum similarity score (default: 0.3) |
| `temperature` | `src/llm_answer.py` | LLM temperature (default: 0.2) |
| `max_tokens` | `src/llm_answer.py` | Max response tokens (default: 1024) |

---

## 🧠 Design Notes

- **Dual vector store** — separating background KB from user PDFs gives clean isolation and ensures the user's own document always takes retrieval priority.
- **Milvus Standalone** — production-grade vector database running locally via Docker, replacing lightweight in-process stores for better scalability and semantic search performance.
- **L2 → similarity conversion** — `1 / (1 + distance)` produces better score spread than linear subtraction.
- **Strict prompt** — the LLM is forbidden from using general knowledge, keeping answers grounded and verifiable.
- **Simulated streaming** — answers are split word-by-word with a 15 ms delay for a real-time feel without async infrastructure.
- **uv** — used instead of pip/conda for fast, reproducible dependency resolution and lockfile support.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It demonstrates a full RAG pipeline on medical data and is **not a substitute for professional medical advice**.
