# 🩺 Medical Report Intelligence

Advanced clinical analysis and document intelligence dashboard.  
Upload medical PDFs, ask questions, and get AI-powered answers backed by an upgraded RAG pipeline with biomedical embeddings, cross-encoder reranking, and HyDE query expansion.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat&logo=nextdotjs&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-1C3C3C?style=flat)
![Milvus](https://img.shields.io/badge/Milvus-Standalone-00A1EA?style=flat&logo=milvus&logoColor=white)
![Groq](https://img.shields.io/badge/LLaMA_3.3_70B-Groq-F55036?style=flat)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=flat)

---

## 📌 Project Scope

| Property | Detail |
|---|---|
| Dataset | MTSamples (~4,000 transcripts) + PubMedQA + MedQA (USMLE) + MedMCQA |
| Problem Type | Retrieval-Augmented Generation (RAG) |
| Input | User-uploaded PDF medical report + natural language question |
| Output | Cited, plain-language answer grounded in the document |
| Deployment | Next.js frontend + FastAPI backend |

---

## ✨ Features

- **PDF upload & indexing** — drag-and-drop any medical PDF; chunked and embedded instantly
- **Dual knowledge base** — uploaded PDF takes priority; background KB fills any gaps
- **Biomedical embeddings** — `ncbi/MedCPT-Query-Encoder` trained on PubMed search logs
- **Cross-encoder reranking** — `ncbi/MedCPT-Cross-Encoder` reranks retrieved chunks before LLM sees them
- **HyDE query expansion** — generates a hypothetical answer first, uses it for retrieval (dramatically improves results)
- **LLaMA 3.3-70B answers** — fast, precise responses via the Groq API
- **Source citations** — every answer references which chunk it drew from
- **Relevance scoring** — interactive bar chart shows reranker scores per chunk
- **Multi-session history** — left-panel session cards to switch between past conversations
- **Strict medical prompt** — LLM is instructed never to answer from general knowledge, reducing hallucination risk
- **Extended knowledge base** — PubMedQA, MedQA (USMLE), and MedMCQA indexed alongside MTSamples

---

## 🏗️ Project Structure

```
medical-qa/
│
├── src/                          # Python backend source
│   ├── api.py                    # FastAPI app — REST endpoints
│   ├── rag_pipeline.py           # RAG engine — Milvus + MedCPT + reranker
│   ├── llm_answer.py             # LLM integration — HyDE + Groq / LLaMA 3.3-70B
│   ├── medical_datasets.py       # Free dataset downloader — PubMedQA, MedQA, MedMCQA
│   └── utils.py                  # Helpers — PDF extract, CSV load, text clean, chart
│
├── frontend/                     # Next.js frontend
│   ├── app/
│   │   ├── layout.js             # Root layout + Google Fonts
│   │   ├── page.js               # Main page — wires all components
│   │   └── globals.css           # Tailwind directives + scrollbar styles
│   ├── components/
│   │   ├── Sidebar.js            # Session list + new session button
│   │   ├── UploadZone.js         # PDF drag-and-drop upload
│   │   ├── ChatBox.js            # Consultation chat UI
│   │   └── RelevanceChart.js     # Recharts bar chart + source chunks tabs
│   ├── lib/
│   │   └── api.js                # All axios calls to FastAPI centralised here
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── data/
│   └── mtsamples.csv             # MTSamples dataset from Kaggle (not included)
│
├── main.py                       # Entry point — runs uvicorn
├── pyproject.toml                # Python dependencies (managed by uv)
└── .env                          # API keys — never commit this
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, Recharts |
| Backend | FastAPI, Uvicorn |
| RAG Pipeline | LangChain, Milvus (Docker) |
| Embeddings | `ncbi/MedCPT-Query-Encoder` (biomedical, HuggingFace) |
| Reranker | `ncbi/MedCPT-Cross-Encoder` (biomedical cross-encoder) |
| Query Expansion | HyDE — Hypothetical Document Embeddings |
| LLM | LLaMA 3.3-70B via Groq API |
| Vector DB | Milvus Standalone (Docker container) |
| Package Manager | uv (Python), npm (Node) |

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.11
- Node.js LTS
- Docker Desktop (for Milvus)
- uv installed (`pip install uv`)
- A HuggingFace account (free) — for MedCPT model access

### 1. Clone and install Python deps
```bash
cd medical-qa
uv sync
```

### 2. Add your API keys
Create a `.env` file in the root:
```env
GROQ_API_KEY=your_groq_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
```

Get a free Groq key at: https://console.groq.com  
Get a free HuggingFace token at: https://huggingface.co/settings/tokens (Read permission is enough)

### 3. Log in to HuggingFace CLI (first time only)
```bash
uv run huggingface-cli login
```
Paste your token when prompted. This is required to download the MedCPT models.

### 4. Start Milvus via Docker
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

### 5. Add MTSamples dataset
Download `mtsamples.csv` from Kaggle and place it at:
```
medical-qa/data/mtsamples.csv
```
Link: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

### 6. Install frontend deps (once only)
```bash
cd frontend
npm install
```

### 7. Index data — run in this order (first time only)

**Step 1 — Index MTSamples CSV into Milvus:**
```bash
uv run python src/rag_pipeline.py
```
> ⏳ First run takes 3–7 minutes on CPU. Subsequent runs skip automatically.

**Step 2 — Download and index free medical datasets:**
```bash
uv run python src/medical_datasets.py
```
This pulls PubMedQA, MedQA (USMLE), and MedMCQA from HuggingFace automatically — no manual downloads needed. You can also index just one dataset:
```bash
uv run python src/medical_datasets.py --dataset pubmedqa
uv run python src/medical_datasets.py --dataset medqa
uv run python src/medical_datasets.py --dataset medmcqa --limit 500
```

### 8. Run both servers

**Terminal 1 — Backend:**
```bash
uv run python main.py
```
Runs on: http://localhost:8000

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```
Runs on: http://localhost:3000

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/sessions` | List all sessions |
| POST | `/sessions` | Create new session |
| DELETE | `/sessions/{id}` | Delete a session |
| POST | `/sessions/{id}/upload` | Upload PDF to session |
| POST | `/chat` | Send message, get RAG answer |
| GET | `/sessions/{id}/results` | Get last relevance scores |

---

## 🧠 How It Works

```
User asks question
       ↓
HyDE expansion — LLM writes a hypothetical clinical paragraph
       ↓                 (used for retrieval only)
retrieve_chunks() — searches Milvus with expanded query
   ├── PDF collection first (user's uploaded docs)
   └── Background collection fallback (MTSamples + PubMedQA + MedQA + MedMCQA)
       ↓
Cross-encoder reranker — MedCPT-Cross-Encoder rescores all candidates
       ↓
Top 5 chunks by reranker score
       ↓
get_answer() — builds prompt with ORIGINAL question + sends to Groq
       ↓
LLaMA 3.3-70B generates answer citing sources
       ↓
Answer + reranker scores + source chunks returned to UI
```

---

## 📁 Data Flow

```
mtsamples.csv      →  load_csv_data()       →  chunk  →  MedCPT embed  →  Milvus (background)
PubMedQA           →  index_pubmedqa()      →  chunk  →  MedCPT embed  →  Milvus (background)
MedQA (USMLE)      →  index_medqa()         →  chunk  →  MedCPT embed  →  Milvus (background)
MedMCQA            →  index_medmcqa()       →  chunk  →  MedCPT embed  →  Milvus (background)
PDF upload         →  index_pdf_document()  →  chunk  →  MedCPT embed  →  Milvus (pdf uploads)
```

---

## 🔍 How Retrieval Works

| Step | Description |
|---|---|
| 1 | **HyDE** — LLM generates a hypothetical answer; this is used as the search query instead of the raw question |
| 2 | **Bi-encoder search** — MedCPT-Query-Encoder embeds the query; Milvus finds nearest vectors |
| 3 | PDF collection searched first; background KB searched if PDF results are sparse |
| 4 | Results merged and deduplicated by content hash |
| 5 | **Cross-encoder reranking** — MedCPT-Cross-Encoder scores each (query, chunk) pair together for precise relevance |
| 6 | Top 5 chunks by reranker score sent to LLM |

---

## ⚙️ Configuration

| Variable | Location | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` | Groq API key |
| `HUGGINGFACE_TOKEN` | `.env` | HuggingFace token for MedCPT download |
| `EMBED_MODEL` | `src/rag_pipeline.py` | Embedding model (default: `ncbi/MedCPT-Query-Encoder`) |
| `RERANKER_MODEL` | `src/rag_pipeline.py` | Reranker model (default: `ncbi/MedCPT-Cross-Encoder`) |
| `chunk_size` | `src/rag_pipeline.py` | Characters per chunk (default: 500) |
| `chunk_overlap` | `src/rag_pipeline.py` | Overlap between chunks (default: 100) |
| `RELEVANCE_THRESHOLD` | `src/rag_pipeline.py` | Minimum bi-encoder score (default: 0.3) |
| `LIMITS` | `src/medical_datasets.py` | Records per dataset (default: 2000 each) |
| `temperature` | `src/llm_answer.py` | LLM temperature (default: 0.2) |
| `max_tokens` | `src/llm_answer.py` | Max response tokens (default: 1024) |

---

## ✅ Project Checklist

- [x] **Data ingestion** — MTSamples CSV loading, PDF text extraction via PyMuPDF
- [x] **Extended datasets** — PubMedQA, MedQA (USMLE), MedMCQA via HuggingFace (auto-download)
- [x] **Text cleaning** — unicode normalisation, whitespace collapse, control-character removal
- [x] **Chunking** — RecursiveCharacterTextSplitter (500 chars / 100 overlap)
- [x] **Biomedical embeddings** — `ncbi/MedCPT-Query-Encoder` trained on PubMed search logs
- [x] **Vector storage** — two persistent Milvus collections (background + PDF) via Standalone Docker
- [x] **Two-stage retrieval** — bi-encoder search + cross-encoder reranking
- [x] **HyDE query expansion** — hypothetical document embeddings for better retrieval
- [x] **LLM integration** — LLaMA 3.3-70B via Groq with structured medical prompt
- [x] **REST API** — FastAPI backend with session management
- [x] **Frontend** — Next.js 14 + Tailwind + Recharts dashboard
- [x] **Visualisation** — reranker score bar chart + source chunk tabs
- [x] **Documentation** — README, inline comments, and docstrings throughout
- [x] **Version control** — organised as a GitHub repository with `uv` for dependency management

---

## ⚠️ Important Notes

- `node_modules/` and `.next/` are git-ignored — run `npm install` after cloning
- `.env` is git-ignored — never commit API keys
- Milvus Docker container must be running before starting the backend
- **Re-indexing required** if you previously used `BAAI/bge-base-en-v1.5` — MedCPT vectors are incompatible with bge vectors. Set `drop_old=True` in `initialize_chromadb()` once, re-run, then set it back to `False`
- Do **not** name any file `datasets.py` in the `src/` folder — it shadows the HuggingFace `datasets` library and causes a circular import. The dataset indexer is named `medical_datasets.py`
- CSV indexing is idempotent — safe to restart the server anytime
- In-memory sessions reset on server restart (by design)

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It demonstrates a full RAG pipeline on medical data and is **not a substitute for professional medical advice**.
