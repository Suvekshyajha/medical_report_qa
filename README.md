# 🩺 Medical Report Q&A System

A RAG-based web application that lets patients and clinicians upload PDF medical reports and ask natural-language questions about them. Built as a capstone project covering the full pipeline — from document ingestion to a live Streamlit deployment.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-1C3C3C?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6.3-orange?style=flat)
![Groq](https://img.shields.io/badge/LLaMA_3.3_70B-Groq-F55036?style=flat)

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
├── app.py               # Streamlit UI — entry point
├── rag_pipeline.py      # Embedding, indexing, retrieval
├── llm_answer.py        # Prompt building, Groq API call
├── utils.py             # CSV/PDF loading, text cleaning, chart
├── style.css            # Custom Streamlit theme
├── requirements.txt     # Python dependencies
├── .env                 # Your API key (not committed)
├── .env.example         # API key template (committed)
├── data/
│   └── mtsamples.csv    # Download from Kaggle (see Setup)
└── vectorstore/         # Created automatically on first run
    ├── background/      # MTSamples ChromaDB collection
    └── pdf/             # Uploaded PDFs ChromaDB collection
```

---

## 🚀 Setup

### 1. Prerequisites

- Python 3.10 or later
- A free Groq API key — https://console.groq.com
- `mtsamples.csv` from Kaggle — search "medicaltranscriptions"

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add the dataset

Place the downloaded CSV at:

```
data/mtsamples.csv
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

> ⏳ **First run only:** embedding the full MTSamples dataset (~4,000 transcripts) takes **3–7 minutes** on CPU. Every subsequent start is instant — vectors are cached in `vectorstore/background/`.

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
| 1 | Query the **PDF store** first (the uploaded document) |
| 2 | Convert ChromaDB L2 distances to `[0,1]` similarity: `score = 1 / (1 + distance)` |
| 3 | If 3+ PDF chunks score above `0.3`, return those immediately |
| 4 | Otherwise also query the **background KB** and merge, deduplicated by content hash |
| 5 | Return top-5 chunks sorted by descending score |

---

## ✅ Project Checklist

- [x] **Data ingestion** — MTSamples CSV loading, PDF text extraction via PyMuPDF
- [x] **Text cleaning** — unicode normalisation, whitespace collapse, control-character removal
- [x] **Chunking** — RecursiveCharacterTextSplitter (500 chars / 100 overlap)
- [x] **Embedding** — BAAI/bge-base-en-v1.5 via HuggingFace (CPU, normalised)
- [x] **Vector storage** — two persistent ChromaDB collections (background + PDF)
- [x] **Retrieval** — priority cascade with relevance threshold and deduplication
- [x] **LLM integration** — LLaMA 3.3-70B via Groq with structured medical prompt
- [x] **Deployment** — Streamlit app in `app.py`
- [x] **Visualisation** — Plotly relevance score bar chart + source chunk expanders
- [x] **Documentation** — README, inline comments, and docstrings throughout
- [x] **Version control** — organised as a GitHub repository

---

## ⚙️ Configuration

| Variable | Location | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` | Groq API key |
| `EMBED_MODEL` | `rag_pipeline.py` | HuggingFace embedding model name |
| `chunk_size` | `rag_pipeline.py` | Characters per chunk (default: 500) |
| `chunk_overlap` | `rag_pipeline.py` | Overlap between chunks (default: 100) |
| `RELEVANCE_THRESHOLD` | `rag_pipeline.py` | Minimum similarity score (default: 0.3) |
| `temperature` | `llm_answer.py` | LLM temperature (default: 0.2) |
| `max_tokens` | `llm_answer.py` | Max response tokens (default: 1024) |

---

## 🧠 Design Notes

- **Dual vector store** — separating background KB from user PDFs gives clean isolation and ensures the user's own document always takes retrieval priority.
- **L2 → similarity conversion** — `1 / (1 + distance)` produces better score spread than linear subtraction.
- **Strict prompt** — the LLM is forbidden from using general knowledge, keeping answers grounded and verifiable.
- **Simulated streaming** — answers are split word-by-word with a 15 ms delay for a real-time feel without async infrastructure.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It demonstrates a full RAG pipeline on medical data and is **not a substitute for professional medical advice**.
