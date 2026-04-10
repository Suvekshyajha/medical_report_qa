
🩺 Medical Report Q&A System

README


A Retrieval-Augmented Generation (RAG) application that lets patients and clinicians upload PDF medical reports and ask natural-language questions about them. Answers are grounded in the uploaded document and backed by a medical knowledge base derived from the MTSamples dataset.

✨ Features

•	PDF upload & indexing — drag-and-drop any medical PDF; it is chunked and embedded instantly.
•	Dual knowledge base — your PDF takes priority; the MTSamples background KB fills any gaps.
•	LLaMA 3.3-70B answers — fast, precise responses via the Groq API.
•	Source citations — every answer references which chunk it drew from.
•	Relevance scoring — interactive Plotly bar chart shows how relevant each retrieved chunk was.
•	Multi-session history — left-panel session cards let you switch between past conversations.
•	Strict medical prompt — the LLM is instructed never to answer from general knowledge, reducing hallucination risk.

🏗️ Architecture


<img width="701" height="553" alt="image" src="https://github.com/user-attachments/assets/0fa5d616-c822-4400-956e-ef5e05b70bdd" />



<img width="687" height="261" alt="image" src="https://github.com/user-attachments/assets/3adfd5d5-a485-4bfa-87af-13259d149585" />


📁 Project Structure


<img width="648" height="218" alt="image" src="https://github.com/user-attachments/assets/8d530afc-2f98-4752-9b79-e2d503e6938d" />


🚀 Setup

1. Prerequisites
•	Python 3.10 or later
•	A free Groq API key — https://console.groq.com
•	mtsamples.csv from Kaggle — search "medicaltranscriptions"

2. Clone & install
git clone https://github.com/Suvekshyajha/medical_report_qa.git
cd medical-qa
pip install -r requirements.txt

3. Add your API key
Create a .env file in the project root:

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

4. Add the dataset
Place the downloaded CSV at:

data/mtsamples.csv

5. Run
streamlit run app.py

The app opens at http://localhost:8501.

⏳  First run only: embedding the full MTSamples dataset (~4,000 transcripts) takes 3–7 minutes on CPU. Every subsequent start is instant — the vectors are cached in vectorstore/background/.

🖥️ Usage

1.	Click  ＋ New Session  in the left panel.
2.	Drag and drop a medical PDF (lab report, discharge summary, etc.) into the upload zone.
3.	Wait for the indexing confirmation, then type a question in the chat input.
4.	Review the answer, then expand the Relevance Scores or Source Chunks tabs to verify what the model used.

🔍 How Retrieval Works

Retrieval follows a priority cascade:

5.	Query the PDF store first (the document you uploaded).
6.	Convert raw ChromaDB L2 distances to [0, 1] similarity:  score = 1 / (1 + distance).
7.	If 3 or more PDF chunks score above 0.3, return those immediately.
8.	Otherwise, also query the background KB and merge results, deduplicated by content hash.
9.	Return the top-5 chunks sorted by descending score.

This ensures your uploaded report always takes priority while the background knowledge base fills gaps when the PDF lacks relevant content.

🧪 Testing Modules Independently

Each backend module has a built-in test block:

# Test helpers (CSV loading, PDF extraction, text cleaning, chart)
python utils.py

# Test RAG pipeline (embedding init, CSV indexing, retrieval)
python rag_pipeline.py

# Test full pipeline end-to-end (retrieval → Groq → answer)
python llm_answer.py

⚙️ Configuration


Variable	Location	Description
GROQ_API_KEY	.env	Groq API key
EMBED_MODEL	rag_pipeline.py	HuggingFace embedding model name
chunk_size	rag_pipeline.py	Characters per chunk (default: 500)
chunk_overlap	rag_pipeline.py	Overlap between chunks (default: 100)
RELEVANCE_THRESHOLD	rag_pipeline.py	Minimum similarity score (default: 0.3)
temperature	llm_answer.py	LLM temperature (default: 0.2)
max_tokens	llm_answer.py	Max response tokens (default: 1024)

⚠️ Limitations

•	The PDF vector store is shared across all sessions — not suitable for multi-user production deployments without namespacing.
•	Streaming is simulated (word-split with sleep delay); the full LLM response must arrive before display begins.
•	No authentication or document access control is implemented.
•	The system is designed for informational purposes only and should not replace professional medical advice.

🗺️ Roadmap

☐	Per-session ChromaDB namespacing for true isolation
☐	Real token streaming via Groq's native streaming API
☐	Cross-encoder re-ranking for improved retrieval quality
☐	RAGAS evaluation pipeline (faithfulness, answer relevancy, context recall)
☐	Document management panel (view / delete indexed PDFs)





