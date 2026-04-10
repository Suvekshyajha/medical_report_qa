# ============================================================
# utils.py
# PURPOSE: Helper toolbox used by all other files
#
# FUNCTION 1: load_csv_data()         → read mtsamples.csv
# FUNCTION 2: extract_text_from_pdf() → read uploaded PDF
# FUNCTION 3: clean_text()            → clean messy text
# FUNCTION 4: build_relevance_chart() → Plotly bar chart
# ============================================================

import re                              # built-in: text cleaning
import os                              # built-in: file/folder operations
import fitz                            # PyMuPDF: read PDF files
import pandas as pd                    # read CSV file
import plotly.graph_objects as go      # build interactive charts


# ---------------------------------------------------------------
# FUNCTION 1: Load CSV Dataset (MTSamples from Kaggle)
# ---------------------------------------------------------------
# What it does:
#   - Reads mtsamples.csv from data/ folder
#   - Combines medical_specialty + description + transcription
#     into one rich text per row
#   - Returns a list of strings ready for ChromaDB indexing
#
# Example:
#   Input  → "data/mtsamples.csv"
#   Output → ["Specialty: Cardiology\nDescription: Chest pain...",
#              "Specialty: Neurology\nDescription: Headache...",
#              ...]
# ---------------------------------------------------------------
def load_csv_data(csv_path: str = "data/mtsamples.csv") -> list:
    """
    Loads MTSamples CSV and returns list of combined text strings.

    Args:
        csv_path: path to mtsamples.csv file

    Returns:
        List of cleaned text strings (one per CSV row)
    """

    # Check if CSV file exists before trying to open
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found at: {csv_path}")
        print("   Please put mtsamples.csv inside the data/ folder")
        return []

    print(f"📂 Loading CSV from: {csv_path}")

    # Read the CSV into a pandas DataFrame
    # DataFrame is like an Excel table in Python
    df = pd.read_csv(csv_path)

    print(f"   ✅ Loaded {len(df)} rows")
    print(f"   Columns found: {list(df.columns)}")

    # ---------------------------------------------------------------
    # MTSamples CSV has these columns:
    #   description       → short title e.g. "Chest Pain - ER Visit"
    #   medical_specialty → e.g. "Cardiology", "Surgery"
    #   sample_name       → name of the sample
    #   transcription     → full doctor notes (MOST IMPORTANT)
    #   keywords          → tags e.g. "chest, pain, ecg"
    # ---------------------------------------------------------------

    # Combine 3 columns into one rich text per row
    # This gives ChromaDB more context when searching
    texts = []

    for index, row in df.iterrows():

        # Safely get each column value
        # fillna("") means if the value is empty, use empty string
        specialty    = str(row.get("medical_specialty", "")).strip()
        description  = str(row.get("description", "")).strip()
        transcription = str(row.get("transcription", "")).strip()
        keywords     = str(row.get("keywords", "")).strip()

        # Skip rows where transcription is empty or NaN
        if not transcription or transcription == "nan":
            continue

        # Build combined text string for this row
        combined = (
            f"Specialty: {specialty}\n"
            f"Description: {description}\n"
            f"Keywords: {keywords}\n"
            f"Transcription: {transcription}"
        )

        # Clean the text and add to list
        texts.append(clean_text(combined))

    print(f"   ✅ {len(texts)} valid rows ready for ChromaDB\n")
    return texts


# ---------------------------------------------------------------
# FUNCTION 2: Extract Text from PDF
# ---------------------------------------------------------------
# What it does:
#   - Takes a PDF file path OR uploaded file bytes
#   - Opens it using PyMuPDF (fitz)
#   - Loops through every page
#   - Extracts all text from each page
#   - Returns one clean string of all text
#
# Example:
#   Input  → "data/sample_reports/lab_report.pdf"
#   Output → "Patient: John Doe\nAge: 45\nBlood Pressure: 120/80..."
# ---------------------------------------------------------------
def extract_text_from_pdf(pdf_source) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_source: either a file path (string)
                    or uploaded file bytes (from Streamlit)

    Returns:
        Single clean string of all text from PDF
    """

    # PyMuPDF can open both file paths and raw bytes
    # This handles both cases:
    #   - file path  → fitz.open("path/to/file.pdf")
    #   - bytes      → fitz.open(stream=bytes, filetype="pdf")
    if isinstance(pdf_source, str):
        # It's a file path
        if not os.path.exists(pdf_source):
            print(f"❌ PDF not found: {pdf_source}")
            return ""
        doc = fitz.open(pdf_source)
    else:
        # It's bytes from Streamlit file uploader
        doc = fitz.open(stream=pdf_source, filetype="pdf")

    full_text = []  # collect text from each page

    # Loop through every page
    for page_num, page in enumerate(doc):

        # Extract plain text from this page
        page_text = page.get_text("text")

        # Only keep pages with actual content
        if page_text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{page_text}")

    doc.close()  # free memory

    # Join all pages and clean
    combined = "\n".join(full_text)
    return clean_text(combined)


# ---------------------------------------------------------------
# FUNCTION 3: Clean Text
# ---------------------------------------------------------------
# What it does:
#   - Removes weird characters from PDFs
#   - Fixes multiple spaces → single space
#   - Fixes multiple blank lines → single newline
#   - Strips leading/trailing whitespace
#
# Example:
#   Input  → "Patient   Name:  John\n\n\n\nAge:  45  "
#   Output → "Patient Name: John\nAge: 45"
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans raw text from PDF or CSV.

    Args:
        text: raw messy text string

    Returns:
        Clean normalized text string
    """

    if not text:
        return ""

    # Normalize unicode (e.g. curly quotes → straight, ligatures → letters)
    # but preserve medical characters like µg, °C, ≥, ≤
    import unicodedata
    text = unicodedata.normalize("NFKC", text)

    # Remove non-printable control characters (keep newlines and tabs)
    text = re.sub(r'[^\S\n\t ]+', ' ', text)   # collapse weird whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # strip control chars

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)

    # Remove lines that are just whitespace or dashes
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line and line != '-']

    # Join back and strip
    return '\n'.join(lines).strip()


# ---------------------------------------------------------------
# FUNCTION 4: Build Relevance Score Chart
# ---------------------------------------------------------------
# What it does:
#   - Takes top-5 chunk scores from ChromaDB retrieval
#   - Builds a colorful Plotly bar chart
#   - Green bar  = very relevant (score > 0.8)
#   - Orange bar = somewhat relevant (score > 0.6)
#   - Red bar    = low relevance (score < 0.6)
#   - This chart is shown in Streamlit dashboard
#
# Example:
#   Input  → scores=[0.92, 0.87, 0.75, 0.65, 0.58]
#             labels=["Chunk 1","Chunk 2","Chunk 3","Chunk 4","Chunk 5"]
#   Output → Plotly bar chart object
# ---------------------------------------------------------------
def build_relevance_chart(scores: list, labels: list):
    """
    Builds Plotly bar chart of relevance scores.

    Args:
        scores: list of float scores  e.g. [0.92, 0.87, 0.75]
        labels: list of chunk labels  e.g. ["Chunk 1", "Chunk 2"]

    Returns:
        Plotly Figure object
        → used in app.py as: st.plotly_chart(fig)
    """

    if not scores or not labels:
        # Return empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="📊 Relevance Scores — No data",
            height=400
        )
        return fig

    # Assign color based on score value
    colors = []
    for s in scores:
        if s > 0.8:
            colors.append("#2ecc71")   # green  = highly relevant
        elif s > 0.6:
            colors.append("#f39c12")   # orange = somewhat relevant
        else:
            colors.append("#e74c3c")   # red    = low relevance

    # Build the bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,                           # X axis: chunk labels
                y=scores,                           # Y axis: scores
                marker_color=colors,                # bar colors
                text=[f"{s:.2f}" for s in scores],  # score label on bar
                textposition="auto"
            )
        ]
    )

    # Style the chart
    fig.update_layout(
        title="📊 Relevance Scores — Top Retrieved Chunks",
        xaxis_title="Retrieved Chunks",
        yaxis_title="Relevance Score (0 to 1)",
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
        height=400,
        font=dict(size=13)
    )

    return fig


# ---------------------------------------------------------------
# TEST — run this file directly to verify everything works
# Command: python utils.py
# ---------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("TESTING utils.py")
    print("=" * 50)

    # Test 1: Load CSV
    print("\n📋 TEST 1: Load CSV")
    texts = load_csv_data("data/mtsamples.csv")
    if texts:
        print(f"✅ First entry preview:\n{texts[0][:300]}\n")

    # Test 2: Extract PDF (if any PDF exists)
    print("\n📄 TEST 2: Extract PDF")
    pdf_dir = "data/sample_reports"
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")] if os.path.exists(pdf_dir) else []
    if pdfs:
        pdf_path = os.path.join(pdf_dir, pdfs[0])
        text = extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters")
        print(f"Preview:\n{text[:300]}\n")
    else:
        print("⚠️  No PDFs in data/sample_reports/ — skipping PDF test")

    # Test 3: Clean text
    print("\n🧹 TEST 3: Clean Text")
    dirty = "Patient   Name:  John\n\n\n\nAge:  45  "
    clean = clean_text(dirty)
    print(f"Before: {repr(dirty)}")
    print(f"After:  {repr(clean)}")
    print("✅ Clean text works!")

    print("\n" + "=" * 50)
    print("✅ utils.py ALL TESTS DONE!")
    print("=" * 50)