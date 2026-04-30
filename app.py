# ============================================================
# app.py
# PURPOSE: Streamlit dashboard for Medical Report Q&A System
# UI: Left panel (session history) + Right panel (upload + chat)
#
# RUN: streamlit run app.py
# ============================================================

import streamlit as st
from datetime import datetime

from rag_pipeline import initialize_chromadb, index_csv_data, index_pdf_document
from llm_answer import load_llm, get_answer
from utils import build_relevance_chart


# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Medical Report Q&A",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------

with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)





# ---------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------
def fmt_date():
    return datetime.now().strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")

if "sessions" not in st.session_state:
    st.session_state.sessions = [{
        "id": 0, "name": "New Session", "created": fmt_date(),
        "messages": [], "indexed_files": [], "last_result": None
    }]

if "active_session"   not in st.session_state: st.session_state.active_session   = 0
if "vectorstore_bg"   not in st.session_state: st.session_state.vectorstore_bg   = None
if "vectorstore_pdf"  not in st.session_state: st.session_state.vectorstore_pdf  = None
if "embeddings"       not in st.session_state: st.session_state.embeddings       = None
if "llm"              not in st.session_state: st.session_state.llm              = None
if "initialized"      not in st.session_state: st.session_state.initialized      = False


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def get_active():
    return st.session_state.sessions[st.session_state.active_session]

def add_session():
    new_id = len(st.session_state.sessions)
    st.session_state.sessions.append({
        "id": new_id, "name": "New Session", "created": fmt_date(),
        "messages": [], "indexed_files": [], "last_result": None
    })
    st.session_state.active_session = new_id
    st.rerun()


# ---------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_all_models():
    vectorstore_bg, vectorstore_pdf, embeddings = initialize_chromadb()
    try:
        index_csv_data(vectorstore_bg, embeddings)
    except (FileNotFoundError, ValueError) as e:
        st.warning(f"⚠️ CSV indexing skipped: {e}")
    llm = load_llm()
    return vectorstore_bg, vectorstore_pdf, llm


# ---------------------------------------------------------------
# TOP HEADER
# ---------------------------------------------------------------
st.markdown("""
<div class="top-header">
    <div class="logo-wrap">🩺</div>
    <div>
        <h1>Medical Report Q&amp;A</h1>
        <p>Upload your medical report and ask questions</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------
if not st.session_state.initialized:
    with st.spinner("⚙️ Loading models… first run may take a few minutes"):
        try:
            vs_bg, vs_pdf, llm = load_all_models()
            st.session_state.vectorstore_bg  = vs_bg
            st.session_state.vectorstore_pdf = vs_pdf
            st.session_state.llm             = llm
            st.session_state.initialized     = True
        except Exception as e:
            st.error(f"❌ Startup error: {e}")
            st.stop()


# ---------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------
left, right = st.columns([1, 2.6], gap="small")


# ── LEFT PANEL ──────────────────────────────────────────────────
with left:
    st.markdown('<div class="left-panel-header">Session History</div>', unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 12px;">', unsafe_allow_html=True)

    st.markdown('<div class="new-session-wrap">', unsafe_allow_html=True)
    if st.button("＋  New Session", key="new_session_btn", use_container_width=True):
        add_session()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="session-cards-wrap">', unsafe_allow_html=True)
    for i, sess in enumerate(st.session_state.sessions):
        label = sess["indexed_files"][0] if sess["indexed_files"] else "New Session"
        count = len(sess["messages"])
        btn_label = f"📄  {label}\n{sess['created']} • {count} message{'s' if count != 1 else ''}"
        if st.button(btn_label, key=f"sess_{i}", use_container_width=True):
            st.session_state.active_session = i
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close padding wrapper


# ── RIGHT PANEL ─────────────────────────────────────────────────
with right:
    session = get_active()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Upload Medical Report</div>', unsafe_allow_html=True)

    # Decorative top of upload zone
    st.markdown("""
    <div class="upload-zone">
        <span class="upicon">⬆️</span>
        <p class="up-text">Drag and drop your medical report PDF here</p>
        <p class="up-or">or</p>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit uploader — visually stitched below the dashed box
    uploaded_file = st.file_uploader(
        label="pdf", type=["pdf"],
        key=f"uploader_{session['id']}",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if uploaded_file.name not in session["indexed_files"]:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                try:
                    pdf_bytes  = uploaded_file.read()
                    num_chunks = index_pdf_document(
                        pdf_source=pdf_bytes,
                        vectorstore_pdf=st.session_state.vectorstore_pdf,
                        filename=uploaded_file.name
                    )
                    session["indexed_files"].append(uploaded_file.name)
                    session["name"] = uploaded_file.name
                    st.success(f"✅ '{uploaded_file.name}' indexed — {num_chunks} chunks added.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to index: {e}")
        else:
            st.info(f"ℹ️ '{uploaded_file.name}' is already indexed in this session.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── CONVERSATION ──
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    chat_box = st.container(height=200)
    with chat_box:
        if not session["messages"]:
            st.markdown("""
            <div class="chat-empty">
                <span class="ce-icon">💬</span>
                <p>Upload a report and ask any questions.<br>
                I'll help you understand your medical information.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in session["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    question = st.chat_input(
        placeholder="Ask a question about your report…",
        disabled=not st.session_state.initialized
    )

    # In app.py, replace the get_answer call with streaming
    if question:
        session["messages"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                # Stream the response
                result = get_answer(
                    question=question,
                    vectorstore_bg=st.session_state.vectorstore_bg,
                    vectorstore_pdf=st.session_state.vectorstore_pdf,
                    llm=st.session_state.llm
                )
                # Simulate streaming from result for better UX
                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    placeholder.markdown(full_response + "▌")
                    import time;

                    time.sleep(0.015)
                placeholder.markdown(full_response)

                session["messages"].append({"role": "assistant", "content": full_response})
                session["last_result"] = result
            except Exception as e:
                session["messages"].append({"role": "assistant", "content": f"❌ Error: {e}"})
        st.rerun()

    # ── CHART + SOURCES ──
    if session["last_result"]:
        result = session["last_result"]
        st.divider()

        tab1, tab2 = st.tabs(["📊  Relevance Scores", "📄  Source Chunks"])

        with tab1:
            if result.get("scores"):
                fig = build_relevance_chart(result["scores"], result["labels"])
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if result.get("sources"):
                for i, (src, score) in enumerate(zip(result["sources"], result["scores"])):
                    badge = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                    with st.expander(f"{badge}  Chunk {i+1}  ·  Score: {score:.3f}"):
                        st.markdown(
                            f"<div style='font-size:13.5px; line-height:1.7; "
                            f"color:#374151; font-family:DM Sans,sans-serif'>{src}</div>",
                            unsafe_allow_html=True
                        )