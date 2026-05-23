import os
import logging
import streamlit as st
from dotenv import load_dotenv

from ingest import ingest, vector_store_exists
from retriever import retrieve_relevant_chunks, format_context
from llm import get_answer

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Financial Report Intelligence",
    page_icon="",
    layout="wide",
)

st.title("Financial Report Intelligence")
st.caption(
    "Query Infosys, TCS, and Reliance FY2024 annual reports "
    "using natural language — powered by RAG and Mistral LLM"
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses **Retrieval Augmented Generation (RAG)** to answer
    questions about corporate annual reports.

    **How it works:**
    1. PDFs are chunked into overlapping segments
    2. Each chunk is embedded using sentence-transformers
    3. Your question is matched to the most relevant chunks
    4. Mistral LLM generates an answer grounded in the source text

    **Documents loaded:**
    - Infosys FY2024 Annual Report
    - TCS FY2024 Annual Report
    - Reliance Industries FY2024 Annual Report
    """)

    st.divider()
    k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)
    show_sources = st.checkbox("Show source chunks", value=True)

    st.divider()
    if st.button("Rebuild Vector Store", type="secondary"):
        with st.spinner("Rebuilding..."):
            st.session_state.pop("vector_store", None)
            ingest(force_rebuild=True)
        st.success("Vector store rebuilt")

# ── Load vector store ─────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    with st.spinner("Loading document index..."):
        try:
            from ingest import load_vector_store, get_embeddings
            embeddings = get_embeddings()
            st.session_state["vector_store"] = load_vector_store(embeddings)
            st.success("Documents loaded and ready")
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")
            st.stop()

vector_store = st.session_state["vector_store"]

# ── Sample questions ──────────────────────────────────────────────────
st.subheader("Sample Questions")
sample_questions = [
    "What was Infosys revenue in FY2024?",
    "How many employees does TCS have?",
    "What is Reliance Industries net profit for FY2024?",
    "What are the key risks mentioned in the Infosys annual report?",
    "What is TCS dividend per share?",
]

cols = st.columns(3)
for i, q in enumerate(sample_questions):
    if cols[i % 3].button(q, use_container_width=True):
        st.session_state["question"] = q

# ── Question input ────────────────────────────────────────────────────
st.subheader("Ask a Question")
question = st.text_input(
    "Type your question about any of the annual reports:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What was the operating margin of TCS in FY2024?",
)

if st.button("Get Answer", type="primary") and question:
    with st.spinner("Searching documents and generating answer..."):
        try:
            # Retrieve
            results = retrieve_relevant_chunks(question, vector_store, k=k)
            context, sources = format_context(results)

            # Generate
            answer = get_answer(question, context)

            # Display answer
            st.divider()
            st.subheader("Answer")
            st.markdown(answer)

            # Display sources
            if show_sources:
                st.divider()
                st.subheader("Source Chunks Used")
                for src in sources:
                    with st.expander(
                        f"Source {src['index']}: {src['company']} "
                        f"— Page {src['page']} "
                        f"(relevance: {src['score']})"
                    ):
                        st.caption(src["preview"])

        except Exception as e:
            st.error(f"Error generating answer: {e}")