import os
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
DATA_PATH   = "."

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Load sentence-transformer embedding model. Runs locally, no API needed."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdfs(data_path: str = DATA_PATH) -> list:
    """Load all PDFs from the data directory."""
    documents = []
    pdf_files = list(Path(data_path).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path}/")

    for pdf_path in pdf_files:
        logger.info("Loading %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Tag each page with source company name
        company = pdf_path.stem.replace("_", " ").title()
        for doc in docs:
            doc.metadata["company"] = company
            doc.metadata["source_file"] = pdf_path.name

        documents.extend(docs)
        logger.info("Loaded %d pages from %s", len(docs), pdf_path.name)

    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d pages", len(chunks), len(documents))
    return chunks


def build_vector_store(chunks: list, embeddings) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    logger.info("Building vector store with %d chunks...", len(chunks))
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    logger.info("Vector store built and persisted to %s", CHROMA_PATH)
    return vector_store


def load_vector_store(embeddings) -> Chroma:
    """Load existing ChromaDB vector store from disk."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )


def vector_store_exists() -> bool:
    """Check if ChromaDB has already been built."""
    return Path(CHROMA_PATH).exists() and any(Path(CHROMA_PATH).iterdir())


def ingest(force_rebuild: bool = False):
    """
    Main ingestion function.
    Builds vector store from PDFs if not already built.
    Set force_rebuild=True to re-ingest after adding new PDFs.
    """
    embeddings = get_embeddings()

    if vector_store_exists() and not force_rebuild:
        logger.info("Vector store already exists — loading from disk")
        return load_vector_store(embeddings)

    logger.info("Building vector store from scratch...")
    documents = load_pdfs()
    chunks    = chunk_documents(documents)
    store     = build_vector_store(chunks, embeddings)
    return store