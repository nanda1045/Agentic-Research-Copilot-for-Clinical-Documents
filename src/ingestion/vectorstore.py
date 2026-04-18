"""
FAISS vector store management.
Uses HuggingFace sentence-transformers for embeddings (FREE — no API cost).
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_MODEL, FAISS_INDEX_DIR


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace embedding model.
    Uses all-MiniLM-L6-v2: fast, lightweight, and free.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector store from document chunks.
    
    Args:
        chunks: List of chunked Document objects
        
    Returns:
        FAISS vector store instance
    """
    embeddings = get_embeddings()
    print(f"  🧮 Building FAISS index with {len(chunks)} chunks...")
    print(f"     Embedding model: {EMBEDDING_MODEL}")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"  ✅ FAISS index built successfully!")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Optional[str] = None) -> str:
    """Save FAISS index to disk."""
    save_path = path or str(FAISS_INDEX_DIR)
    vectorstore.save_local(save_path)
    print(f"  💾 FAISS index saved to: {save_path}")
    return save_path


def load_vectorstore(path: Optional[str] = None) -> FAISS:
    """Load FAISS index from disk."""
    load_path = path or str(FAISS_INDEX_DIR)
    
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at: {load_path}. "
            "Run ingestion first: python main.py ingest"
        )
    
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"  📂 FAISS index loaded from: {load_path}")
    return vectorstore
