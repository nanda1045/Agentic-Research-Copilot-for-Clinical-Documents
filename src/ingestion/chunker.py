"""
Text chunking for clinical documents.
Uses RecursiveCharacterTextSplitter for intelligent splitting that respects
document structure (paragraphs, sentences, etc.)
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks for embedding.
    
    Args:
        documents: List of loaded Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked Document objects with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_total"] = len(chunks)
    
    print(f"  🔪 Split {len(documents)} documents → {len(chunks)} chunks")
    print(f"     Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    return chunks
