"""
Document loader for clinical PDFs and text files.
Supports both PDF (via PyPDFLoader) and plain text files.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_dir: str) -> List[Document]:
    """
    Load all PDF and text files from the specified directory.
    
    Args:
        data_dir: Path to directory containing clinical documents
        
    Returns:
        List of Document objects with content and metadata
    """
    data_path = Path(data_dir)
    documents = []
    supported_extensions = {".pdf", ".txt", ".text"}
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = sorted(data_path.iterdir())
    loaded_count = 0
    
    for filepath in files:
        if filepath.suffix.lower() not in supported_extensions:
            continue
            
        try:
            if filepath.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
            else:
                loader = TextLoader(str(filepath), encoding="utf-8")
                docs = loader.load()
            
            # Enrich metadata
            for doc in docs:
                doc.metadata.update({
                    "source": filepath.name,
                    "file_type": filepath.suffix.lower(),
                    "doc_id": f"DOC-{loaded_count + 1:04d}",
                    "full_path": str(filepath),
                })
            
            documents.extend(docs)
            loaded_count += 1
            
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load {filepath.name}: {e}")
            continue
    
    print(f"  📄 Loaded {loaded_count} files → {len(documents)} document pages/sections")
    return documents
