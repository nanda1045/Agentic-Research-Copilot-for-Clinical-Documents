"""
Semantic similarity retrieval from the FAISS vector store.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TOP_K


def get_retriever(vectorstore: FAISS, k: int = TOP_K):
    """
    Create a retriever from the vector store.
    
    Args:
        vectorstore: FAISS vector store instance
        k: Number of documents to retrieve
        
    Returns:
        VectorStoreRetriever instance
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_documents(
    vectorstore: FAISS, query: str, k: int = TOP_K
) -> List[Document]:
    """
    Retrieve top-k documents similar to the query.
    
    Args:
        vectorstore: FAISS vector store instance
        query: User's search query
        k: Number of results to return
        
    Returns:
        List of relevant Document objects
    """
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def retrieve_with_scores(
    vectorstore: FAISS, query: str, k: int = TOP_K
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k documents with their similarity scores.
    
    Args:
        vectorstore: FAISS vector store instance
        query: User's search query
        k: Number of results to return
        
    Returns:
        List of (Document, score) tuples, sorted by relevance
    """
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    return docs_and_scores
