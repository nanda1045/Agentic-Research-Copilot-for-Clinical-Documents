"""
LangGraph agent state definition.
Defines the shared state that flows through all agent nodes.
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document
import operator


class AgentState(TypedDict):
    """State shared across all nodes in the agentic RAG graph."""
    
    # Input
    query: str                          # User's original question
    
    # Retrieval
    retrieved_docs: List[Document]      # Chunks from FAISS retrieval
    retrieval_scores: List[float]       # Similarity scores for retrieved docs
    
    # Verification
    verified_docs: List[Document]       # Chunks that passed relevance verification
    verification_details: List[dict]    # Per-chunk verification reasoning
    
    # Contradiction Detection
    contradictions: List[dict]          # Detected contradictions between chunks
    has_contradictions: bool            # Whether any contradictions were found
    
    # Confidence & Decision
    confidence_score: float             # Overall confidence (0.0 - 1.0)
    should_abstain: bool                # Whether to refuse answering
    needs_more_evidence: bool           # Whether to trigger multi-hop retrieval
    
    # Multi-hop
    hop_count: int                      # Current retrieval iteration
    reformulated_query: str             # Query rewritten for additional retrieval
    
    # Output
    answer: str                         # Final generated answer
    citations: List[str]               # Source citations
    reasoning_trace: List[str]         # Step-by-step reasoning log
