"""
LangGraph agentic graph construction.
Builds the multi-node RAG pipeline with conditional routing for:
  - Multi-hop retrieval loops
  - Confidence-based abstention
  - Grounded answer generation

Graph Flow:
  START → retrieve → verify → route_after_verify
    ├─ needs_more_evidence → reformulate → retrieve (loop)
    └─ has_enough → contradict → route_after_contradict
                      ├─ low confidence → abstain → END
                      └─ sufficient confidence → answer → END
"""

import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agents.state import AgentState
from src.agents.nodes import (
    retrieve_node,
    verify_node,
    contradict_node,
    abstain_node,
    answer_node,
    reformulate_query_node,
)
from config.settings import CONFIDENCE_THRESHOLD, MAX_HOPS


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_verify(state: AgentState) -> str:
    """Decide whether to retrieve more evidence or proceed to contradiction check."""
    needs_more = state.get("needs_more_evidence", False)
    hop_count = state.get("hop_count", 0)
    should_abstain = state.get("should_abstain", False)
    
    if should_abstain and hop_count >= MAX_HOPS:
        return "abstain"
    
    if needs_more and hop_count < MAX_HOPS:
        return "reformulate"
    
    return "contradict"


def route_after_contradict(state: AgentState) -> str:
    """Decide whether to abstain or generate an answer based on confidence."""
    confidence = state.get("confidence_score", 0.0)
    should_abstain = state.get("should_abstain", False)
    
    if should_abstain or confidence < CONFIDENCE_THRESHOLD:
        return "abstain"
    
    return "answer"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph() -> StateGraph:
    """
    Build and compile the agentic RAG graph.
    
    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # --- Add Nodes ---
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("contradict", contradict_node)
    workflow.add_node("reformulate", reformulate_query_node)
    workflow.add_node("abstain", abstain_node)
    workflow.add_node("answer", answer_node)
    
    # --- Add Edges ---
    # Entry point
    workflow.add_edge(START, "retrieve")
    
    # Retrieve → Verify (always)
    workflow.add_edge("retrieve", "verify")
    
    # Verify → conditional routing
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "reformulate": "reformulate",
            "contradict": "contradict",
            "abstain": "abstain",
        },
    )
    
    # Reformulate → Retrieve (multi-hop loop)
    workflow.add_edge("reformulate", "retrieve")
    
    # Contradict → conditional routing
    workflow.add_conditional_edges(
        "contradict",
        route_after_contradict,
        {
            "abstain": "abstain",
            "answer": "answer",
        },
    )
    
    # Terminal nodes
    workflow.add_edge("abstain", END)
    workflow.add_edge("answer", END)
    
    # Compile the graph
    compiled = workflow.compile()
    
    return compiled


def run_agent(query: str) -> dict:
    """
    Run the full agentic RAG pipeline on a query.
    
    Args:
        query: User's clinical research question
        
    Returns:
        Final agent state with answer, citations, reasoning trace
    """
    graph = build_graph()
    
    # Initialize state
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "retrieval_scores": [],
        "verified_docs": [],
        "verification_details": [],
        "contradictions": [],
        "has_contradictions": False,
        "confidence_score": 0.0,
        "should_abstain": False,
        "needs_more_evidence": False,
        "hop_count": 0,
        "reformulated_query": "",
        "answer": "",
        "citations": [],
        "reasoning_trace": [],
    }
    
    # Execute the graph
    final_state = graph.invoke(initial_state)
    
    return final_state


# ============================================================
# GRAPH VISUALIZATION (optional utility)
# ============================================================

def get_graph_mermaid() -> str:
    """Get Mermaid diagram representation of the graph."""
    graph = build_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        return """
graph TD
    START --> retrieve
    retrieve --> verify
    verify -->|needs more evidence| reformulate
    verify -->|enough evidence| contradict
    verify -->|should abstain| abstain
    reformulate --> retrieve
    contradict -->|low confidence| abstain
    contradict -->|sufficient confidence| answer
    abstain --> END
    answer --> END
"""
