"""
Agent nodes for the Agentic RAG pipeline.
Each node is a function that takes AgentState and returns state updates.

Nodes:
  1. retrieve_node     — Fetch relevant chunks from FAISS
  2. verify_node       — LLM judges chunk relevance
  3. contradict_node   — Detect conflicting evidence
  4. abstain_node      — Refuse when confidence is too low
  5. answer_node       — Generate grounded answer with citations
"""

import json
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    VERIFICATION_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_HOPS, TOP_K,
)
from src.agents.state import AgentState
from src.ingestion.vectorstore import load_vectorstore
from src.retrieval.retriever import retrieve_documents


def _get_llm():
    """Initialize Claude LLM (Haiku for cost efficiency)."""
    return ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


def _format_docs(docs: List[Document]) -> str:
    """Format documents into a numbered context string."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        formatted.append(
            f"[Chunk {i+1} | Source: {source} | ChunkID: {chunk_id}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# ============================================================
# NODE 1: RETRIEVE
# ============================================================

def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve relevant document chunks from the FAISS vector store.
    On multi-hop iterations, uses a reformulated query.
    """
    hop_count = state.get("hop_count", 0)
    query = state.get("reformulated_query", state["query"]) if hop_count > 0 else state["query"]
    
    trace = list(state.get("reasoning_trace", []))
    trace.append(f"[Retrieve] Hop {hop_count + 1}: Searching for '{query}'")
    
    try:
        vectorstore = load_vectorstore()
        docs = retrieve_documents(vectorstore, query, k=TOP_K)
        
        # Merge with previously retrieved docs (for multi-hop)
        existing_docs = list(state.get("retrieved_docs", []))
        seen_contents = {d.page_content for d in existing_docs}
        
        new_docs = []
        for doc in docs:
            if doc.page_content not in seen_contents:
                new_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        all_docs = existing_docs + new_docs
        trace.append(f"[Retrieve] Found {len(new_docs)} new chunks (total: {len(all_docs)})")
        
        return {
            "retrieved_docs": all_docs,
            "hop_count": hop_count + 1,
            "reasoning_trace": trace,
        }
    except Exception as e:
        trace.append(f"[Retrieve] ERROR: {str(e)}")
        return {
            "retrieved_docs": state.get("retrieved_docs", []),
            "hop_count": hop_count + 1,
            "reasoning_trace": trace,
        }


# ============================================================
# NODE 2: VERIFY
# ============================================================

def verify_node(state: AgentState) -> dict:
    """
    Verify that each retrieved chunk actually supports answering the query.
    Uses LLM-as-judge to score relevance.
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    trace = list(state.get("reasoning_trace", []))
    
    if not docs:
        trace.append("[Verify] No documents to verify.")
        return {
            "verified_docs": [],
            "verification_details": [],
            "confidence_score": 0.0,
            "should_abstain": True,
            "needs_more_evidence": state.get("hop_count", 0) < MAX_HOPS,
            "reasoning_trace": trace,
        }
    
    llm = _get_llm()
    verified_docs = []
    verification_details = []
    
    # Batch all chunks into a single LLM call to save API cost
    chunks_text = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        chunks_text += f"\n\n--- CHUNK {i+1} (Source: {source}) ---\n{doc.page_content}"
    
    prompt = f"""You are a clinical document relevance evaluator. Given a question and multiple document chunks, evaluate EACH chunk's relevance to answering the question.

QUESTION: {query}

CHUNKS:{chunks_text}

For each chunk, respond with a JSON array where each element has:
- "chunk_index": the chunk number (1-based)
- "relevant": true/false
- "score": relevance score from 0.0 to 1.0
- "reason": brief explanation (1 sentence)

Respond ONLY with the JSON array, no other text."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse LLM response
        response_text = response.content.strip()
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        evaluations = json.loads(response_text)
        
        for eval_item in evaluations:
            idx = eval_item.get("chunk_index", 0) - 1
            score = eval_item.get("score", 0.0)
            relevant = eval_item.get("relevant", False)
            reason = eval_item.get("reason", "")
            
            if 0 <= idx < len(docs):
                verification_details.append({
                    "chunk_index": idx,
                    "source": docs[idx].metadata.get("source", "Unknown"),
                    "score": score,
                    "relevant": relevant,
                    "reason": reason,
                })
                
                if score >= VERIFICATION_THRESHOLD and relevant:
                    verified_docs.append(docs[idx])
        
    except (json.JSONDecodeError, Exception) as e:
        trace.append(f"[Verify] LLM parsing error: {str(e)}. Keeping all docs.")
        verified_docs = docs
        verification_details = [
            {"chunk_index": i, "source": d.metadata.get("source", ""), 
             "score": 0.6, "relevant": True, "reason": "Fallback: kept due to parsing error"}
            for i, d in enumerate(docs)
        ]
    
    # Calculate confidence based on verification
    if verification_details:
        # Use scores from VERIFIED docs only (not all docs)
        verified_scores = [v["score"] for v in verification_details if v["relevant"]]
        if verified_scores:
            avg_score = sum(verified_scores) / len(verified_scores)
        else:
            avg_score = 0.0
        verified_ratio = len(verified_docs) / len(docs)
        # Weight: 70% on avg quality of verified chunks, 30% on coverage
        confidence = min(1.0, avg_score * 0.7 + verified_ratio * 0.3 + 0.1)
    else:
        confidence = 0.0
    
    needs_more = (
        len(verified_docs) < 2 
        and state.get("hop_count", 0) < MAX_HOPS
    )
    
    trace.append(
        f"[Verify] {len(verified_docs)}/{len(docs)} chunks verified "
        f"(confidence: {confidence:.2f}, needs_more: {needs_more})"
    )
    
    return {
        "verified_docs": verified_docs,
        "verification_details": verification_details,
        "confidence_score": confidence,
        "needs_more_evidence": needs_more,
        "should_abstain": confidence < CONFIDENCE_THRESHOLD and not needs_more,
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 3: CONTRADICT
# ============================================================

def contradict_node(state: AgentState) -> dict:
    """
    Check for contradictions between verified document chunks.
    Flags conflicting evidence and adjusts confidence accordingly.
    """
    verified_docs = state.get("verified_docs", [])
    trace = list(state.get("reasoning_trace", []))
    confidence = state.get("confidence_score", 0.5)
    
    if len(verified_docs) < 2:
        trace.append("[Contradict] Not enough chunks to check for contradictions.")
        return {
            "contradictions": [],
            "has_contradictions": False,
            "reasoning_trace": trace,
        }
    
    llm = _get_llm()
    context = _format_docs(verified_docs)
    
    prompt = f"""You are a clinical evidence contradiction detector. Analyze the following document chunks and identify ANY contradictory claims between them.

QUESTION CONTEXT: {state['query']}

DOCUMENT CHUNKS:
{context}

Look for:
1. Contradictory efficacy findings (one says drug works, another says it doesn't)
2. Conflicting safety data (different adverse event rates, conflicting safety conclusions)
3. Opposing recommendations or conclusions
4. Inconsistent statistical results for the same drug/condition pair

Respond with a JSON array of contradictions found. Each element should have:
- "chunk_a": first chunk number involved
- "chunk_b": second chunk number involved  
- "claim_a": the claim from chunk A
- "claim_b": the contradicting claim from chunk B
- "severity": "high", "medium", or "low"
- "explanation": brief explanation of the contradiction

If NO contradictions are found, respond with an empty array: []

Respond ONLY with the JSON array."""

    contradictions = []
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        contradictions = json.loads(response_text)
        
    except (json.JSONDecodeError, Exception) as e:
        trace.append(f"[Contradict] Parsing error: {str(e)}. Assuming no contradictions.")
        contradictions = []
    
    has_contradictions = len(contradictions) > 0
    
    # Adjust confidence if contradictions found
    if has_contradictions:
        high_count = sum(1 for c in contradictions if c.get("severity") == "high")
        med_count = sum(1 for c in contradictions if c.get("severity") == "medium")
        penalty = high_count * 0.15 + med_count * 0.08
        confidence = max(0.0, confidence - penalty)
        trace.append(
            f"[Contradict] Found {len(contradictions)} contradictions "
            f"({high_count} high, {med_count} medium). "
            f"Confidence adjusted to {confidence:.2f}"
        )
    else:
        trace.append("[Contradict] No contradictions detected.")
    
    return {
        "contradictions": contradictions,
        "has_contradictions": has_contradictions,
        "confidence_score": confidence,
        "should_abstain": confidence < CONFIDENCE_THRESHOLD and not state.get("needs_more_evidence", False),
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 4: ABSTAIN
# ============================================================

def abstain_node(state: AgentState) -> dict:
    """
    Generate an abstention response when confidence is too low.
    Explains why the system cannot provide a reliable answer.
    """
    trace = list(state.get("reasoning_trace", []))
    verified_docs = state.get("verified_docs", [])
    contradictions = state.get("contradictions", [])
    confidence = state.get("confidence_score", 0.0)
    
    reasons = []
    if not verified_docs:
        reasons.append("No relevant evidence was found in the clinical document corpus.")
    elif len(verified_docs) < 2:
        reasons.append("Insufficient supporting evidence (only 1 relevant chunk found).")
    
    if contradictions:
        reasons.append(
            f"Contradictory evidence detected ({len(contradictions)} conflicts), "
            "making a reliable conclusion impossible."
        )
    
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Overall confidence ({confidence:.2f}) is below the reliability "
            f"threshold ({CONFIDENCE_THRESHOLD})."
        )
    
    reason_text = "\n".join(f"  • {r}" for r in reasons)
    
    # Mention what was found, if anything
    found_info = ""
    if verified_docs:
        sources = set(d.metadata.get("source", "Unknown") for d in verified_docs)
        found_info = (
            f"\n\nPartial evidence was found in: {', '.join(sources)}. "
            "However, the evidence is not sufficient to provide a confident answer."
        )
    
    answer = (
        f"⚠️ **ABSTENTION**: I cannot provide a reliable answer to this question.\n\n"
        f"**Reasons:**\n{reason_text}"
        f"{found_info}\n\n"
        f"**Recommendation:** Consider consulting additional clinical databases, "
        f"peer-reviewed literature, or a domain expert for this query."
    )
    
    trace.append(f"[Abstain] Abstaining due to: {'; '.join(reasons)}")
    
    return {
        "answer": answer,
        "citations": [],
        "should_abstain": True,
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 5: ANSWER
# ============================================================

def answer_node(state: AgentState) -> dict:
    """
    Generate a grounded answer with citations from verified evidence.
    Notes any contradictions found for transparency.
    """
    query = state["query"]
    verified_docs = state.get("verified_docs", [])
    contradictions = state.get("contradictions", [])
    confidence = state.get("confidence_score", 0.5)
    trace = list(state.get("reasoning_trace", []))
    
    llm = _get_llm()
    context = _format_docs(verified_docs)
    
    contradiction_note = ""
    if contradictions:
        contra_details = []
        for c in contradictions:
            contra_details.append(
                f"- {c.get('explanation', 'Conflicting claims found')}"
            )
        contradiction_note = (
            "\n\nIMPORTANT: The following contradictions were detected in the evidence:\n"
            + "\n".join(contra_details)
            + "\n\nYou MUST acknowledge these contradictions in your answer and present both sides."
        )
    
    prompt = f"""You are a clinical research assistant providing evidence-based answers. 
Answer the following question using ONLY the provided document chunks as evidence.

RULES:
1. Base your answer EXCLUSIVELY on the provided evidence.
2. Cite sources inline using [Source: filename] format.
3. If evidence is limited, say so explicitly.
4. Include specific data points (p-values, effect sizes, percentages) when available.
5. Structure your answer with clear sections if the question is complex.
6. Do NOT make claims not supported by the evidence.
{contradiction_note}

QUESTION: {query}

EVIDENCE:
{context}

Provide a comprehensive, well-structured answer with inline citations:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer_text = response.content.strip()
    except Exception as e:
        trace.append(f"[Answer] LLM error: {str(e)}")
        answer_text = f"Error generating answer: {str(e)}"
    
    # Extract citations from verified docs
    citations = []
    seen_sources = set()
    for doc in verified_docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen_sources:
            citations.append(source)
            seen_sources.add(source)
    
    # Add confidence badge
    if confidence >= 0.85:
        badge = "🟢 **High Confidence**"
    elif confidence >= 0.7:
        badge = "🟡 **Moderate Confidence**"
    else:
        badge = "🟠 **Low Confidence**"
    
    full_answer = f"{badge} (Score: {confidence:.2f})\n\n{answer_text}"
    
    if contradictions:
        full_answer += (
            f"\n\n---\n⚠️ **Note:** {len(contradictions)} contradiction(s) "
            "were detected in the source evidence. The answer above attempts to "
            "present balanced findings from conflicting sources."
        )
    
    full_answer += f"\n\n---\n📚 **Sources:** {', '.join(citations)}"
    
    trace.append(f"[Answer] Generated answer with {len(citations)} citations (confidence: {confidence:.2f})")
    
    return {
        "answer": full_answer,
        "citations": citations,
        "reasoning_trace": trace,
    }


# ============================================================
# ROUTING HELPER: REFORMULATE QUERY FOR MULTI-HOP
# ============================================================

def reformulate_query_node(state: AgentState) -> dict:
    """
    Reformulate the query for multi-hop retrieval based on what's
    been found so far and what gaps remain.
    """
    query = state["query"]
    verified_docs = state.get("verified_docs", [])
    trace = list(state.get("reasoning_trace", []))
    
    llm = _get_llm()
    
    existing_info = ""
    if verified_docs:
        for doc in verified_docs[:3]:  # Limit context to save tokens
            existing_info += f"\n- {doc.page_content[:200]}..."
    
    prompt = f"""Given the original question and the evidence found so far, generate a NEW search query to find ADDITIONAL evidence that would help answer the question more completely.

ORIGINAL QUESTION: {query}

EVIDENCE FOUND SO FAR:{existing_info if existing_info else " None yet."}

Generate a single, focused search query that targets the GAPS in the current evidence. Respond with ONLY the new query text, nothing else."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        new_query = response.content.strip().strip('"').strip("'")
        trace.append(f"[Reformulate] New query: '{new_query}'")
    except Exception as e:
        new_query = query  # Fallback to original
        trace.append(f"[Reformulate] Error: {str(e)}. Using original query.")
    
    return {
        "reformulated_query": new_query,
        "reasoning_trace": trace,
    }
