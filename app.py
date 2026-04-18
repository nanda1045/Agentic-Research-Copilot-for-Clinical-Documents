"""
Streamlit web interface for the Agentic Research Copilot.
Provides a chat-style interface with real-time reasoning trace display.
"""

import streamlit as st
import sys
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_DIR, FAISS_INDEX_DIR, CONFIDENCE_THRESHOLD, 
    MAX_HOPS, TOP_K, LLM_MODEL,
)


# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Research Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1e293b 0%, #1a1a2e 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Confidence badges */
    .badge-high {
        background: linear-gradient(90deg, #10b981, #34d399);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        color: #1a1a2e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background: linear-gradient(90deg, #ef4444, #f87171);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Trace steps */
    .trace-step {
        background: #1e293b;
        border-left: 3px solid #667eea;
        padding: 8px 16px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    /* Sidebar styling */
    .sidebar-stat {
        background: linear-gradient(135deg, #1e293b 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    
    /* Contradiction warning */
    .contradiction-box {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        color: #fecaca;
    }
    
    /* Abstain box */
    .abstain-box {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #d97706;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        color: #fef3c7;
    }
    
    /* Example queries */
    .example-btn {
        background: #1e293b;
        border: 1px solid #475569;
        color: #e2e8f0;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
        width: 100%;
        text-align: left;
        margin: 4px 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def get_system_status():
    """Check the status of the system components."""
    status = {
        "docs_count": 0,
        "index_exists": False,
    }
    
    data_path = Path(DATA_DIR)
    if data_path.exists():
        status["docs_count"] = len(list(data_path.glob("*.txt")) + list(data_path.glob("*.pdf")))
    
    index_path = Path(FAISS_INDEX_DIR)
    status["index_exists"] = (index_path / "index.faiss").exists()
    
    return status


def run_ingestion():
    """Run the document ingestion pipeline."""
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.vectorstore import build_vectorstore, save_vectorstore
    
    with st.spinner("Loading documents..."):
        documents = load_documents(str(DATA_DIR))
    
    with st.spinner("Chunking documents..."):
        chunks = chunk_documents(documents)
    
    with st.spinner("Building FAISS index (embedding chunks)..."):
        vectorstore = build_vectorstore(chunks)
    
    with st.spinner("Saving index..."):
        save_vectorstore(vectorstore)
    
    return len(documents), len(chunks)


# --- Sidebar ---

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    
    status = get_system_status()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="stat-number">{status['docs_count']}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        index_emoji = "✅" if status['index_exists'] else "❌"
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="stat-number">{index_emoji}</div>
            <div class="stat-label">FAISS Index</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate Data button
    st.markdown("### 📄 Data Management")
    if st.button("🔄 Generate Sample Data", use_container_width=True):
        with st.spinner("Generating 50 clinical documents..."):
            from scripts.generate_sample_data import main as gen_main
            gen_main()
        st.success("✅ 50 clinical documents generated!")
        st.rerun()
    
    # Ingest button
    if st.button("📥 Ingest Documents", use_container_width=True):
        if status["docs_count"] == 0:
            st.error("No documents found. Generate data first!")
        else:
            try:
                n_docs, n_chunks = run_ingestion()
                st.success(f"✅ Indexed {n_docs} docs → {n_chunks} chunks")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {str(e)}")
    
    st.markdown("---")
    
    # Settings display
    st.markdown("### 🎛️ Configuration")
    st.markdown(f"""
    - **LLM:** `{LLM_MODEL}`
    - **Top-K:** `{TOP_K}` chunks
    - **Confidence Threshold:** `{CONFIDENCE_THRESHOLD}`
    - **Max Hops:** `{MAX_HOPS}`
    - **Embeddings:** Local (free)
    """)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.8rem;'>"
        "Built with LangGraph + Claude + FAISS<br>"
        "Agentic Research Copilot v1.0"
        "</p>",
        unsafe_allow_html=True,
    )


# --- Main Content ---

st.markdown('<h1 class="main-header">🏥 Agentic Research Copilot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered clinical document analysis with verification, '
    'contradiction detection, and evidence-based answers</p>',
    unsafe_allow_html=True,
)

# --- Example Queries ---
st.markdown("#### 💡 Example Questions")
example_queries = [
    "What are the side effects of Metformin in Type 2 Diabetes trials?",
    "Compare the efficacy of Metformin vs Sitagliptin for Type 2 Diabetes",
    "Is Semaglutide effective for treating Obesity?",
    "What is the recommended dosage of Zyxoplatin for cancer?",
]

cols = st.columns(2)
for i, query in enumerate(example_queries):
    with cols[i % 2]:
        if st.button(f"📌 {query[:55]}...", key=f"example_{i}", use_container_width=True):
            st.session_state["query_input"] = query

# --- Query Input ---
st.markdown("---")

query = st.text_area(
    "🔍 Ask a clinical research question:",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="e.g., What are the comparative results of Lisinopril vs Losartan for Hypertension?",
    key="main_query_input",
)

col_run, col_clear = st.columns([3, 1])
with col_run:
    run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["query_input"] = ""
        st.rerun()

# --- Process Query ---
if run_btn and query.strip():
    if not status["index_exists"]:
        st.error("⚠️ FAISS index not found. Please ingest documents first (use sidebar buttons).")
    else:
        st.markdown("---")
        
        # Reasoning trace (live)
        trace_container = st.expander("📋 Agent Reasoning Trace", expanded=True)
        answer_container = st.container()
        
        with st.spinner("🤖 Agent is thinking..."):
            try:
                from src.agents.graph import run_agent
                result = run_agent(query.strip())
                
                # Display reasoning trace
                with trace_container:
                    for step in result.get("reasoning_trace", []):
                        if "[Retrieve]" in step:
                            icon = "🔍"
                        elif "[Verify]" in step:
                            icon = "✅"
                        elif "[Contradict]" in step:
                            icon = "⚡"
                        elif "[Reformulate]" in step:
                            icon = "🔄"
                        elif "[Abstain]" in step:
                            icon = "⚠️"
                        elif "[Answer]" in step:
                            icon = "💡"
                        else:
                            icon = "📌"
                        
                        st.markdown(
                            f'<div class="trace-step">{icon} {step}</div>',
                            unsafe_allow_html=True,
                        )
                
                # Display answer
                with answer_container:
                    answer = result.get("answer", "No answer generated.")
                    should_abstain = result.get("should_abstain", False)
                    contradictions = result.get("contradictions", [])
                    confidence = result.get("confidence_score", 0.0)
                    
                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        badge_class = "badge-high" if confidence >= 0.85 else "badge-medium" if confidence >= 0.7 else "badge-low"
                        st.markdown(f'<span class="{badge_class}">Confidence: {confidence:.2f}</span>', unsafe_allow_html=True)
                    with m2:
                        st.metric("Hops", result.get("hop_count", 0))
                    with m3:
                        st.metric("Verified Chunks", len(result.get("verified_docs", [])))
                    with m4:
                        st.metric("Contradictions", len(contradictions))
                    
                    st.markdown("---")
                    
                    # Answer display
                    if should_abstain:
                        st.markdown(f'<div class="abstain-box">{answer}</div>', unsafe_allow_html=True)
                    elif contradictions:
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                        with st.expander("⚡ Contradiction Details", expanded=False):
                            for c in contradictions:
                                st.warning(
                                    f"**{c.get('severity', 'unknown').upper()}**: "
                                    f"{c.get('explanation', 'Conflicting evidence found')}"
                                )
                    else:
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Citations
                    citations = result.get("citations", [])
                    if citations:
                        with st.expander("📚 Sources", expanded=False):
                            for cite in citations:
                                st.markdown(f"- `{cite}`")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif run_btn and not query.strip():
    st.warning("Please enter a question.")
