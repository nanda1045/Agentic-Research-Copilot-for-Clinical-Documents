"""
Streamlit web interface for the Agentic Research Copilot.
Premium dark-theme UI with glassmorphism, ambient animations, and editorial layout.
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
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Inject Premium CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ========================================
   GLOBAL RESET & AMBIENT BACKGROUND
   ======================================== */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: #08080d;
    overflow-x: hidden;
}

/* Ensure Streamlit main content sits above ambient orbs */
.stMain, div[data-testid="stAppViewContainer"],
div[data-testid="stMainBlockContainer"],
section[data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}

/* Ambient floating orbs — gives depth without being distracting */
.stApp::before,
.stApp::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.12;
    pointer-events: none;
    z-index: 0;
}
.stApp::before {
    width: 600px; height: 600px;
    top: -120px; right: -100px;
    background: radial-gradient(circle, #6d5bf7 0%, transparent 70%);
    animation: orbFloat1 22s ease-in-out infinite;
}
.stApp::after {
    width: 500px; height: 500px;
    bottom: -80px; left: -60px;
    background: radial-gradient(circle, #2dd4bf 0%, transparent 70%);
    animation: orbFloat2 28s ease-in-out infinite;
}
@keyframes orbFloat1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(40px, 60px) scale(1.1); }
    66% { transform: translate(-30px, 30px) scale(0.95); }
}
@keyframes orbFloat2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(50px, -40px) scale(1.08); }
}

/* ========================================
   HIDE STREAMLIT CHROME
   ======================================== */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }

/* ========================================
   SIDEBAR — FROSTED PANEL
   ======================================== */
section[data-testid="stSidebar"] {
    background: rgba(12, 12, 18, 0.85) !important;
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border-right: 1px solid rgba(255,255,255,0.03) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #7a7e94 !important;
    font-size: 0.85rem;
    line-height: 1.65;
}
section[data-testid="stSidebar"] h3 {
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
    margin-top: 1.8rem;
    margin-bottom: 0.5rem;
}
section[data-testid="stSidebar"] code {
    color: #a78bfa !important;
    background: rgba(167,139,250,0.06) !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.78rem;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.04) !important;
    margin: 1.4rem 0;
}

/* Sidebar buttons — ghost style */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: #7a7e94 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
    position: relative;
    overflow: hidden;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(109,91,247,0.06) !important;
    border-color: rgba(109,91,247,0.2) !important;
    color: #b4a8f8 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(109,91,247,0.08);
}

/* ========================================
   TEXT AREA — GLOWING INPUT
   ======================================== */
.stTextArea textarea {
    background: rgba(16,16,24,0.8) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    color: #e2e4ef !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 18px 20px !important;
    transition: all 0.4s cubic-bezier(.4,0,.2,1);
    box-shadow: 0 0 0 0 rgba(109,91,247,0);
}
.stTextArea textarea:focus {
    border-color: rgba(109,91,247,0.4) !important;
    box-shadow: 0 0 0 4px rgba(109,91,247,0.08), 0 0 30px rgba(109,91,247,0.06) !important;
}
.stTextArea textarea::placeholder {
    color: #2e2e42 !important;
}
.stTextArea label { display: none !important; }

/* ========================================
   PRIMARY BUTTON — GRADIENT GLOW
   ======================================== */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6d5bf7 0%, #7c6cf0 40%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1.8rem !important;
    letter-spacing: 0.01em;
    transition: all 0.35s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 20px rgba(109,91,247,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    position: relative;
    overflow: hidden;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 8px 32px rgba(109,91,247,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transform: translateY(-2px);
}
div.stButton > button[kind="primary"]:active,
div.stButton > button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(109,91,247,0.25) !important;
}

/* Secondary buttons */
div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.02) !important;
    color: #55586e !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.3s ease !important;
}
div.stButton > button[kind="secondary"]:hover,
div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.04) !important;
    color: #8b8fa3 !important;
    border-color: rgba(255,255,255,0.08) !important;
}

/* ========================================
   EXPANDER — GLASS PANEL
   ======================================== */
div[data-testid="stExpander"] {
    background: rgba(14,14,22,0.6) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-radius: 16px !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    color: #7a7e94 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 6px 0;
}
div[data-testid="stExpander"] summary:hover {
    color: #b4a8f8 !important;
}

/* ========================================
   METRICS — MINIMAL
   ======================================== */
div[data-testid="stMetric"] {
    background: rgba(14,14,22,0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.25) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #e2e4ef !important;
    font-weight: 700;
    font-size: 1.4rem !important;
}

/* Alert styling */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: none !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6d5bf7 !important;
}

/* HR */
hr {
    border-color: rgba(255,255,255,0.03) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(109,91,247,0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(109,91,247,0.3); }

/* ========================================
   HERO SECTION
   ======================================== */
.hero-wrap {
    text-align: center;
    padding: 3rem 0 1.2rem 0;
    position: relative;
    z-index: 1;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(109,91,247,0.06);
    border: 1px solid rgba(109,91,247,0.12);
    color: #a78bfa;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 8px 18px;
    border-radius: 100px;
    margin-bottom: 1.2rem;
    animation: fadeSlideDown 0.6s ease-out;
}
.hero-badge .dot {
    width: 6px; height: 6px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f2f2f8;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.8rem;
    animation: fadeSlideDown 0.6s ease-out 0.1s both;
}
.hero-title .accent {
    background: linear-gradient(135deg, #6d5bf7 0%, #a78bfa 45%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #4e5168;
    font-size: 1.02rem;
    font-weight: 400;
    line-height: 1.7;
    max-width: 540px;
    margin: 0 auto;
    animation: fadeSlideDown 0.6s ease-out 0.2s both;
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ========================================
   PIPELINE STEPPER
   ======================================== */
.pipeline-strip {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 18px 0;
    margin: 0.4rem 0 0.8rem 0;
    animation: fadeSlideDown 0.6s ease-out 0.3s both;
}
.pipe-node {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #4e5168;
    transition: all 0.3s ease;
    position: relative;
}
.pipe-node .node-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #2a2a3d;
    transition: all 0.3s ease;
}
.pipe-node:hover {
    color: #a78bfa;
    background: rgba(109,91,247,0.04);
}
.pipe-node:hover .node-dot {
    background: #6d5bf7;
    box-shadow: 0 0 8px rgba(109,91,247,0.4);
}
.pipe-connector {
    width: 32px; height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.08), rgba(255,255,255,0.04));
}

/* ========================================
   SECTION LABELS
   ======================================== */
.section-lbl {
    color: rgba(255,255,255,0.2);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 700;
    margin: 2rem 0 0.8rem 0;
    padding-left: 2px;
}

/* ========================================
   EXAMPLE QUERIES
   ======================================== */
.example-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 1.2rem;
}

/* ========================================
   STAT PILL ROW (sidebar)
   ======================================== */
.side-stat {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    margin-bottom: 8px;
}
.side-stat .ss-icon {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.side-stat .ss-icon.docs {
    background: rgba(109,91,247,0.08);
}
.side-stat .ss-icon.idx {
    background: rgba(45,212,191,0.08);
}
.side-stat .ss-meta .ss-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e4ef;
    line-height: 1;
}
.side-stat .ss-meta .ss-lbl {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.25);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-top: 2px;
}

/* ========================================
   PIPELINE CONFIG ITEMS (sidebar)
   ======================================== */
.cfg-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.025);
}
.cfg-item:last-child { border-bottom: none; }
.cfg-item .cfg-key {
    color: #4e5168;
    font-size: 0.8rem;
    font-weight: 500;
}
.cfg-item .cfg-val {
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ========================================
   RESULTS METRICS ROW
   ======================================== */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 0.6rem 0 1.2rem 0;
}
.metric-tile {
    background: rgba(14,14,22,0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    transition: all 0.25s ease;
}
.metric-tile:hover {
    border-color: rgba(109,91,247,0.1);
    background: rgba(14,14,22,0.7);
}
.metric-tile .mt-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-tile .mt-val.purple { color: #a78bfa; }
.metric-tile .mt-val.teal { color: #2dd4bf; }
.metric-tile .mt-val.amber { color: #fbbf24; }
.metric-tile .mt-val.red { color: #f87171; }
.metric-tile .mt-lbl {
    color: rgba(255,255,255,0.2);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

/* ========================================
   CONFIDENCE BADGE
   ======================================== */
.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.conf-badge .conf-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
}
.conf-badge.high {
    background: rgba(45,212,191,0.06);
    color: #2dd4bf;
    border: 1px solid rgba(45,212,191,0.12);
}
.conf-badge.high .conf-dot { background: #2dd4bf; box-shadow: 0 0 6px rgba(45,212,191,0.4); }
.conf-badge.moderate {
    background: rgba(251,191,36,0.06);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.12);
}
.conf-badge.moderate .conf-dot { background: #fbbf24; box-shadow: 0 0 6px rgba(251,191,36,0.4); }
.conf-badge.low {
    background: rgba(248,113,113,0.06);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.12);
}
.conf-badge.low .conf-dot { background: #f87171; box-shadow: 0 0 6px rgba(248,113,113,0.4); }

/* ========================================
   REASONING TRACE
   ======================================== */
.trace-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 10px 16px;
    margin: 1px 0;
    border-radius: 12px;
    transition: background 0.2s ease;
    animation: traceIn 0.35s ease-out both;
}
.trace-step:hover { background: rgba(255,255,255,0.015); }

.trace-pip {
    flex-shrink: 0;
    width: 30px; height: 30px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    margin-top: 1px;
}
.trace-pip.retrieve { background: rgba(59,130,246,0.08); color: #60a5fa; }
.trace-pip.verify   { background: rgba(45,212,191,0.08); color: #2dd4bf; }
.trace-pip.contradict { background: rgba(251,191,36,0.08); color: #fbbf24; }
.trace-pip.reformulate { background: rgba(167,139,250,0.08); color: #a78bfa; }
.trace-pip.abstain  { background: rgba(248,113,113,0.08); color: #f87171; }
.trace-pip.answer   { background: rgba(45,212,191,0.08); color: #2dd4bf; }

.trace-body {
    flex: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.6;
    color: #5a5e73;
    padding-top: 5px;
}
@keyframes traceIn {
    from { opacity: 0; transform: translateX(-8px); }
    to { opacity: 1; transform: translateX(0); }
}

/* ========================================
   ANSWER CARD — FROSTED GLASS
   ======================================== */
.answer-glass {
    position: relative;
    background: rgba(14,14,22,0.5);
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 32px;
    margin: 12px 0 20px 0;
    overflow: hidden;
}
.answer-glass::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6d5bf7, #a78bfa, #2dd4bf, #6d5bf7);
    background-size: 300% 100%;
    animation: gradientSlide 4s linear infinite;
}
@keyframes gradientSlide {
    0% { background-position: 300% 0; }
    100% { background-position: -300% 0; }
}
.answer-glass .answer-body {
    color: #b8bbd0;
    font-size: 0.93rem;
    line-height: 1.85;
    letter-spacing: 0.005em;
}
.answer-glass .answer-body strong {
    color: #e2e4ef;
    font-weight: 600;
}

/* ========================================
   ABSTAIN CARD
   ======================================== */
.abstain-glass {
    position: relative;
    background: rgba(30,22,12,0.5);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(251,191,36,0.1);
    border-radius: 20px;
    padding: 32px;
    margin: 12px 0 20px 0;
    color: #d4a853;
    font-size: 0.93rem;
    line-height: 1.8;
    overflow: hidden;
}
.abstain-glass::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #fbbf24, #f59e0b, #d97706, #fbbf24);
    background-size: 300% 100%;
    animation: gradientSlide 4s linear infinite;
}

/* ========================================
   CONTRADICTION CARD
   ======================================== */
.conflict-card {
    background: rgba(248,113,113,0.03);
    border: 1px solid rgba(248,113,113,0.08);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 8px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.conflict-card .cc-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    flex-shrink: 0;
    margin-top: 2px;
}
.conflict-card .cc-pill.high { background: rgba(248,113,113,0.12); color: #f87171; }
.conflict-card .cc-pill.medium { background: rgba(251,191,36,0.12); color: #fbbf24; }
.conflict-card .cc-pill.low { background: rgba(45,212,191,0.08); color: #2dd4bf; }
.conflict-card .cc-text {
    color: #8b8fa3;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ========================================
   SOURCE TAGS
   ======================================== */
.src-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(109,91,247,0.04);
    border: 1px solid rgba(109,91,247,0.08);
    color: #8b82d6;
    padding: 5px 12px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    transition: all 0.2s ease;
}
.src-tag:hover {
    border-color: rgba(109,91,247,0.2);
    background: rgba(109,91,247,0.08);
}

/* ========================================
   FOOTER BRAND
   ======================================== */
.brand-footer {
    text-align: center;
    padding: 1rem 0 0;
    color: rgba(255,255,255,0.08);
    font-size: 0.68rem;
    letter-spacing: 1px;
    font-weight: 500;
}

/* ========================================
   RESULTS DIVIDER
   ======================================== */
.results-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(109,91,247,0.15), rgba(45,212,191,0.1), transparent);
    margin: 1.5rem 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def get_system_status():
    """Check the status of the system components."""
    status = {"docs_count": 0, "index_exists": False}
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
    with st.spinner("Building FAISS index..."):
        vectorstore = build_vectorstore(chunks)
    with st.spinner("Saving index..."):
        save_vectorstore(vectorstore)
    return len(documents), len(chunks)


# --- Sidebar ---

status = get_system_status()

with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.2rem 0;">
        <div style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 44px; height: 44px;
            background: linear-gradient(135deg, rgba(109,91,247,0.15), rgba(45,212,191,0.1));
            border-radius: 13px;
            font-size: 1.3rem;
            margin-bottom: 8px;
        ">🧬</div>
        <div style="color: rgba(255,255,255,0.2); font-size: 0.62rem; letter-spacing: 3px; text-transform: uppercase; font-weight: 700;">
            Research Copilot
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Status")

    # Status cards
    idx_color = "#2dd4bf" if status['index_exists'] else "#fbbf24"
    idx_icon = "✓" if status['index_exists'] else "○"
    st.markdown(f"""
    <div class="side-stat">
        <div class="ss-icon docs">📄</div>
        <div class="ss-meta">
            <div class="ss-val">{status['docs_count']}</div>
            <div class="ss-lbl">Documents</div>
        </div>
    </div>
    <div class="side-stat">
        <div class="ss-icon idx" style="color: {idx_color}">⬡</div>
        <div class="ss-meta">
            <div class="ss-val" style="color: {idx_color}">{idx_icon}</div>
            <div class="ss-lbl">FAISS Index</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("⟳  Generate Sample Data", use_container_width=True):
        with st.spinner("Generating 50 clinical documents..."):
            from scripts.generate_sample_data import main as gen_main
            gen_main()
        st.success("✓ 50 clinical documents generated")
        st.rerun()

    if st.button("↓  Ingest Documents", use_container_width=True):
        if status["docs_count"] == 0:
            st.error("No documents found.")
        else:
            try:
                n_docs, n_chunks = run_ingestion()
                st.success(f"✓ {n_docs} docs → {n_chunks} chunks")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("### Configuration")

    model_short = LLM_MODEL.split('-20')[0] if '-20' in LLM_MODEL else LLM_MODEL
    st.markdown(f"""
    <div style="padding: 4px 0;">
        <div class="cfg-item">
            <span class="cfg-key">Model</span>
            <span class="cfg-val">{model_short}</span>
        </div>
        <div class="cfg-item">
            <span class="cfg-key">Retrieval</span>
            <span class="cfg-val">Top-{TOP_K}</span>
        </div>
        <div class="cfg-item">
            <span class="cfg-key">Threshold</span>
            <span class="cfg-val">{CONFIDENCE_THRESHOLD}</span>
        </div>
        <div class="cfg-item">
            <span class="cfg-key">Max Hops</span>
            <span class="cfg-val">{MAX_HOPS}</span>
        </div>
        <div class="cfg-item">
            <span class="cfg-key">Embeddings</span>
            <span class="cfg-val">Local</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="brand-footer">
        LangGraph · Claude · FAISS · v1.0
    </div>
    """, unsafe_allow_html=True)


# ===== MAIN CONTENT =====

# Hero
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge"><span class="dot"></span> AI-Powered Clinical Analysis</div>
    <div class="hero-title">Agentic Research <span class="accent">Copilot</span></div>
    <div class="hero-sub">
        Multi-hop reasoning over clinical documents with verification,
        contradiction detection, and evidence-based answers.
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline visualization
st.markdown("""
<div class="pipeline-strip">
    <div class="pipe-node"><span class="node-dot"></span> Retrieve</div>
    <div class="pipe-connector"></div>
    <div class="pipe-node"><span class="node-dot"></span> Verify</div>
    <div class="pipe-connector"></div>
    <div class="pipe-node"><span class="node-dot"></span> Contradict</div>
    <div class="pipe-connector"></div>
    <div class="pipe-node"><span class="node-dot"></span> Answer</div>
</div>
""", unsafe_allow_html=True)

# --- Example Queries ---
st.markdown('<div class="section-lbl">Try an example</div>', unsafe_allow_html=True)

example_queries = [
    ("💊", "What are the side effects of Metformin in Type 2 Diabetes trials?"),
    ("⚖️", "Compare the efficacy of Metformin vs Sitagliptin for Type 2 Diabetes"),
    ("🔬", "Is Semaglutide effective for treating Obesity?"),
    ("❓", "What is the recommended dosage of Zyxoplatin for cancer?"),
]

cols = st.columns(2)
for i, (icon, query_text) in enumerate(example_queries):
    with cols[i % 2]:
        if st.button(f"{icon}  {query_text[:52]}{'...' if len(query_text) > 52 else ''}", key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = query_text

# --- Query Input ---
st.markdown('<div class="section-lbl">Your question</div>', unsafe_allow_html=True)

query = st.text_area(
    "query_input_label",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="Ask a clinical research question...",
    key="main_query_input",
    label_visibility="collapsed",
)

col_run, col_clear = st.columns([4, 1])
with col_run:
    run_btn = st.button("Run Agent  →", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state["query_input"] = ""
        st.rerun()


# --- Process Query ---
if run_btn and query.strip():
    if not status["index_exists"]:
        st.error("FAISS index not found. Use sidebar to ingest documents first.")
    else:
        st.markdown('<div class="results-divider"></div>', unsafe_allow_html=True)

        with st.spinner(""):
            try:
                from src.agents.graph import run_agent
                result = run_agent(query.strip())

                answer = result.get("answer", "No answer generated.")
                should_abstain = result.get("should_abstain", False)
                contradictions = result.get("contradictions", [])
                confidence = result.get("confidence_score", 0.0)
                trace = result.get("reasoning_trace", [])
                verified_docs = result.get("verified_docs", [])
                citations = result.get("citations", [])
                hop_count = result.get("hop_count", 0)

                # --- Metrics Row ---
                st.markdown('<div class="section-lbl">Pipeline Results</div>', unsafe_allow_html=True)

                # Confidence
                if confidence >= 0.85:
                    pill_cls, pill_label = "high", "High"
                elif confidence >= 0.7:
                    pill_cls, pill_label = "moderate", "Moderate"
                else:
                    pill_cls, pill_label = "low", "Low"

                conf_color = {"high": "teal", "moderate": "amber", "low": "red"}[pill_cls]

                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-tile">
                        <div class="mt-val {conf_color}">{confidence:.0%}</div>
                        <div class="mt-lbl">Confidence</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-val purple">{hop_count}</div>
                        <div class="mt-lbl">Hops</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-val teal">{len(verified_docs)}</div>
                        <div class="mt-lbl">Verified</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-val {'red' if len(contradictions) > 0 else 'amber'}">{len(contradictions)}</div>
                        <div class="mt-lbl">Conflicts</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence badge
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span class="conf-badge {pill_cls}">
                        <span class="conf-dot"></span>
                        {pill_label} Confidence — {confidence:.0%}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # --- Reasoning Trace ---
                with st.expander("Agent Reasoning Trace", expanded=False):
                    for idx, step in enumerate(trace):
                        if "[Retrieve]" in step:
                            icon_cls, icon_char = "retrieve", "⌕"
                        elif "[Verify]" in step:
                            icon_cls, icon_char = "verify", "✓"
                        elif "[Contradict]" in step:
                            icon_cls, icon_char = "contradict", "⚡"
                        elif "[Reformulate]" in step:
                            icon_cls, icon_char = "reformulate", "↻"
                        elif "[Abstain]" in step:
                            icon_cls, icon_char = "abstain", "✕"
                        elif "[Answer]" in step:
                            icon_cls, icon_char = "answer", "✓"
                        else:
                            icon_cls, icon_char = "retrieve", "•"

                        delay = f"{idx * 0.06:.2f}s"
                        st.markdown(f"""
                        <div class="trace-step" style="animation-delay: {delay};">
                            <div class="trace-pip {icon_cls}">{icon_char}</div>
                            <div class="trace-body">{step}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # --- Answer ---
                st.markdown('<div class="section-lbl">Answer</div>', unsafe_allow_html=True)

                if should_abstain:
                    st.markdown(f'<div class="abstain-glass">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="answer-glass"><div class="answer-body">{answer}</div></div>', unsafe_allow_html=True)

                # --- Contradictions ---
                if contradictions:
                    with st.expander(f"⚡ {len(contradictions)} Contradiction(s) Detected", expanded=True):
                        for c in contradictions:
                            severity = c.get("severity", "medium")
                            explanation = c.get("explanation", "Conflicting evidence found")
                            st.markdown(f"""
                            <div class="conflict-card">
                                <span class="cc-pill {severity}">{severity}</span>
                                <span class="cc-text">{explanation}</span>
                            </div>
                            """, unsafe_allow_html=True)

                # --- Sources ---
                if citations:
                    st.markdown('<div class="section-lbl">Sources</div>', unsafe_allow_html=True)
                    sources_html = "".join(f'<span class="src-tag">📎 {c}</span>' for c in citations)
                    st.markdown(sources_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif run_btn and not query.strip():
    st.warning("Please enter a question.")
