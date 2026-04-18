"""
Centralized configuration for the Agentic Research Copilot.
Loads environment variables and defines all tunable parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "clinical_docs"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- LLM Configuration ---
# Using Claude 3 Haiku for cost efficiency (~$0.25/1M input, $1.25/1M output)
LLM_MODEL = "claude-3-haiku-20240307"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2048

# --- Embedding Configuration ---
# Using free local HuggingFace model (no API cost!)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval Configuration ---
TOP_K = 5  # Number of chunks to retrieve per query

# --- Agent Configuration ---
CONFIDENCE_THRESHOLD = 0.7  # Below this → abstain
MAX_HOPS = 3  # Maximum multi-hop retrieval iterations
VERIFICATION_THRESHOLD = 0.5  # Chunk relevance cutoff

# --- Ensure directories exist ---
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
