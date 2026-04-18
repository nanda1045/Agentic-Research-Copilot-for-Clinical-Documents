"""
Centralized configuration for the Agentic Research Copilot.
Loads environment variables and defines all tunable parameters.

All settings can be overridden via environment variables (see .env.example).
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "clinical_docs"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- LLM Configuration ---
# Using Claude 3 Haiku for cost efficiency (~$0.25/1M input, $1.25/1M output)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2048

# --- Embedding Configuration ---
# Using free local HuggingFace model (no API cost!)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Chunking Configuration ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval Configuration ---
TOP_K = int(os.getenv("TOP_K", "5"))  # Number of chunks to retrieve per query

# --- Agent Configuration ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))  # Below this → abstain
MAX_HOPS = int(os.getenv("MAX_HOPS", "3"))  # Maximum multi-hop retrieval iterations
VERIFICATION_THRESHOLD = 0.5  # Chunk relevance cutoff

# --- Ensure directories exist ---
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> list[str]:
    """
    Validate the configuration and return a list of warnings/errors.
    
    Returns:
        List of warning messages. Empty list means all is well.
    """
    warnings = []

    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
        warnings.append(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key from https://console.anthropic.com"
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        warnings.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})"
        )

    if CONFIDENCE_THRESHOLD < 0 or CONFIDENCE_THRESHOLD > 1:
        warnings.append(
            f"CONFIDENCE_THRESHOLD ({CONFIDENCE_THRESHOLD}) must be between 0.0 and 1.0"
        )

    if MAX_HOPS < 1:
        warnings.append(f"MAX_HOPS ({MAX_HOPS}) must be at least 1")

    if TOP_K < 1:
        warnings.append(f"TOP_K ({TOP_K}) must be at least 1")

    return warnings
