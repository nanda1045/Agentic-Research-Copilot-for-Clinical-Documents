"""
CLI entry point for the Agentic Research Copilot.

Usage:
    python main.py ingest          # Ingest clinical documents into FAISS
    python main.py query "..."     # Ask a clinical research question
    python main.py evaluate        # Run evaluation pipeline
    python main.py generate-data   # Generate sample clinical documents
    python main.py generate-eval   # Generate evaluation Q&A dataset
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_generate_data():
    """Generate synthetic clinical documents."""
    print("\n🏥 Generating Synthetic Clinical Documents")
    print("=" * 50)
    from scripts.generate_sample_data import main as gen_main
    gen_main()


def cmd_ingest():
    """Ingest documents into FAISS vector store."""
    print("\n📥 Document Ingestion Pipeline")
    print("=" * 50)
    
    from config.settings import DATA_DIR
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.vectorstore import build_vectorstore, save_vectorstore
    
    # Check if data exists
    data_dir = Path(DATA_DIR)
    doc_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf"))
    
    if not doc_files:
        print(f"  ⚠ No documents found in {DATA_DIR}")
        print("  Run `python main.py generate-data` first.")
        return
    
    print(f"  Found {len(doc_files)} document files")
    
    # Load → Chunk → Embed → Store
    print("\n  Step 1: Loading documents...")
    documents = load_documents(str(DATA_DIR))
    
    print("\n  Step 2: Chunking documents...")
    chunks = chunk_documents(documents)
    
    print("\n  Step 3: Building FAISS vector store...")
    vectorstore = build_vectorstore(chunks)
    
    print("\n  Step 4: Saving index...")
    save_vectorstore(vectorstore)
    
    print(f"\n✅ Ingestion complete! {len(doc_files)} files → {len(chunks)} chunks indexed.")


def cmd_query(question: str):
    """Run a query through the agentic pipeline."""
    print("\n🔍 Agentic Research Copilot")
    print("=" * 50)
    print(f"  Query: {question}")
    print("-" * 50)
    
    from src.agents.graph import run_agent
    
    result = run_agent(question)
    
    # Display reasoning trace
    print("\n📋 Reasoning Trace:")
    for step in result.get("reasoning_trace", []):
        print(f"  {step}")
    
    # Display answer
    print("\n" + "=" * 50)
    print("💡 ANSWER:")
    print("=" * 50)
    print(result.get("answer", "No answer generated."))
    print()


def cmd_evaluate(use_ragas: bool = False):
    """Run evaluation pipeline."""
    print("\n📊 Evaluation Pipeline")
    print("=" * 50)
    
    from src.evaluation.evaluate import main as eval_main
    eval_main(run_ragas=use_ragas)


def cmd_generate_eval():
    """Generate evaluation dataset."""
    print("\n📝 Generating Evaluation Dataset")
    print("=" * 50)
    from src.evaluation.eval_data_generator import main as eval_gen_main
    eval_gen_main()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "generate-data":
        cmd_generate_data()
    
    elif command == "ingest":
        cmd_ingest()
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("  Usage: python main.py query \"Your question here\"")
            return
        question = " ".join(sys.argv[2:])
        cmd_query(question)
    
    elif command == "evaluate":
        use_ragas = "--ragas" in sys.argv
        cmd_evaluate(use_ragas=use_ragas)
    
    elif command == "generate-eval":
        cmd_generate_eval()
    
    else:
        print(f"  Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
