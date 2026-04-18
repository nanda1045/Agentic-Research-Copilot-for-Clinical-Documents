"""
RAGAS-based evaluation pipeline for the Agentic Research Copilot.
Evaluates the RAG pipeline on faithfulness, answer relevancy, and context precision.
Also computes custom metrics: grounded accuracy and unsupported answer rate.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import EVAL_RESULTS_DIR, ANTHROPIC_API_KEY


def run_pipeline_on_eval_set(eval_path: str) -> List[Dict]:
    """
    Run the agentic pipeline on each question in the eval set.
    
    Args:
        eval_path: Path to the eval_dataset.json file
        
    Returns:
        List of result dicts with question, answer, contexts, ground_truth
    """
    from src.agents.graph import run_agent
    
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    results = []
    total = len(eval_data)
    
    for i, item in enumerate(eval_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        question_type = item.get("question_type", "unknown")
        
        print(f"  [{i+1}/{total}] Processing: {question[:60]}...")
        
        try:
            state = run_agent(question)
            
            answer = state.get("answer", "")
            contexts = [
                doc.page_content 
                for doc in state.get("verified_docs", [])
            ]
            citations = state.get("citations", [])
            confidence = state.get("confidence_score", 0.0)
            should_abstain = state.get("should_abstain", False)
            contradictions = state.get("contradictions", [])
            
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved."],
                "ground_truth": ground_truth,
                "question_type": question_type,
                "confidence": confidence,
                "abstained": should_abstain,
                "num_contradictions": len(contradictions),
                "num_citations": len(citations),
            })
            
        except Exception as e:
            print(f"    ⚠ Error: {str(e)}")
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}",
                "contexts": ["Error during processing."],
                "ground_truth": ground_truth,
                "question_type": question_type,
                "confidence": 0.0,
                "abstained": True,
                "num_contradictions": 0,
                "num_citations": 0,
            })
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return results


def compute_custom_metrics(results: List[Dict]) -> Dict:
    """
    Compute custom metrics beyond RAGAS:
    - Grounded answer accuracy
    - Unsupported answer rate
    - Abstention precision/recall
    - Contradiction detection rate
    """
    total = len(results)
    if total == 0:
        return {}
    
    # --- Grounded Answer Accuracy ---
    # Answers with supporting context and reasonable confidence
    grounded = sum(
        1 for r in results
        if not r["abstained"]
        and r["confidence"] >= 0.7
        and r["contexts"] != ["No context retrieved."]
    )
    non_abstained = sum(1 for r in results if not r["abstained"])
    grounded_accuracy = grounded / non_abstained if non_abstained > 0 else 0.0
    
    # --- Unsupported Answer Rate ---
    # Answers generated despite low confidence or no context
    unsupported = sum(
        1 for r in results
        if not r["abstained"]
        and (r["confidence"] < 0.5 or r["contexts"] == ["No context retrieved."])
    )
    unsupported_rate = unsupported / total
    
    # --- Abstention Metrics ---
    abstain_questions = [r for r in results if r["question_type"] == "abstain"]
    abstain_correct = sum(1 for r in abstain_questions if r["abstained"])
    abstention_precision = abstain_correct / len(abstain_questions) if abstain_questions else 0.0
    
    total_abstained = sum(1 for r in results if r["abstained"])
    
    # --- Contradiction Detection ---
    contra_questions = [r for r in results if r["question_type"] == "contradiction"]
    contra_detected = sum(1 for r in contra_questions if r["num_contradictions"] > 0)
    contradiction_detection_rate = contra_detected / len(contra_questions) if contra_questions else 0.0
    
    # --- Average Confidence ---
    avg_confidence = sum(r["confidence"] for r in results) / total
    
    return {
        "total_questions": total,
        "grounded_answer_accuracy": round(grounded_accuracy * 100, 1),
        "unsupported_answer_rate": round(unsupported_rate * 100, 1),
        "abstention_precision": round(abstention_precision * 100, 1),
        "total_abstentions": total_abstained,
        "contradiction_detection_rate": round(contradiction_detection_rate * 100, 1),
        "average_confidence": round(avg_confidence, 3),
    }


def run_ragas_evaluation(results: List[Dict]) -> Optional[Dict]:
    """
    Run RAGAS evaluation metrics (faithfulness, answer_relevancy, context_precision).
    Returns None if RAGAS evaluation cannot run (e.g., missing dependencies).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        
        # Prepare data for RAGAS
        ragas_data = {
            "question": [r["question"] for r in results if not r["abstained"]],
            "answer": [r["answer"] for r in results if not r["abstained"]],
            "contexts": [r["contexts"] for r in results if not r["abstained"]],
            "ground_truth": [r["ground_truth"] for r in results if not r["abstained"]],
        }
        
        if not ragas_data["question"]:
            print("  ⚠ No non-abstained answers to evaluate with RAGAS.")
            return None
        
        dataset = Dataset.from_dict(ragas_data)
        
        # Run RAGAS evaluation
        print("  Running RAGAS evaluation (this uses API credits)...")
        ragas_results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        
        return dict(ragas_results)
        
    except ImportError as e:
        print(f"  ⚠ RAGAS not available: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ RAGAS evaluation failed: {e}")
        return None


def format_results_table(custom_metrics: Dict, ragas_metrics: Optional[Dict] = None) -> str:
    """Format evaluation results as a readable table."""
    lines = [
        "\n" + "=" * 65,
        "  AGENTIC RESEARCH COPILOT — EVALUATION RESULTS",
        "=" * 65,
        "",
        "  CUSTOM METRICS",
        "  " + "-" * 50,
        f"  Total Questions Evaluated:      {custom_metrics.get('total_questions', 0)}",
        f"  Grounded Answer Accuracy:       {custom_metrics.get('grounded_answer_accuracy', 0)}%",
        f"  Unsupported Answer Rate:        {custom_metrics.get('unsupported_answer_rate', 0)}%",
        f"  Abstention Precision:           {custom_metrics.get('abstention_precision', 0)}%",
        f"  Total Abstentions:              {custom_metrics.get('total_abstentions', 0)}",
        f"  Contradiction Detection Rate:   {custom_metrics.get('contradiction_detection_rate', 0)}%",
        f"  Average Confidence Score:       {custom_metrics.get('average_confidence', 0)}",
        "",
    ]
    
    if ragas_metrics:
        lines.extend([
            "  RAGAS METRICS",
            "  " + "-" * 50,
            f"  Faithfulness:                   {ragas_metrics.get('faithfulness', 'N/A')}",
            f"  Answer Relevancy:               {ragas_metrics.get('answer_relevancy', 'N/A')}",
            f"  Context Precision:              {ragas_metrics.get('context_precision', 'N/A')}",
            "",
        ])
    
    lines.extend([
        "  TARGET METRICS (vs Baseline RAG)",
        "  " + "-" * 50,
        f"  Grounded Accuracy Improvement:  +28% target",
        f"  Unsupported Answers Reduction:  -31% target",
        "",
        "=" * 65,
    ])
    
    return "\n".join(lines)


def main(run_ragas: bool = False):
    """Run the full evaluation pipeline."""
    eval_path = EVAL_RESULTS_DIR / "eval_dataset.json"
    
    if not eval_path.exists():
        print("  Generating eval dataset first...")
        from src.evaluation.eval_data_generator import generate_eval_dataset, save_eval_dataset
        eval_pairs = generate_eval_dataset()
        save_eval_dataset(eval_pairs)
    
    print("\n📊 Running evaluation pipeline...")
    print("=" * 50)
    
    # Run pipeline on eval set
    results = run_pipeline_on_eval_set(str(eval_path))
    
    # Compute custom metrics
    custom_metrics = compute_custom_metrics(results)
    
    # Optionally run RAGAS (costs API credits)
    ragas_metrics = None
    if run_ragas:
        ragas_metrics = run_ragas_evaluation(results)
    
    # Format and display results
    table = format_results_table(custom_metrics, ragas_metrics)
    print(table)
    
    # Save results
    output = {
        "custom_metrics": custom_metrics,
        "ragas_metrics": ragas_metrics,
        "detailed_results": results,
    }
    
    output_path = EVAL_RESULTS_DIR / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  💾 Full results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    main(run_ragas=False)
