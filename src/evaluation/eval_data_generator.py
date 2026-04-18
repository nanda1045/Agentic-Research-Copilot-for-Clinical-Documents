"""
Generate synthetic evaluation Q&A pairs from the ingested clinical documents.
Creates a labeled eval set for RAGAS evaluation without using API calls —
questions and answers are generated from templates based on document content.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, EVAL_RESULTS_DIR


def _extract_doc_info(content: str) -> dict:
    """Extract key information from a clinical document."""
    info = {}
    lines = content.split("\n")
    
    for line in lines:
        line_lower = line.lower().strip()
        if "drug:" in line_lower or "study title:" in line_lower:
            info["title"] = line.strip()
        if "document id:" in line_lower:
            info["doc_id"] = line.split(":")[-1].strip()
        if "conclusion" in line_lower:
            idx = lines.index(line)
            # Grab the next few lines as conclusion
            conclusion_lines = []
            for j in range(idx + 1, min(idx + 5, len(lines))):
                if lines[j].strip():
                    conclusion_lines.append(lines[j].strip())
            info["conclusion"] = " ".join(conclusion_lines)
    
    return info


def generate_eval_dataset(n_questions: int = 20) -> List[Dict]:
    """
    Generate synthetic Q&A evaluation pairs from clinical documents.
    
    Creates different question types:
    - Single-hop factual (40%)
    - Multi-hop reasoning (30%)
    - Contradiction-triggering (15%)
    - Impossible/abstain-triggering (15%)
    
    Returns:
        List of eval dicts with: question, ground_truth, question_type
    """
    random.seed(42)
    
    # Read all documents
    doc_contents = {}
    for filepath in sorted(DATA_DIR.iterdir()):
        if filepath.suffix == ".txt":
            doc_contents[filepath.name] = filepath.read_text()
    
    if not doc_contents:
        print("  ⚠ No documents found. Run data generation first.")
        return []
    
    eval_pairs = []
    doc_names = list(doc_contents.keys())
    
    # --- Single-hop factual questions (40%) ---
    single_hop_templates = [
        ("What were the primary results of the {drug} trial for {condition}?",
         "efficacy_trial"),
        ("What adverse events were reported for {drug}?",
         "adverse_event"),
        ("What is the recommended dose of {drug} for {condition}?",
         "dosing_study"),
        ("How does {drug1} compare to {drug2} for {condition}?",
         "comparative_study"),
        ("What did the meta-analysis conclude about {drug} for {condition}?",
         "meta_analysis"),
        ("What was the safety profile observed in the {drug} trial?",
         "efficacy_trial"),
        ("What was the patient population in the {drug} study for {condition}?",
         "efficacy_trial"),
        ("What was the p-value and effect size in the {drug} trial?",
         "efficacy_trial"),
    ]
    
    drugs = ["Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Semaglutide",
             "Dapagliflozin", "Empagliflozin", "Tirzepatide", "Sitagliptin"]
    conditions = ["Type 2 Diabetes Mellitus", "Hypertension", "Hyperlipidemia",
                  "Heart Failure", "Obesity"]
    
    n_single = int(n_questions * 0.4)
    for i in range(n_single):
        template, doc_type = random.choice(single_hop_templates)
        drug = random.choice(drugs)
        drug2 = random.choice([d for d in drugs if d != drug])
        condition = random.choice(conditions)
        
        question = template.format(drug=drug, drug1=drug, drug2=drug2, condition=condition)
        
        # Find a matching document for ground truth
        ground_truth = f"Based on clinical trial data for {drug} in {condition}."
        source_doc = None
        for name, content in doc_contents.items():
            if drug.lower() in content.lower() and doc_type.replace("_", " ") in name.replace("_", " "):
                info = _extract_doc_info(content)
                if "conclusion" in info:
                    ground_truth = info["conclusion"]
                source_doc = name
                break
        
        eval_pairs.append({
            "question": question,
            "ground_truth": ground_truth,
            "question_type": "single_hop",
            "source_doc": source_doc,
        })
    
    # --- Multi-hop reasoning questions (30%) ---
    multi_hop_templates = [
        "Compare the efficacy of {drug1} and {drug2} for {condition}, considering both their clinical trial results and adverse event profiles.",
        "Based on all available evidence, what is the best treatment option for {condition} and why?",
        "Synthesize the findings from multiple studies on {drug} to assess its overall benefit-risk profile for {condition}.",
        "What do the dose-finding and efficacy data together suggest about the optimal use of {drug} for {condition}?",
        "Considering both the meta-analysis and individual trial results, how strong is the evidence for {drug} in treating {condition}?",
    ]
    
    n_multi = int(n_questions * 0.3)
    for i in range(n_multi):
        template = random.choice(multi_hop_templates)
        drug1 = random.choice(drugs)
        drug2 = random.choice([d for d in drugs if d != drug1])
        condition = random.choice(conditions)
        
        question = template.format(drug=drug1, drug1=drug1, drug2=drug2, condition=condition)
        
        eval_pairs.append({
            "question": question,
            "ground_truth": f"A comprehensive assessment of {drug1} for {condition} requires considering multiple evidence sources.",
            "question_type": "multi_hop",
            "source_doc": None,
        })
    
    # --- Contradiction-triggering questions (15%) ---
    contradiction_questions = [
        "Is Semaglutide effective for treating Obesity? What does the evidence show?",
        "Which is more effective for Type 2 Diabetes Mellitus: Metformin or Sitagliptin?",
        "What is the safety profile of Empagliflozin based on post-marketing surveillance?",
    ]
    
    n_contra = int(n_questions * 0.15)
    for i in range(n_contra):
        question = contradiction_questions[i % len(contradiction_questions)]
        eval_pairs.append({
            "question": question,
            "ground_truth": "Evidence is contradictory. Different studies show conflicting results.",
            "question_type": "contradiction",
            "source_doc": None,
        })
    
    # --- Impossible/abstain questions (15%) ---
    abstain_questions = [
        "What are the effects of Zyxoplatin (a novel compound) on pancreatic cancer?",
        "What is the recommended dosage of Unobtanium for chronic fatigue syndrome?",
        "How does quantum neural therapy compare to standard SSRI treatment for anxiety?",
    ]
    
    n_abstain = n_questions - len(eval_pairs)
    for i in range(n_abstain):
        question = abstain_questions[i % len(abstain_questions)]
        eval_pairs.append({
            "question": question,
            "ground_truth": "ABSTAIN - No relevant evidence available in the corpus.",
            "question_type": "abstain",
            "source_doc": None,
        })
    
    return eval_pairs[:n_questions]


def save_eval_dataset(eval_pairs: List[Dict], filename: str = "eval_dataset.json"):
    """Save evaluation dataset to JSON file."""
    output_path = EVAL_RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(eval_pairs, f, indent=2)
    
    print(f"  ✅ Saved {len(eval_pairs)} eval pairs to: {output_path}")
    return output_path


def main():
    """Generate and save the evaluation dataset."""
    print("Generating evaluation dataset...")
    eval_pairs = generate_eval_dataset(n_questions=20)
    save_eval_dataset(eval_pairs)
    
    # Print summary
    types = {}
    for pair in eval_pairs:
        qt = pair["question_type"]
        types[qt] = types.get(qt, 0) + 1
    
    print(f"\n  Question type distribution:")
    for qt, count in types.items():
        print(f"    {qt}: {count}")


if __name__ == "__main__":
    main()
