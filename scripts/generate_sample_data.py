"""
Generate 50 synthetic clinical trial documents for testing the RAG pipeline.
Documents cover diverse clinical scenarios with some deliberate contradictions
to test the contradiction detection capability.
"""

import os
import random
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

# --- Clinical Document Templates ---

DRUGS = [
    "Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole",
    "Dapagliflozin", "Semaglutide", "Empagliflozin", "Tirzepatide", "Canagliflozin",
    "Sitagliptin", "Losartan", "Valsartan", "Ramipril", "Rosuvastatin"
]

CONDITIONS = [
    "Type 2 Diabetes Mellitus", "Hypertension", "Hyperlipidemia",
    "Chronic Kidney Disease", "Heart Failure", "Obesity",
    "Coronary Artery Disease", "Atrial Fibrillation", "COPD",
    "Major Depressive Disorder"
]

ADVERSE_EVENTS = [
    "nausea", "headache", "dizziness", "fatigue", "gastrointestinal discomfort",
    "hypoglycemia", "urinary tract infection", "upper respiratory infection",
    "diarrhea", "constipation", "muscle pain", "joint pain", "insomnia",
    "peripheral edema", "cough", "rash", "weight gain", "elevated liver enzymes"
]

STUDY_PHASES = ["Phase I", "Phase II", "Phase III", "Phase IV"]


def generate_efficacy_trial(doc_id, drug, condition, positive=True):
    """Generate a drug efficacy trial document."""
    n_patients = random.randint(200, 5000)
    duration_weeks = random.choice([12, 24, 36, 48, 52])
    mean_age = random.randint(45, 68)
    female_pct = random.randint(35, 65)
    phase = random.choice(STUDY_PHASES[1:])
    
    if positive:
        primary_result = f"significantly reduced" if condition in ["Type 2 Diabetes Mellitus", "Hypertension", "Hyperlipidemia"] else "significantly improved"
        p_value = round(random.uniform(0.001, 0.04), 4)
        effect_size = round(random.uniform(0.3, 0.8), 2)
        conclusion = f"{drug} demonstrated statistically significant efficacy in the treatment of {condition}."
    else:
        primary_result = "did not significantly differ from placebo in"
        p_value = round(random.uniform(0.06, 0.45), 4)
        effect_size = round(random.uniform(0.01, 0.15), 2)
        conclusion = f"{drug} did not demonstrate statistically significant efficacy in the treatment of {condition} compared to placebo."
    
    primary_endpoint_map = {
        "Type 2 Diabetes Mellitus": f"HbA1c reduction from baseline (primary: {round(random.uniform(0.5, 1.8), 1)}% vs placebo)",
        "Hypertension": f"change in systolic blood pressure ({round(random.uniform(5, 18), 1)} mmHg reduction vs placebo)",
        "Hyperlipidemia": f"LDL-C reduction ({round(random.uniform(15, 55), 1)}% from baseline)",
        "Heart Failure": f"reduction in heart failure hospitalization (HR: {round(random.uniform(0.65, 0.85), 2)})",
        "Obesity": f"body weight reduction ({round(random.uniform(3, 15), 1)}% from baseline)",
    }
    primary_endpoint = primary_endpoint_map.get(condition, f"improvement in {condition} composite score")
    
    ae1, ae2, ae3 = random.sample(ADVERSE_EVENTS, 3)
    ae1_pct = round(random.uniform(2, 15), 1)
    ae2_pct = round(random.uniform(1, 10), 1)
    ae3_pct = round(random.uniform(0.5, 8), 1)
    
    return f"""CLINICAL TRIAL REPORT
Document ID: CTR-{doc_id:04d}
Study Title: {phase} Randomized, Double-Blind, Placebo-Controlled Trial of {drug} for {condition}

STUDY OVERVIEW
This was a {phase.lower()}, multicenter, randomized, double-blind, placebo-controlled clinical trial evaluating the efficacy and safety of {drug} in patients with {condition}. The study was conducted across {random.randint(15, 120)} clinical sites in {random.randint(5, 20)} countries.

PATIENT POPULATION
A total of {n_patients} patients were enrolled and randomized 1:1 to {drug} or placebo. Key inclusion criteria included age 18-75 years with confirmed diagnosis of {condition}. The mean age of participants was {mean_age} years, with {female_pct}% female patients. {random.randint(60, 85)}% of patients were White, {random.randint(8, 20)}% Black or African American, and {random.randint(5, 15)}% Hispanic or Latino.

STUDY DESIGN
Patients received {drug} at a dose of {random.choice(['5mg', '10mg', '25mg', '50mg', '100mg', '250mg', '500mg'])} {random.choice(['once daily', 'twice daily'])} or matching placebo for {duration_weeks} weeks. The primary endpoint was {primary_endpoint}. Secondary endpoints included safety assessments, quality of life measures (SF-36), and time to first clinical event.

RESULTS
Primary Endpoint: {drug} {primary_result} the primary endpoint compared to placebo (p={p_value}, effect size={effect_size}, 95% CI [{round(effect_size - random.uniform(0.05, 0.15), 2)}, {round(effect_size + random.uniform(0.05, 0.15), 2)}]).

The treatment group showed a mean change from baseline of {round(random.uniform(-2.5, -0.3), 2)} compared to {round(random.uniform(-0.8, 0.2), 2)} in the placebo group.

Safety Profile:
The most common adverse events in the {drug} group were:
- {ae1}: {ae1_pct}% (vs {round(ae1_pct * random.uniform(0.3, 0.7), 1)}% placebo)
- {ae2}: {ae2_pct}% (vs {round(ae2_pct * random.uniform(0.3, 0.7), 1)}% placebo)
- {ae3}: {ae3_pct}% (vs {round(ae3_pct * random.uniform(0.3, 0.7), 1)}% placebo)

Serious adverse events occurred in {round(random.uniform(1, 5), 1)}% of the {drug} group vs {round(random.uniform(0.5, 4), 1)}% of the placebo group. {random.randint(0, 3)} deaths occurred during the study, none considered related to the study drug.

CONCLUSION
{conclusion} The safety profile of {drug} was consistent with prior studies, with no new safety signals identified. Based on these results, {drug} {'represents a promising therapeutic option' if positive else 'requires further investigation with modified endpoints or dosing'} for patients with {condition}.

REFERENCES
1. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])} et al. ({random.randint(2019, 2024)}). {drug} in {condition}: A systematic review. The Lancet. {random.randint(380, 410)}:{random.randint(100, 999)}-{random.randint(1000, 1200)}.
2. {random.choice(['Anderson', 'Thomas', 'Jackson', 'White', 'Harris'])} et al. ({random.randint(2019, 2024)}). Safety profile of {drug}: Pooled analysis. NEJM. {random.randint(370, 395)}:{random.randint(100, 999)}-{random.randint(1000, 1200)}.
"""


def generate_adverse_event_report(doc_id, drug):
    """Generate an adverse event surveillance report."""
    n_reports = random.randint(500, 10000)
    aes = random.sample(ADVERSE_EVENTS, random.randint(4, 7))
    
    ae_table = ""
    for ae in aes:
        count = random.randint(10, 500)
        rate = round(count / n_reports * 100, 2)
        serious = random.randint(0, int(count * 0.2))
        ae_table += f"  - {ae}: {count} reports ({rate}%), {serious} serious\n"
    
    return f"""ADVERSE EVENT SURVEILLANCE REPORT
Document ID: AER-{doc_id:04d}
Drug: {drug}
Reporting Period: {random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {random.randint(2022, 2024)}

SUMMARY
This post-marketing surveillance report summarizes adverse event data for {drug} collected through the FDA Adverse Event Reporting System (FAERS) and manufacturer safety databases. A total of {n_reports} adverse event reports were analyzed during the reporting period.

ADVERSE EVENT DISTRIBUTION
The following adverse events were reported:
{ae_table}
SIGNAL DETECTION
{"A new safety signal was identified for " + random.choice(aes) + " which exceeded the proportional reporting ratio (PRR) threshold of 2.0 (observed PRR: " + str(round(random.uniform(2.1, 4.5), 2)) + ")." if random.random() > 0.5 else "No new safety signals were identified during this reporting period. All adverse event rates remain within expected ranges based on clinical trial data."}

RISK-BENEFIT ASSESSMENT
The overall benefit-risk profile of {drug} remains {'favorable' if random.random() > 0.3 else 'under review'} based on current data. {random.choice(['Continued monitoring is recommended.', 'Label updates may be warranted for specific populations.', 'No regulatory action is recommended at this time.'])}

DEMOGRAPHICS OF REPORTERS
- Age range: {random.randint(18, 30)}-{random.randint(70, 90)} years (median: {random.randint(50, 65)})
- Sex: {random.randint(40, 60)}% Female, {random.randint(35, 55)}% Male, {random.randint(1, 5)}% Not reported
- Geographic distribution: {random.randint(50, 70)}% North America, {random.randint(15, 30)}% Europe, {random.randint(5, 15)}% Asia-Pacific
"""


def generate_dosing_protocol(doc_id, drug, condition):
    """Generate a dosing protocol study."""
    doses = sorted(random.sample([2.5, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 500], 3))
    optimal_dose = doses[1]  # Middle dose usually optimal
    
    return f"""DOSE-FINDING STUDY REPORT
Document ID: DFS-{doc_id:04d}
Study Title: Dose-Response Evaluation of {drug} in {condition}

OBJECTIVE
To determine the optimal therapeutic dose of {drug} for the treatment of {condition} by evaluating efficacy and safety across multiple dose levels.

METHODS
This was a {random.choice(['Phase IIa', 'Phase IIb'])} dose-ranging study. {random.randint(150, 800)} patients with {condition} were randomized to receive {drug} at doses of {doses[0]}mg, {doses[1]}mg, or {doses[2]}mg daily, or placebo, for {random.choice([8, 12, 16, 24])} weeks.

PHARMACOKINETIC PROFILE
{drug} demonstrated linear pharmacokinetics across the dose range studied:
- Tmax: {round(random.uniform(1, 4), 1)} hours
- Half-life: {round(random.uniform(4, 24), 1)} hours
- Bioavailability: {random.randint(40, 95)}%
- Steady-state achieved by Day {random.randint(3, 14)}
- No significant food effect observed

DOSE-RESPONSE RESULTS
{doses[0]}mg: Modest improvement over placebo (effect size: {round(random.uniform(0.1, 0.3), 2)}, p={round(random.uniform(0.05, 0.15), 3)})
{doses[1]}mg: Significant improvement (effect size: {round(random.uniform(0.4, 0.7), 2)}, p={round(random.uniform(0.001, 0.01), 4)}) — RECOMMENDED DOSE
{doses[2]}mg: Similar efficacy to {doses[1]}mg but with increased adverse events (effect size: {round(random.uniform(0.4, 0.65), 2)}, p={round(random.uniform(0.001, 0.01), 4)})

SAFETY BY DOSE
Treatment-emergent adverse events increased with dose:
- {doses[0]}mg: {random.randint(15, 25)}% of patients
- {doses[1]}mg: {random.randint(20, 35)}% of patients
- {doses[2]}mg: {random.randint(35, 55)}% of patients
- Placebo: {random.randint(10, 20)}% of patients

RECOMMENDATION
The optimal dose of {drug} for {condition} is {optimal_dose}mg {random.choice(['once daily', 'twice daily'])}, providing the best balance of efficacy and tolerability. Dose adjustment may be needed for patients with {'renal impairment (eGFR < 30 mL/min)' if random.random() > 0.5 else 'hepatic impairment (Child-Pugh B or C)'}.
"""


def generate_comparative_study(doc_id, drug1, drug2, condition, drug1_better=True):
    """Generate a head-to-head comparative effectiveness study."""
    n_patients = random.randint(300, 3000)
    
    if drug1_better:
        d1_effect = round(random.uniform(0.5, 0.9), 2)
        d2_effect = round(random.uniform(0.2, 0.45), 2)
        winner = drug1
    else:
        d1_effect = round(random.uniform(0.2, 0.45), 2)
        d2_effect = round(random.uniform(0.5, 0.9), 2)
        winner = drug2
    
    return f"""COMPARATIVE EFFECTIVENESS STUDY
Document ID: CES-{doc_id:04d}
Study Title: Head-to-Head Comparison of {drug1} vs {drug2} for {condition}

BACKGROUND
Both {drug1} and {drug2} are approved for the treatment of {condition}. This study aimed to compare their relative efficacy and safety in a real-world clinical setting.

STUDY DESIGN
Multicenter, randomized, active-controlled, open-label trial. {n_patients} patients with {condition} were randomized 1:1 to {drug1} or {drug2} for {random.choice([24, 36, 48, 52])} weeks.

PATIENT BASELINE CHARACTERISTICS
- Mean age: {random.randint(48, 65)} years
- Mean BMI: {round(random.uniform(25, 35), 1)} kg/m²
- Mean disease duration: {round(random.uniform(2, 15), 1)} years
- Prior treatment failure: {random.randint(20, 60)}%
- Concomitant medications: {random.randint(40, 80)}% on at least one

PRIMARY RESULTS
{drug1}: Effect size {d1_effect} (95% CI [{round(d1_effect-0.1, 2)}, {round(d1_effect+0.1, 2)}])
{drug2}: Effect size {d2_effect} (95% CI [{round(d2_effect-0.1, 2)}, {round(d2_effect+0.1, 2)}])

Between-group difference: {round(abs(d1_effect - d2_effect), 2)} (p={'<0.001' if abs(d1_effect - d2_effect) > 0.3 else '=' + str(round(random.uniform(0.01, 0.04), 3))})

{winner} demonstrated superior efficacy compared to {drug1 if winner == drug2 else drug2} on the primary endpoint.

SAFETY COMPARISON
Treatment discontinuation due to adverse events:
- {drug1}: {round(random.uniform(3, 12), 1)}%
- {drug2}: {round(random.uniform(3, 12), 1)}%

Notable differences in adverse event profiles:
- {drug1} was associated with higher rates of {random.choice(ADVERSE_EVENTS)} ({round(random.uniform(5, 15), 1)}% vs {round(random.uniform(1, 5), 1)}%)
- {drug2} was associated with higher rates of {random.choice(ADVERSE_EVENTS)} ({round(random.uniform(5, 15), 1)}% vs {round(random.uniform(1, 5), 1)}%)

CONCLUSION
In this head-to-head comparison, {winner} demonstrated statistically significant superiority over {drug1 if winner == drug2 else drug2} for the treatment of {condition}. The safety profiles of both drugs were generally acceptable, though they differed in specific adverse event patterns. Treatment selection should consider individual patient factors and tolerability preferences.
"""


def generate_meta_analysis(doc_id, drug, condition):
    """Generate a meta-analysis summary document."""
    n_studies = random.randint(8, 30)
    n_patients = random.randint(5000, 50000)
    
    return f"""META-ANALYSIS REPORT
Document ID: MAR-{doc_id:04d}
Title: Systematic Review and Meta-Analysis of {drug} for {condition}

OBJECTIVE
To synthesize evidence from randomized controlled trials (RCTs) evaluating the efficacy and safety of {drug} in patients with {condition}.

SEARCH STRATEGY
Electronic databases (PubMed, Embase, Cochrane Library) were searched from inception to {random.choice(['January', 'March', 'June', 'September'])} {random.randint(2023, 2024)}. {random.randint(200, 1500)} records were screened, and {n_studies} RCTs meeting inclusion criteria were included in the final analysis.

INCLUDED STUDIES
- Number of RCTs: {n_studies}
- Total patients: {n_patients}
- Study duration range: {random.randint(8, 12)} to {random.randint(48, 104)} weeks
- {random.choice(['Phase II', 'Phase III'])} trials: {random.randint(60, 80)}%
- Geographic coverage: {random.randint(20, 50)} countries

POOLED RESULTS
Primary Outcome:
- Pooled effect size (random-effects model): {round(random.uniform(0.3, 0.8), 2)} (95% CI [{round(random.uniform(0.2, 0.35), 2)}, {round(random.uniform(0.75, 0.95), 2)}])
- p-value: <0.001
- I² heterogeneity: {random.randint(10, 65)}% ({'low' if random.randint(10, 65) < 25 else 'moderate' if random.randint(10, 65) < 50 else 'substantial'})
- Number needed to treat (NNT): {random.randint(4, 20)}

Safety Outcomes:
- Risk of serious adverse events: OR {round(random.uniform(0.8, 1.3), 2)} (95% CI [{round(random.uniform(0.6, 0.8), 2)}, {round(random.uniform(1.2, 1.6), 2)}])
- Treatment discontinuation: OR {round(random.uniform(1.0, 1.8), 2)} (95% CI [{round(random.uniform(0.8, 1.0), 2)}, {round(random.uniform(1.5, 2.2), 2)}])

SUBGROUP ANALYSES
- Age ≥65 years: Similar benefit (interaction p={round(random.uniform(0.2, 0.8), 2)})
- Severe disease at baseline: Greater benefit (effect size: {round(random.uniform(0.5, 1.0), 2)})
- Prior treatment failure: Maintained efficacy (effect size: {round(random.uniform(0.3, 0.7), 2)})

CONCLUSION
This meta-analysis of {n_studies} RCTs involving {n_patients} patients provides {'strong' if n_studies > 15 else 'moderate'} evidence supporting the efficacy of {drug} for {condition}. The safety profile is acceptable though ongoing surveillance is warranted.

QUALITY ASSESSMENT
- Risk of bias: {random.choice(['Low', 'Low to moderate', 'Moderate'])} (Cochrane RoB 2 tool)
- GRADE certainty of evidence: {random.choice(['High', 'Moderate', 'Low'])}
- Publication bias: {'Not detected (Egger test p=' + str(round(random.uniform(0.1, 0.8), 2)) + ')' if random.random() > 0.3 else 'Possible (Egger test p=' + str(round(random.uniform(0.01, 0.05), 3)) + ')'}
"""


def generate_all_documents():
    """Generate the full set of 50 clinical documents."""
    documents = []
    doc_id = 1
    
    # --- 15 Efficacy Trials (12 positive, 3 negative) ---
    for i in range(12):
        drug = DRUGS[i % len(DRUGS)]
        condition = CONDITIONS[i % len(CONDITIONS)]
        documents.append((f"efficacy_trial_{doc_id:03d}.txt", 
                          generate_efficacy_trial(doc_id, drug, condition, positive=True)))
        doc_id += 1
    
    for i in range(3):
        drug = DRUGS[(i + 5) % len(DRUGS)]
        condition = CONDITIONS[(i + 3) % len(CONDITIONS)]
        documents.append((f"efficacy_trial_{doc_id:03d}.txt",
                          generate_efficacy_trial(doc_id, drug, condition, positive=False)))
        doc_id += 1
    
    # --- 8 Adverse Event Reports ---
    for i in range(8):
        drug = DRUGS[i % len(DRUGS)]
        documents.append((f"adverse_event_report_{doc_id:03d}.txt",
                          generate_adverse_event_report(doc_id, drug)))
        doc_id += 1
    
    # --- 7 Dosing Protocol Studies ---
    for i in range(7):
        drug = DRUGS[i % len(DRUGS)]
        condition = CONDITIONS[i % len(CONDITIONS)]
        documents.append((f"dosing_study_{doc_id:03d}.txt",
                          generate_dosing_protocol(doc_id, drug, condition)))
        doc_id += 1
    
    # --- 10 Comparative Effectiveness Studies ---
    for i in range(10):
        drug1 = DRUGS[i % len(DRUGS)]
        drug2 = DRUGS[(i + 3) % len(DRUGS)]
        condition = CONDITIONS[i % len(CONDITIONS)]
        documents.append((f"comparative_study_{doc_id:03d}.txt",
                          generate_comparative_study(doc_id, drug1, drug2, condition, 
                                                   drug1_better=(i % 3 != 0))))
        doc_id += 1
    
    # --- 5 Meta-Analyses ---
    for i in range(5):
        drug = DRUGS[i % len(DRUGS)]
        condition = CONDITIONS[i % len(CONDITIONS)]
        documents.append((f"meta_analysis_{doc_id:03d}.txt",
                          generate_meta_analysis(doc_id, drug, condition)))
        doc_id += 1
    
    # --- 5 Deliberately Contradictory Pairs ---
    # These create conflicting evidence about the same drug-condition pair
    # Pair 1: Conflicting efficacy of Semaglutide for Obesity
    documents.append((f"contradiction_positive_{doc_id:03d}.txt",
                      generate_efficacy_trial(doc_id, "Semaglutide", "Obesity", positive=True)))
    doc_id += 1
    documents.append((f"contradiction_negative_{doc_id:03d}.txt",
                      generate_efficacy_trial(doc_id, "Semaglutide", "Obesity", positive=False)))
    doc_id += 1
    
    # Pair 2: Conflicting comparative results
    documents.append((f"contradiction_comp_a_{doc_id:03d}.txt",
                      generate_comparative_study(doc_id, "Metformin", "Sitagliptin", 
                                                "Type 2 Diabetes Mellitus", drug1_better=True)))
    doc_id += 1
    documents.append((f"contradiction_comp_b_{doc_id:03d}.txt",
                      generate_comparative_study(doc_id, "Metformin", "Sitagliptin",
                                                "Type 2 Diabetes Mellitus", drug1_better=False)))
    doc_id += 1
    
    # Pair 3: Conflicting safety data
    documents.append((f"contradiction_safety_{doc_id:03d}.txt",
                      generate_adverse_event_report(doc_id, "Empagliflozin")))
    doc_id += 1
    
    return documents


def main():
    """Generate and save all clinical documents."""
    print(f"Generating synthetic clinical documents in: {DATA_DIR}")
    
    random.seed(42)  # Reproducibility
    documents = generate_all_documents()
    
    for filename, content in documents:
        filepath = DATA_DIR / filename
        filepath.write_text(content.strip())
        print(f"  ✓ Created: {filename}")
    
    print(f"\n✅ Generated {len(documents)} clinical documents successfully!")
    print(f"   Location: {DATA_DIR}")
    return len(documents)


if __name__ == "__main__":
    main()
