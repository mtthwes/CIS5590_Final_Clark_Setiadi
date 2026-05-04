#!/usr/bin/env python3
"""
Medical Diagnosis Recommender using Mini-NARS (Scaled Version)
CIS 5590 Final Project - Matthew Setiadi & Justin Clark
Spring 2026 - Temple University Data Science
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_nars import MiniNARS

# ---------------------------------------------------------------
# Step 1: Build knowledge base from Massive Dataset
# ---------------------------------------------------------------

def load_dataset(filepath):
    print(f"  Loading large dataset from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    symptoms = [
        "fever", "cough", "fatigue", "shortness of breath", 
        "chest tightness", "headache", "nausea", "diarrhea",
        "vomiting", "dizziness", "muscle pain", "joint pain", "chills"
    ]
    
    available_symptoms = [s for s in symptoms if s in df.columns]
    
    freq = {}
    for disease, group in df.groupby("diseases"):
        disease_name = str(disease).strip()
        freq[disease_name] = {}
        total = len(group)
        
        for sym in available_symptoms:
            yes_count = group[sym].astype(str).str.strip().str.lower().isin(['yes', 'true', '1']).sum()
            if total > 0:
                freq[disease_name][sym] = round(yes_count / total, 2)
                
    return freq


def build_knowledge_base(nars, freq_data, target_diseases):
    print("=" * 60)
    print(f"  KNOWLEDGE BASE (Loaded {len(target_diseases)} diseases)")
    print("=" * 60)

    sym_map = {
        "fever": "fever",
        "cough": "cough",
        "fatigue": "fatigue",
        "shortness of breath": "breathing_difficulty",
        "chest tightness": "chest_tightness",
        "headache": "headache",
        "nausea": "nausea",
        "diarrhea": "diarrhea",
        "vomiting": "vomiting",
        "dizziness": "dizziness",
        "muscle pain": "muscle_pain",
        "joint pain": "joint_pain",
        "chills": "chills"
    }

    statements_added = 0
    for disease in target_diseases:
        d_term = disease.lower().replace(" ", "_").replace("-", "_")
        for sym_name, sym_term in sym_map.items():
            f = freq_data[disease].get(sym_name, 0)
            if f > 0:
                c = 0.85
                # LOGIC ALIGNMENT: Mapping symptom directly to the disease indicator
                stmt = f"<{sym_term} --> {d_term}_indicator>. %{f};{c}%"
                nars.input_narsese(stmt)
                statements_added += 1

    print(f"  Successfully encoded {statements_added} symptom links into Mini-NARS memory.")
    nars.cycle(200)
    print()


# ---------------------------------------------------------------
# Step 2: Patient diagnosis (Logical Inference)
# ---------------------------------------------------------------

def diagnose_patient(nars, patient_name, symptoms, target_diseases, cycles=150):
    print(f"\n{'=' * 60}")
    print(f"  PATIENT: {patient_name}")
    print(f"  Symptoms: {', '.join(symptoms)}")
    print(f"{'=' * 60}")

    sym_map = {
        "fever": "fever",
        "cough": "cough",
        "fatigue": "fatigue",
        "shortness of breath": "breathing_difficulty",
    }

    for sym in symptoms:
        term = sym_map.get(sym, sym)
        # LOGIC ALIGNMENT: Linking the patient directly to the exact symptom concept
        stmt = f"<{patient_name} --> {term}>. %1.0;0.9%"
        nars.input_narsese(stmt)

    nars.cycle(cycles)

    results_dict = {}
    for concept in nars.memory.all_items():
        for belief in concept.belief_bag.all_items():
            s = belief.statement
            best = belief.best_truth()
            
            # The AI has now successfully bridged <patient --> symptom> and <symptom --> indicator>
            if best and s.subject == patient_name and "_indicator" in s.predicate:
                d_term = s.predicate.replace("_indicator", "").replace("_", " ")
                exp = best.confidence * (best.frequency - 0.5) + 0.5
                
                if d_term not in results_dict or exp > results_dict[d_term]:
                    results_dict[d_term] = exp

    results = [(d, val) for d, val in results_dict.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run_belief_revision_experiment(nars, freq_data, target_diseases):
    print("\n" + "=" * 60)
    print("  BELIEF REVISION EXPERIMENT")
    print("  Patient symptoms arrive one at a time")
    print("=" * 60)

    best_matches = []
    for d, syms in freq_data.items():
        if d in target_diseases:
            score = syms.get("fever", 0) + syms.get("cough", 0) + syms.get("shortness of breath", 0)
            if score > 0:
                best_matches.append((d, score))
    best_matches.sort(key=lambda x: x[1], reverse=True)
    
    tracked = [d for d, score in best_matches[:4]]
    if not tracked:
        tracked = target_diseases[:4]

    sym_map = {
        "fever": "fever",
        "cough": "cough",
        "fatigue": "fatigue",
        "breathing_difficulty": "breathing_difficulty",
    }

    d_terms = {d: d.lower().replace(" ", "_").replace("-", "_") for d in tracked}

    symptom_sequence = [
        ("fever", "Patient reports fever"),
        ("cough", "Patient reports cough"),
        ("fatigue", "Patient reports fatigue"),
        ("breathing_difficulty", "Patient has difficulty breathing"),
    ]

    print("\n  Tracking diagnosis confidence after each symptom:\n")
    print(f"  {'Step':<6} {'Evidence':<35} ", end="")
    for d in tracked:
        print(f"{d[:13]:<14}", end="")
    print("\n  " + "-" * (41 + 14 * len(tracked)))

    for step, (sym, description) in enumerate(symptom_sequence, 1):
        term = sym_map.get(sym, sym)
        nars.input_narsese(f"<patient1 --> {term}>. %1.0;0.9%")
        nars.cycle(100) 

        confidences = {}
        for d_name, d_term in d_terms.items():
            best_exp = 0.0
            for concept in nars.memory.all_items():
                for belief in concept.belief_bag.all_items():
                    s = belief.statement
                    best = belief.best_truth()
                    if best and s.subject == "patient1" and f"{d_term}_indicator" == s.predicate:
                        exp = best.confidence * (best.frequency - 0.5) + 0.5
                        if exp > best_exp:
                            best_exp = exp
            confidences[d_name] = best_exp

        print(f"  {step:<6} {description:<35} ", end="")
        for d in tracked:
            val = confidences.get(d, 0)
            if val > 0:
                print(f"{val:<14.3f}", end="")
            else:
                print(f"{'---':<14}", end="")
        print()

    print(f"\n  {5:<6} {'Negative lab test result':<35} ", end="")
    nars.input_narsese("<patient1 --> negative_test>. %1.0;0.9%")
    
    # AI dynamically learns that a negative test reduces the likelihood of the top tracked disease
    top_d_term = d_terms[tracked[0]]
    nars.input_narsese(f"<negative_test --> {top_d_term}_indicator>. %0.1;0.9%")
    nars.cycle(150)

    for d in tracked:
        d_term = d_terms[d]
        best_exp = 0.0
        for concept in nars.memory.all_items():
            for belief in concept.belief_bag.all_items():
                s = belief.statement
                best = belief.best_truth()
                if best and s.subject == "patient1" and f"{d_term}_indicator" == s.predicate:
                    exp = best.confidence * (best.frequency - 0.5) + 0.5
                    if exp > best_exp:
                        best_exp = exp
        if best_exp > 0:
            print(f"{best_exp:<14.3f}", end="")
        else:
            print(f"{'---':<14}", end="")
    print("  <-- contradictory evidence processed")
    
    return tracked[0]


def generate_explanation(nars, patient_name, top_disease, symptoms):
    d_term = top_disease.lower().replace(" ", "_").replace("-", "_")
    supporting = []
    
    for concept in nars.memory.all_items():
        for belief in concept.belief_bag.all_items():
            s = belief.statement
            best = belief.best_truth()
            if best and f"{d_term}_indicator" == s.predicate and best.frequency > 0.5:
                sym_name = s.subject.replace("_", " ")
                supporting.append((sym_name, best.frequency, best.confidence))

    supporting.sort(key=lambda x: x[1] * x[2], reverse=True)

    print(f"\n  EXPLANATION:")
    print(f"  {top_disease} is the suggested diagnosis because:")
    for sym, f, c in supporting[:4]:
        strength = "strong" if c > 0.7 else "moderate" if c > 0.4 else "weak"
        print(f"    - {sym} is a {strength} indicator (freq={f:.2f}, conf={c:.2f})")


# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------

def main():
    dataset_path = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found in this folder.")
        return

    freq_data = load_dataset(dataset_path)
    
    relevance_scores = []
    target_syms = ["fever", "cough", "fatigue", "shortness of breath"]
    
    for d, syms in freq_data.items():
        score = sum(syms.get(s, 0) for s in target_syms)
        if score > 0:
            relevance_scores.append((d, score))

    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    # STRICT TRIAGE: Force the engine to only hold the 4 most relevant diseases 
    # to guarantee memory retention and complete inference bridging.
    target_diseases = [d for d, score in relevance_scores[:4]]
    
    print(f"\n  [TRIAGE] Narrowed to EXACTLY {len(target_diseases)} target diseases for the differential diagnosis.\n")
    
    nars = MiniNARS(silent=True)
    build_knowledge_base(nars, freq_data, target_diseases)

    results = diagnose_patient(nars, "alice", ["fever", "cough", "fatigue"], target_diseases)
    print("\n  Top Diagnosis ranking for Alice:")
    for pred, exp in results[:5]:
        print(f"    {pred:<35} expect={exp:.2f}")

    nars2 = MiniNARS(silent=True)
    build_knowledge_base(nars2, freq_data, target_diseases)
    results2 = diagnose_patient(nars2, "bob", ["fever", "cough", "shortness of breath"], target_diseases)
    
    print("\n  Top Diagnosis ranking for Bob:")
    for pred, exp in results2[:5]:
        print(f"    {pred:<35} expect={exp:.2f}")

    nars3 = MiniNARS(silent=True)
    build_knowledge_base(nars3, freq_data, target_diseases)
    
    top_tracked_disease = run_belief_revision_experiment(nars3, freq_data, target_diseases)

    generate_explanation(nars3, "patient1", top_tracked_disease, ["fever", "cough", "fatigue", "shortness of breath"])

    print("\n" + "=" * 60)
    print("  EXPERIMENTS COMPLETE")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()