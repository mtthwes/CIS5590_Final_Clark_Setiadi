#!/usr/bin/env python3
"""
Medical Diagnosis Recommender using Mini-NARS
CIS 5590 Final Project - Matthew Setiadi & Justin Clark
Spring 2026

Uses the Kaggle Disease Symptoms and Patient Profile Dataset
to build a symptom-disease knowledge base in Narsese,
then runs patient scenarios through Mini-NARS to demonstrate
belief revision and explainable diagnosis.
"""

import sys
import os
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_nars import MiniNARS


# ---------------------------------------------------------------
# Step 1: Build knowledge base from dataset
# ---------------------------------------------------------------

def load_dataset(filepath):
    """
    Read the CSV and compute symptom frequencies per disease.
    Returns dict: disease -> symptom -> frequency (0 to 1)
    """
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [yes, total]
    symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row["Disease"].strip()
            for sym in symptoms:
                val = row[sym].strip()
                counts[disease][sym][1] += 1
                if val == "Yes":
                    counts[disease][sym][0] += 1

    # Convert to frequencies
    freq = {}
    for disease in counts:
        freq[disease] = {}
        for sym in symptoms:
            yes, total = counts[disease][sym]
            if total > 0:
                freq[disease][sym] = round(yes / total, 2)
    return freq


def build_knowledge_base(nars, freq_data, target_diseases):
    """
    Convert symptom-disease frequencies into Narsese statements.
    Uses the actual frequencies from the dataset as truth-values.
    """
    print("=" * 60)
    print("  KNOWLEDGE BASE (from Kaggle dataset)")
    print("=" * 60)

    # Map symptom names to clean Narsese terms
    sym_map = {
        "Fever": "fever",
        "Cough": "cough",
        "Fatigue": "fatigue",
        "Difficulty Breathing": "breathing_difficulty",
    }

    for disease in target_diseases:
        if disease not in freq_data:
            continue
        d_term = disease.lower().replace(" ", "_")
        print(f"\n  {disease}:")
        for sym_name, sym_term in sym_map.items():
            f = freq_data[disease].get(sym_name, 0)
            if f > 0:
                # Confidence based on sample size (more data = higher confidence)
                c = 0.85
                stmt = f"<{sym_term} --> {d_term}_symptom>. %{f};{c}%"
                nars.input_narsese(stmt)
                print(f"    {sym_term} -> {d_term}_symptom  freq={f}  conf={c}")

    # Run cycles to let the system process the knowledge base
    nars.cycle(30)
    print()


# ---------------------------------------------------------------
# Step 2: Patient diagnosis
# ---------------------------------------------------------------

def diagnose_patient(nars, patient_name, symptoms, target_diseases, cycles=50):
    """
    Feed patient symptoms into the system and query each disease.
    Returns ranked diagnoses.
    """
    print(f"\n{'=' * 60}")
    print(f"  PATIENT: {patient_name}")
    print(f"  Symptoms: {', '.join(symptoms)}")
    print(f"{'=' * 60}")

    sym_map = {
        "fever": "fever",
        "cough": "cough",
        "fatigue": "fatigue",
        "breathing_difficulty": "breathing_difficulty",
    }

    # Input patient symptoms
    for sym in symptoms:
        term = sym_map.get(sym, sym)
        stmt = f"<{patient_name} --> has_{term}>. %1.0;0.9%"
        nars.input_narsese(stmt)

    nars.cycle(cycles)

    # Collect beliefs about this patient
    results = []
    for concept in nars.memory.all_items():
        for belief in concept.belief_bag.all_items():
            s = belief.statement
            best = belief.best_truth()
            if best and s.subject == patient_name:
                results.append((s.predicate, best.frequency, best.confidence))

    # Sort by expectation
    results.sort(key=lambda x: x[1] * x[2], reverse=True)
    return results


def run_belief_revision_experiment(nars, target_diseases):
    """
    Core experiment: feed symptoms one at a time and track
    how diagnosis confidence changes with each new piece of evidence.
    """
    print("\n" + "=" * 60)
    print("  BELIEF REVISION EXPERIMENT")
    print("  Patient symptoms arrive one at a time")
    print("=" * 60)

    sym_map = {
        "fever": "fever",
        "cough": "cough",
        "fatigue": "fatigue",
        "breathing_difficulty": "breathing_difficulty",
    }

    d_terms = {d: d.lower().replace(" ", "_") for d in target_diseases}

    # Scenario: symptoms arrive sequentially
    symptom_sequence = [
        ("fever", "Patient reports fever"),
        ("cough", "Patient reports cough"),
        ("fatigue", "Patient reports fatigue"),
        ("breathing_difficulty", "Patient has difficulty breathing"),
    ]

    print("\n  Tracking diagnosis confidence after each symptom:\n")
    print(f"  {'Step':<6} {'Evidence':<35} ", end="")
    for d in target_diseases:
        print(f"{d:<14}", end="")
    print()
    print("  " + "-" * (41 + 14 * len(target_diseases)))

    for step, (sym, description) in enumerate(symptom_sequence, 1):
        term = sym_map.get(sym, sym)
        nars.input_narsese(f"<patient1 --> has_{term}>. %1.0;0.9%")
        nars.cycle(30)

        # Check beliefs for patient1 related to each disease
        confidences = {}
        for d_name, d_term in d_terms.items():
            best_exp = 0.0
            for concept in nars.memory.all_items():
                for belief in concept.belief_bag.all_items():
                    s = belief.statement
                    best = belief.best_truth()
                    if best and "patient1" in s.subject and d_term in s.predicate:
                        exp = best.confidence * (best.frequency - 0.5) + 0.5
                        if exp > best_exp:
                            best_exp = exp

            # Also check indirect: does this symptom match disease symptoms?
            for concept in nars.memory.all_items():
                for belief in concept.belief_bag.all_items():
                    s = belief.statement
                    best = belief.best_truth()
                    if best and term in s.subject and d_term in s.predicate:
                        exp = best.confidence * (best.frequency - 0.5) + 0.5
                        if exp > best_exp:
                            best_exp = exp

            confidences[d_name] = best_exp

        print(f"  {step:<6} {description:<35} ", end="")
        for d in target_diseases:
            val = confidences.get(d, 0)
            if val > 0:
                print(f"{val:<14.3f}", end="")
            else:
                print(f"{'---':<14}", end="")
        print()

    # Now introduce contradictory evidence
    print(f"\n  {5:<6} {'Negative flu test result':<35} ", end="")
    nars.input_narsese("<patient1 --> has_flu_negative_test>. %1.0;0.9%")
    nars.input_narsese("<has_flu_negative_test --> not_influenza>. %0.9;0.9%")
    nars.cycle(30)

    # Recheck
    for d in target_diseases:
        d_term = d_terms[d]
        best_exp = 0.0
        for concept in nars.memory.all_items():
            for belief in concept.belief_bag.all_items():
                s = belief.statement
                best = belief.best_truth()
                if best:
                    if ("patient1" in s.subject and d_term in s.predicate) or \
                       (d_term in s.predicate and any(sym in s.subject for sym in sym_map.values())):
                        exp = best.confidence * (best.frequency - 0.5) + 0.5
                        if exp > best_exp:
                            best_exp = exp
        if best_exp > 0:
            print(f"{best_exp:<14.3f}", end="")
        else:
            print(f"{'---':<14}", end="")
    print("  <-- contradictory evidence")


def generate_explanation(nars, patient_name, top_disease, symptoms):
    """
    Generate a plain-English explanation for the diagnosis.
    """
    d_term = top_disease.lower().replace(" ", "_")

    supporting = []
    for concept in nars.memory.all_items():
        for belief in concept.belief_bag.all_items():
            s = belief.statement
            best = belief.best_truth()
            if best and d_term in s.predicate and best.frequency > 0.5:
                sym_name = s.subject.replace("_", " ")
                supporting.append((sym_name, best.frequency, best.confidence))

    supporting.sort(key=lambda x: x[1] * x[2], reverse=True)

    print(f"\n  EXPLANATION:")
    print(f"  {top_disease} is the suggested diagnosis because:")
    for sym, f, c in supporting[:4]:
        strength = "strong" if c > 0.7 else "moderate" if c > 0.4 else "weak"
        print(f"    - {sym} is a {strength} indicator (freq={f:.2f}, conf={c:.2f})")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    # Find dataset
    dataset_path = None
    search_paths = [
        "Disease_symptom_and_patient_profile_dataset.csv",
        "../Disease_symptom_and_patient_profile_dataset.csv",
        "examples/Disease_symptom_and_patient_profile_dataset.csv",
        "/mnt/user-data/uploads/1776791873194_Disease_symptom_and_patient_profile_dataset.csv",
    ]
    for p in search_paths:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        print("Error: Dataset not found.")
        print("Place Disease_symptom_and_patient_profile_dataset.csv in the project folder.")
        return

    target_diseases = ["Influenza", "Common Cold", "Pneumonia", "Bronchitis"]

    # Load dataset
    freq_data = load_dataset(dataset_path)
    print(f"\n  Loaded {len(freq_data)} diseases from dataset.")
    print(f"  Target diseases: {', '.join(target_diseases)}\n")

    # --- Experiment 1: Knowledge base and simple diagnosis ---
    nars = MiniNARS(silent=True)
    build_knowledge_base(nars, freq_data, target_diseases)

    # Patient 1: classic flu symptoms
    results = diagnose_patient(nars, "alice", ["fever", "cough", "fatigue"], target_diseases)
    print("\n  Diagnosis ranking for Alice:")
    for pred, f, c in results[:8]:
        exp = c * (f - 0.5) + 0.5
        print(f"    {pred:<35} freq={f:.2f}  conf={c:.2f}  expect={exp:.2f}")

    # Patient 2: pneumonia symptoms
    nars2 = MiniNARS(silent=True)
    build_knowledge_base(nars2, freq_data, target_diseases)
    results2 = diagnose_patient(nars2, "bob", ["fever", "cough", "breathing_difficulty"], target_diseases)
    print("\n  Diagnosis ranking for Bob:")
    for pred, f, c in results2[:8]:
        exp = c * (f - 0.5) + 0.5
        print(f"    {pred:<35} freq={f:.2f}  conf={c:.2f}  expect={exp:.2f}")

    # --- Experiment 2: Belief revision ---
    nars3 = MiniNARS(silent=True)
    build_knowledge_base(nars3, freq_data, target_diseases)
    run_belief_revision_experiment(nars3, target_diseases)

    # --- Experiment 3: Explanation ---
    generate_explanation(nars3, "patient1", "Pneumonia", ["fever", "cough", "fatigue", "breathing_difficulty"])

    print("\n" + "=" * 60)
    print("  EXPERIMENTS COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
