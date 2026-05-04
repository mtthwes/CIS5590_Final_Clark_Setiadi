"""Experiment drivers.

Each function here runs one scripted experiment that exercises a specific
aspect of the project's reasoning goals:

  - `run_case`               : diagnose a single patient and explain the result.
  - `run_all_cases`          : diagnose the whole cohort and report accuracy.
  - `belief_revision_conflict`: feed supportive evidence, then contradictory
                                evidence, and track how the belief changes.
  - `compare_with_pynars`    : cross-check the deterministic NAL engine
                                against the actual PyNARS Reasoner on the
                                same inputs.

All experiments print their results to stdout so the output can be pasted
into reports or saved to a text file.
"""

from typing import List, Optional

from . import knowledge_base as kb
from .dataset import PATIENT_CASES, PatientCase, get_case
from .diagnosis_engine import DiagnosisEngine, EvidenceItem
from .explanation import (
    explain_diagnosis_summary,
    explain_trajectory,
    explain_top_diagnosis,
    _pretty_disease,
)
from .nal_truth import Truth


# ---------- Single-case diagnosis ----------

def run_case(case: PatientCase, verbose: bool = True) -> DiagnosisEngine:
    engine = DiagnosisEngine()
    engine.ingest_record(
        case.patient_id,
        symptoms_present=case.symptoms_present,
        symptoms_absent=case.symptoms_absent,
        labs_positive=case.labs_positive,
        labs_negative=case.labs_negative,
    )
    if verbose:
        print(f"Case: {case.description}")
        print(f"Ground truth: {_pretty_disease(case.ground_truth)}")
        print()
        print(explain_diagnosis_summary(engine, case.patient_id))
        print()
    return engine


# ---------- Cohort accuracy ----------

def run_all_cases(cases: Optional[List[PatientCase]] = None
                  ) -> List[dict]:
    cases = cases or PATIENT_CASES
    results = []
    correct = 0
    print("Running full patient cohort")
    print("=" * 72)
    for case in cases:
        engine = DiagnosisEngine()
        engine.ingest_record(
            case.patient_id,
            symptoms_present=case.symptoms_present,
            symptoms_absent=case.symptoms_absent,
            labs_positive=case.labs_positive,
            labs_negative=case.labs_negative,
        )
        ranked = engine.diagnose(case.patient_id)
        top = ranked[0]
        is_correct = top.disease == case.ground_truth
        if is_correct:
            correct += 1
        results.append({
            "patient": case.patient_id,
            "ground_truth": case.ground_truth,
            "predicted": top.disease,
            "correct": is_correct,
            "top_freq": top.frequency,
            "top_conf": top.confidence,
            "top_exp": top.expectation,
        })
        marker = "OK " if is_correct else "X  "
        print(f"{marker} {case.patient_id:18s}  "
              f"truth={case.ground_truth:6s}  "
              f"pred={top.disease:6s}  "
              f"f={top.frequency:.2f}  c={top.confidence:.2f}  "
              f"exp={top.expectation:.2f}")
    print("=" * 72)
    print(f"Accuracy: {correct}/{len(cases)} "
          f"({100.0 * correct / len(cases):.1f}%)")
    return results


# ---------- Scripted belief-revision experiment ----------

def belief_revision_conflict(verbose: bool = True) -> DiagnosisEngine:
    """Manually interleave supportive and contradictory evidence to observe
    belief revision at each step.

    Scenario: a patient presents with fever and sore throat, which is highly
    compatible with strep. A strep rapid test is then performed and returns
    NEGATIVE. A second piece of supportive evidence (body aches) follows.
    We expect the belief about strep to drop sharply when the negative lab
    arrives, while the belief about flu should rise.
    """
    engine = DiagnosisEngine()
    patient = "revision_demo"

    # Assemble the evidence items one at a time using the engine's own helper
    # so the negative lab is rewritten to its paired "_negative" feature
    # (which is how it gets deduction-carried as informative evidence).
    items = (
        DiagnosisEngine._evidence_from(patient, symptoms_present=["fever"])
        + DiagnosisEngine._evidence_from(patient, symptoms_present=["sore_throat"])
        + DiagnosisEngine._evidence_from(patient,
                                         labs_negative=["strep_rapid_test_positive"])
        + DiagnosisEngine._evidence_from(patient, symptoms_present=["body_aches"])
    )
    steps = engine.ingest_sequence(patient, items)

    if verbose:
        print("Belief-revision under conflicting evidence")
        print("=" * 72)
        print()
        print(explain_trajectory(patient, steps))
        print()
        ranked = engine.diagnose(patient)
        print("Final ranked diagnoses:")
        for b in ranked:
            print(f"  {_pretty_disease(b.disease):16s}  "
                  f"f={b.frequency:.3f}  c={b.confidence:.3f}  "
                  f"exp={b.expectation:.3f}")
        print()
        print(explain_top_diagnosis(ranked[0]))
        print()

    return engine


# ---------- Comparison with PyNARS ----------

def compare_with_pynars(case: PatientCase,
                        cycles_per_disease: int = 1500,
                        verbose: bool = True) -> dict:
    """Run the same patient through both the deterministic engine and the
    actual PyNARS Reasoner, and show their outputs side by side."""
    # deterministic engine
    eng = DiagnosisEngine()
    eng.ingest_record(
        case.patient_id,
        symptoms_present=case.symptoms_present,
        symptoms_absent=case.symptoms_absent,
        labs_positive=case.labs_positive,
        labs_negative=case.labs_negative,
    )
    det_ranked = eng.diagnose(case.patient_id)
    det_by_disease = {b.disease: b for b in det_ranked}

    # PyNARS reasoner
    from .nars_backend import PyNarsBackend   # local import: optional
    nars = PyNarsBackend()
    nars.load_knowledge_base()
    nars.add_patient_evidence(
        case.patient_id,
        symptoms_present=case.symptoms_present,
        symptoms_absent=case.symptoms_absent,
        labs_positive=case.labs_positive,
        labs_negative=case.labs_negative,
    )
    pynars_result = nars.query_all(case.patient_id,
                                   cycles_per_disease=cycles_per_disease)

    if verbose:
        print(f"Case: {case.patient_id} — {case.description}")
        print(f"Ground truth: {_pretty_disease(case.ground_truth)}")
        print()
        print(f"{'disease':10s}  {'NAL engine (f, c, exp)':28s}  "
              f"{'PyNARS (f, c)':20s}")
        print("-" * 72)
        for d in kb.DISEASES:
            b = det_by_disease[d]
            n = pynars_result.get(d)
            det_str = f"({b.frequency:.2f}, {b.confidence:.2f}, {b.expectation:.2f})"
            n_str = (f"({n.frequency:.2f}, {n.confidence:.2f})"
                     if n is not None else "not derived")
            print(f"{d:10s}  {det_str:28s}  {n_str:20s}")
        print()

    return {
        "deterministic": {d: (b.frequency, b.confidence) for d, b in det_by_disease.items()},
        "pynars": {d: (t.frequency, t.confidence) if t else None
                   for d, t in pynars_result.items()},
    }


# ---------- Aggregate runner ----------

def run_everything():
    print()
    print("##" * 36)
    print("## PART 1: Per-patient diagnoses with explanations")
    print("##" * 36)
    for case in PATIENT_CASES:
        run_case(case)

    print("##" * 36)
    print("## PART 2: Cohort accuracy")
    print("##" * 36)
    run_all_cases()

    print()
    print("##" * 36)
    print("## PART 3: Scripted belief-revision experiment")
    print("##" * 36)
    belief_revision_conflict()


if __name__ == "__main__":
    run_everything()
