"""Translate Python-level medical data into Narsese statements.

Naming conventions (kept simple so NARS term tables stay readable):
  - Diseases, symptoms, labs and patients are all atomic terms:
        flu, cold, fever, cough, patient1, ...
  - "Patient has symptom X" is represented with the inheritance copula:
        <patient1 --> fever>.
    i.e. patient1 is-a fever-instance.  We could instead use the intension-set
    property form  <patient1 --> [fever]>, but PyNARS' KanrenEngine currently
    throws an "Empty" exception on some derivations involving [...] brackets,
    so we stick to plain atomic terms for robustness.
  - Association rules use NAL implication with a universal variable:
        <<$x --> fever> ==> <$x --> flu>>. %0.85;0.85%

Contradictory / negative evidence is encoded via a low frequency and high
confidence, which is the standard NAL way of saying "I strongly believe X is
not the case" without needing an explicit negation operator. NARS then
revises these against positive evidence automatically.
"""

from typing import Iterable, List, Tuple

from . import knowledge_base as kb


def fmt_truth(f: float, c: float) -> str:
    return f"%{f:.3f};{c:.3f}%"


# ---------- Rules (knowledge base → Narsese) ----------

def symptom_rule(symptom: str, disease: str, f: float, c: float) -> str:
    """`<<$x --> symptom> ==> <$x --> disease>>. %f;c%`"""
    return f"<<$x --> {symptom}> ==> <$x --> {disease}>>. {fmt_truth(f, c)}"


def lab_rule(lab: str, disease: str, f: float, c: float) -> str:
    return f"<<$x --> {lab}> ==> <$x --> {disease}>>. {fmt_truth(f, c)}"


def knowledge_base_rules() -> List[str]:
    """All symptom→disease and lab→disease rules from the KB."""
    rules = []
    for s, d, f, c in kb.all_symptom_disease_rules():
        rules.append(symptom_rule(s, d, f, c))
    for l, d, f, c in kb.all_lab_disease_rules():
        rules.append(lab_rule(l, d, f, c))
    return rules


# ---------- Patient evidence ----------

def has_symptom(patient: str, symptom: str,
                f: float = 1.0, c: float = 0.90) -> str:
    return f"<{patient} --> {symptom}>. {fmt_truth(f, c)}"


def lacks_symptom(patient: str, symptom: str, c: float = 0.90) -> str:
    """Patient is *not* observed to have the symptom (f=0)."""
    return f"<{patient} --> {symptom}>. {fmt_truth(0.0, c)}"


def lab_positive(patient: str, lab: str, c: float = 0.95) -> str:
    return f"<{patient} --> {lab}>. {fmt_truth(1.0, c)}"


def lab_negative(patient: str, lab: str, c: float = 0.95) -> str:
    return f"<{patient} --> {lab}>. {fmt_truth(0.0, c)}"


# ---------- Target term (for direct belief lookup) ----------

def diagnosis_term_str(patient: str, disease: str) -> str:
    """The target term whose belief table we inspect to read the current
    confidence that `patient` has `disease`."""
    return f"<{patient} --> {disease}>"


# ---------- Helpers over a patient record ----------

def patient_evidence(
    patient: str,
    symptoms_present: Iterable[str] = (),
    symptoms_absent: Iterable[str] = (),
    labs_positive: Iterable[str] = (),
    labs_negative: Iterable[str] = (),
    symptom_confidence: float = 0.90,
    lab_confidence: float = 0.95,
) -> List[Tuple[str, str]]:
    """Return [(label, narsese)] pairs in the order they should be fed in.

    Absent symptoms and negative labs are rewritten to their paired
    "no_<feature>" / "<feature>_negative" terms (see `knowledge_base.py`)
    so the deduction rules for their negative form fire directly, which is
    how NAL makes negative evidence informative.
    """
    out: List[Tuple[str, str]] = []
    for s in symptoms_present:
        out.append((f"symptom:{s}",
                    has_symptom(patient, s, 1.0, symptom_confidence)))
    for s in symptoms_absent:
        # If the absence of this symptom is specifically encoded in the KB,
        # use the paired negative feature; otherwise fall back to f=0.
        neg = kb.SYMPTOM_ABSENCE_FEATURES.get(s)
        if neg is not None:
            out.append((f"no_symptom:{s}",
                        has_symptom(patient, neg, 1.0, symptom_confidence)))
        else:
            out.append((f"no_symptom:{s}",
                        lacks_symptom(patient, s, symptom_confidence)))
    for l in labs_positive:
        out.append((f"lab+:{l}",
                    lab_positive(patient, l, lab_confidence)))
    for l in labs_negative:
        # Labs are always paired with an explicit negative feature.
        neg = kb.LAB_FEATURE_PAIRS.get(l)
        if neg is not None:
            out.append((f"lab-:{l}",
                        lab_positive(patient, neg, lab_confidence)))
        else:
            out.append((f"lab-:{l}",
                        lab_negative(patient, l, lab_confidence)))
    return out
