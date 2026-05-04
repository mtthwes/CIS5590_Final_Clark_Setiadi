"""Knowledge base for a small explainable medical diagnosis recommender.

Scope is intentionally narrow: four respiratory / throat conditions with
overlapping symptoms, a handful of labs, and statistical-association style
weights that will become NAL truth-values.

Each symptom-disease and lab-disease edge carries:
  - frequency  f in [0, 1]  — how characteristic the feature is of the disease
  - confidence c in [0, 1]  — how strongly we believe the rule itself

The values are rough clinical priors chosen so that overlapping evidence and
conflicting labs exercise NAL's belief revision machinery. They are NOT meant
to be medically authoritative.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


DISEASES: List[str] = [
    "flu",
    "cold",
    "covid",
    "strep",
]


SYMPTOMS: List[str] = [
    "fever",
    "high_fever",
    "cough",
    "sore_throat",
    "fatigue",
    "headache",
    "runny_nose",
    "loss_of_smell",
    "body_aches",
    "swollen_lymph_nodes",
    "nasal_congestion",
    # A handful of "absent-X" pseudo-features used for the few symptoms whose
    # absence is diagnostically informative. Their presence in the patient's
    # evidence is implied by absence of the positive feature (see translator).
    "no_cough",
    "no_runny_nose",
    "no_loss_of_smell",
]


LAB_TESTS: List[str] = [
    "strep_rapid_test_positive",
    "strep_rapid_test_negative",
    "throat_culture_positive",
    "throat_culture_negative",
    "covid_pcr_positive",
    "covid_pcr_negative",
    "influenza_antigen_positive",
    "influenza_antigen_negative",
]


# (frequency, confidence) per (symptom, disease) pair.
#
# The intended reading of each entry is, for the implication
#   <<$x --> symptom> ==> <$x --> disease>>. %f; c%
# that the rule's frequency is the characteristic strength of this symptom
# as a signal for this disease — i.e. "how characteristic is this symptom of
# this disease when observed in isolation, relative to the other candidate
# diagnoses in the KB". The engine combines these by NAL revision across
# all observed features, so characteristic features like "loss of smell"
# for COVID or "swollen lymph nodes" for strep correctly dominate ambiguous
# presentations.
#
# Low-frequency entries (f < 0.30, c high) are deliberately encoded as
# "evidence of absence" — they are equivalent under NAL negation to
# "symptom implies NOT disease" with high f — and the engine applies
# `deduction_via_negation` to them so that, for example, a cough argues
# confidently against strep.
SYMPTOM_DISEASE_WEIGHTS: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Flu: abrupt onset, systemic
    ("fever",              "flu"):   (0.80, 0.80),
    ("high_fever",         "flu"):   (0.75, 0.80),
    ("cough",              "flu"):   (0.65, 0.75),
    ("fatigue",            "flu"):   (0.75, 0.80),
    ("headache",           "flu"):   (0.60, 0.70),
    ("body_aches",         "flu"):   (0.85, 0.85),   # characteristic
    ("sore_throat",        "flu"):   (0.35, 0.65),

    # Common cold: milder, upper-respiratory
    ("runny_nose",         "cold"):  (0.90, 0.90),   # characteristic
    ("nasal_congestion",   "cold"):  (0.85, 0.85),
    ("sore_throat",        "cold"):  (0.55, 0.70),
    ("cough",              "cold"):  (0.55, 0.70),
    ("headache",           "cold"):  (0.35, 0.60),
    # cold rarely causes fever — low-f, high-c: fires as negation-deduction
    ("fever",              "cold"):  (0.10, 0.80),
    ("high_fever",         "cold"):  (0.05, 0.85),
    ("body_aches",         "cold"):  (0.15, 0.75),

    # COVID-19
    ("fever",              "covid"): (0.70, 0.80),
    ("cough",              "covid"): (0.70, 0.75),
    ("fatigue",            "covid"): (0.70, 0.75),
    ("loss_of_smell",      "covid"): (0.90, 0.92),   # highly distinctive
    ("headache",           "covid"): (0.55, 0.70),
    ("body_aches",         "covid"): (0.50, 0.70),
    ("sore_throat",        "covid"): (0.40, 0.70),
    ("nasal_congestion",   "covid"): (0.30, 0.65),
    ("high_fever",         "covid"): (0.55, 0.75),

    # Strep throat (group A strep pharyngitis)
    ("sore_throat",           "strep"): (0.90, 0.85),  # very characteristic
    ("fever",                 "strep"): (0.70, 0.80),
    ("high_fever",            "strep"): (0.55, 0.75),
    ("swollen_lymph_nodes",   "strep"): (0.85, 0.85),  # very characteristic
    ("headache",              "strep"): (0.45, 0.65),
    # Cough / runny nose rarely happen with strep — low-f, high-c argues
    # confidently against strep via negation-deduction.
    ("cough",                 "strep"): (0.10, 0.85),
    ("runny_nose",            "strep"): (0.05, 0.85),
    ("nasal_congestion",      "strep"): (0.10, 0.80),

    # --- Absence-as-evidence rules for the few highly diagnostic symptoms.
    # These fire when the translator converts a symptoms_absent entry into
    # an observation of the matching 'no_*' pseudo-feature. They push
    # probability *down* on the associated disease when the characteristic
    # symptom is notably missing.
    ("no_cough",         "cold"):   (0.10, 0.70),  # cold almost always has cough
    ("no_cough",         "flu"):    (0.20, 0.65),
    ("no_cough",         "covid"):  (0.25, 0.60),
    ("no_runny_nose",    "cold"):   (0.15, 0.70),
    ("no_loss_of_smell", "covid"):  (0.40, 0.70),  # mild absence signal
}


# Map diagnostic-symptom name -> "no_X" pseudo-feature name.
SYMPTOM_ABSENCE_FEATURES: Dict[str, str] = {
    "cough":         "no_cough",
    "runny_nose":    "no_runny_nose",
    "loss_of_smell": "no_loss_of_smell",
}


# Labs are more decisive than symptoms, so confidence is higher.
#
# NAL deduction multiplies frequencies, so an evidence truth value with f=0
# contributes zero confidence toward the conclusion — meaning a naive
# encoding cannot use a negative lab to lower the belief in a disease.
# To handle negative labs as informative evidence, we treat each lab as
# having TWO possible observations — its positive and its negative feature —
# and encode a forward rule for each. A negative lab thus produces a
# positive observation of `<patient --> lab_..._negative>` which, via the
# negative rule below, *deducts* toward "probably not disease".
LAB_DISEASE_WEIGHTS: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Positive-result rules: observing the positive feature strongly implies
    # the disease.
    ("strep_rapid_test_positive",    "strep"): (0.95, 0.95),
    ("throat_culture_positive",      "strep"): (0.98, 0.97),
    ("covid_pcr_positive",           "covid"): (0.98, 0.97),
    ("influenza_antigen_positive",   "flu"):   (0.95, 0.93),

    # Negative-result rules: observing the negative feature strongly implies
    # absence of the disease (low frequency, high confidence).
    ("strep_rapid_test_negative",    "strep"): (0.08, 0.92),
    ("throat_culture_negative",      "strep"): (0.03, 0.96),
    ("covid_pcr_negative",           "covid"): (0.03, 0.96),
    ("influenza_antigen_negative",   "flu"):   (0.10, 0.90),
}


# Paired feature names: (positive_feature, negative_feature).
# Used by the translator to map `labs_negative=[positive_name]` inputs to
# observations of the corresponding negative feature.
LAB_FEATURE_PAIRS: Dict[str, str] = {
    "strep_rapid_test_positive":    "strep_rapid_test_negative",
    "throat_culture_positive":      "throat_culture_negative",
    "covid_pcr_positive":           "covid_pcr_negative",
    "influenza_antigen_positive":   "influenza_antigen_negative",
}


@dataclass
class DiseaseInfo:
    """Human-readable info used by the explanation layer."""
    name: str
    display: str
    description: str


DISEASE_INFO: Dict[str, DiseaseInfo] = {
    "flu":   DiseaseInfo("flu",   "Influenza",
                         "A viral infection with abrupt fever, body aches, and fatigue."),
    "cold":  DiseaseInfo("cold",  "Common Cold",
                         "A mild upper respiratory viral infection."),
    "covid": DiseaseInfo("covid", "COVID-19",
                         "SARS-CoV-2 infection; loss of smell is notably characteristic."),
    "strep": DiseaseInfo("strep", "Strep Throat",
                         "Group A streptococcal pharyngitis; bacterial, treated with antibiotics."),
}


SYMPTOM_DISPLAY: Dict[str, str] = {
    "fever":                "a fever",
    "high_fever":           "a high fever (>= 39 C)",
    "cough":                "a cough",
    "sore_throat":          "a sore throat",
    "fatigue":              "fatigue",
    "headache":             "a headache",
    "runny_nose":           "a runny nose",
    "loss_of_smell":        "loss of smell",
    "body_aches":           "body aches",
    "swollen_lymph_nodes":  "swollen lymph nodes",
    "nasal_congestion":     "nasal congestion",
    "no_cough":             "no cough",
    "no_runny_nose":        "no runny nose",
    "no_loss_of_smell":     "a normal sense of smell",
}


LAB_DISPLAY: Dict[str, str] = {
    "strep_rapid_test_positive":    "a positive rapid strep test",
    "strep_rapid_test_negative":    "a negative rapid strep test",
    "throat_culture_positive":      "a positive group-A strep throat culture",
    "throat_culture_negative":      "a negative group-A strep throat culture",
    "covid_pcr_positive":           "a positive COVID-19 PCR test",
    "covid_pcr_negative":           "a negative COVID-19 PCR test",
    "influenza_antigen_positive":   "a positive influenza antigen test",
    "influenza_antigen_negative":   "a negative influenza antigen test",
}


def all_symptom_disease_rules() -> List[Tuple[str, str, float, float]]:
    return [(s, d, f, c) for (s, d), (f, c) in SYMPTOM_DISEASE_WEIGHTS.items()]


def all_lab_disease_rules() -> List[Tuple[str, str, float, float]]:
    return [(l, d, f, c) for (l, d), (f, c) in LAB_DISEASE_WEIGHTS.items()]


def associations_for_disease(disease: str) -> List[Tuple[str, float, float]]:
    """Return list of (symptom, freq, conf) supporting `disease`, sorted by
    frequency * confidence desc — used by the explanation layer."""
    out = []
    for (s, d), (f, c) in SYMPTOM_DISEASE_WEIGHTS.items():
        if d == disease:
            out.append((s, f, c))
    out.sort(key=lambda x: x[1] * x[2], reverse=True)
    return out
