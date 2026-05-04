"""Simulated patient cases.

Each `PatientCase` carries:
  - a unique `patient_id` (also used as the NAL term),
  - the *ordered* list of symptoms and labs as they are "observed", which
    matters for the belief-revision trajectory,
  - the ground-truth diagnosis (for evaluation purposes — the engine does
    not see this).

The cases are hand-constructed to exercise the reasoning capabilities the
project is designed to study:

  - clear cases where one disease obviously fits,
  - ambiguous cases where two diseases both partially fit,
  - cases where a later lab result should override an earlier symptom-based
    leaning (belief revision under conflicting evidence).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PatientCase:
    patient_id: str
    description: str
    ground_truth: str
    symptoms_present: List[str] = field(default_factory=list)
    symptoms_absent:  List[str] = field(default_factory=list)
    labs_positive:    List[str] = field(default_factory=list)
    labs_negative:    List[str] = field(default_factory=list)

    def ordered_features(self) -> List[str]:
        """Evidence order as presented. Symptoms first, then labs."""
        return (list(self.symptoms_present)
                + list(self.symptoms_absent)
                + list(self.labs_positive)
                + list(self.labs_negative))


# ---------- The patient cohort ----------

PATIENT_CASES: List[PatientCase] = [
    PatientCase(
        patient_id="patient_alpha",
        description="Classic flu presentation.",
        ground_truth="flu",
        symptoms_present=["fever", "body_aches", "fatigue", "cough", "headache"],
    ),
    PatientCase(
        patient_id="patient_beta",
        description="Mild upper respiratory — common cold.",
        ground_truth="cold",
        symptoms_present=["runny_nose", "nasal_congestion", "sore_throat", "cough"],
        symptoms_absent=["high_fever", "body_aches"],
    ),
    PatientCase(
        patient_id="patient_gamma",
        description="Strep throat with characteristic findings.",
        ground_truth="strep",
        symptoms_present=["sore_throat", "fever", "swollen_lymph_nodes"],
        symptoms_absent=["cough", "runny_nose"],
        labs_positive=["strep_rapid_test_positive"],
    ),
    PatientCase(
        patient_id="patient_delta",
        description="COVID-19 with loss of smell.",
        ground_truth="covid",
        symptoms_present=["fever", "cough", "fatigue", "loss_of_smell"],
        labs_positive=["covid_pcr_positive"],
    ),

    # --- Ambiguous / belief-revision focused cases ---

    PatientCase(
        patient_id="patient_epsilon",
        description=(
            "Fever + sore throat — initially flu/strep candidate. "
            "Strep rapid test later returns NEGATIVE, which should "
            "revise the belief away from strep."),
        ground_truth="flu",
        symptoms_present=["fever", "sore_throat", "body_aches"],
        labs_negative=["strep_rapid_test_positive"],
    ),
    PatientCase(
        patient_id="patient_zeta",
        description=(
            "Fever + cough + fatigue — initially flu-leaning. "
            "Loss of smell arrives late, shifting belief toward COVID."),
        ground_truth="covid",
        symptoms_present=["fever", "cough", "fatigue", "loss_of_smell"],
    ),
    PatientCase(
        patient_id="patient_eta",
        description=(
            "Overlapping cold symptoms. Sore throat leans toward strep "
            "initially but a negative rapid strep test revises the belief "
            "back toward cold."),
        ground_truth="cold",
        symptoms_present=["cough", "sore_throat", "headache", "runny_nose"],
        labs_negative=["strep_rapid_test_positive"],
    ),
    PatientCase(
        patient_id="patient_theta",
        description=(
            "Sore throat + fever initially suggest strep. A positive "
            "influenza antigen test then revises toward flu."),
        ground_truth="flu",
        symptoms_present=["sore_throat", "fever", "body_aches"],
        labs_positive=["influenza_antigen_positive"],
    ),
]


def get_case(patient_id: str) -> PatientCase:
    for c in PATIENT_CASES:
        if c.patient_id == patient_id:
            return c
    raise KeyError(f"Unknown patient id: {patient_id}")
