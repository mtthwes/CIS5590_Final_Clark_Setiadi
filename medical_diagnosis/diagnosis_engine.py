"""Medical diagnosis reasoning engine.

This project's main contribution is studying belief revision and evidential
truth-values in a medical reasoning setting. The NAL semantics (deduction +
revision under (frequency, confidence) truth-values) are implemented directly
in `nal_truth.py` so the engine's behavior is deterministic and every step of
the inference can be traced for explanation.

The engine works in three phases for each piece of evidence:

  1. For each rule `<<$x --> symptom> ==> <$x --> disease>> %f_r;c_r%` that
     matches the incoming evidence `<patient --> symptom>. %f_e;c_e%`, apply
     NAL deduction to get a per-rule derived truth about `<patient --> disease>`.

  2. Collect all per-rule derived truths for a given disease (across all
     evidence the patient has provided so far) and combine them via NAL
     revision, producing a single `(f, c)` belief about the diagnosis.

  3. Record the full trajectory — which evidence produced which derivation,
     how the revised belief evolved — so the explanation layer can reconstruct
     a plain-English rationale.

Negative evidence (e.g. "lab was negative") is encoded with `f=0`, which NAL's
revision correctly combines with positive evidence to shift the belief down.

A separate `nars_backend.py` exposes the actual PyNARS Reasoner running the
same Narsese rules and evidence, so the pipeline can be cross-checked against
the reference implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import knowledge_base as kb
from . import narsese_translator as nt
from .nal_truth import Truth, deduction, deduction_via_negation, revise_many


# ---------- Result containers ----------

@dataclass
class DiseaseBelief:
    disease: str
    truth: Truth
    # Per-rule derivations that were revised to produce `truth`. Each item is
    # (evidence_label, rule_feature, rule_truth, derived_truth).
    support: List[Tuple[str, str, Truth, Truth]] = field(default_factory=list)

    @property
    def frequency(self) -> float:
        return self.truth.frequency

    @property
    def confidence(self) -> float:
        return self.truth.confidence

    @property
    def expectation(self) -> float:
        return self.truth.expectation

    def __repr__(self) -> str:
        return (f"DiseaseBelief({self.disease}, f={self.frequency:.3f}, "
                f"c={self.confidence:.3f}, exp={self.expectation:.3f})")


@dataclass
class EvidenceItem:
    """One piece of evidence fed into the engine."""
    label: str            # e.g. 'symptom:fever', 'lab-:strep_rapid_test_positive'
    feature: str          # the symptom/lab name
    kind: str             # 'symptom' or 'lab'
    truth: Truth          # truth-value attached to the observation
    narsese: str          # the exact Narsese string (for the NARS backend)


@dataclass
class EvidenceStep:
    """Snapshot of the diagnostic state after one evidence item."""
    evidence: EvidenceItem
    beliefs: Dict[str, DiseaseBelief] = field(default_factory=dict)


# ---------- The engine ----------

class DiagnosisEngine:
    """Deterministic NAL-based reasoner over the medical KB."""

    def __init__(self) -> None:
        # patient -> ordered list of evidence items
        self.patient_evidence: Dict[str, List[EvidenceItem]] = {}
        # patient -> ordered list of (evidence_item, beliefs_snapshot)
        self.history: Dict[str, List[EvidenceStep]] = {}

    # ---- Build evidence items from raw inputs ----
    @staticmethod
    def _evidence_from(
        patient: str,
        symptoms_present=(),
        symptoms_absent=(),
        labs_positive=(),
        labs_negative=(),
        symptom_confidence: float = 0.90,
        lab_confidence: float = 0.95,
    ) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        for s in symptoms_present:
            items.append(EvidenceItem(
                label=f"symptom:{s}", feature=s, kind="symptom",
                truth=Truth(1.0, symptom_confidence),
                narsese=nt.has_symptom(patient, s, 1.0, symptom_confidence),
            ))
        for s in symptoms_absent:
            neg = kb.SYMPTOM_ABSENCE_FEATURES.get(s)
            if neg is not None:
                # Positive observation of the paired "no_X" feature lets NAL
                # deduction fire with real signal instead of a vacuous f=0.
                items.append(EvidenceItem(
                    label=f"no_symptom:{s}", feature=neg, kind="symptom",
                    truth=Truth(1.0, symptom_confidence),
                    narsese=nt.has_symptom(patient, neg, 1.0, symptom_confidence),
                ))
            else:
                items.append(EvidenceItem(
                    label=f"no_symptom:{s}", feature=s, kind="symptom",
                    truth=Truth(0.0, symptom_confidence),
                    narsese=nt.lacks_symptom(patient, s, symptom_confidence),
                ))
        for l in labs_positive:
            items.append(EvidenceItem(
                label=f"lab+:{l}", feature=l, kind="lab",
                truth=Truth(1.0, lab_confidence),
                narsese=nt.lab_positive(patient, l, lab_confidence),
            ))
        for l in labs_negative:
            neg = kb.LAB_FEATURE_PAIRS.get(l)
            if neg is not None:
                # Observe the paired negative feature as positive (f=1).
                items.append(EvidenceItem(
                    label=f"lab-:{l}", feature=neg, kind="lab",
                    truth=Truth(1.0, lab_confidence),
                    narsese=nt.lab_positive(patient, neg, lab_confidence),
                ))
            else:
                items.append(EvidenceItem(
                    label=f"lab-:{l}", feature=l, kind="lab",
                    truth=Truth(0.0, lab_confidence),
                    narsese=nt.lab_negative(patient, l, lab_confidence),
                ))
        return items

    # ---- Rule lookup ----
    @staticmethod
    def _rule_truth(feature: str, disease: str, kind: str) -> Optional[Truth]:
        table = (kb.SYMPTOM_DISEASE_WEIGHTS
                 if kind == "symptom" else kb.LAB_DISEASE_WEIGHTS)
        if (feature, disease) in table:
            f, c = table[(feature, disease)]
            return Truth(f, c)
        return None

    # ---- Core inference ----
    NEG_F_THRESHOLD = 0.30
    NEG_C_THRESHOLD = 0.70

    @classmethod
    def _should_use_negation_deduction(
        cls, rule: Truth, evidence: Truth, feature: str
    ) -> bool:
        """Decide when to apply `deduction_via_negation` instead of the
        standard NAL-1 deduction.

        A rule like  <<$x --> cough> ==> <$x --> strep>> %0.15;0.85%  is
        semantically equivalent — by NAL negation — to
                     <<$x --> cough> ==> <$x --> (--, strep)>> %0.85;0.85%.
        If we see the feature positively (f_e high), the second form is the
        one that carries the rule's confidence; standard deduction on the
        first form collapses the confidence via multiplication by f_rule.
        We therefore use the negation variant when the rule encodes
        "evidence against" (low f_rule, non-trivial c_rule) and the
        evidence is a positive observation of the feature.

        Explicit negative features (`no_*` / `*_negative`) always use the
        negation variant, regardless of thresholds.
        """
        if feature.startswith("no_") or feature.endswith("_negative"):
            return True
        return (rule.frequency < cls.NEG_F_THRESHOLD
                and rule.confidence >= cls.NEG_C_THRESHOLD
                and evidence.frequency >= 0.5)

    def _derive_belief(
        self,
        evidence_items: List[EvidenceItem],
        disease: str,
    ) -> DiseaseBelief:
        """Apply deduction per rule, then revise across all matches."""
        per_rule: List[Truth] = []
        support: List[Tuple[str, str, Truth, Truth]] = []
        for ev in evidence_items:
            rule = self._rule_truth(ev.feature, disease, ev.kind)
            if rule is None:
                continue
            if self._should_use_negation_deduction(rule, ev.truth, ev.feature):
                derived = deduction_via_negation(rule, ev.truth)
            else:
                derived = deduction(rule, ev.truth)
            per_rule.append(derived)
            support.append((ev.label, ev.feature, rule, derived))
        combined = revise_many(per_rule) if per_rule else Truth(0.5, 0.0)
        return DiseaseBelief(disease=disease, truth=combined, support=support)

    # ---- Public API ----
    def add_evidence(self, patient: str, item: EvidenceItem) -> None:
        self.patient_evidence.setdefault(patient, []).append(item)

    def diagnose(self, patient: str,
                 diseases: Optional[List[str]] = None
                 ) -> List[DiseaseBelief]:
        """Return current diagnoses sorted by expectation desc."""
        diseases = diseases or kb.DISEASES
        ev = self.patient_evidence.get(patient, [])
        beliefs = [self._derive_belief(ev, d) for d in diseases]
        beliefs.sort(key=lambda b: b.expectation, reverse=True)
        return beliefs

    def ingest_record(
        self,
        patient: str,
        symptoms_present=(),
        symptoms_absent=(),
        labs_positive=(),
        labs_negative=(),
        symptom_confidence: float = 0.90,
        lab_confidence: float = 0.95,
    ) -> List[EvidenceStep]:
        """Feed a structured patient record in a single call, snapshotting
        the diagnosis trajectory after each evidence item."""
        items = self._evidence_from(
            patient,
            symptoms_present=symptoms_present,
            symptoms_absent=symptoms_absent,
            labs_positive=labs_positive,
            labs_negative=labs_negative,
            symptom_confidence=symptom_confidence,
            lab_confidence=lab_confidence,
        )
        return self.ingest_sequence(patient, items)

    def ingest_sequence(
        self,
        patient: str,
        items: List[EvidenceItem],
        diseases: Optional[List[str]] = None,
    ) -> List[EvidenceStep]:
        """Feed a pre-built list of evidence items one at a time, recording
        belief snapshots so the caller can see how each piece of evidence
        shifted the diagnosis (this is the belief-revision trajectory)."""
        diseases = diseases or kb.DISEASES
        steps: List[EvidenceStep] = []
        for item in items:
            self.add_evidence(patient, item)
            snapshot = {b.disease: b for b in self.diagnose(patient, diseases)}
            steps.append(EvidenceStep(evidence=item, beliefs=snapshot))
        self.history[patient] = steps
        return steps

    # ---- Convenience for the explanation layer ----
    def current_belief(self, patient: str, disease: str) -> DiseaseBelief:
        ev = self.patient_evidence.get(patient, [])
        return self._derive_belief(ev, disease)
