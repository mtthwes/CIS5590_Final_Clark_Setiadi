"""Plain-English explanation layer.

Given a `DiagnosisEngine` trajectory, produce a short natural-language
rationale describing:
  - the top-ranked diagnosis and its confidence,
  - which pieces of evidence most strongly supported it (based on the
    `support` field attached to each belief, which records the per-rule
    deduction results that were revised together),
  - any evidence that weighed against it (low-frequency rules, negative labs),
  - how the belief changed over time (the belief-revision story).

The explainer never invents clinical facts — it only surfaces the rules from
the knowledge base that actually fired, described with the human-readable
display names in `knowledge_base.py`.
"""

from typing import List, Optional

from . import knowledge_base as kb
from .diagnosis_engine import DiagnosisEngine, DiseaseBelief, EvidenceStep


def _pretty_feature(feature: str, kind: str) -> str:
    if kind == "symptom":
        return kb.SYMPTOM_DISPLAY.get(feature, feature.replace("_", " "))
    return kb.LAB_DISPLAY.get(feature, feature.replace("_", " "))


def _pretty_disease(disease: str) -> str:
    info = kb.DISEASE_INFO.get(disease)
    return info.display if info else disease


def explain_top_diagnosis(
    belief: DiseaseBelief,
    max_reasons: int = 4,
) -> str:
    """Describe why the current top diagnosis is what it is."""
    info = kb.DISEASE_INFO.get(belief.disease)
    name = info.display if info else belief.disease

    # Sort support by contribution: positive deductions that raise the belief
    # come first (derived frequency * confidence), then negative / contrarian
    # derivations.
    support = list(belief.support)
    support.sort(key=lambda s: s[3].frequency * s[3].confidence, reverse=True)

    positive = [s for s in support if s[3].frequency >= 0.5]
    negative = [s for s in support if s[3].frequency < 0.5]

    lines: List[str] = []
    lines.append(
        f"Most likely diagnosis: {name} "
        f"(frequency {belief.frequency:.2f}, confidence {belief.confidence:.2f}, "
        f"expectation {belief.expectation:.2f})."
    )

    if positive:
        bits = []
        for label, feature, rule_truth, _derived in positive[:max_reasons]:
            kind = "lab" if label.startswith("lab") else "symptom"
            pretty = _pretty_feature(feature, kind)
            strength = _qualitative(rule_truth.frequency)
            bits.append(f"{pretty} ({strength} associated)")
        lines.append("Supporting evidence: " + ", ".join(bits) + ".")

    if negative:
        bits = []
        for label, feature, rule_truth, _derived in negative[:max_reasons]:
            kind = "lab" if label.startswith("lab") else "symptom"
            pretty = _pretty_feature(feature, kind)
            note = ("absent" if label.startswith("no_symptom")
                    or label.startswith("lab-")
                    else "rarely seen with this diagnosis")
            bits.append(f"{pretty} ({note})")
        lines.append("Contrary / weakening evidence: " + ", ".join(bits) + ".")

    if info:
        lines.append(f"About {name}: {info.description}")

    return "\n".join(lines)


def explain_trajectory(
    patient: str,
    steps: List[EvidenceStep],
    diseases: Optional[List[str]] = None,
) -> str:
    """Narrate how each evidence item shifted the ranking — the core of the
    belief-revision story that this project is designed to study."""
    diseases = diseases or kb.DISEASES
    if not steps:
        return f"No evidence has been recorded for {patient}."

    lines = [f"Belief-revision trajectory for {patient}:"]
    prev_top: Optional[str] = None
    prev_exp: Optional[float] = None
    for i, step in enumerate(steps, start=1):
        ev = step.evidence
        kind = ev.kind
        pretty = _pretty_feature(ev.feature, kind)
        is_positive_obs = ev.truth.frequency >= 0.5
        # The feature name encodes whether this is a negative/absence feature,
        # so choose the verb from the *feature* rather than the truth-value —
        # otherwise "observed the negative test (f=1)" reads as "returned positive".
        is_absence_feature = (ev.feature.startswith("no_")
                              or ev.feature.endswith("_negative"))
        if kind == "symptom":
            if is_absence_feature or not is_positive_obs:
                verb = "noted"
            else:
                verb = "observed"
        else:  # lab
            if is_absence_feature or not is_positive_obs:
                verb = "returned"
            else:
                verb = "returned"
        ranked = sorted(step.beliefs.values(),
                        key=lambda b: b.expectation, reverse=True)
        top = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None

        line = f"  {i}. {pretty} {verb}"
        line += f"  ->  top: {_pretty_disease(top.disease)} "
        line += f"(exp={top.expectation:.2f}, c={top.confidence:.2f})"
        if runner is not None:
            line += (f", runner-up: {_pretty_disease(runner.disease)} "
                     f"(exp={runner.expectation:.2f})")
        if prev_top is not None and prev_top != top.disease:
            line += f"  [ranking flipped from {_pretty_disease(prev_top)}]"
        elif prev_exp is not None:
            delta = top.expectation - prev_exp
            if abs(delta) >= 0.01:
                arrow = "up" if delta > 0 else "down"
                line += f"  [exp {arrow} {abs(delta):.2f}]"
        lines.append(line)
        prev_top = top.disease
        prev_exp = top.expectation
    return "\n".join(lines)


def explain_diagnosis_summary(
    engine: DiagnosisEngine,
    patient: str,
    max_reasons: int = 4,
) -> str:
    """Full explanation: trajectory + top diagnosis rationale + all ranks."""
    steps = engine.history.get(patient, [])
    ranked = engine.diagnose(patient)

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"Patient: {patient}")
    lines.append("=" * 60)
    lines.append("")
    if steps:
        lines.append(explain_trajectory(patient, steps))
        lines.append("")

    lines.append("Final ranked diagnoses:")
    for b in ranked:
        lines.append(
            f"  {_pretty_disease(b.disease):16s}  "
            f"f={b.frequency:.3f}  c={b.confidence:.3f}  "
            f"exp={b.expectation:.3f}"
        )
    lines.append("")

    if ranked:
        lines.append(explain_top_diagnosis(ranked[0], max_reasons=max_reasons))
    return "\n".join(lines)


def _qualitative(f: float) -> str:
    if f >= 0.85:
        return "strongly"
    if f >= 0.70:
        return "commonly"
    if f >= 0.50:
        return "sometimes"
    if f >= 0.30:
        return "weakly"
    return "uncharacteristically"
