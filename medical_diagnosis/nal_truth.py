"""NAL truth-value functions (frequency, confidence).

Implements the core NAL-1 / NAL-5 truth functions used by this project:
  - deduction  : from an implication rule and a premise, derive a conclusion
  - revision   : combine two beliefs about the *same* statement
  - expectation: ranking score used to pick "most likely" diagnosis

Formulas follow Wang's NAL textbook (the same ones PyNARS itself uses
internally):

    deduction(T1, T2):  f = f1 * f2
                        c = f1 * f2 * c1 * c2

    revision(T1, T2):   w1 = c1 / (1 - c1),  w2 = c2 / (1 - c2)
                        f  = (w1 * f1 + w2 * f2) / (w1 + w2)
                        c  = (w1 + w2) / (w1 + w2 + 1)

    expectation(f, c):  c * (f - 0.5) + 0.5

Using these directly gives us a deterministic engine that is faithful to NAL
semantics without being at the mercy of PyNARS' attention-bag scheduler,
which in a crowded rule base often fails to fire the specific chain we want
within a bounded number of cycles. The PyNARS `Reasoner` is still exposed
through `nars_backend.py` for demonstration / comparison purposes.
"""

from dataclasses import dataclass


EPS = 1e-9


@dataclass(frozen=True)
class Truth:
    frequency: float
    confidence: float

    @property
    def expectation(self) -> float:
        return self.confidence * (self.frequency - 0.5) + 0.5

    def __repr__(self) -> str:
        return f"%{self.frequency:.3f};{self.confidence:.3f}%"


def deduction(rule: Truth, premise: Truth) -> Truth:
    """NAL deduction truth function.

    Used for:   <<$x --> S> ==> <$x --> D>> (rule)  +  <p --> S> (premise)
                => <p --> D>.
    """
    f = rule.frequency * premise.frequency
    c = rule.frequency * premise.frequency * rule.confidence * premise.confidence
    return Truth(_clip(f), _clip(c))


def deduction_via_negation(rule: Truth, premise: Truth) -> Truth:
    """Deduction for rules whose semantic intent is "evidence of absence".

    A rule like  <<$x --> strep_rapid_test_negative> ==> <$x --> strep>>
    with a low rule-frequency (e.g. %0.08;0.92%) is really asserting, in
    NAL-negation form,
        <<$x --> strep_rapid_test_negative> ==> <$x --> (--, strep)>>
    with truth %0.92;0.92% — and the positive observation of the premise
    should carry the rule's full confidence through to the negated
    conclusion. Flipping it back to the non-negated form gives a belief
    about `strep` whose frequency is `f_rule` but whose confidence is NOT
    discounted by that small frequency:

        f_out = f_rule * f_premise
        c_out = c_rule * c_premise

    This matches the NAL-3/NAL-5 result for applying standard deduction on
    the negated implication and then converting back, and is what lets
    "a negative lab" meaningfully lower the belief in the disease.
    """
    f = rule.frequency * premise.frequency
    c = rule.confidence * premise.confidence
    return Truth(_clip(f), _clip(c))


def revision(a: Truth, b: Truth) -> Truth:
    """NAL revision truth function — combine independent evidence about the
    same statement. Evidence is monotonic: revising two beliefs is always at
    least as confident as either alone (up to the asymptote c=1)."""
    # w = c / (1 - c): positive amount of evidence
    w1 = a.confidence / max(1.0 - a.confidence, EPS)
    w2 = b.confidence / max(1.0 - b.confidence, EPS)
    w = w1 + w2
    if w <= EPS:
        # both beliefs are effectively ignorance; use the higher-confidence one
        return a if a.confidence >= b.confidence else b
    f = (w1 * a.frequency + w2 * b.frequency) / w
    c = w / (w + 1.0)
    return Truth(_clip(f), _clip(c))


def revise_many(truths):
    """Revise a list of independent truth values pairwise. Order-independent
    to the extent NAL revision is (which is up to floating point)."""
    truths = list(truths)
    if not truths:
        return Truth(0.5, 0.0)
    acc = truths[0]
    for t in truths[1:]:
        acc = revision(acc, t)
    return acc


def _clip(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
