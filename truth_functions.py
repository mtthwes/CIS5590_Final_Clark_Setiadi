"""
Truth-value functions for Mini-NARS inference rules.
Implements the functions from Appendix B/C of "Designing a Mind".
"""


def _and(*args):
    """Extended Boolean AND: product of all arguments."""
    result = 1.0
    for a in args:
        result *= a
    return result


def _or(*args):
    """Extended Boolean OR: 1 - product of (1 - xi)."""
    result = 1.0
    for a in args:
        result *= (1.0 - a)
    return 1.0 - result


def _not(x):
    """Extended Boolean NOT."""
    return 1.0 - x


# --- Evidence <-> truth-value conversions ---

K = 1.0  # system parameter


def truth_to_evidence(f, c):
    """Convert (frequency, confidence) to (w+, w)."""
    w = K * c / (1.0 - c) if c < 1.0 else float('inf')
    wp = w * f
    return wp, w


def evidence_to_truth(wp, w):
    """Convert (w+, w) to (frequency, confidence)."""
    f = wp / w if w > 0 else 0.5
    c = w / (w + K)
    return f, c


def expectation(f, c):
    """Expectation value: e = c * (f - 0.5) + 0.5"""
    return c * (f - 0.5) + 0.5


# --- Local inference ---

def revision(f1, c1, f2, c2):
    """Revision rule: pool evidence from two sources."""
    wp1, w1 = truth_to_evidence(f1, c1)
    wp2, w2 = truth_to_evidence(f2, c2)
    wp = wp1 + wp2
    w = w1 + w2
    return evidence_to_truth(wp, w)


# --- Strong syllogism ---

def deduction(f1, c1, f2, c2):
    """Deduction: {M -> P, S -> M} |- S -> P"""
    f = _and(f1, f2)
    c = _and(f1, c1, f2, c2)
    return f, c


def analogy(f1, c1, f2, c2):
    """Analogy: {M <-> P, S -> M} |- S -> P"""
    f = _and(f1, f2)
    c = _and(f2, c1, c2)
    return f, c


def resemblance(f1, c1, f2, c2):
    """Resemblance: {M <-> P, S <-> M} |- S <-> P"""
    f = _and(f1, f2)
    c = _and(_or(f1, f2), c1, c2)
    return f, c


# --- Weak syllogism ---

def abduction(f1, c1, f2, c2):
    """Abduction: {P -> M, S -> M} |- S -> P"""
    wp = _and(f1, f2, c1, c2)
    w = _and(f1, c1, c2)
    return evidence_to_truth(wp, w)


def induction(f1, c1, f2, c2):
    """Induction: {M -> P, M -> S} |- S -> P"""
    wp = _and(f1, f2, c1, c2)
    w = _and(f2, c1, c2)
    return evidence_to_truth(wp, w)


def exemplification(f1, c1, f2, c2):
    """Exemplification: {P -> M, M -> S} |- S -> P"""
    wp = _and(f1, f2, c1, c2)
    w = _and(f1, f2, c1, c2)
    return evidence_to_truth(wp, w)


# --- Immediate inference ---

def negation(f1, c1):
    """Negation: swap positive and negative evidence."""
    return _not(f1), c1


def conversion(f1, c1):
    """Conversion: S -> P to P -> S (weak)."""
    wp = _and(f1, c1)
    w = wp  # w- = 0
    return evidence_to_truth(wp, w)


def contraposition(f1, c1):
    """Contraposition: S => P to ~P => ~S (weak)."""
    wp = 0.0
    w = _and(_not(f1), c1)
    return evidence_to_truth(wp, w)


# --- Composition ---

def intersection(f1, c1, f2, c2):
    """Extensional intersection / conjunction."""
    f = _and(f1, f2)
    c = _and(c1, c2)
    return f, c


def union(f1, c1, f2, c2):
    """Intensional intersection / disjunction."""
    f = _or(f1, f2)
    c = _and(c1, c2)
    return f, c


def difference(f1, c1, f2, c2):
    """Extensional difference."""
    f = _and(f1, _not(f2))
    c = _and(c1, c2)
    return f, c
