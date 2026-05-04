"""
Budget functions for Mini-NARS.
These control resource allocation: how much attention each item receives.
"""

from truth_functions import _or, _and, _not, expectation


def activate(concept, budget):
    """Activate a concept when a task is added to it."""
    concept.priority = _or(concept.priority, budget.priority)
    concept.durability = _or(concept.durability, budget.durability)


def revision_budget(task, judgment, belief):
    """
    Revision rule rewards novel conclusions.
    Returns (new_budget_priority, new_budget_durability).
    """
    pT = task.budget.priority
    dT = task.budget.durability
    pB = belief.priority
    dB = belief.durability

    novelty = abs(task.truth_value.expectation() - expectation(judgment.frequency, judgment.confidence))

    # Feedback to task and belief
    task.budget.priority = _or(pT, novelty)
    task.budget.durability = _or(dT, novelty)
    task.priority = task.budget.priority
    task.durability = task.budget.durability

    belief.priority = _or(pB, _not(novelty))
    belief.durability = _or(dB, _not(novelty))

    # New task budget
    priority = _or(pT, pB, novelty)
    durability = _or(dT, dB, novelty)
    return priority, durability


def choice_budget(task, belief, quality):
    """
    Choice rule rewards better answers.
    Returns (new_budget_priority, new_budget_durability).
    """
    pT = task.budget.priority
    dT = task.budget.durability
    pB = belief.priority
    dB = belief.durability

    # Feedback: task priority decreases, belief increases
    task.budget.priority = _and(pT, _not(quality))
    task.budget.durability = _and(dT, _not(quality))
    task.priority = task.budget.priority
    task.durability = task.budget.durability

    belief.priority = _or(pB, quality)
    belief.durability = _or(dB, quality)

    priority = _or(pT, pB, quality)
    durability = _or(dT, dB, quality)
    return priority, durability


def syllogistic_budget(task, belief, quality):
    """
    Syllogistic rules reward strong conclusions.
    Returns (new_budget_priority, new_budget_durability).
    """
    pT = task.budget.priority
    dT = task.budget.durability
    pB = belief.priority
    dB = belief.durability

    # Feedback
    task.budget.priority = _and(pT, _not(quality))
    task.budget.durability = _and(dT, _not(quality))
    task.priority = task.budget.priority
    task.durability = task.budget.durability

    belief.priority = _or(pB, quality)
    belief.durability = _or(dB, quality)

    priority = _or(pT, pB, quality)
    durability = _or(dT, dB, quality)
    return priority, durability
