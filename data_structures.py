"""
Data structures for Mini-NARS: Statement, TruthValue, Task, Belief, Concept.
"""

from truth_functions import expectation, _or, _and, _not
from bag import Bag


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class Statement:
    """An inheritance statement: subject -> predicate."""

    def __init__(self, subject, predicate):
        self.subject = subject
        self.predicate = predicate

    def __eq__(self, other):
        return (isinstance(other, Statement)
                and self.subject == other.subject
                and self.predicate == other.predicate)

    def __hash__(self):
        return hash((self.subject, self.predicate))

    def __repr__(self):
        return f"<{self.subject} --> {self.predicate}>"


# ---------------------------------------------------------------------------
# TruthValue
# ---------------------------------------------------------------------------

class TruthValue:
    """A truth-value: (frequency, confidence) plus an evidential base."""

    def __init__(self, frequency=1.0, confidence=0.9, evidential_base=None):
        self.frequency = max(0.0, min(1.0, frequency))
        self.confidence = max(0.001, min(0.999, confidence))
        self.evidential_base = evidential_base if evidential_base else []

    def expectation(self):
        return expectation(self.frequency, self.confidence)

    def __repr__(self):
        return f"%{self.frequency:.2f};{self.confidence:.2f}%"


# ---------------------------------------------------------------------------
# BudgetValue
# ---------------------------------------------------------------------------

class BudgetValue:
    """Budget: priority and durability, both in (0,1]."""

    def __init__(self, priority=0.8, durability=0.9):
        self.priority = max(0.001, min(1.0, priority))
        self.durability = max(0.001, min(0.999, durability))

    def __repr__(self):
        return f"${self.priority:.2f};{self.durability:.2f}$"


# ---------------------------------------------------------------------------
# BagItem base
# ---------------------------------------------------------------------------

class BagItem:
    """Base class for items stored in a Bag."""

    def __init__(self, key, priority=0.5, durability=0.5):
        self.key = key
        self.priority = priority
        self.durability = durability

    def merge(self, other):
        """Merge with another item of the same key: OR on priority/durability."""
        if isinstance(other, BagItem):
            self.priority = _or(self.priority, other.priority)
            self.durability = _or(self.durability, other.durability)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task(BagItem):
    """
    A task is either a judgment-task or a question-task.
    Judgment-task: has statement + truth_value + budget.
    Question-task: has statement (possibly with variable) + budget, no truth_value.
    """

    _serial = 0

    def __init__(self, statement, truth_value=None, budget=None, is_input=False):
        self.statement = statement
        self.truth_value = truth_value  # None for questions
        self.is_question = (truth_value is None)
        self.solution = None  # best answer found (for questions)

        if budget is None:
            if self.is_question:
                budget = BudgetValue(0.9, 0.8)
            else:
                budget = BudgetValue(0.8, 0.9)

        self.budget = budget
        self.is_input = is_input

        Task._serial += 1
        self.serial = Task._serial

        # Build key: statement string + evidential base for judgments
        if truth_value and truth_value.evidential_base:
            eb_str = str(sorted(truth_value.evidential_base))
        else:
            eb_str = str(self.serial)
        key = str(statement) + ("?" if self.is_question else ".") + eb_str

        super().__init__(key, budget.priority, budget.durability)

    def has_variable(self):
        """Check if the statement contains a query variable '?'."""
        return self.statement.subject == "?" or self.statement.predicate == "?"

    def merge(self, other):
        """Merge budgets when same task re-enters a bag."""
        if isinstance(other, Task):
            self.priority = _or(self.priority, other.priority)
            self.durability = _or(self.durability, other.durability)
            self.budget.priority = self.priority
            self.budget.durability = self.durability
        elif isinstance(other, BagItem):
            super().merge(other)

    def __repr__(self):
        if self.is_question:
            sol = ""
            if self.solution:
                sol = f" => {self.solution}"
            return f"{self.statement}?{sol} {self.budget}"
        else:
            return f"{self.statement}. {self.truth_value} {self.budget}"


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------

class Belief(BagItem):
    """
    A belief stores multiple truth-values for the same statement,
    in a priority queue of capacity m (default 4).
    """

    MAX_TRUTHS = 4

    def __init__(self, statement, truth_value=None, budget=None):
        self.statement = statement
        # Priority queue of (truth_value, priority) sorted by confidence desc
        self.truths = []
        if truth_value:
            self.truths.append(truth_value)

        if budget is None:
            budget = BudgetValue(0.5, 0.5)

        key = str(statement)
        super().__init__(key, budget.priority, budget.durability)

    def add_truth(self, tv):
        """Add a truth-value; keep top MAX_TRUTHS by confidence."""
        self.truths.append(tv)
        self.truths.sort(key=lambda t: t.confidence, reverse=True)
        if len(self.truths) > self.MAX_TRUTHS:
            self.truths.pop()

    def select_truth(self, task):
        """
        Return the truth-value with highest confidence that has
        a disjoint evidential base with the task.
        """
        task_eb = set()
        if task.truth_value and task.truth_value.evidential_base:
            task_eb = set(task.truth_value.evidential_base)

        for tv in self.truths:
            tv_eb = set(tv.evidential_base) if tv.evidential_base else set()
            if task_eb.isdisjoint(tv_eb):
                return tv
        return None

    def merge(self, other):
        """Merge with another belief of the same key."""
        if isinstance(other, Belief):
            for tv in other.truths:
                self.add_truth(tv)
            self.priority = _or(self.priority, other.priority)
            self.durability = _or(self.durability, other.durability)
        elif isinstance(other, BagItem):
            super().merge(other)

    def best_truth(self):
        """Return the highest-confidence truth-value."""
        if not self.truths:
            return None
        return self.truths[0]

    def __repr__(self):
        best = self.best_truth()
        return f"Belief({self.statement} {best})"


# ---------------------------------------------------------------------------
# Concept
# ---------------------------------------------------------------------------

class Concept(BagItem):
    """
    A concept clusters tasks and beliefs sharing a common term.
    """

    def __init__(self, term):
        self.term = term
        # taskBag: 10 levels x 3 capacity = 30 items
        self.task_bag = Bag(levels=10, bucket_capacity=3, label=f"tasks@{term}")
        # beliefBag: 100 levels x 5 capacity = 500 items
        self.belief_bag = Bag(levels=20, bucket_capacity=5, label=f"beliefs@{term}")

        key = term
        super().__init__(key, priority=0.001, durability=0.5)

    def fire(self, nars):
        """Select a task and belief, perform inference."""
        task = self.task_bag.select()
        if task is None:
            return

        belief = self.belief_bag.select()
        if belief is None:
            self.task_bag.put_back(task)
            return

        # Get a judgment from the belief with disjoint evidence
        judgment = belief.select_truth(task)
        if judgment is not None:
            nars.inference(task, judgment, belief)

        self.belief_bag.put_back(belief)
        self.task_bag.put_back(task)

    def merge(self, other):
        """Merge with another concept of the same key."""
        if isinstance(other, Concept):
            self.priority = _or(self.priority, other.priority)
            self.durability = _or(self.durability, other.durability)
        elif isinstance(other, BagItem):
            super().merge(other)

    def __repr__(self):
        return (f"Concept({self.term}, tasks={len(self.task_bag)}, "
                f"beliefs={len(self.belief_bag)}, pri={self.priority:.3f})")
