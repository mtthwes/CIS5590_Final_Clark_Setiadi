"""
Mini-NARS: A minimal implementation of NARS.
Implements the architecture, inference engine, and working cycle
as specified in Chapter 4 of "Designing a Mind" (Wang, 2026).
"""

import heapq
from data_structures import Statement, TruthValue, BudgetValue, Task, Belief, Concept
from bag import Bag
from truth_functions import (
    revision as tv_revision, deduction, induction, abduction,
    exemplification, expectation, _or, _and, _not
)
from budget_functions import (
    activate, revision_budget, choice_budget, syllogistic_budget
)


# ---------------------------------------------------------------------------
# Narsese Parser
# ---------------------------------------------------------------------------

class NarseseParser:
    """Parse Narsese text into Task objects."""

    @staticmethod
    def parse(text, serial_counter):
        """
        Parse a single Narsese sentence.
        Formats:
          <subject --> predicate>. %f;c% $p;d$     (judgment)
          <subject --> predicate>?  $p;d$           (question)
          <? --> predicate>?                        (question with variable)
          <subject --> ?>?                          (question with variable)
        """
        text = text.strip()
        if not text:
            return None

        # Determine sentence type
        is_question = False
        truth_value = None
        budget = None

        # Extract budget if present: $p;d$
        if '$' in text:
            parts = text.split('$')
            for i, p in enumerate(parts):
                if ';' in p and i > 0:
                    try:
                        vals = p.split(';')
                        priority = float(vals[0])
                        durability = float(vals[1])
                        budget = BudgetValue(priority, durability)
                    except (ValueError, IndexError):
                        pass
            # Remove budget from text
            first_dollar = text.index('$')
            text = text[:first_dollar].strip()

        # Extract truth-value if present: %f;c%
        if '%' in text:
            parts = text.split('%')
            for p in parts:
                if ';' in p:
                    try:
                        vals = p.split(';')
                        freq = float(vals[0])
                        conf = float(vals[1])
                        truth_value = TruthValue(freq, conf, [serial_counter])
                    except (ValueError, IndexError):
                        pass
            # Remove truth-value from text
            first_pct = text.index('%')
            text = text[:first_pct].strip()

        # Determine punctuation
        if text.endswith('?'):
            is_question = True
            text = text[:-1].strip()
        elif text.endswith('.'):
            text = text[:-1].strip()

        # Parse statement: <subject --> predicate>
        text = text.strip('<>() ')
        if '-->' in text:
            parts = text.split('-->')
            subject = parts[0].strip()
            predicate = parts[1].strip()
        elif '->' in text:
            parts = text.split('->')
            subject = parts[0].strip()
            predicate = parts[1].strip()
        else:
            return None

        statement = Statement(subject, predicate)

        if is_question:
            task = Task(statement, truth_value=None, budget=budget, is_input=True)
        else:
            if truth_value is None:
                truth_value = TruthValue(1.0, 0.9, [serial_counter])
            task = Task(statement, truth_value=truth_value, budget=budget, is_input=True)

        return task


# ---------------------------------------------------------------------------
# MiniNARS
# ---------------------------------------------------------------------------

class MiniNARS:
    """
    The Mini-NARS reasoning system.

    Architecture:
      - Buffer: I/O interface holding new tasks
      - Memory: bag of concepts forming a concept graph
      - Inference engine: local, forward, backward rules
      - Control: priority-based selection in bags
    """

    REPORT_THRESHOLD = 0.3
    BUFFER_SIZE = 10
    BUFFER_TASKS_PER_CYCLE = 2

    def __init__(self, silent=False):
        # Memory: bag of concepts (100 levels x 10 capacity)
        self.memory = Bag(levels=100, bucket_capacity=10, label="memory")
        # Buffer: priority queue of tasks
        self.buffer = []
        self.cycle_count = 0
        self.output_log = []
        self.silent = silent
        self.parser = NarseseParser()
        self._input_serial = 0
        self._reported = {}  # dedup: statement -> best confidence reported

    # --- I/O ---

    def input_narsese(self, text):
        """Accept a Narsese sentence as input."""
        self._input_serial += 1
        task = self.parser.parse(text, self._input_serial)
        if task:
            self._add_to_buffer(task)
            if not self.silent:
                label = "IN" if not task.is_question else "IN"
                print(f"  [{self.cycle_count}] IN: {self._format_task(task)}")
            return task
        return None

    def _format_task(self, task):
        """Format a task for display."""
        s = task.statement
        if task.is_question:
            return f"<{s.subject} --> {s.predicate}>?"
        else:
            tv = task.truth_value
            return f"<{s.subject} --> {s.predicate}>. %{tv.frequency:.2f};{tv.confidence:.2f}%"

    def _format_answer(self, statement, tv):
        """Format an answer for display."""
        return f"<{statement.subject} --> {statement.predicate}>. %{tv.frequency:.2f};{tv.confidence:.2f}%"

    def _add_to_buffer(self, task):
        """Add task to the buffer (priority queue)."""
        # Use negative priority for min-heap to get max-priority first
        heapq.heappush(self.buffer, (-task.priority, task.serial, task))
        # Trim buffer
        while len(self.buffer) > self.BUFFER_SIZE:
            heapq.heappop(self.buffer)

    def _report(self, label, text):
        """Output a message."""
        msg = f"  [{self.cycle_count}] {label}: {text}"
        self.output_log.append(msg)
        if not self.silent:
            print(msg)

    # --- Memory operations ---

    def _get_or_create_concept(self, term):
        """Get concept from memory, create if not exists."""
        if term == "?":
            return None
        concept = self.memory.get(term)
        if concept is None:
            concept = Concept(term)
        return concept

    def _new_task(self, term, task):
        """Add a new task to the concept identified by term."""
        if term == "?":
            return
        concept = self._get_or_create_concept(term)
        if concept is None:
            return

        concept.task_bag.put(task)

        # If the task is a judgment, also add as belief
        if task.truth_value is not None:
            existing_belief = concept.belief_bag.peek(str(task.statement))
            if existing_belief:
                belief_item = concept.belief_bag.get(str(task.statement))
                if belief_item:
                    belief_item.add_truth(task.truth_value)
                    concept.belief_bag.put(belief_item)
            else:
                belief = Belief(task.statement, task.truth_value,
                                BudgetValue(task.budget.priority, task.budget.durability))
                concept.belief_bag.put(belief)

        activate(concept, task.budget)
        self.memory.put(concept)

    # --- Working cycle ---

    def cycle(self, n=1):
        """Run n working cycles."""
        for _ in range(n):
            self.cycle_count += 1
            self._memory_cycle()
            self._buffer_cycle()

    def _memory_cycle(self):
        """Select a concept, fire it."""
        concept = self.memory.select()
        if concept is None:
            return
        concept.fire(self)
        self.memory.put_back(concept)

    def _buffer_cycle(self):
        """Move top tasks from buffer to concepts."""
        for _ in range(self.BUFFER_TASKS_PER_CYCLE):
            if not self.buffer:
                break
            neg_pri, serial, task = heapq.heappop(self.buffer)
            self._new_task(task.statement.subject, task)
            self._new_task(task.statement.predicate, task)

    # --- Inference engine ---

    def inference(self, task, judgment, belief):
        """
        Carry out an inference step from task + judgment (from belief).
        Dispatches to local, forward, or backward inference.
        """
        if self._match(task, judgment, belief):
            # Local rules
            if task.is_question:
                self._choice(task, judgment, belief)
            else:
                self._revise(task, judgment, belief)
        else:
            # Syllogistic rules
            if task.is_question:
                self._backward(task, judgment, belief)
            else:
                self._forward(task, judgment, belief)

    def _match(self, task, judgment, belief):
        """Check if task and belief have matching statements."""
        st = task.statement
        sb = belief.statement
        return (self._term_match(st.subject, sb.subject)
                and self._term_match(st.predicate, sb.predicate))

    def _term_match(self, term_t, term_b):
        """Two terms match if identical or one is a variable."""
        return term_t == "?" or term_t == term_b

    # --- Local rules ---

    def _revise(self, task, judgment, belief):
        """Revision: merge evidence from two sources for same statement."""
        tv_t = task.truth_value
        tv_b = judgment
        f, c = tv_revision(tv_t.frequency, tv_t.confidence,
                            tv_b.frequency, tv_b.confidence)

        # Merge evidential bases
        eb = list(set((tv_t.evidential_base or []) + (tv_b.evidential_base or [])))
        new_tv = TruthValue(f, c, eb)

        p, d = revision_budget(task, judgment, belief)
        new_budget = BudgetValue(p, d)
        new_task = Task(belief.statement, new_tv, new_budget)
        self._post_processing(task, belief, new_task, "Revised")

    def _choice(self, task, judgment, belief):
        """Choice: check if belief provides a better answer for the question."""
        quality = self._better_solution(task, judgment, belief)
        if quality > 0:
            p, d = choice_budget(task, belief, quality)
            new_budget = BudgetValue(p, d)
            new_task = Task(belief.statement, judgment, new_budget)
            self._add_to_buffer(new_task)

            # Report answer (dedup)
            key = "ANS:" + str(belief.statement)
            prev_conf = self._reported.get(key, -1)
            if judgment.confidence > prev_conf + 0.01:
                self._reported[key] = judgment.confidence
                answer_text = self._format_answer(belief.statement, judgment)
                self._report("ANS", answer_text)

    def _better_solution(self, task, judgment, belief):
        """Check if judgment is a better answer than stored solution."""
        quality = 0.0
        if task.solution is None:
            task.solution = judgment
            if task.has_variable():
                quality = judgment.expectation()
            else:
                quality = judgment.confidence
        else:
            prev = task.solution
            if task.has_variable():
                if prev.expectation() < judgment.expectation():
                    task.solution = judgment
                    quality = judgment.expectation()
            else:
                if prev.confidence < judgment.confidence:
                    task.solution = judgment
                    quality = judgment.confidence
        return quality

    # --- Syllogistic: figure determination ---

    def _get_figure(self, st, sb):
        """
        Determine the figure of the syllogism:
        1: sT.subject matches sB.predicate  (S->M, M->P => S->P ded)
        2: sT.predicate matches sB.predicate (S->M, P->M => ...)
        3: sT.subject matches sB.subject    (M->S, M->P => ...)
        4: sT.predicate matches sB.subject  (M->P, S->M => ...)
        """
        if (self._term_match(st.subject, sb.predicate)
                and not self._term_match(st.predicate, sb.subject)):
            return 1
        if (self._term_match(st.predicate, sb.predicate)
                and not self._term_match(st.subject, sb.subject)):
            return 2
        if (self._term_match(st.subject, sb.subject)
                and not self._term_match(st.predicate, sb.predicate)):
            return 3
        if (self._term_match(st.predicate, sb.subject)
                and not self._term_match(st.subject, sb.predicate)):
            return 4
        return 0

    # --- Forward inference (task is judgment) ---

    def _forward(self, task, judgment, belief):
        """Forward syllogistic inference: task is a judgment."""
        st = task.statement
        sb = belief.statement
        figure = self._get_figure(st, sb)
        if figure == 0:
            return

        f1 = task.truth_value.frequency
        c1 = task.truth_value.confidence
        f2 = judgment.frequency
        c2 = judgment.confidence

        # Merge evidential bases
        eb = list(set((task.truth_value.evidential_base or [])
                      + (judgment.evidential_base or [])))

        s1 = s2 = None
        tv1 = tv2 = None

        if figure == 1:
            # Task: S -> M, Belief: M -> P
            # Actually per the book: figure 1 is sT.sub matches sB.pred
            # That means task has ?->X and belief has X->?
            # task: A -> M (sub=A, pred=M), belief: X -> M (sub=X, pred=M... no)
            # Let me re-read carefully.
            #
            # From the code in Ch4:
            # Figure 1: termMatch(sT.subject, sB.predicate)
            #   task: S -> M, belief: ? -> S  where S = sT.subject = sB.predicate
            #   Actually: sT = (subT -> preT), sB = (subB -> preB)
            #   sT.subject == sB.predicate => subT == preB
            #   So shared term M = subT = preB
            #   task:  M -> preT     belief: subB -> M
            #   This is: {M->P, S->M} which is deduction figure
            #
            # Conclusions: S->P (ded) and P->S (exe')
            s1 = Statement(sb.subject, st.predicate)      # subB -> preT (S->P)
            s2 = Statement(st.predicate, sb.subject)       # preT -> subB (P->S)
            fv1, cv1 = deduction(f1, c1, f2, c2)
            fv2, cv2 = exemplification(f1, c1, f2, c2)
            tv1 = TruthValue(fv1, cv1, eb)
            tv2 = TruthValue(fv2, cv2, eb)

        elif figure == 2:
            # sT.predicate == sB.predicate, shared = M = preT = preB
            # task: subT -> M, belief: subB -> M
            # This is abduction figure: {P->M, S->M} => S->P
            s1 = Statement(sb.subject, st.subject)         # subB -> subT
            s2 = Statement(st.subject, sb.subject)          # subT -> subB
            fv1, cv1 = abduction(f1, c1, f2, c2)
            fv2, cv2 = abduction(f2, c2, f1, c1)
            tv1 = TruthValue(fv1, cv1, eb)
            tv2 = TruthValue(fv2, cv2, eb)

        elif figure == 3:
            # sT.subject == sB.subject, shared = M = subT = subB
            # task: M -> preT, belief: M -> preB
            # This is induction figure: {M->P, M->S} => S->P
            s1 = Statement(sb.predicate, st.predicate)     # preB -> preT
            s2 = Statement(st.predicate, sb.predicate)      # preT -> preB
            fv1, cv1 = induction(f1, c1, f2, c2)
            fv2, cv2 = induction(f2, c2, f1, c1)
            tv1 = TruthValue(fv1, cv1, eb)
            tv2 = TruthValue(fv2, cv2, eb)

        elif figure == 4:
            # sT.predicate == sB.subject, shared = M = preT = subB
            # task: subT -> M, belief: M -> preB
            # = {S -> M, M -> P} => S -> P by deduction, P -> S by exemplification
            s1 = Statement(st.subject, sb.predicate)       # subT -> preB (S->P)
            s2 = Statement(sb.predicate, st.subject)        # preB -> subT (P->S)
            fv1, cv1 = deduction(f2, c2, f1, c1)
            fv2, cv2 = exemplification(f2, c2, f1, c1)
            tv1 = TruthValue(fv1, cv1, eb)
            tv2 = TruthValue(fv2, cv2, eb)

        if s1 and tv1:
            quality1 = tv1.expectation()
            p1, d1 = syllogistic_budget(task, belief, quality1)
            new_task1 = Task(s1, tv1, BudgetValue(p1, d1))
            self._post_processing(task, belief, new_task1, "Derived")

        if s2 and tv2:
            quality2 = tv2.expectation()
            p2, d2 = syllogistic_budget(task, belief, quality2)
            new_task2 = Task(s2, tv2, BudgetValue(p2, d2))
            self._post_processing(task, belief, new_task2, "Derived")

    # --- Backward inference (task is question) ---

    def _backward(self, task, judgment, belief):
        """Backward syllogistic inference: task is a question."""
        st = task.statement
        sb = belief.statement
        figure = self._get_figure(st, sb)
        if figure == 0:
            return

        s1 = s2 = None

        if figure == 1:
            s1 = Statement(sb.subject, st.predicate)
            s2 = Statement(st.predicate, sb.subject)
        elif figure == 2:
            s1 = Statement(sb.subject, st.subject)
            s2 = Statement(st.subject, sb.subject)
        elif figure == 3:
            s1 = Statement(sb.predicate, st.predicate)
            s2 = Statement(st.predicate, sb.predicate)
        elif figure == 4:
            s1 = Statement(st.subject, sb.predicate)
            s2 = Statement(sb.predicate, st.subject)

        if s1:
            p1, d1 = syllogistic_budget(task, belief, 0.5)
            q1 = Task(s1, None, BudgetValue(p1, d1))
            self._post_processing(task, belief, q1, "Derived?")

        if s2:
            p2, d2 = syllogistic_budget(task, belief, 0.5)
            q2 = Task(s2, None, BudgetValue(p2, d2))
            self._post_processing(task, belief, q2, "Derived?")

    # --- Post-processing ---

    def _post_processing(self, task, belief, new_task, label=""):
        """Add derived task to buffer; report if priority above threshold."""
        self._add_to_buffer(new_task)

        if new_task.priority > self.REPORT_THRESHOLD and new_task.truth_value:
            tv = new_task.truth_value
            s = new_task.statement
            if tv.confidence > 0.1:
                # Dedup: only report if new or better confidence
                key = str(s)
                prev_conf = self._reported.get(key, -1)
                if tv.confidence > prev_conf + 0.01:
                    self._reported[key] = tv.confidence
                    self._report("OUT", self._format_answer(s, tv))

    # --- Inspection ---

    def show_concepts(self):
        """Display all concepts in memory."""
        concepts = self.memory.all_items()
        if not concepts:
            print("  (memory empty)")
            return
        concepts.sort(key=lambda c: c.priority, reverse=True)
        for c in concepts:
            print(f"  {c}")
            for b in c.belief_bag.all_items():
                best = b.best_truth()
                if best:
                    print(f"    Belief: {b.statement} {best}")

    def show_beliefs(self, term=None):
        """Show beliefs, optionally for a specific concept."""
        concepts = self.memory.all_items()
        for c in concepts:
            if term and c.term != term:
                continue
            beliefs = c.belief_bag.all_items()
            if beliefs:
                for b in beliefs:
                    best = b.best_truth()
                    if best:
                        print(f"  {b.statement} {best}")
