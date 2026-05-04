"""PyNARS reference backend.

Runs the same medical knowledge base and patient evidence through the actual
PyNARS `Reasoner` (the KanrenEngine-backed NAL implementation). This lets the
project demonstrate that the Python-side NAL semantics in `nal_truth.py` match
what the reference implementation derives when given the chance to settle.

Usage: build the backend, call `load_knowledge_base`, feed evidence via Narsese
strings (or use `add_patient_evidence`), then call `query` to read NARS' final
belief about `<patient --> disease>`. Results depend on the attention-bag
scheduler so `cycles_per_query` may need to be large for all diseases to get
derivations in a crowded rule base.
"""

from typing import Dict, List, Optional, Tuple

from pynars.NARS import Reasoner
from pynars import Narsese

from . import knowledge_base as kb
from . import narsese_translator as nt
from .nal_truth import Truth


class PyNarsBackend:
    def __init__(self, memory: int = 5000, capacity: int = 5000) -> None:
        self.nars = Reasoner(memory, capacity)
        self._kb_loaded = False

    def _run_cycles(self, n: int) -> None:
        """Swallow PyNARS KanrenEngine internal exceptions so a bad
        derivation doesn't abort the whole run."""
        for _ in range(n):
            try:
                self.nars.cycle()
            except Exception:  # noqa: BLE001
                continue

    def load_knowledge_base(self, warmup_cycles: int = 200) -> None:
        if self._kb_loaded:
            return
        for rule in nt.knowledge_base_rules():
            success, _task, _ = self.nars.input_narsese(rule)
            if not success:
                raise RuntimeError(f"Failed to parse rule: {rule}")
        self._run_cycles(warmup_cycles)
        self._kb_loaded = True

    def add_evidence(self, narsese: str, cycles: int = 100) -> None:
        success, _task, _ = self.nars.input_narsese(narsese)
        if not success:
            raise RuntimeError(f"Failed to parse evidence: {narsese}")
        self._run_cycles(cycles)

    def add_patient_evidence(
        self,
        patient: str,
        symptoms_present=(),
        symptoms_absent=(),
        labs_positive=(),
        labs_negative=(),
        cycles_per_item: int = 100,
    ) -> None:
        items = nt.patient_evidence(
            patient,
            symptoms_present=symptoms_present,
            symptoms_absent=symptoms_absent,
            labs_positive=labs_positive,
            labs_negative=labs_negative,
        )
        for _label, stmt in items:
            self.add_evidence(stmt, cycles=cycles_per_item)

    def query(self, patient: str, disease: str,
              cycles: int = 1000) -> Optional[Truth]:
        """Ask `<patient --> disease>?` and read NARS' current belief."""
        self.nars.input_narsese(nt.diagnosis_term_str(patient, disease) + "?")
        self._run_cycles(cycles)
        term = Narsese.parse(
            nt.diagnosis_term_str(patient, disease) + "."
        ).term
        concept = self.nars.memory.take_by_key(term, remove=False)
        if concept is None or concept.belief_table.empty:
            return None
        best = concept.belief_table.first()
        if best.truth is None:
            return None
        return Truth(best.truth.f, best.truth.c)

    def query_all(self, patient: str,
                  cycles_per_disease: int = 1000
                  ) -> Dict[str, Optional[Truth]]:
        # Prompt all diseases first so the concept bag activates them together,
        # then read all beliefs.
        for d in kb.DISEASES:
            self.nars.input_narsese(nt.diagnosis_term_str(patient, d) + "?")
        self._run_cycles(cycles_per_disease * len(kb.DISEASES))
        out: Dict[str, Optional[Truth]] = {}
        for d in kb.DISEASES:
            term = Narsese.parse(nt.diagnosis_term_str(patient, d) + ".").term
            concept = self.nars.memory.take_by_key(term, remove=False)
            if concept is None or concept.belief_table.empty:
                out[d] = None
                continue
            best = concept.belief_table.first()
            out[d] = Truth(best.truth.f, best.truth.c) if best.truth else None
        return out
