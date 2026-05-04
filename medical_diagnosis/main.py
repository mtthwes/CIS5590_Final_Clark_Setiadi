"""Entry point for the medical-diagnosis experiment.

Run with:
    python -m Experiments.MedicalDiagnosis.main            # default: full demo
    python -m Experiments.MedicalDiagnosis.main cohort     # cohort accuracy only
    python -m Experiments.MedicalDiagnosis.main revision   # belief-revision demo
    python -m Experiments.MedicalDiagnosis.main case patient_epsilon
    python -m Experiments.MedicalDiagnosis.main compare patient_alpha
"""

import sys

from .dataset import PATIENT_CASES, get_case
from . import experiments


USAGE = (
    "Usage:\n"
    "  python -m Experiments.MedicalDiagnosis.main [command] [args]\n\n"
    "Commands:\n"
    "  full                    Run the complete demo (default).\n"
    "  cohort                  Diagnose every patient and report accuracy.\n"
    "  revision                Run the scripted belief-revision experiment.\n"
    "  case <patient_id>       Diagnose & explain a single patient case.\n"
    "  compare <patient_id>    Cross-check against the PyNARS reasoner.\n"
    "  list                    List available patient ids.\n"
)


def main(argv):
    if not argv:
        experiments.run_everything()
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd in ("full", "all"):
        experiments.run_everything()
    elif cmd == "cohort":
        experiments.run_all_cases()
    elif cmd == "revision":
        experiments.belief_revision_conflict()
    elif cmd == "case":
        if not rest:
            print("Missing patient id.\n" + USAGE, file=sys.stderr)
            return 2
        experiments.run_case(get_case(rest[0]))
    elif cmd == "compare":
        if not rest:
            print("Missing patient id.\n" + USAGE, file=sys.stderr)
            return 2
        experiments.compare_with_pynars(get_case(rest[0]))
    elif cmd == "list":
        for c in PATIENT_CASES:
            print(f"  {c.patient_id:20s}  [truth={c.ground_truth}]  {c.description}")
    elif cmd in ("-h", "--help", "help"):
        print(USAGE)
    else:
        print(f"Unknown command: {cmd}\n" + USAGE, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
