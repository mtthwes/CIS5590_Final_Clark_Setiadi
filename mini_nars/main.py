#!/usr/bin/env python3
"""
Mini-NARS Interactive Shell
A minimal implementation of the Non-Axiomatic Reasoning System.

Usage:
  python main.py              Interactive mode
  python main.py --test       Run built-in test suite
  python main.py FILE         Load experience from file

Input format:
  <subject --> predicate>. %f;c%   Judgment (truth-value optional, default %1.0;0.9%)
  <subject --> predicate>?         Question
  <? --> predicate>?               Question with variable (what inherits predicate?)
  <subject --> ?>?                 Question with variable (what does subject inherit?)
  123                              Run 123 inference cycles
  :memory                         Show all concepts
  :beliefs                        Show all beliefs
  :beliefs term                   Show beliefs for a term
  :reset                          Reset the system
  :help                           Show this help
  :quit / :exit                   Exit

Example session:
  <bird --> animal>. %1.0;0.9%
  <robin --> bird>. %1.0;0.9%
  5
  <robin --> animal>?
"""

import sys
import os

# Add the directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_nars import MiniNARS


BANNER = r"""
  MINI NARS

  Non-Axiomatic Reasoning System (Minimal Implementation)
  Based on: "Designing a Mind" by Pei Wang (2026)
  Type :help for commands, :quit to exit.
"""


def run_tests(nars=None):
    """Run built-in test examples demonstrating NARS capabilities."""
    print("\n" + "=" * 60)
    print("  MINI-NARS TEST SUITE")
    print("=" * 60)

    # ---- Test 1: Deduction ----
    print("\n--- Test 1: Deduction ---")
    print("  Given: bird -> animal, robin -> bird")
    print("  Expected: robin -> animal (by deduction)\n")

    nars = MiniNARS()
    nars.input_narsese("<bird --> animal>. %1.0;0.9%")
    nars.input_narsese("<robin --> bird>. %1.0;0.9%")
    nars.cycle(5)
    nars.input_narsese("<robin --> animal>?")
    nars.cycle(20)
    print()

    # ---- Test 2: Abduction ----
    print("--- Test 2: Abduction ---")
    print("  Given: bird -> animal, fish -> animal")
    print("  Expected: fish -> bird (weak, by abduction)\n")

    nars = MiniNARS()
    nars.input_narsese("<bird --> animal>. %1.0;0.9%")
    nars.input_narsese("<fish --> animal>. %1.0;0.9%")
    nars.cycle(5)
    nars.input_narsese("<fish --> bird>?")
    nars.cycle(20)
    print()

    # ---- Test 3: Induction ----
    print("--- Test 3: Induction ---")
    print("  Given: water -> liquid, water -> transparent")
    print("  Expected: transparent -> liquid (weak, by induction)\n")

    nars = MiniNARS()
    nars.input_narsese("<water --> liquid>. %1.0;0.9%")
    nars.input_narsese("<water --> transparent>. %1.0;0.9%")
    nars.cycle(5)
    nars.input_narsese("<transparent --> liquid>?")
    nars.cycle(20)
    print()

    # ---- Test 4: Revision ----
    print("--- Test 4: Revision ---")
    print("  Given: robin -> bird %0.7;0.9%, robin -> bird %0.9;0.8%")
    print("  Expected: robin -> bird with merged evidence\n")

    nars = MiniNARS()
    nars.input_narsese("<robin --> bird>. %0.7;0.9%")
    nars.input_narsese("<robin --> bird>. %0.9;0.8%")
    nars.cycle(10)
    nars.input_narsese("<robin --> bird>?")
    nars.cycle(20)
    print()

    # ---- Test 5: Chained deduction ----
    print("--- Test 5: Chained Deduction ---")
    print("  Given: robin -> bird, bird -> animal, animal -> living")
    print("  Expected: robin -> living (multi-step deduction)\n")

    nars = MiniNARS()
    nars.input_narsese("<robin --> bird>. %1.0;0.9%")
    nars.input_narsese("<bird --> animal>. %1.0;0.9%")
    nars.input_narsese("<animal --> living>. %1.0;0.9%")
    nars.cycle(30)
    nars.input_narsese("<robin --> living>?")
    nars.cycle(50)
    print()

    # ---- Test 6: Mixed evidence ----
    print("--- Test 6: Mixed Evidence ---")
    print("  Given: penguin -> bird %1.0;0.9%, penguin -> flyer %0.0;0.9%")
    print("  Query: bird -> flyer?\n")

    nars = MiniNARS()
    nars.input_narsese("<penguin --> bird>. %1.0;0.9%")
    nars.input_narsese("<penguin --> flyer>. %0.0;0.9%")
    nars.cycle(10)
    nars.input_narsese("<bird --> flyer>?")
    nars.cycle(30)
    print()

    # ---- Test 7: Active learning via backward inference ----
    print("--- Test 7: Backward Inference (Question Derivation) ---")
    print("  Given: bird -> animal")
    print("  Query: robin -> animal?")
    print("  Expected: derived question robin -> bird?\n")

    nars = MiniNARS()
    nars.input_narsese("<bird --> animal>. %1.0;0.9%")
    nars.cycle(3)
    nars.input_narsese("<robin --> animal>?")
    nars.cycle(20)
    print()

    # ---- Test 8: Medical domain (relevant to CIS 5590 project) ----
    print("--- Test 8: Medical Diagnosis Domain ---")
    print("  Given: fever -> symptom, cough -> symptom,")
    print("         flu -> disease, flu -> causesFever,")
    print("         cold -> disease, cold -> causesCough\n")

    nars = MiniNARS()
    nars.input_narsese("<fever --> symptom>. %1.0;0.9%")
    nars.input_narsese("<cough --> symptom>. %1.0;0.9%")
    nars.input_narsese("<flu --> disease>. %1.0;0.9%")
    nars.input_narsese("<flu --> causesFever>. %1.0;0.9%")
    nars.input_narsese("<cold --> disease>. %1.0;0.9%")
    nars.input_narsese("<cold --> causesCough>. %1.0;0.9%")
    nars.cycle(30)
    print("\n  Querying: flu -> cold?")
    nars.input_narsese("<flu --> cold>?")
    nars.cycle(30)
    print()

    print("=" * 60)
    print("  Tests complete.")
    print("=" * 60)


def load_file(nars, filepath):
    """Load an experience file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                if line.startswith('OUT:'):
                    continue  # Skip expected output lines
                try:
                    cycles = int(line)
                    nars.cycle(cycles)
                except ValueError:
                    nars.input_narsese(line)
        print(f"  Loaded: {filepath}")
    except FileNotFoundError:
        print(f"  Error: file not found: {filepath}")


def interactive(nars=None):
    """Run the interactive REPL."""
    if nars is None:
        nars = MiniNARS()
    print(BANNER)

    while True:
        try:
            raw = input("nars> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not raw:
            continue

        # Commands
        if raw in (":quit", ":exit", ":q"):
            print("  Goodbye.")
            break
        elif raw == ":help":
            print(__doc__)
            continue
        elif raw == ":memory":
            nars.show_concepts()
            continue
        elif raw.startswith(":beliefs"):
            parts = raw.split()
            if len(parts) > 1:
                nars.show_beliefs(parts[1])
            else:
                nars.show_beliefs()
            continue
        elif raw == ":reset":
            nars = MiniNARS()
            print("  System reset.")
            continue
        elif raw == ":test":
            run_tests()
            nars = MiniNARS()
            continue

        # Try parsing as cycle count
        try:
            cycles = int(raw)
            if cycles > 0:
                nars.cycle(cycles)
                continue
        except ValueError:
            pass

        # Try loading a file
        if raw.startswith(":load "):
            filepath = raw[6:].strip()
            load_file(nars, filepath)
            continue

        # Parse as Narsese
        result = nars.input_narsese(raw)
        if result is None:
            print("  Parse error. Type :help for format.")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--test":
            run_tests()
        elif os.path.isfile(arg):
            nars = MiniNARS()
            load_file(nars, arg)
            interactive(nars)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python main.py [--test | FILE]")
    else:
        interactive()


if __name__ == "__main__":
    main()
