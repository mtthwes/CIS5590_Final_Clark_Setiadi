# Explainable Medical Diagnosis Recommender using NARS

Justin Clark and Matthew Setiadi
CIS 5590: Artificial General Intelligence, Spring 2026
Professor Pei Wang, Temple University

---

## What this project is

We built two independent medical diagnosis systems grounded in Non-Axiomatic Logic and used them to study how belief revision works when the evidence is incomplete or contradictory. NARS is designed for exactly this kind of problem under what the textbook calls AIKR, the Assumption of Insufficient Knowledge and Resources. Medicine fits naturally because doctors deal with overlapping symptoms, late test results, and ambiguous presentations every day.

The project has two parts. Matthew implemented Mini-NARS from scratch following Chapter 4 of *Designing a Mind* and applied it to a real patient dataset from Kaggle. Justin built a deterministic NAL diagnosis engine with hand-crafted clinical rules and validated it against the PyNARS reference implementation. Both parts demonstrate the same core behavior: deduction produces strong conclusions, revision pools evidence and increases confidence, and contradictory evidence revises beliefs without erasing what came before.

---

## How to run it

Requires Python 3. No external libraries needed for Matthew's implementation.

**Matthew's Mini-NARS with Kaggle data:**
```
cd mini_nars
python diagnosis_demo.py
```
Make sure the CSV file is in the same folder. The demo loads the dataset, builds the knowledge base, runs patient scenarios, and shows the belief revision experiment step by step.

**Justin's NAL diagnosis engine:**
```
python -m medical_diagnosis.main
```
Runs all 8 patient cases, the cohort accuracy test, and the scripted belief revision experiment. Requires PyNARS for the optional cross-check backend (`pip install pynars`), but the core engine runs without it.

---

## What we found

### Belief revision works as the theory predicts

Both systems show confidence building gradually as symptoms accumulate, not jumping to a conclusion from one piece of evidence. In Matthew's system, a single symptom gives expectation around 0.64 to 0.74. Four symptoms push it to 0.93. In Justin's system, the revision trajectory shows the top diagnosis flipping twice as evidence arrives and then contradicts.

### Contradictory evidence revises, it does not replace

Matthew's negative flu test drops Influenza from 0.75 to 0.45 expectation, but the fever and cough evidence is still there. Justin's negative strep test drops Strep from 0.76 to 0.34 expectation while its confidence actually increases to 0.91, because now there is more total evidence even though most of it points away from Strep. In NAL you can simultaneously be very confident and say "probably not." That is distinctively NAL behavior.

### Strong vs weak inference is enforced by the math

Deduction (following a direct inheritance chain) gives confidence up to 0.81. Abduction and induction (reasoning from shared symptoms) always give confidence below 0.5. The system expresses appropriate uncertainty without us having to program special cases.

### Some diseases cannot be separated by symptoms alone

Matthew's system cannot distinguish Pneumonia from Bronchitis because their symptom profiles in the Kaggle data are nearly identical. Justin's system cannot distinguish Flu from COVID-19 without lab tests. Both observations are clinically realistic and show the system correctly expressing uncertainty rather than guessing.

---

## Project structure

```
README.md                   this file
final_report.pdf            the project report
presentation.pptx           slides from the April 23 presentation

mini_nars/                   Matthew's implementation
  mini_nars.py               core NARS engine (parser, memory, inference cycle)
  data_structures.py         Statement, TruthValue, Task, Belief, Concept
  bag.py                     probabilistic priority queue (Bag)
  truth_functions.py         NAL truth-value functions
  budget_functions.py        resource allocation functions
  diagnosis_demo.py          medical application with Kaggle data
  Disease_symptom_and_patient_profile_dataset.csv

medical_diagnosis/           Justin's implementation
  nal_truth.py               NAL truth functions (deduction, revision)
  knowledge_base.py          hand-crafted clinical rules for 4 diseases
  narsese_translator.py      converts Python data to Narsese statements
  diagnosis_engine.py        deterministic NAL reasoning engine
  explanation.py             plain-English explanation generator
  experiments.py             experiment drivers (cohort, belief revision)
  dataset.py                 simulated patient cases
  nars_backend.py            PyNARS reference backend for cross-checking
  main.py                    entry point
```

---

## How the two approaches differ

Matthew's implementation uses the full NARS architecture from Chapter 4: Bags with probabilistic selection, the working cycle, concept-centered memory, and all four syllogistic figures. The knowledge base uses real statistical frequencies computed from the Kaggle dataset (348 patient records). Inference timing is non-deterministic because it depends on which concepts the Bag selects each cycle.

Justin's implementation applies the NAL truth functions directly and deterministically. The knowledge base uses hand-written clinical priors for flu, cold, COVID-19, and strep throat. It also includes a deduction_via_negation function for handling negative evidence (like a negative lab test) without the confidence collapsing to zero, and an explanation layer that traces every diagnosis back to the evidence that contributed.

Both use the same underlying math: f = f1*f2 and c = f1*f2*c1*c2 for deduction, evidence pooling via w = c/(1-c) for revision, and e = c(f-0.5)+0.5 for ranking.

---

## References

Wang, P. (2026). *Designing a Mind: The Implementations of NARS*. Draft.

Wang, P. (2025). *Non-Axiomatic Logic: A Model of Intelligent Reasoning*. 2nd ed. World Scientific.

OpenNARS-4 / PyNARS: https://github.com/opennars/OpenNARS-4/tree/dev/pynars

Kaggle Dataset: https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset
