# Question 1: Word Segmentation and POS Tagging

## Overview
This directory contains the modularized, production-quality implementation of **Question 1 (Word Segmentation and Part-of-Speech Tagging)** for English and German, structured directly around the official marking criteria (100 Marks).

---

## Directory & File Structure

```text
Q1/
├── __init__.py
├── data_handling.py           # Criterion 1: Train/test split & data handling (10 Marks)
├── segmentation.py            # Criterion 2: Segmentation model (Trigram LM + Viterbi DP) (15 Marks)
├── tagger.py                  # Criterion 3: POS tagging model (Emission + Transition Trigram HMM) (15 Marks)
├── morphology.py              # Criterion 4: Morphology-aware tagging extension (15 Marks)
├── baselines.py               # Criterion 5: Baseline models (Greedy match & Most-Frequent-Tag) (13 Marks)
├── evaluate.py                # Criterion 6: Evaluation metrics, confusion matrix, error breakdown (15 Marks)
├── run_q1.py                  # Full pipeline runner executing all evaluations and test sentences
├── README_Q1.md               # Project documentation and criteria mapping
└── outputs/                   # Generated reports, metrics, matrices, and benchmark outputs
    ├── comparative_analysis_report.md  # Criterion 7: Comparative analysis report (17 Marks)
    ├── comparison_summary.csv          # Side-by-side tabular comparison of English vs German
    ├── english_results.txt             # English model vs baseline metrics & error breakdown
    ├── german_results.txt              # German model vs baseline metrics & error breakdown
    ├── english_confusion_matrix.csv    # English POS confusion matrix
    ├── german_confusion_matrix.csv     # German POS confusion matrix
    └── sample_test_outputs.txt         # Outputs on benchmark and additional test sentences
```

---

## Marking Scheme Alignment

| Marking Component | Marks | Module File | Description |
|---|---|---|---|
| **1. Train/test split & data handling** | 10 | `data_handling.py` | Brown corpus (80/20 train/test split) & UD German-GSD (official train/dev/test splits), vocabulary generation |
| **2. Segmentation model (trigram + DP)** | 15 | `segmentation.py` | Trigram Language Model with Jelinek-Mercer linear interpolation + Viterbi DP decoder with beam search |
| **3. POS tagging model (emission + transition)** | 15 | `tagger.py` | Trigram HMM with emission/transition probabilities and fast Viterbi decoding |
| **4. Morphology-aware tagging extension** | 15 | `morphology.py` | Brown morphology tag mapping (Sg/Pl/Tense) and German UD features (Gender/Number/Agreement) |
| **5. Baseline models + comparison** | 13 | `baselines.py` | Greedy longest-match segmentation baseline and Most-Frequent-Tag (MFT) baseline |
| **6. Evaluation & error analysis** | 15 | `evaluate.py` | Accuracy, confusion matrix generation, error separation (segmentation-caused vs. genuine POS error) |
| **7. Comparative analysis report** | 17 | `outputs/comparative_analysis_report.md` | Formal report answering the 4 assignment questions with empirical data |
| **Total** | **100** | | |

---

## How to Run

To run the complete evaluation pipeline and regenerate all outputs:

```bash
python Q1/run_q1.py
```

All results, confusion matrices, and benchmark sentence predictions will be saved directly into `Q1/outputs/`.
