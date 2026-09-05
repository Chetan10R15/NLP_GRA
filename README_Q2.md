# Question 2 — Transition-Based Dependency Parser

## Overview

This project implements a simple data-driven transition-based dependency parser from scratch using the Universal Dependencies English-EWT treebank.

The parser uses an arc-standard transition system and a machine-learning classifier to predict parser transitions.

The implementation is divided into three parts:

1. CoNLL-U data processing and oracle simulation
2. Feature extraction and transition classification
3. Dependency parsing and evaluation using UAS and LAS

---

# Dataset

The project uses the Universal Dependencies English-EWT treebank.

### Training Dataset

```text
UD_English-EWT/en_ewt-ud-train.conllu
