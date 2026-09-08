## Files

- `main.py`
- `spelling_corrector_artifacts.pkl` 

## Stop
Type:

'exit'

to stop.

## What it implements

- Brown Corpus vocabulary and unigram frequencies from the supplied artifact
- Method A: standard edit-distance-1 generation
  - deletion
  - transposition
  - replacement
  - insertion
- Method B: symmetric-delete lookup
- Non-word correction using unigram frequency
- Real-word correction using local bigram probabilities
- ANSI highlighting of changed words
- Per-sentence latency reporting
