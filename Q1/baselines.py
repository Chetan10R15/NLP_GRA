"""
Q1: Word Segmentation and POS Tagging
Criterion 5: Baseline models + comparison (13 Marks)

Implements:
- Segmentation baseline: Greedy Longest-Match against vocabulary.
- Tagging baseline: Most Frequent Tag (MFT) with overall majority class fallback.
"""

from collections import defaultdict, Counter

MAX_WORD_LEN = 25


def greedy_longest_match_segment(text, vocab, max_word_len=MAX_WORD_LEN):
    """
    Baseline segmentation: greedy longest-match algorithm.
    At each index, matches the longest candidate substring in vocab.
    Falls back to single character consumption if no vocabulary word matches.
    """
    text = text.lower()
    n = len(text)
    words = []
    i = 0
    while i < n:
        matched = False
        for j in range(min(n, i + max_word_len), i, -1):
            if text[i:j] in vocab:
                words.append(text[i:j])
                i = j
                matched = True
                break
        if not matched:
            # Consume 1 character if no vocab match found
            words.append(text[i:i + 1])
            i += 1
    return words


def build_mft_baseline(tagged_sents):
    """
    Baseline tagger: Most-Frequent-Tag.
    Builds a dictionary mapping word -> most frequent tag in training data.
    Also returns the default majority tag for out-of-vocabulary words.
    """
    word_tag_counts = defaultdict(Counter)
    overall_tag_counts = Counter()

    for sent in tagged_sents:
        for w, t in sent:
            w_lower = w.lower()
            word_tag_counts[w_lower][t] += 1
            overall_tag_counts[t] += 1

    mft_baseline = {
        word: counts.most_common(1)[0][0]
        for word, counts in word_tag_counts.items()
    }
    default_tag = overall_tag_counts.most_common(1)[0][0] if overall_tag_counts else 'NOUN'
    return mft_baseline, default_tag


def tag_with_baseline(words, mft_baseline, default_tag):
    """Tags a list of words using the Most Frequent Tag baseline."""
    return [(w, mft_baseline.get(w.lower(), default_tag)) for w in words]
