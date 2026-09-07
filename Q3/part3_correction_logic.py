"""
Part 3: Spelling Correction Logic

### 3.1 Non-Word Error Correction
For a word not found in the vocabulary, use both Method
A and Method B to generate candidate sets. The best correction is the candidate with the
highest frequency (unigram probability).
"""

from part1_corpus_and_model import vocab, vocab_freq, bigram_prob
from part2_candidate_generation import method_a_candidates, method_b_candidates, deletion_dict


def correct_nonword(word, method='B', deletion_dict=None):

    if word in vocab:
        return word

    if method == 'A':
        candidates = method_a_candidates(word, vocab)
    elif method == 'B':
        candidates = method_b_candidates(word, deletion_dict)
    else:
        raise ValueError("method must be 'A' or 'B'")

    if not candidates:
        return word  # no correction found within edit distance 1

    return max(candidates, key=lambda w: vocab_freq[w])


"""
### 3.2 Real-Word Error Correction

1. Generate edit-distance-1 candidates for the (valid) word itself.
2. Score the original word and every candidate using the surrounding bigram context
3. Only replace the original word if a candidate's phrase probability exceeds the original's by a configurable margin
"""

def correct_realword(sent_words, idx, deletion_dict, threshold=1.5):

    # sent_words: list of lowercase tokens (already possibly containing an error at `idx`)
    # idx: index of the word to check/correct
    # threshold: how much higher a candidate's phrase probability must be to trigger a correction

    word = sent_words[idx]
    if word not in vocab:
        return word  # this isn't a real case

    candidates = method_b_candidates(word, deletion_dict)
    candidates.discard(word)
    candidates = {c for c in candidates if c in vocab}
    if not candidates:
        return word

    def phrase_score(w):
        score = 1.0
        if idx > 0:
            score *= bigram_prob(sent_words[idx - 1], w)
        if idx < len(sent_words) - 1:
            score *= bigram_prob(w, sent_words[idx + 1])
        return score

    original_score = phrase_score(word)
    best_word, best_score = word, original_score

    for c in candidates:
        s = phrase_score(c)
        if s > best_score:
            best_word, best_score = c, s

    if best_word != word and best_score > threshold * original_score:
        return best_word
    return word


# Demo
for w in ['helo', 'wrold', 'jsut', 'zzzzz']:
    print(f"{w!r:10} -> {correct_nonword(w, method='B', deletion_dict=deletion_dict)!r}")

#testing
demo_sent = ['i', 'ate', 'an', 'apply']
corrected_idx = 3
print("Original:", demo_sent)
fix = correct_realword(demo_sent, corrected_idx, deletion_dict)
print("Correction for position 3:", fix)
