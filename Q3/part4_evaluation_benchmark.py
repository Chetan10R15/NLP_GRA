"""
Part 4: Evaluation and "Speed Demon" Benchmark

### 4.1 Test Set Generation
We hold out 10% of Brown corpus sentences. For each held-out sentence we pick one random word and
introduce a single-edit spelling mistake
"""

import random
import string
import time

from part1_corpus_and_model import sentences, vocab
from part2_candidate_generation import deletion_dict
from part3_correction_logic import correct_nonword, correct_realword


def introduce_error(word):
    # Apply random edit
    if len(word) < 2:
        return word
    letters = string.ascii_lowercase
    op = random.choice(['delete', 'insert', 'replace', 'transpose'])
    i = random.randint(0, len(word) - 1)

    if op == 'delete':
        return word[:i] + word[i + 1:]
    elif op == 'insert':
        c = random.choice(letters)
        return word[:i] + c + word[i:]
    elif op == 'replace':
        c = random.choice(letters)
        return word[:i] + c + word[i + 1:]
    elif op == 'transpose':
        if len(word) < 2:
            return word
        j = min(i + 1, len(word) - 1)
        w = list(word)
        w[i], w[j] = w[j], w[i]
        return ''.join(w)
    return word


random.shuffle(sentences)
test_sentences = sentences[: max(1, int(0.1 * len(sentences)))]
print(f"Held-out test sentences: {len(test_sentences):,}")

nonword_test = []   # (sentence_tokens, idx, original_word, misspelled_word)
realword_test = []   # (sentence_tokens, idx, original_word, misspelled_word)

MAX_ATTEMPTS = 15

for sent in test_sentences:
    candidates_idx = [i for i, w in enumerate(sent) if len(w) >= 3]
    if not candidates_idx:
        continue
    idx = random.choice(candidates_idx)
    orig_word = sent[idx]

    # Non-word error: keep retrying an edit until the result is OOV
    for _ in range(MAX_ATTEMPTS):
        m = introduce_error(orig_word)
        if m != orig_word and m not in vocab:
            nonword_test.append((sent, idx, orig_word, m))
            break

    # Real-word error: keep retrying an edit until the result IS a different valid word
    for _ in range(MAX_ATTEMPTS):
        m = introduce_error(orig_word)
        if m != orig_word and m in vocab:
            realword_test.append((sent, idx, orig_word, m))
            break

print(f"Non-word test cases generated: {len(nonword_test):,}")
print(f"Real-word test cases generated: {len(realword_test):,}")


"""
### 4.2 Accuracy: Non-Word Correction (Method A vs. Method B)
Since both methods search the *same* edit-distance-1 neighborhood and pick by the *same* frequency
criterion, we expect their accuracy to be identical (or nearly so) - the whole point of Method B is
that it gets to the same answer faster, not a different one.
"""

correct_a = 0
correct_b = 0

for sent, idx, orig, misspelled in nonword_test:
    pred_a = correct_nonword(misspelled, method='A')
    pred_b = correct_nonword(misspelled, method='B', deletion_dict=deletion_dict)
    correct_a += (pred_a == orig)
    correct_b += (pred_b == orig)

n = len(nonword_test)
acc_a = correct_a / n if n else 0
acc_b = correct_b / n if n else 0
print(f"Non-word correction accuracy — Method A: {acc_a:.2%} ({correct_a}/{n})")
print(f"Non-word correction accuracy — Method B: {acc_b:.2%} ({correct_b}/{n})")


"""
### 4.3 Accuracy: Real-Word Correction
"""

correct_real = 0

for sent, idx, orig, misspelled in realword_test:
    corrupted_sent = list(sent)
    corrupted_sent[idx] = misspelled
    pred = correct_realword(corrupted_sent, idx, deletion_dict)
    correct_real += (pred == orig)

n_real = len(realword_test)
acc_real = correct_real / n_real if n_real else 0
print(f"Real-word correction accuracy: {acc_real:.2%} ({correct_real}/{n_real})")


"""
### 4.4 Speed Demon Benchmark

We isolate the non-word candidate generation + scoring logic, build a fixed batch of exactly
1,000 misspelled words, and run the *same* batch through Method A and Method B, timing each pass
end-to-end.
"""

vocab_list = [w for w in vocab if len(w) >= 4]  # avoid trivial 1-2 letter words
misspelled_batch = []

while len(misspelled_batch) < 1000:
    w = random.choice(vocab_list)
    m = introduce_error(w)
    if m != w and m not in vocab:
        misspelled_batch.append(m)

misspelled_batch = misspelled_batch[:1000]
print(f"Prepared batch of {len(misspelled_batch)} misspelled words.")

# Method A timing
t0 = time.perf_counter()
results_a = [correct_nonword(w, method='A') for w in misspelled_batch]
time_a = time.perf_counter() - t0

# Method B timing
t0 = time.perf_counter()
results_b = [correct_nonword(w, method='B', deletion_dict=deletion_dict) for w in misspelled_batch]
time_b = time.perf_counter() - t0

print(f"Method A (edit-distance-1 generation): {time_a:.4f}s total "
      f"({time_a/len(misspelled_batch)*1000:.4f} ms/word)")
print(f"Method B (SymSpell lookup):            {time_b:.4f}s total "
      f"({time_b/len(misspelled_batch)*1000:.4f} ms/word)")
print(f"Speedup (A / B): {time_a / time_b:.2f}x")

# Confirm both methods agree
agreement = sum(1 for a, b in zip(results_a, results_b) if a == b) / len(results_a)
print(f"Output agreement between methods: {agreement:.2%}")

"""
### 4.5 Conclusion — Why Method B Is Faster

Method B (Symmetric Delete) is dramatically faster than Method A (standard
edit-distance-1 generation)

Why?

1. For a word of length `L`, Method A's `edits1()` builds
   roughly "54*L + 25" candidate strings every single time it is called, Method B only needs to generate "L" deletions
2. Method B moves almost all of the heavy lifting into a
   one-time preprocessing step (build_deletion_dict), which is computed once for the entire
   vocabulary.
3. Method B pays for speed with memory, the deletion dictionary stores an
   entry for every one-character deletion of every vocabulary word.

In short, Method A does "O(alphabet_size * L)" work every query, but Method B does only "O(L)" work per
query after a "O(alphabet_size' * L * vocab)" index is built.
"""
