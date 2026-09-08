"""
Part 2: Candidate Generation Methods

### 2.1 Method A - Standard Edit-Distance-1 Generation
For a word of length `L`, this generates every possible deletion, transposition, replacement, and
insertion (the classic Norvig-style `edits1` function), then filters down to the ones that are
actual words.
"""

import string
import time
from collections import defaultdict

from part1_corpus_and_model import vocab


def edits1(word):
    letters = string.ascii_lowercase

    # Split the word at every possible position
    splits = [
        (word[:i], word[i:])
        for i in range(len(word) + 1)
    ]

    # Delete one character
    deletes = [
        left + right[1:]
        for left, right in splits
        if right
    ]

    # Swap two adjacent characters
    transposes = [
        left + right[1] + right[0] + right[2:]
        for left, right in splits
        if len(right) > 1
    ]

    # Replace one character with every letter
    replaces = [
        left + char + right[1:]
        for left, right in splits
        if right
        for char in letters
    ]

    # Insert every letter at every position
    inserts = [
        left + char + right
        for left, right in splits
        for char in letters
    ]

    return set(
        deletes +
        transposes +
        replaces +
        inserts
    )


def method_a_candidates(word, vocab):

    # return the edits that are valid vocabulary words.

    candidates = {
        candidate
        for candidate in edits1(word)
        if candidate in vocab
    }

    if word in vocab:
        candidates.add(word)

    return candidates


"""
### 2.2 Method B - Symmetric Delete Spelling Correction (SymSpell)

Instead of generating all edits of the misspelled word at query time, we precompute all single-character deletions of every word, and index them in a dictionary
that maps "deleted_variant" --- [original_words].
At query time we only need to generate the deletions of the misspelled word.

#### Preprocessing step: build the deletion dictionary
"""

def build_deletion_dict(vocab):
    #Map every one-character deletion of every vocab word back to that

    deletion_dict = defaultdict(set)
    for word in vocab:
        deletion_dict[word].add(word)
        for i in range(len(word)):
            deleted = word[:i] + word[i + 1:]
            deletion_dict[deleted].add(word)
    return deletion_dict


"""
#### Candidate generation step

We look up the query word and its own one-character deletions against the deletion dictionary apply the edit-distance-1 verification
"""

def is_edit_distance_le_1(a, b):
    #Check for whether Damerau-Levenshtein distance between a and b is <= 1.
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 0:
            return True  # identical
        if len(diffs) == 1:
            return True  # single substitution
        if (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]):
            return True  # adjacent transposition
        return False

    # when lengths differ by exactly 1: check if the shorter is the longer with one char removed
    short, long_ = (a, b) if la < lb else (b, a)
    i = j = 0

    used_skip = False

    while i < len(short) and j < len(long_):
        if short[i] == long_[j]:
            i += 1
            j += 1
        else:
            if used_skip:
                return False
            used_skip = True
            j += 1

    return True


def method_b_candidates(word, deletion_dict):
    #Look up candidates for `word` using the precomputed deletion dictionary
    superset = set()

    # If the word (or one of its own deletions) is a dictionary key already
    if word in deletion_dict:
        superset.update(deletion_dict[word])

    # generate deletions of the misspelled word itself and look those up too
    for i in range(len(word)):
        deleted = word[:i] + word[i + 1:]
        if deleted in deletion_dict:
            superset.update(deletion_dict[deleted])

    # verify true edit distance to remove any distance-2 false positives
    return {c for c in superset if is_edit_distance_le_1(word, c)}


# test
candidates = method_a_candidates("hte", vocab)
print(sorted(candidates)[:10])

print("Building deletion dictionary...")
t0 = time.time()
deletion_dict = build_deletion_dict(vocab)
build_time = time.time() - t0
print(f"Done in {build_time:.2f}s. Dictionary has {len(deletion_dict):,} keys "
      f"(vs. {len(vocab):,} vocab words).")

# Quick test - should now match Method A's output exactly
print(sorted(method_b_candidates('hte', deletion_dict))[:10])

"""
### 2.3 Check: Do Method A and Method B agree?
They should produce almst the same candidate sets for edit-distance-1 errors, B
is kust  a faster way of finding the same answers.
"""

test_words = ['helo', 'wrold', 'wrok', 'jsut', 'hte', 'aple', 'fone']
for w in test_words:
    a = method_a_candidates(w, vocab)
    b = method_b_candidates(w, deletion_dict)
    print(f"{w!r:10} A={sorted(a)!s:45} B={sorted(b)!s:45} match={a == b}")
