import pickle
import re
import time
import math
from collections import Counter
from pathlib import Path


ARTIFACT_FILE = Path(__file__).with_name("spelling_corrector_artifacts.pkl")

with open(ARTIFACT_FILE, "rb") as f:
    artifacts = pickle.load(f)

VOCAB = artifacts["vocab"]
FREQ = artifacts["vocab_freq"]
BIGRAMS = artifacts["bigram_counts"]
TOTAL_WORDS = sum(FREQ.values())

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def edits1(word):

    #Edit distance 1 words

    word = word.lower()
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [
        L + R[1] + R[0] + R[2:]
        for L, R in splits
        if len(R) > 1
    ]
    replaces = [
        L + c + R[1:]
        for L, R in splits
        if R
        for c in LETTERS
        if c != R[0]
    ]
    inserts = [
        L + c + R
        for L, R in splits
        for c in LETTERS
    ]

    return set(deletes + transposes + replaces + inserts)


def method_a(word):
    #Standard ED1 candidate generation
    return edits1(word) & VOCAB


def method_b(word):
    #Sym delete candidate generation
    candidates = set()

    for i in range(len(word)):
        deleted = word[:i] + word[i + 1:]
        candidates.update(artifacts["deletion_dict"].get(deleted, []))

    return candidates


def unigram_score(word):
    return FREQ.get(word, 0)


def best_nonword_candidate(word):
    
    # Using method B first
    word = word.lower()

    candidates = method_b(word)

    # Keep method A for fallback
    if not candidates:
        candidates = method_a(word)

    if not candidates:
        return None

    return max(candidates, key=unigram_score)


def bigram_logprob(left, right, k=0.1):
    
    # Smoothed bigram probability
    left = left.lower()
    right = right.lower()

    row = BIGRAMS.get(left, {})
    count = row.get(right, 0)
    total_after_left = sum(row.values())

    # Add-k smoothing
    return math.log((count + k) / (total_after_left + k * len(VOCAB)))


def context_score(prev_word, word, next_word):
    score = 0.0

    if prev_word:
        score += bigram_logprob(prev_word, word)

    if next_word:
        score += bigram_logprob(word, next_word)

    return score


def best_realword_candidate(prev_word, word, next_word):
    
    #Check ED1 alternatives and replace the word only when better local bigram score.
    
    word = word.lower()
    candidates = method_b(word)

    if not candidates:
        return None

    original_score = context_score(prev_word, word, next_word)

    best = word
    best_score = original_score

    for candidate in candidates:
        score = context_score(prev_word, candidate, next_word)
        if score > best_score:
            best = candidate
            best_score = score

    # threshold 
    if best != word and best_score - original_score >= 1.0:
        return best

    return None


def tokenize(text):
    # Keep original punctuation
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z\s]", text)


def preserve_case(original, replacement):
    # Preserve the case of the original word
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement


def correct_sentence(text):
    # Correct non-word errors and real-word errors
    
    start = time.perf_counter()

    tokens = tokenize(text)
    words = [t for t in tokens if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", t)]

    # Map word positions to token positions.
    word_token_positions = [
        i for i, t in enumerate(tokens)
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", t)
    ]

    changes = []

    # Returns corrected text, changed words, and latency
    for word_index, token_index in enumerate(word_token_positions):
        original = tokens[token_index]
        lower = original.lower()

        prev_word = None
        next_word = None

        if word_index > 0:
            prev_word = tokens[word_token_positions[word_index - 1]].lower()
        if word_index + 1 < len(word_token_positions):
            next_word = tokens[word_token_positions[word_index + 1]].lower()

        replacement = None
        reason = None

        # Non-word error.
        if lower not in VOCAB:
            replacement = best_nonword_candidate(lower)
            reason = "non-word"

        # Real-word error.
        else:
            replacement = best_realword_candidate(prev_word, lower, next_word)
            reason = "real-word"

        if replacement and replacement != lower:
            replacement = preserve_case(original, replacement)
            tokens[token_index] = replacement
            changes.append((original, replacement, reason))

    elapsed_ms = (time.perf_counter() - start) * 1000

    # reconstruction.
    result = ""
    for token in tokens:
        if not result:
            result = token
        elif re.fullmatch(r"[,.!?;:%)\]}]", token):
            result += token
        elif token in ["'", "’"]:
            result += token
        else:
            result += " " + token

    return result, changes, elapsed_ms


def highlight(changes):
    # Highlight changed words with ANSI green + bold
    out = []
    for original, replacement, reason in changes:
        out.append(
            f"\033[1;32m{replacement}\033[0m"
            f" ({original} -> {replacement}, {reason})"
        )
    return out


def main():
    print("CLI Spelling Corrector")
    print("Type a sentence and press Enter.")
    print("Changed words are highlighted. Type 'exit' to quit.")
    print()

    while True:
        try:
            text = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.strip().lower() == "exit":
            break

        if not text.strip():
            continue

        corrected, changes, latency = correct_sentence(text)

        print("Corrected:", corrected)
        if changes:
            print("Changed:", " | ".join(highlight(changes)))
        else:
            print("Changed: none")
        print(f"Latency: {latency:.3f} ms")
        print()


if __name__ == "__main__":
    main()
