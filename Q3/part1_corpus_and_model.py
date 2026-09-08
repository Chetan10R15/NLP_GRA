"""
Part 1: Corpus and Model Preparation

### 1.1 Vocabulary and Frequencies
Lowercase all alphabetic tokens from the Brown corpus and make unigram model
"""

from collections import Counter, defaultdict

from setup import brown

sentences_raw = brown.sents()

#keep only alphabets, lowercase them
sentences = [[w.lower() for w in sent if w.isalpha()] for sent in sentences_raw]
sentences = [s for s in sentences if len(s) >= 2]  #At least 2 tokens for bigrams

all_words = [w for sent in sentences for w in sent]

vocab_freq = Counter(all_words)
vocab = set(vocab_freq.keys())

"""
### 1.2 Bigram Model
"""

bigram_counts = defaultdict(Counter)

for sent in sentences:
    for w1, w2 in zip(sent, sent[1:]):
        bigram_counts[w1][w2] += 1

VOCAB_SIZE = len(vocab)

def bigram_prob(w1, w2, k=1):
    #Add-k smoothed P(w2 | w1)

    count_w1 = vocab_freq.get(w1, 0)
    if w1 in bigram_counts:
      count_bigram = bigram_counts[w1][w2]
    else:
      count_bigram = 0
    return (count_bigram + k) / (count_w1 + k * VOCAB_SIZE)


print(f"Total sentences: {len(sentences):,}")
print(f"Total tokens: {len(all_words):,}")
print(f"Vocabulary size (unique words): {len(vocab):,}")

# check
print("P(apple | an) ~", bigram_prob('an', 'apple'))
print("P(apply | an) ~", bigram_prob('an', 'apply'))
