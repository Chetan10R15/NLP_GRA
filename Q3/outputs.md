# Recorded Outputs (from the original notebook run)

These are the outputs that were saved.
Re-running `main.py` reproduce the same outputs, except for
figures that depend on `random`/timing (test-set sizes, accuracies, benchmark timings), which vary slightly from run to run.

## setup.py

```
[nltk_data] Downloading package brown to /root/nltk_data...
[nltk_data]   Unzipping corpora/brown.zip.
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.

Setup complete.
```

## part1_corpus_and_model.py

```
Total sentences: 55,820
Total tokens: 980,770
Vocabulary size (unique words): 40,189

P(apple | an) ~ 4.552801110883471e-05
P(apply | an) ~ 2.2764005554417357e-05
```

## part2_candidate_generation.py

```
['ate', 'hate', 'he', 'hee', 'hue', 'the']

Building deletion dictionary...
Done in 0.88s. Dictionary has 310,276 keys (vs. 40,189 vocab words).

['ate', 'hate', 'he', 'hee', 'hue', 'the']

'helo'     A=['halo', 'hel', 'held', 'hell', 'hello', 'helm', 'help', 'hero', 'hilo'] B=['halo', 'hel', 'held', 'hell', 'hello', 'helm', 'help', 'hero', 'hilo'] match=True
'wrold'    A=['wold', 'world']                             B=['wold', 'world']                             match=True
'wrok'     A=['brok', 'grok', 'rok', 'work']               B=['brok', 'grok', 'rok', 'work']               match=True
'jsut'     A=['just', 'sut']                               B=['just', 'sut']                               match=True
'hte'      A=['ate', 'hate', 'he', 'hee', 'hue', 'the']    B=['ate', 'hate', 'he', 'hee', 'hue', 'the']    match=True
'aple'     A=['able', 'ale', 'ample', 'ape', 'apple', 'axle', 'maple', 'pale'] B=['able', 'ale', 'ample', 'ape', 'apple', 'axle', 'maple', 'pale'] match=True
'fone'     A=['bone', 'cone', 'done', 'fine', 'foe', 'fond', 'fore', 'gone', 'hone', 'ione', 'lone', 'none', 'one', 'tone', 'zone'] B=['bone', 'cone', 'done', 'fine', 'foe', 'fond', 'fore', 'gone', 'hone', 'ione', 'lone', 'none', 'one', 'tone', 'zone'] match=True
```

## part3_correction_logic.py

```
'helo'     -> 'help'
'wrold'    -> 'world'
'jsut'     -> 'just'
'zzzzz'    -> 'zzzzz'

Original: ['i', 'ate', 'an', 'apply']
Correction for position 3: apple
```

## part4_evaluation_benchmark.py

```
Held-out test sentences: 5,582

Non-word test cases generated: 5,582
Real-word test cases generated: 2,843

Non-word correction accuracy — Method A: 84.81% (4734/5582)
Non-word correction accuracy — Method B: 84.81% (4734/5582)

Real-word correction accuracy: 74.18% (2109/2843)

Prepared batch of 1000 misspelled words.

Method A (edit-distance-1 generation): 0.1796s total (0.1796 ms/word)
Method B (SymSpell lookup):            0.0117s total (0.0117 ms/word)
Speedup (A / B): 15.37x
Output agreement between methods: 99.90%
```
