from dataclasses import dataclass
from typing import List


@dataclass
class Token:
    """Represents one token in a CoNLL-U sentence."""
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str


def parse_conllu(path: str) -> List[List[Token]]:
    """
    Read a CoNLL-U file and return a list of sentences.

    Each sentence is represented as a list of Token objects.

    Multi-word token lines (e.g. 1-2) and empty-node lines
    (e.g. 3.1) are ignored.
    """
    sentences = []
    current_sentence = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Blank line = end of sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Metadata/comment line
            if line.startswith("#"):
                continue

            columns = line.split("\t")

            # A valid CoNLL-U token line has 10 columns
            if len(columns) != 10:
                continue

            token_id = columns[0]

            # Ignore multi-word tokens such as 1-2
            if "-" in token_id:
                continue

            # Ignore empty nodes such as 3.1
            if "." in token_id:
                continue

            token = Token(
                id=int(token_id),
                form=columns[1],
                lemma=columns[2],
                upos=columns[3],
                xpos=columns[4],
                feats=columns[5],
                head=int(columns[6]),
                deprel=columns[7],
            )

            current_sentence.append(token)

    # Handle a file that does not end with a blank line
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def add_root(sentence: List[Token]) -> List[Token]:
    """
    Add an artificial ROOT token at position 0.

    The original token IDs remain unchanged.
    """
    root = Token(
        id=0,
        form="ROOT",
        lemma="ROOT",
        upos="ROOT",
        xpos="ROOT",
        feats="_",
        head=-1,
        deprel="root",
    )

    return [root] + sentence


if __name__ == "__main__":
    # Small test using the English-EWT training data.
    path = "UD_English-EWT/en_ewt-ud-train.conllu"

    sentences = parse_conllu(path)

    print(f"Number of sentences: {len(sentences)}")

    if sentences:
        sentence = add_root(sentences[0])

        print("\nFirst sentence:")
        for token in sentence:
            print(
                f"{token.id:>2}  "
                f"{token.form:<15} "
                f"HEAD={token.head:<2} "
                f"REL={token.deprel}"
            )
