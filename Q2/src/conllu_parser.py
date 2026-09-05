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
    """Read a CoNLL-U file and return its sentences."""
    sentences = []
    current_sentence = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            columns = line.split("\t")

            if len(columns) != 10:
                continue

            token_id = columns[0]

            if "-" in token_id or "." in token_id:
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

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def add_root(sentence: List[Token]) -> List[Token]:
    """Add an artificial ROOT token at position 0."""
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
