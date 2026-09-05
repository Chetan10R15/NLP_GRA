from typing import Dict, List

from conllu_parser import Token


def build_token_map(
    sentence: List[Token],
) -> Dict[int, Token]:
    """Map token IDs to tokens."""
    return {
        token.id: token
        for token in sentence
    }


def get_token_features(
    token_id: int,
    token_map: Dict[int, Token],
) -> Dict[str, str]:
    """Get the parser features for one token."""
    if token_id == 0:
        return {
            "form": "ROOT",
            "lemma": "ROOT",
            "upos": "ROOT",
            "xpos": "ROOT",
        }

    if token_id not in token_map:
        return {
            "form": "NULL",
            "lemma": "NULL",
            "upos": "NULL",
            "xpos": "NULL",
        }

    token = token_map[token_id]

    return {
        "form": token.form,
        "lemma": token.lemma,
        "upos": token.upos,
        "xpos": token.xpos,
    }


def extract_features(
    stack: List[int],
    buffer: List[int],
    sentence: List[Token],
) -> Dict[str, str]:
    """Extract features from the top of the stack and buffer."""

    token_map = build_token_map(sentence)
    features = {}

    stack_positions = {
        "S0": stack[-1] if len(stack) >= 1 else None,
        "S1": stack[-2] if len(stack) >= 2 else None,
        "S2": stack[-3] if len(stack) >= 3 else None,
    }

    for position, token_id in stack_positions.items():
        if token_id is None:
            token_features = {
                "form": "NULL",
                "lemma": "NULL",
                "upos": "NULL",
                "xpos": "NULL",
            }
        else:
            token_features = get_token_features(
                token_id,
                token_map,
            )

        for attribute, value in token_features.items():
            features[f"{position}_{attribute}"] = value

    buffer_positions = {
        "B0": buffer[0] if len(buffer) >= 1 else None,
        "B1": buffer[1] if len(buffer) >= 2 else None,
        "B2": buffer[2] if len(buffer) >= 3 else None,
    }

    for position, token_id in buffer_positions.items():
        if token_id is None:
            token_features = {
                "form": "NULL",
                "lemma": "NULL",
                "upos": "NULL",
                "xpos": "NULL",
            }
        else:
            token_features = get_token_features(
                token_id,
                token_map,
            )

        for attribute, value in token_features.items():
            features[f"{position}_{attribute}"] = value

    return features


def print_features(
    stack: List[int],
    buffer: List[int],
    sentence: List[Token],
):
    """Print the extracted features."""

    features = extract_features(
        stack,
        buffer,
        sentence,
    )

    for name, value in features.items():
        print(
            f"{name:<20} = {value}"
        )


if __name__ == "__main__":
    from conllu_parser import (
        add_root,
        parse_conllu,
    )

    path = (
        "UD_English-EWT/"
        "en_ewt-ud-train.conllu"
    )

    sentences = parse_conllu(path)

    if not sentences:
        raise ValueError(
            "No sentences found."
        )

    sentence = add_root(
        sentences[0]
    )

    stack = [
        0,
        1,
        3,
    ]

    buffer = [
        4,
        5,
        6,
        7,
        8,
    ]

    print("Sentence:")
    print(
        " ".join(
            token.form
            for token in sentence[1:]
        )
    )

    print()

    print("Parser configuration:")
    print(
        f"Stack  = {stack}"
    )
    print(
        f"Buffer = {buffer}"
    )

    print()

    print("Extracted features:")
    print_features(
        stack,
        buffer,
        sentence,
    )
