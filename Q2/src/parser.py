from typing import Dict, List, Tuple

from conllu_parser import Token, parse_conllu
from model import TransitionClassifier


class DependencyParser:
    """
    Transition-based dependency parser using an
    arc-standard transition system.

    The parser predicts:
        SHIFT
        LEFT-ARC(label)
        RIGHT-ARC(label)
    """

    def __init__(self, classifier: TransitionClassifier):
        self.classifier = classifier

    @staticmethod
    def is_legal_transition(
        transition: str,
        stack: List[int],
        buffer: List[int],
    ) -> bool:

        if transition == "SHIFT":
            return len(buffer) > 0

        if transition.startswith("LEFT-ARC:"):
            return (
                len(stack) >= 2
                and stack[-2] != 0
            )

        if transition.startswith("RIGHT-ARC:"):
            return len(stack) >= 2

        return False

    @staticmethod
    def apply_transition(
        transition: str,
        stack: List[int],
        buffer: List[int],
        arcs: Dict[int, Tuple[int, str]],
    ):

        if transition == "SHIFT":

            if not buffer:
                raise ValueError(
                    "Cannot SHIFT with an empty buffer."
                )

            token_id = buffer.pop(0)
            stack.append(token_id)
            return

        if transition.startswith("LEFT-ARC:"):

            if len(stack) < 2:
                raise ValueError(
                    "LEFT-ARC requires at least two "
                    "items on the stack."
                )

            label = transition.split(":", 1)[1]

            dependent = stack[-2]
            head = stack[-1]

            arcs[dependent] = (
                head,
                label,
            )

            stack.pop(-2)
            return

        if transition.startswith("RIGHT-ARC:"):

            if len(stack) < 2:
                raise ValueError(
                    "RIGHT-ARC requires at least two "
                    "items on the stack."
                )

            label = transition.split(":", 1)[1]

            dependent = stack[-1]
            head = stack[-2]

            arcs[dependent] = (
                head,
                label,
            )

            stack.pop()
            return

        raise ValueError(
            f"Unknown transition: {transition}"
        )

    def parse(
        self,
        sentence: List[Token],
        max_steps: int = 1000,
    ) -> Dict[int, Tuple[int, str]]:

        stack = [0]

        buffer = [
            token.id
            for token in sentence
            if token.id != 0
        ]

        arcs = {}

        steps = 0

        while buffer or len(stack) > 1:

            steps += 1

            if steps > max_steps:
                raise RuntimeError(
                    "Parser exceeded maximum number "
                    "of transitions."
                )

            prediction = self.classifier.predict(
                stack,
                buffer,
                sentence,
            )

            if self.is_legal_transition(
                prediction,
                stack,
                buffer,
            ):

                self.apply_transition(
                    prediction,
                    stack,
                    buffer,
                    arcs,
                )

                continue

            # ------------------------------------------------
            # Fallback for an illegal classifier prediction
            # ------------------------------------------------

            alternatives = []

            for candidate in (
                "RIGHT-ARC:dep",
                "LEFT-ARC:dep",
                "SHIFT",
            ):

                if candidate == prediction:
                    continue

                if self.is_legal_transition(
                    candidate,
                    stack,
                    buffer,
                ):
                    alternatives.append(candidate)

            if not alternatives:
                raise RuntimeError(
                    "No legal transition available. "
                    f"Stack={stack}, "
                    f"Buffer={buffer}, "
                    f"Prediction={prediction}"
                )

            fallback = alternatives[0]

            self.apply_transition(
                fallback,
                stack,
                buffer,
                arcs,
            )

        if stack != [0]:
            raise RuntimeError(
                "Parser did not terminate with ROOT only. "
                f"Final stack={stack}"
            )

        if buffer:
            raise RuntimeError(
                "Parser terminated with a non-empty buffer."
            )

        return arcs


# ============================================================
# TEST SENTENCE CREATION
# ============================================================


def create_test_sentence(
    words: List[str],
) -> List[Token]:
    """
    Convert a raw test sentence into Token objects.

    The assignment provides raw sentences without
    dependency annotations.

    Since no POS tagger is included in this project,
    plausible UPOS tags are assigned manually so that
    the test input is compatible with the POS-based
    features used during training.
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

    # --------------------------------------------------------
    # Simple POS dictionary for the three required sentences
    # --------------------------------------------------------

    pos_map = {
        "The": "DET",
        "the": "DET",
        "a": "DET",

        "cat": "NOUN",
        "mat": "NOUN",
        "salad": "NOUN",
        "man": "NOUN",
        "telescope": "NOUN",

        "sat": "VERB",
        "eats": "VERB",
        "saw": "VERB",

        "on": "ADP",
        "with": "ADP",

        "She": "PRON",
        "I": "PRON",

        "green": "ADJ",

        ".": "PUNCT",
    }

    tokens = [root]

    for index, word in enumerate(
        words,
        start=1,
    ):

        upos = pos_map.get(
            word,
            "X",
        )

        token = Token(
            id=index,
            form=word,
            lemma=word.lower(),
            upos=upos,
            xpos=upos,
            feats="_",
            head=-1,
            deprel="_",
        )

        tokens.append(token)

    return tokens


# ============================================================
# PRINT RESULTS
# ============================================================


def print_test_result(
    sentence: List[Token],
    predicted_arcs: Dict[int, Tuple[int, str]],
):

    print()
    print("Predicted dependency arcs:")
    print()

    token_map = {
        token.id: token
        for token in sentence
    }

    for dependent_id in sorted(
        predicted_arcs
    ):

        head_id, label = (
            predicted_arcs[dependent_id]
        )

        dependent_form = token_map[
            dependent_id
        ].form

        head_form = token_map[
            head_id
        ].form

        print(
            f"{dependent_id:>2} "
            f"{dependent_form:<15} "
            f"<- "
            f"{head_id:>2} "
            f"{head_form:<15} "
            f"({label})"
        )

    print()

    print(
        f"Number of predicted arcs: "
        f"{len(predicted_arcs)}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    classifier_path = (
        "outputs/models/"
        "transition_classifier.joblib"
    )

    vectorizer_path = (
        "outputs/models/"
        "feature_vectorizer.joblib"
    )

    print(
        "Loading saved transition classifier..."
    )

    classifier = TransitionClassifier()

    classifier.load(
        classifier_path=classifier_path,
        vectorizer_path=vectorizer_path,
    )

    parser = DependencyParser(
        classifier
    )

    # ========================================================
    # REQUIRED TEST SENTENCES
    # ========================================================

    test_sentences = [

        [
            "The",
            "cat",
            "sat",
            "on",
            "the",
            "mat",
            ".",
        ],

        [
            "She",
            "eats",
            "a",
            "green",
            "salad",
            ".",
        ],

        [
            "I",
            "saw",
            "the",
            "man",
            "with",
            "a",
            "telescope",
            ".",
        ],
    ]

    print()

    print("=" * 60)
    print(
        "REQUIRED TEST SENTENCES"
    )
    print("=" * 60)

    for test_number, words in enumerate(
        test_sentences,
        start=1,
    ):

        sentence = create_test_sentence(
            words
        )

        print()
        print(
            f"Test Sentence {test_number}:"
        )

        print(
            " ".join(words)
        )

        try:

            predicted_arcs = parser.parse(
                sentence
            )

            print_test_result(
                sentence,
                predicted_arcs,
            )

        except (
            RuntimeError,
            ValueError,
        ) as exc:

            print(
                f"Parser error: {exc}"
            )

    print()

    print("=" * 60)
    print(
        "TESTING COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
