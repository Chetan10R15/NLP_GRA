from typing import Dict, List, Tuple

from conllu_parser import (
    Token,
    add_root,
    parse_conllu,
)

from model import TransitionClassifier
from parser import DependencyParser


# ============================================================
# EVALUATION METRICS
# ============================================================


def calculate_scores(
    gold_arcs: Dict[int, Tuple[int, str]],
    predicted_arcs: Dict[int, Tuple[int, str]],
) -> Tuple[float, float, int, int]:
    """
    Calculate UAS and LAS.

    UAS (Unlabelled Attachment Score):
        Percentage of tokens whose predicted HEAD
        matches the gold HEAD.

    LAS (Labelled Attachment Score):
        Percentage of tokens whose predicted HEAD
        and dependency label both match the gold tree.
    """

    total = len(gold_arcs)

    if total == 0:
        return 0.0, 0.0, 0, 0

    correct_head = 0
    correct_labelled = 0

    for dependent_id, (
        gold_head,
        gold_label,
    ) in gold_arcs.items():

        if dependent_id not in predicted_arcs:
            continue

        predicted_head, predicted_label = (
            predicted_arcs[dependent_id]
        )

        if predicted_head == gold_head:
            correct_head += 1

            if predicted_label == gold_label:
                correct_labelled += 1

    uas = (
        correct_head / total
    ) * 100

    las = (
        correct_labelled / total
    ) * 100

    return (
        uas,
        las,
        correct_head,
        correct_labelled,
    )


# ============================================================
# GOLD DEPENDENCY ARCS
# ============================================================


def get_gold_arcs(
    sentence: List[Token],
) -> Dict[int, Tuple[int, str]]:
    """
    Convert the gold CoNLL-U annotations into:

        dependent_id -> (head_id, label)
    """

    gold_arcs = {}

    for token in sentence:

        if token.id == 0:
            continue

        gold_arcs[token.id] = (
            token.head,
            token.deprel,
        )

    return gold_arcs


# ============================================================
# PARSER EVALUATION
# ============================================================


def evaluate(
    parser: DependencyParser,
    sentences: List[List[Token]],
):
    """
    Evaluate the dependency parser on a collection
    of development sentences.
    """

    total_tokens = 0
    correct_heads = 0
    correct_labels = 0

    evaluated_sentences = 0
    skipped_sentences = 0

    for sentence_number, sentence in enumerate(
        sentences,
        start=1,
    ):

        sentence_with_root = add_root(
            sentence
        )

        gold_arcs = get_gold_arcs(
            sentence_with_root
        )

        try:

            predicted_arcs = parser.parse(
                sentence_with_root
            )

        except (
            RuntimeError,
            ValueError,
        ) as exc:

            skipped_sentences += 1

            print(
                f"Skipping evaluation sentence "
                f"{sentence_number}: {exc}"
            )

            continue

        evaluated_sentences += 1

        for dependent_id, (
            gold_head,
            gold_label,
        ) in gold_arcs.items():

            total_tokens += 1

            if dependent_id not in predicted_arcs:
                continue

            predicted_head, predicted_label = (
                predicted_arcs[dependent_id]
            )

            if predicted_head == gold_head:

                correct_heads += 1

                if predicted_label == gold_label:
                    correct_labels += 1

        if sentence_number % 100 == 0:

            print(
                f"Evaluated "
                f"{sentence_number} sentences..."
            )

    if total_tokens == 0:

        raise ValueError(
            "No sentences were successfully evaluated."
        )

    uas = (
        correct_heads / total_tokens
    ) * 100

    las = (
        correct_labels / total_tokens
    ) * 100

    return {
        "uas": uas,
        "las": las,
        "total_tokens": total_tokens,
        "correct_heads": correct_heads,
        "correct_labels": correct_labels,
        "evaluated_sentences": evaluated_sentences,
        "skipped_sentences": skipped_sentences,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    dev_path = (
        "UD_English-EWT/"
        "en_ewt-ud-dev.conllu"
    )

    classifier_path = (
        "outputs/models/"
        "transition_classifier.joblib"
    )

    vectorizer_path = (
        "outputs/models/"
        "feature_vectorizer.joblib"
    )

    # ========================================================
    # LOAD SAVED MODEL
    # ========================================================

    print(
        "Loading saved transition classifier..."
    )

    classifier = TransitionClassifier()

    classifier.load(
        classifier_path=classifier_path,
        vectorizer_path=vectorizer_path,
    )

    # ========================================================
    # CREATE PARSER
    # ========================================================

    parser = DependencyParser(
        classifier
    )

    # ========================================================
    # LOAD DEVELOPMENT DATA
    # ========================================================

    print()

    print(
        "Loading development dataset..."
    )

    dev_sentences = parse_conllu(
        dev_path
    )

    print(
        f"Loaded {len(dev_sentences)} "
        f"development sentences."
    )

    # ========================================================
    # FULL DEVELOPMENT SET EVALUATION
    # ========================================================

    evaluation_subset = dev_sentences

    print()

    print(
        f"Evaluating on "
        f"{len(evaluation_subset)} "
        f"development sentences..."
    )

    print()

    results = evaluate(
        parser,
        evaluation_subset,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print("=" * 60)

    print(
        "DEPENDENCY PARSER EVALUATION"
    )

    print("=" * 60)

    print(
        "Training model           : "
        "Saved full EWT model"
    )

    print(
        f"Development sentences    : "
        f"{len(evaluation_subset)}"
    )

    print(
        f"Evaluated sentences      : "
        f"{results['evaluated_sentences']}"
    )

    print(
        f"Skipped sentences        : "
        f"{results['skipped_sentences']}"
    )

    print(
        f"Total evaluated tokens   : "
        f"{results['total_tokens']}"
    )

    print(
        f"Correct heads            : "
        f"{results['correct_heads']}"
    )

    print(
        f"Correct labelled arcs    : "
        f"{results['correct_labels']}"
    )

    print()

    print(
        f"UAS: {results['uas']:.2f}%"
    )

    print(
        f"LAS: {results['las']:.2f}%"
    )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()
