from typing import Dict, List, Tuple

from conllu_parser import Token


# ============================================================
# TRANSITION REPRESENTATION
# ============================================================
#
# SHIFT
# LEFT-ARC(label)
# RIGHT-ARC(label)
#
# Internally:
# ("SHIFT", None)
# ("LEFT-ARC", label)
# ("RIGHT-ARC", label)
# ============================================================


def build_gold_dependencies(
    sentence: List[Token],
) -> Dict[int, Tuple[int, str]]:
    """
    Build a gold dependency mapping.

    Returns:
        dependent_id -> (head_id, dependency_label)

    Example:
        {
            1: (0, "root"),
            2: (3, "punct"),
            3: (1, "flat")
        }
    """

    gold = {}

    for token in sentence:
        if token.id == 0:
            continue

        gold[token.id] = (
            token.head,
            token.deprel,
        )

    return gold


def has_unprocessed_dependents(
    token_id: int,
    buffer: List[int],
    gold: Dict[int, Tuple[int, str]],
) -> bool:
    """
    Check whether token_id still has a gold dependent
    remaining in the buffer.

    An arc should not be created until all required
    dependents of the dependent have been processed.
    """

    for dependent_id, (head_id, _) in gold.items():

        if head_id == token_id and dependent_id in buffer:
            return True

    return False


def oracle_transition(
    stack: List[int],
    buffer: List[int],
    gold: Dict[int, Tuple[int, str]],
) -> Tuple[str, str | None]:
    """
    Determine the correct transition for the current
    arc-standard parser configuration.

    Possible transitions:

        SHIFT
        LEFT-ARC(label)
        RIGHT-ARC(label)

    Priority:

        1. LEFT-ARC
        2. RIGHT-ARC
        3. SHIFT
    """

    # --------------------------------------------------------
    # ARC TRANSITIONS REQUIRE AT LEAST TWO STACK ITEMS
    # --------------------------------------------------------

    if len(stack) >= 2:

        second_top = stack[-2]
        top = stack[-1]

        # ====================================================
        # LEFT-ARC
        # ====================================================
        #
        # second_top is dependent of top.
        #
        # Example:
        #
        # Stack = [ROOT, head, dependent]
        #
        # LEFT-ARC means:
        #
        # head -> dependent
        #
        # In arc-standard notation, the second-top item
        # becomes the dependent and is removed.
        # ====================================================

        if second_top != 0 and second_top in gold:

            head, label = gold[second_top]

            if head == top:

                if not has_unprocessed_dependents(
                    second_top,
                    buffer,
                    gold,
                ):
                    return (
                        "LEFT-ARC",
                        label,
                    )

        # ====================================================
        # RIGHT-ARC
        # ====================================================
        #
        # top is dependent of second_top.
        #
        # Example:
        #
        # Stack = [ROOT, head, dependent]
        #
        # RIGHT-ARC means:
        #
        # head -> dependent
        #
        # The top item is removed.
        # ====================================================

        if top in gold:

            head, label = gold[top]

            if head == second_top:

                if not has_unprocessed_dependents(
                    top,
                    buffer,
                    gold,
                ):
                    return (
                        "RIGHT-ARC",
                        label,
                    )

    # ========================================================
    # SHIFT
    # ========================================================

    if buffer:

        return (
            "SHIFT",
            None,
        )

    # ========================================================
    # NO VALID TRANSITION
    # ========================================================

    raise ValueError(
        "No valid oracle transition. "
        f"Stack={stack}, Buffer={buffer}. "
        "This sentence may contain a non-projective "
        "dependency structure that cannot be derived "
        "using the arc-standard transition system."
    )


def simulate_oracle(
    sentence: List[Token],
):
    """
    Simulate the complete gold transition sequence
    for one sentence.

    Initial configuration:

        Stack  = [ROOT]
        Buffer = [all words]

    Returns:

        [
            (
                stack_before_transition,
                buffer_before_transition,
                transition
            ),
            ...
        ]
    """

    gold = build_gold_dependencies(sentence)

    # --------------------------------------------------------
    # INITIAL CONFIGURATION
    # --------------------------------------------------------

    stack = [0]

    buffer = [
        token.id
        for token in sentence
        if token.id != 0
    ]

    transitions = []

    # --------------------------------------------------------
    # SIMULATE UNTIL ALL WORDS ARE ATTACHED TO ROOT
    # --------------------------------------------------------

    while buffer or len(stack) > 1:

        try:

            transition = oracle_transition(
                stack,
                buffer,
                gold,
            )

        except ValueError as exc:

            raise ValueError(
                "Oracle failed for sentence. "
                f"Stack={stack}, Buffer={buffer}. "
                "The sentence is not derivable by the "
                "arc-standard oracle."
            ) from exc

        # ----------------------------------------------------
        # Store the configuration BEFORE applying transition
        # ----------------------------------------------------

        transitions.append(
            (
                stack.copy(),
                buffer.copy(),
                transition,
            )
        )

        action, label = transition

        # ====================================================
        # SHIFT
        # ====================================================

        if action == "SHIFT":

            if not buffer:
                raise ValueError(
                    "SHIFT requested with empty buffer."
                )

            stack.append(
                buffer.pop(0)
            )

        # ====================================================
        # LEFT-ARC
        # ====================================================

        elif action == "LEFT-ARC":

            if len(stack) < 2:
                raise ValueError(
                    "LEFT-ARC requires at least two "
                    "items on the stack."
                )

            # Remove second-top item.
            stack.pop(-2)

        # ====================================================
        # RIGHT-ARC
        # ====================================================

        elif action == "RIGHT-ARC":

            if len(stack) < 2:
                raise ValueError(
                    "RIGHT-ARC requires at least two "
                    "items on the stack."
                )

            # Remove top item.
            stack.pop()

        else:

            raise ValueError(
                f"Unknown transition: {action}"
            )

    # --------------------------------------------------------
    # FINAL SANITY CHECK
    # --------------------------------------------------------

    if stack != [0] or buffer:

        raise ValueError(
            "Oracle simulation did not terminate correctly. "
            f"Final Stack={stack}, Final Buffer={buffer}"
        )

    return transitions


def print_oracle_derivation(
    sentence: List[Token],
):
    """
    Print the complete oracle derivation in a readable format.
    """

    transitions = simulate_oracle(sentence)

    print(
        f"Number of transitions: "
        f"{len(transitions)}"
    )

    print()

    for step, (
        stack,
        buffer,
        transition,
    ) in enumerate(
        transitions,
        start=1,
    ):

        action, label = transition

        if label is not None:

            transition_text = (
                f"{action}({label})"
            )

        else:

            transition_text = action

        print(
            f"{step:>3}: "
            f"Stack={str(stack):<30} "
            f"Buffer={str(buffer):<30} "
            f"-> {transition_text}"
        )


# ============================================================
# TEST THE ORACLE
# ============================================================

if __name__ == "__main__":

    from conllu_parser import (
        add_root,
        parse_conllu,
    )

    path = (
        "UD_English-EWT/"
        "en_ewt-ud-train.conllu"
    )

    print("Loading dataset...")

    sentences = parse_conllu(path)

    print(
        f"Loaded {len(sentences)} sentences."
    )

    if not sentences:

        raise ValueError(
            "No sentences found in dataset."
        )

    # --------------------------------------------------------
    # Test the first sentence
    # --------------------------------------------------------

    sentence = add_root(
        sentences[0]
    )

    print()

    print("Sentence:")

    print(
        " ".join(
            token.form
            for token in sentence[1:]
        )
    )

    print()

    print("Oracle derivation:")

    print_oracle_derivation(
        sentence
    )
