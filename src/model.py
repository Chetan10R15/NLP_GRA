
from typing import Dict, List, Tuple

import joblib
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from conllu_parser import (
    Token,
    add_root,
    parse_conllu,
)

from features import extract_features
from oracle import simulate_oracle


# ============================================================
# TRANSITION CLASSIFIER
# ============================================================


class TransitionClassifier:
    """
    Machine-learning classifier for predicting the next
    transition of an arc-standard dependency parser.

    Possible transitions:
        SHIFT
        LEFT-ARC(label)
        RIGHT-ARC(label)
    """

    def __init__(self):
        # Convert dictionary-based features into a sparse
        # numerical representation.
        self.vectorizer = DictVectorizer()

        # Logistic Regression classifier.
        self.classifier = LogisticRegression(
            max_iter=1000,
            solver="saga",
        )

        self.is_trained = False

    # ========================================================
    # CONVERT SPARSE MATRIX TO 32-BIT INDEX FORMAT
    # ========================================================

    @staticmethod
    def convert_sparse_indices_to_int32(X):
        """
        Convert a SciPy sparse matrix's index arrays from
        int64 to int32.

        scikit-learn 1.9 requires 32-bit sparse indices for
        this type of sparse matrix.
        """
        X = X.tocsr(copy=True)

        X.indices = np.asarray(
            X.indices,
            dtype=np.int32,
        )

        X.indptr = np.asarray(
            X.indptr,
            dtype=np.int32,
        )

        return X

    # ========================================================
    # BUILD TRAINING DATA
    # ========================================================

    def build_training_data(
        self,
        sentences: List[List[Token]],
    ) -> Tuple[
        List[Dict[str, str]],
        List[str],
    ]:
        """
        Generate supervised training examples.

        For each sentence:

            Gold dependency tree
                    ↓
                  Oracle
                    ↓
            Parser configurations
                    ↓
             Feature extraction
                    ↓
             Gold transition label
        """

        X = []
        y = []

        skipped_sentences = 0

        # ----------------------------------------------------
        # Process every sentence.
        # ----------------------------------------------------

        for sentence_number, sentence in enumerate(
            sentences,
            start=1,
        ):
            sentence_with_root = add_root(sentence)

            # ------------------------------------------------
            # Generate oracle transitions.
            #
            # Arc-standard cannot derive non-projective
            # structures, so such sentences are skipped.
            # ------------------------------------------------

            try:
                oracle_steps = simulate_oracle(
                    sentence_with_root
                )

            except ValueError:
                skipped_sentences += 1

                print(
                    f"Skipping sentence "
                    f"{sentence_number}: "
                    f"not derivable by arc-standard oracle."
                )

                continue

            # ------------------------------------------------
            # Convert every oracle configuration into a
            # supervised learning example.
            # ------------------------------------------------

            for (
                stack,
                buffer,
                transition,
            ) in oracle_steps:

                features = extract_features(
                    stack,
                    buffer,
                    sentence_with_root,
                )

                action, label = transition

                if label is not None:
                    transition_label = (
                        f"{action}:{label}"
                    )
                else:
                    transition_label = action

                X.append(features)
                y.append(
                    transition_label
                )

            # ------------------------------------------------
            # Progress information.
            # ------------------------------------------------

            if sentence_number % 1000 == 0:
                print(
                    f"Processed "
                    f"{sentence_number} sentences..."
                )

        print()

        print(
            f"Skipped sentences: "
            f"{skipped_sentences}"
        )

        return X, y

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train(
        self,
        sentences: List[List[Token]],
    ):
        """
        Train the Logistic Regression transition classifier.

        The trained classifier and feature vectorizer are
        saved to disk after successful training.
        """

        print(
            "Generating training examples..."
        )

        X, y = self.build_training_data(
            sentences
        )

        # ----------------------------------------------------
        # Validate training data.
        # ----------------------------------------------------

        if not X:
            raise ValueError(
                "No training examples were generated."
            )

        if len(set(y)) < 2:
            raise ValueError(
                "Training data contains fewer than "
                "two transition classes."
            )

        print()

        print(
            f"Training examples: "
            f"{len(X)}"
        )

        print(
            f"Unique transitions: "
            f"{len(set(y))}"
        )

        # ----------------------------------------------------
        # Display transition classes.
        # ----------------------------------------------------

        print()

        print(
            "Transition classes:"
        )

        for transition in sorted(
            set(y)
        ):
            print(
                f"  {transition}"
            )

        # ----------------------------------------------------
        # Convert feature dictionaries into sparse numerical
        # matrix.
        # ----------------------------------------------------

        print()

        print(
            "Vectorizing features..."
        )

        X_vectorized = (
            self.vectorizer.fit_transform(X)
        )

        print(
            "Original feature matrix shape: "
            f"{X_vectorized.shape}"
        )

        # ----------------------------------------------------
        # Fix sparse matrix index type for scikit-learn 1.9.
        # ----------------------------------------------------

        X_vectorized = (
            self.convert_sparse_indices_to_int32(
                X_vectorized
            )
        )

        print(
            "Converted sparse matrix indices "
            "to int32."
        )

        # ----------------------------------------------------
        # Train classifier.
        # ----------------------------------------------------

        print(
            "Training Logistic Regression classifier..."
        )

        self.classifier.fit(
            X_vectorized,
            y,
        )

        self.is_trained = True

        # ----------------------------------------------------
        # Save trained classifier and feature vectorizer.
        # ----------------------------------------------------

        joblib.dump(
            self.classifier,
            "outputs/models/transition_classifier.joblib",
        )

        joblib.dump(
            self.vectorizer,
            "outputs/models/feature_vectorizer.joblib",
        )

        print()

        print(
            "Training completed."
        )

        print(
            "Saved classifier to "
            "outputs/models/"
            "transition_classifier.joblib"
        )

        print(
            "Saved vectorizer to "
            "outputs/models/"
            "feature_vectorizer.joblib"
        )

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    def load(
        self,
        classifier_path: str = (
            "outputs/models/"
            "transition_classifier.joblib"
        ),
        vectorizer_path: str = (
            "outputs/models/"
            "feature_vectorizer.joblib"
        ),
    ):
        """
        Load a previously trained classifier and vectorizer.

        This avoids retraining the model when evaluating or
        running the parser.
        """

        self.classifier = joblib.load(
            classifier_path
        )

        self.vectorizer = joblib.load(
            vectorizer_path
        )

        self.is_trained = True

        print(
            "Loaded trained classifier from "
            f"{classifier_path}"
        )

        print(
            "Loaded feature vectorizer from "
            f"{vectorizer_path}"
        )

    # ========================================================
    # PREDICT TRANSITION
    # ========================================================

    def predict(
        self,
        stack: List[int],
        buffer: List[int],
        sentence: List[Token],
    ) -> str:
        """
        Predict the next parser transition.

        Returns:

            SHIFT
            LEFT-ARC:label
            RIGHT-ARC:label
        """

        if not self.is_trained:
            raise RuntimeError(
                "Classifier has not been trained."
            )

        # ----------------------------------------------------
        # Extract current parser-state features.
        # ----------------------------------------------------

        features = extract_features(
            stack,
            buffer,
            sentence,
        )

        # ----------------------------------------------------
        # Convert features into sparse vector.
        # ----------------------------------------------------

        X = self.vectorizer.transform(
            [features]
        )

        # ----------------------------------------------------
        # Ensure sparse indices are int32.
        # ----------------------------------------------------

        X = (
            self.convert_sparse_indices_to_int32(
                X
            )
        )

        # ----------------------------------------------------
        # Predict transition.
        # ----------------------------------------------------

        prediction = (
            self.classifier.predict(X)[0]
        )

        return prediction


# ============================================================
# DEVELOPMENT TEST
# ============================================================


def main():

    path = (
        "UD_English-EWT/"
        "en_ewt-ud-train.conllu"
    )

    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    print(
        "Loading training dataset..."
    )

    sentences = parse_conllu(
        path
    )

    print(
        f"Loaded {len(sentences)} sentences."
    )

    # --------------------------------------------------------
    # Use the complete EWT training set.
    # --------------------------------------------------------

    training_subset = sentences

    print()

    print(
        f"Using {len(training_subset)} "
        f"sentences for training."
    )

    print()

    # --------------------------------------------------------
    # Create classifier.
    # --------------------------------------------------------

    model = TransitionClassifier()

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    model.train(
        training_subset
    )

    # --------------------------------------------------------
    # Test prediction on the first sentence.
    # --------------------------------------------------------

    test_sentence = add_root(
        sentences[0]
    )

    # Initial parser configuration.
    stack = [0]

    buffer = [
        token.id
        for token in test_sentence
        if token.id != 0
    ]

    # --------------------------------------------------------
    # Predict first transition.
    # --------------------------------------------------------

    prediction = model.predict(
        stack,
        buffer,
        test_sentence
    )

    print()

    print(
        "Test prediction:"
    )

    print(
        f"Stack  = {stack}"
    )

    print(
        f"Buffer = {buffer[:5]}..."
    )

    print(
        f"Predicted transition = "
        f"{prediction}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()


