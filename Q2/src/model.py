from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from conllu_parser import Token, add_root, parse_conllu
from features import extract_features
from oracle import simulate_oracle


class TransitionClassifier:
    """Classifier used to predict parser transitions."""

    def __init__(self):
        self.vectorizer = DictVectorizer()

        self.classifier = LogisticRegression(
            max_iter=1000,
            solver="saga",
        )

        self.is_trained = False

    @staticmethod
    def convert_sparse_indices_to_int32(X):
        """Convert sparse matrix indices to int32."""
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

    def build_training_data(
        self,
        sentences: List[List[Token]],
    ) -> Tuple[
        List[Dict[str, str]],
        List[str],
    ]:
        """Generate feature vectors and gold transitions."""

        X = []
        y = []
        skipped_sentences = 0

        for sentence_number, sentence in enumerate(
            sentences,
            start=1,
        ):
            sentence_with_root = add_root(sentence)

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
                y.append(transition_label)

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

    def train(
        self,
        sentences: List[List[Token]],
    ):
        """Train and save the transition classifier."""

        print(
            "Generating training examples..."
        )

        X, y = self.build_training_data(
            sentences
        )

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

        print()
        print("Transition classes:")

        for transition in sorted(set(y)):
            print(
                f"  {transition}"
            )

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

        X_vectorized = (
            self.convert_sparse_indices_to_int32(
                X_vectorized
            )
        )

        print(
            "Converted sparse matrix indices "
            "to int32."
        )

        print(
            "Training Logistic Regression classifier..."
        )

        self.classifier.fit(
            X_vectorized,
            y,
        )

        self.is_trained = True

        joblib.dump(
            self.classifier,
            "outputs/models/transition_classifier.joblib",
        )

        joblib.dump(
            self.vectorizer,
            "outputs/models/feature_vectorizer.joblib",
        )

        print()
        print("Training completed.")

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
        """Load a saved classifier and vectorizer."""

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

    def predict(
        self,
        stack: List[int],
        buffer: List[int],
        sentence: List[Token],
    ) -> str:
        """Predict the next parser transition."""

        if not self.is_trained:
            raise RuntimeError(
                "Classifier has not been trained."
            )

        features = extract_features(
            stack,
            buffer,
            sentence,
        )

        X = self.vectorizer.transform(
            [features]
        )

        X = (
            self.convert_sparse_indices_to_int32(
                X
            )
        )

        prediction = (
            self.classifier.predict(X)[0]
        )

        return prediction


def main():
    path = (
        "UD_English-EWT/"
        "en_ewt-ud-train.conllu"
    )

    print(
        "Loading training dataset..."
    )

    sentences = parse_conllu(
        path
    )

    print(
        f"Loaded {len(sentences)} sentences."
    )

    training_subset = sentences

    print()
    print(
        f"Using {len(training_subset)} "
        f"sentences for training."
    )

    print()

    model = TransitionClassifier()

    model.train(
        training_subset
    )

    test_sentence = add_root(
        sentences[0]
    )

    stack = [0]

    buffer = [
        token.id
        for token in test_sentence
        if token.id != 0
    ]

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


if __name__ == "__main__":
    main()
