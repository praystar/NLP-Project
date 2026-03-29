"""
Q2 - Emotion Detection from Text (Topic 7)
==========================================
This project converts the original word-embedding task into an
emotion-detection mini project.

What it does:
1. Uses a small, bundled embedding dictionary (no internet dependency)
2. Converts each sentence into an averaged embedding vector
3. Trains a Logistic Regression classifier to predict emotion labels
4. Prints evaluation metrics and sample predictions
5. Saves a confusion matrix plot as results.png
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


EMBED_DIM = 6

# Small semantic embedding table designed for this assignment.
# Words related to similar emotions are close in this vector space.
EMBEDDINGS = {
    "happy": np.array([0.95, 0.10, 0.05, 0.08, 0.90, 0.20]),
    "joy": np.array([0.92, 0.10, 0.06, 0.09, 0.88, 0.22]),
    "excited": np.array([0.90, 0.15, 0.10, 0.11, 0.85, 0.25]),
    "great": np.array([0.88, 0.12, 0.08, 0.10, 0.82, 0.18]),
    "love": np.array([0.90, 0.11, 0.06, 0.10, 0.87, 0.19]),
    "calm": np.array([0.78, 0.16, 0.12, 0.12, 0.80, 0.17]),
    "sad": np.array([0.10, 0.92, 0.08, 0.35, 0.10, 0.18]),
    "down": np.array([0.08, 0.90, 0.10, 0.32, 0.10, 0.16]),
    "cry": np.array([0.09, 0.94, 0.12, 0.38, 0.08, 0.20]),
    "lonely": np.array([0.07, 0.88, 0.15, 0.36, 0.10, 0.14]),
    "tired": np.array([0.12, 0.82, 0.18, 0.30, 0.12, 0.12]),
    "anger": np.array([0.06, 0.20, 0.93, 0.46, 0.10, 0.10]),
    "angry": np.array([0.07, 0.18, 0.95, 0.44, 0.08, 0.10]),
    "furious": np.array([0.04, 0.15, 0.96, 0.50, 0.06, 0.08]),
    "hate": np.array([0.08, 0.25, 0.90, 0.40, 0.12, 0.09]),
    "annoyed": np.array([0.09, 0.22, 0.88, 0.42, 0.10, 0.11]),
    "fear": np.array([0.10, 0.28, 0.22, 0.95, 0.08, 0.10]),
    "afraid": np.array([0.09, 0.30, 0.20, 0.96, 0.07, 0.09]),
    "worried": np.array([0.12, 0.35, 0.24, 0.90, 0.10, 0.11]),
    "panic": np.array([0.05, 0.32, 0.28, 0.94, 0.06, 0.09]),
    "nervous": np.array([0.12, 0.34, 0.20, 0.88, 0.11, 0.12]),
    "surprised": np.array([0.70, 0.20, 0.18, 0.35, 0.65, 0.96]),
    "amazed": np.array([0.72, 0.18, 0.16, 0.32, 0.67, 0.95]),
    "shocked": np.array([0.60, 0.25, 0.22, 0.45, 0.58, 0.98]),
    "unexpected": np.array([0.62, 0.22, 0.20, 0.40, 0.60, 0.93]),
    "wow": np.array([0.75, 0.18, 0.14, 0.30, 0.70, 0.97]),
    "okay": np.array([0.50, 0.40, 0.40, 0.42, 0.50, 0.50]),
    "normal": np.array([0.48, 0.42, 0.38, 0.40, 0.49, 0.49]),
    "fine": np.array([0.52, 0.38, 0.36, 0.38, 0.53, 0.52]),
    "today": np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50]),
    "work": np.array([0.50, 0.50, 0.50, 0.50, 0.48, 0.49]),
    "is": np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50]),
    "my": np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50]),
    "the": np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50]),
}


DATASET = [
    ("i feel happy and excited", "joy"),
    ("this day is great and i love it", "joy"),
    ("wow i am amazed and very happy", "joy"),
    ("i feel calm and fine today", "joy"),
    ("this unexpected gift made me joyful", "joy"),
    ("i am sad and lonely", "sadness"),
    ("i want to cry i feel down", "sadness"),
    ("today i am tired and sad", "sadness"),
    ("i feel lonely and down today", "sadness"),
    ("my mood is sad and tired", "sadness"),
    ("i am angry and annoyed", "anger"),
    ("this makes me furious with anger", "anger"),
    ("i hate this situation", "anger"),
    ("i am very angry today", "anger"),
    ("i feel annoyed and upset", "anger"),
    ("i am afraid and nervous", "fear"),
    ("this makes me worried and scared", "fear"),
    ("i feel panic and fear", "fear"),
    ("i am nervous and afraid today", "fear"),
    ("the news made me worried", "fear"),
    ("i am surprised and shocked", "surprise"),
    ("wow this is unexpected", "surprise"),
    ("i am amazed and surprised", "surprise"),
    ("that was a shocking and unexpected result", "surprise"),
    ("wow i did not expect this", "surprise"),
    ("today is normal and okay", "neutral"),
    ("i feel fine and normal", "neutral"),
    ("my work day is okay", "neutral"),
    ("everything is normal today", "neutral"),
    ("i am fine today", "neutral"),
]


def tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z\\s]", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


def sentence_embedding(text: str) -> np.ndarray:
    vectors = [EMBEDDINGS[token] for token in tokenize(text) if token in EMBEDDINGS]
    if not vectors:
        return np.zeros(EMBED_DIM, dtype=float)
    return np.mean(vectors, axis=0)


def build_features(samples: list[tuple[str, str]]):
    texts = [text for text, _ in samples]
    labels = [label for _, label in samples]
    x = np.vstack([sentence_embedding(text) for text in texts])
    return x, np.array(labels), texts


def main() -> None:
    x, y, texts = build_features(DATASET)

    x_train, x_test, y_train, y_test, texts_train, texts_test = train_test_split(
        x,
        y,
        texts,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 65)
    print("Q2 - EMOTION DETECTION FROM TEXT")
    print("=" * 65)
    print(f"Dataset size: {len(DATASET)}")
    print(f"Train/Test split: {len(x_train)} / {len(x_test)}")
    print(f"Detected classes: {sorted(set(y))}")
    print(f"Test accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nSample predictions:")
    for text, truth, pred in zip(texts_test, y_test, y_pred):
        status = "OK" if truth == pred else "MISS"
        print(f"- {status:4} | text='{text}' | true={truth:<8} pred={pred}")

    counts = Counter(y)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(counts.keys(), counts.values(), color="#4c78a8", edgecolor="white")
    axes[0].set_title("Dataset Label Distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=35)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=axes[1],
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "results.png")
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
