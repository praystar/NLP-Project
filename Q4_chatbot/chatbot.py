"""
Q4 - Chatbot Emotion Response System (Topic 15)
================================================
This chatbot predicts user emotion and responds empathetically.

Features:
1. Emotion classification with TF-IDF + Logistic Regression
2. Emotion-aware responses for joy, sadness, anger, fear, stress, neutral
3. Confidence threshold fallback for uncertain inputs
4. Demo mode and interactive chat mode
"""

import random
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
TRAINING_SAMPLES = [
    ("i am very happy today", "joy"),
    ("this is amazing and exciting", "joy"),
    ("i feel great and thankful", "joy"),
    ("today is wonderful", "joy"),
    ("i am feeling sad and lonely", "sadness"),
    ("i want to cry", "sadness"),
    ("i feel down and empty", "sadness"),
    ("nothing feels good today", "sadness"),
    ("i am angry right now", "anger"),
    ("this made me furious", "anger"),
    ("i hate this situation", "anger"),
    ("i am frustrated and annoyed", "anger"),
    ("i am scared and worried", "fear"),
    ("i feel nervous about tomorrow", "fear"),
    ("this is making me panic", "fear"),
    ("i am afraid of failing", "fear"),
    ("work pressure is too much", "stress"),
    ("i feel overwhelmed and exhausted", "stress"),
    ("i cannot focus because of stress", "stress"),
    ("i am mentally tired", "stress"),
    ("i am okay", "neutral"),
    ("just a normal day", "neutral"),
    ("nothing special to report", "neutral"),
    ("i am fine", "neutral"),
]

EMOTION_RESPONSES = {
    "joy": [
        "That sounds wonderful. I am glad things are going well for you.",
        "I love that positive energy. Keep that momentum going.",
    ],
    "sadness": [
        "I am sorry you are feeling this way. Want to share what is weighing on you?",
        "That sounds heavy. Taking one small step right now can help.",
    ],
    "anger": [
        "I hear your frustration. A short pause and deep breathing can help reset.",
        "That sounds upsetting. Do you want to talk through what triggered it?",
    ],
    "fear": [
        "That sounds scary. Breaking the situation into small actions can make it manageable.",
        "I understand. Would it help if we planned one concrete next step together?",
    ],
    "stress": [
        "You are carrying a lot right now. Try focusing on one priority at a time.",
        "That pressure is real. A short break and a simple checklist can reduce overload.",
    ],
    "neutral": [
        "Thanks for sharing. If you want, we can talk about your day in more detail.",
        "Understood. I am here if you want support or just conversation.",
    ],
}

HELP_TEXT = (
    "You can tell me how you feel, for example: "
    "'I am stressed about exams', 'I feel happy today', or 'I am worried'."
)

STOPWORDS = {
    "i", "am", "is", "are", "the", "a", "an", "to", "of", "and", "for", "it", "this", "that"
}

EMOTION_KEYWORDS = {
    "joy": {"happy", "joy", "excited", "great", "amazing", "wonderful", "glad", "thankful", "love"},
    "sadness": {"sad", "lonely", "down", "cry", "empty", "depressed", "hopeless"},
    "anger": {"angry", "furious", "annoyed", "hate", "frustrated", "mad", "upset"},
    "fear": {"afraid", "scared", "worried", "nervous", "panic", "anxious", "fear"},
    "stress": {"stress", "stressed", "pressure", "overwhelmed", "exhausted", "burnout", "deadline", "deadlines"},
    "neutral": {"okay", "fine", "normal", "alright"},
}


class EmotionResponseBot:
    CONFIDENCE_THRESHOLD = 0.22

    def __init__(self) -> None:
        self._fit_model()

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[^a-z\\s]", " ", text.lower())
        tokens = [tok for tok in text.split() if tok not in STOPWORDS]
        return " ".join(tokens) if tokens else text.lower().strip()

    def _fit_model(self) -> None:
        texts = [self._clean(text) for text, _ in TRAINING_SAMPLES]
        labels = np.array([label for _, label in TRAINING_SAMPLES])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        x = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(max_iter=600)
        self.classifier.fit(x, labels)

    def predict_emotion(self, message: str) -> tuple[str, float]:
        cleaned = self._clean(message)
        x_query = self.vectorizer.transform([cleaned])
        probabilities = self.classifier.predict_proba(x_query)[0]
        best_idx = int(np.argmax(probabilities))
        label = self.classifier.classes_[best_idx]
        confidence = float(probabilities[best_idx])
        return label, confidence

    def _keyword_emotion(self, message: str) -> tuple[str, float] | None:
        tokens = set(self._clean(message).split())
        if not tokens:
            return None

        scores = {
            emotion: len(tokens.intersection(words))
            for emotion, words in EMOTION_KEYWORDS.items()
        }
        best_emotion = max(scores, key=scores.get)
        best_score = scores[best_emotion]

        if best_score == 0:
            return None

        # If multiple emotions tie, let the ML model decide.
        ties = sum(1 for v in scores.values() if v == best_score)
        if ties > 1:
            return None

        confidence = min(0.9, 0.4 + 0.2 * best_score)
        return best_emotion, confidence

    def generate_response(self, message: str) -> dict:
        low = message.lower().strip()
        if low in {"help", "options", "what can you do"}:
            return {
                "emotion": "system",
                "confidence": 1.0,
                "reply": HELP_TEXT,
            }
        if low in {"bye", "goodbye", "exit", "quit"}:
            return {
                "emotion": "farewell",
                "confidence": 1.0,
                "reply": "Take care. I am here whenever you want to talk again.",
            }

        keyword_result = self._keyword_emotion(message)
        if keyword_result is not None:
            emotion, confidence = keyword_result
        else:
            emotion, confidence = self.predict_emotion(message)

        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "emotion": "uncertain",
                "confidence": confidence,
                "reply": (
                    "I may be misreading that. Could you describe your feeling in a few more words? "
                    "You can mention if you feel happy, sad, angry, afraid, or stressed."
                ),
            }

        response = random.choice(EMOTION_RESPONSES[emotion])
        return {
            "emotion": emotion,
            "confidence": confidence,
            "reply": response,
        }


def run_demo() -> None:
    sample_messages = [
        "I feel happy after my exam",
        "I am very stressed with deadlines",
        "This situation makes me angry",
        "I am worried about tomorrow",
        "I feel so lonely today",
        "I am okay",
    ]

    bot = EmotionResponseBot()
    print("=" * 70)
    print("Q4 - CHATBOT EMOTION RESPONSE SYSTEM")
    print("=" * 70)

    for msg in sample_messages:
        result = bot.generate_response(msg)
        print(f"\\nUser : {msg}")
        print(f"Bot  : {result['reply']}")
        print(f"Meta : emotion={result['emotion']}, confidence={result['confidence']:.2f}")


def interactive_mode() -> None:
    bot = EmotionResponseBot()

    print("\\n" + "=" * 70)
    print("Interactive Mode - Emotion Response Chatbot")
    print("Type 'help' for usage, 'exit' to stop.")
    print("=" * 70)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nBot: Session ended. Take care.")
            break

        if not user_input:
            continue

        result = bot.generate_response(user_input)
        print(f"Bot: {result['reply']}")
        print(f"     [emotion={result['emotion']}, confidence={result['confidence']:.2f}]\\n")

        if result["emotion"] == "farewell":
            break


if __name__ == "__main__":
    run_demo()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print("\\nRun interactive mode:")
        print("python chatbot.py --interactive")
