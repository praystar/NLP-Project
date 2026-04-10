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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def _generate_large_training_set():
    """Generate 500+ training samples with variations and negations."""
    samples = []
    
    # JOY samples (125+)
    joy_base = [
        "happy", "joyful", "excited", "thrilled", "delighted", "wonderful",
        "amazing", "fantastic", "great", "excellent", "perfect", "blessed",
        "cheerful", "elated", "over the moon", "on cloud nine", "feeling good",
        "having fun", "laughing", "smiling", "love it", "awesome", "brilliant"
    ]
    joy_contexts = [
        "i am", "i feel", "i am feeling", "im", "i have been", "today i am",
        "right now i am", "just now i felt", "i am so", "im very", "im really"
    ]
    for base in joy_base:
        for context in joy_contexts[:3]:
            samples.append((f"{context} {base}", "joy"))
            samples.append((f"{context} {base} today", "joy"))
            samples.append((f"{context} {base} about everything", "joy"))
    for base in joy_base[:10]:
        samples.append((f"things are {base}", "joy"))
        samples.append((f"this is {base}", "joy"))
        samples.append((f"that was {base}", "joy"))
    
    # SADNESS samples (125+)
    sadness_base = [
        "sad", "unhappy", "miserable", "lonely", "depressed", "down",
        "upset", "heartbroken", "devastated", "hopeless", "empty", "lost",
        "crying", "weeping", "hurting", "suffering", "grieving", "blue",
        "melancholy", "gloomy", "heavy hearted", "sorrowful", "dejected"
    ]
    for base in sadness_base:
        for context in joy_contexts[:3]:
            samples.append((f"{context} {base}", "sadness"))
            samples.append((f"{context} {base} today", "sadness"))
            samples.append((f"{context} {base} lately", "sadness"))
    for base in sadness_base[:10]:
        samples.append((f"everything feels {base}", "sadness"))
        samples.append((f"i want to {base}", "sadness"))
        samples.append((f"life is so {base}", "sadness"))
    
    # ANGER samples (125+)
    anger_base = [
        "angry", "furious", "mad", "enraged", "livid", "irritated",
        "annoyed", "frustrated", "upset", "agitated", "hostile", "ticked off",
        "hate", "despise", "detest", "fed up", "sick of this", "over it",
        "outraged", "insulted", "offended", "infuriated", "seething"
    ]
    for base in anger_base:
        for context in joy_contexts[:3]:
            samples.append((f"{context} {base}", "anger"))
            samples.append((f"{context} {base} right now", "anger"))
            samples.append((f"{context} {base} with this", "anger"))
    for base in anger_base[:10]:
        samples.append((f"i {base} this", "anger"))
        samples.append((f"this makes me {base}", "anger"))
        samples.append((f"im so {base} about it", "anger"))
    
    # FEAR samples (125+)
    fear_base = [
        "afraid", "scared", "terrified", "frightened", "nervous", "anxious",
        "worried", "panic", "panicking", "petrified", "horrified", "dreadful",
        "uneasy", "apprehensive", "concerned", "stressed out", "intimidated",
        "threatened", "alarmed", "anxious", "insecure", "vulnerable"
    ]
    for base in fear_base:
        for context in joy_contexts[:3]:
            samples.append((f"{context} {base}", "fear"))
            samples.append((f"{context} {base} about this", "fear"))
            samples.append((f"{context} {base} right now", "fear"))
    for base in fear_base[:10]:
        samples.append((f"i am {base} of this", "fear"))
        samples.append((f"this makes me {base}", "fear"))
        samples.append((f"im {base} about tomorrow", "fear"))
    
    # STRESS samples (125+)
    stress_base = [
        "stressed", "overwhelmed", "exhausted", "burned out", "pressure",
        "tension", "anxiety", "overloaded", "swamped", "drowning", "rushed",
        "frazzled", "tense", "worn out", "fatigued", "drained", "tired",
        "overwhelm", "deadline", "deadlines", "workload", "chaos"
    ]
    for base in stress_base:
        for context in joy_contexts[:3]:
            samples.append((f"{context} {base}", "stress"))
            samples.append((f"{context} {base} today", "stress"))
            samples.append((f"{context} {base} with work", "stress"))
    for base in stress_base[:10]:
        samples.append((f"i feel {base}", "stress"))
        samples.append((f"everything is {base}", "stress"))
        samples.append((f"work is {base}", "stress"))
    
    # NEUTRAL samples (75+)
    neutral_base = [
        "okay", "fine", "alright", "decent", "normal", "regular",
        "average", "middle", "fair", "pleasant", "nice", "good",
        "manageable", "sustainable", "moderate", "reasonable"
    ]
    neutral_contexts = [
        "i am", "i feel", "things are", "today is", "life is", "im",
        "everything is", "it is", "im doing", "i am doing", "being"
    ]
    for base in neutral_base:
        for context in neutral_contexts[:4]:
            samples.append((f"{context} {base}", "neutral"))
            samples.append((f"{context} {base} today", "neutral"))
    
    # Negation samples - flip opposites
    negations = ["not", "no", "dont", "isnt", "arent", "didnt", "wont", "cant"]
    
    # Not happy = sadness instead
    for neg in negations:
        samples.append((f"i am {neg} happy", "sadness"))
        samples.append((f"im {neg} happy", "sadness"))
        samples.append((f"i {neg} feel happy", "sadness"))
        samples.append((f"i am {neg} excited", "sadness"))
        samples.append((f"im {neg} thrilled", "sadness"))
    
    # Not sad = joy instead
    for neg in negations:
        samples.append((f"i am {neg} sad", "joy"))
        samples.append((f"im {neg} sad", "joy"))
        samples.append((f"i {neg} feel sad", "joy"))
        samples.append((f"i am {neg} depressed", "joy"))
        samples.append((f"im {neg} miserable", "joy"))
    
    # Not angry = calm/neutral
    for neg in negations:
        samples.append((f"i am {neg} angry", "neutral"))
        samples.append((f"im {neg} angry", "neutral"))
        samples.append((f"i {neg} feel angry", "neutral"))
    
    # Not afraid = joy instead
    for neg in negations:
        samples.append((f"i am {neg} afraid", "joy"))
        samples.append((f"im {neg} afraid", "joy"))
        samples.append((f"im {neg} scared", "joy"))
    
    # Not stressed = joy
    for neg in negations:
        samples.append((f"i am {neg} stressed", "joy"))
        samples.append((f"im {neg} stressed", "joy"))
        samples.append((f"im {neg} overwhelmed", "joy"))
    
    return samples


TRAINING_SAMPLES = _generate_large_training_set()

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

    @staticmethod
    def _detect_negation(message: str) -> bool:
        """Detect if message contains negation words."""
        negation_pattern = r"\b(not|no|dont|don't|isnt|isn't|arent|aren't|didnt|didn't|wont|won't|cant|can't|never|neither)\b"
        low = message.lower()
        return bool(re.search(negation_pattern, low))

    @staticmethod
    def _flip_emotion(emotion: str) -> str:
        """Flip emotion based on negation context."""
        flip_map = {
            "joy": "sadness",
            "sadness": "joy",
            "anger": "neutral",
            "fear": "joy",
            "stress": "joy",
            "neutral": "neutral",
        }
        return flip_map.get(emotion, emotion)

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

        has_negation = self._detect_negation(message)
        
        keyword_result = self._keyword_emotion(message)
        if keyword_result is not None:
            emotion, confidence = keyword_result
            # Apply negation flipping for keyword-based detection
            if has_negation:
                emotion = self._flip_emotion(emotion)
        else:
            emotion, confidence = self.predict_emotion(message)
            # ML model already trained on negations, but apply additional confidence adjustment
            if has_negation:
                confidence *= 0.9  # Slightly reduce confidence for negations

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
