"""
Q4 – Intent-based NLP Chatbot
===============================
Features:
  • Python + sklearn TF-IDF + cosine similarity for intent matching
  • Preprocessing: lowercase, punctuation removal, stopword stripping
  • Intents: NLP concepts + AI applications + greetings + farewell
  • Confidence threshold: fallback for unknown queries
  • Interactive terminal loop + sample conversation demo
"""

import re
import random
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────
# 1. INTENT DEFINITIONS
#    Each intent has: tag, patterns (training phrases), responses
# ─────────────────────────────────────────────────────────────────
INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "what's up", "greetings",
        ],
        "responses": [
            "Hello! I'm an NLP Chatbot. Ask me about NLP or AI!",
            "Hi there! Ready to talk about NLP and Artificial Intelligence?",
            "Hey! What NLP or AI topic can I help you with today?",
        ],
    },
    {
        "tag": "farewell",
        "patterns": [
            "bye", "goodbye", "see you", "take care", "quit", "exit",
            "see you later", "farewell", "that's all for now",
        ],
        "responses": [
            "Goodbye! Come back anytime to explore NLP and AI.",
            "Bye! Happy learning about Artificial Intelligence!",
            "See you later! Keep exploring the world of NLP.",
        ],
    },
    {
        "tag": "what_is_nlp",
        "patterns": [
            "what is nlp", "what is natural language processing",
            "define natural language processing", "explain nlp",
            "what does nlp mean", "nlp definition",
            "tell me about natural language processing",
        ],
        "responses": [
            ("Natural Language Processing (NLP) is a branch of Artificial "
             "Intelligence that enables computers to understand, interpret, "
             "and generate human language. It bridges the gap between human "
             "communication and machine understanding."),
        ],
    },
    {
        "tag": "tokenization",
        "patterns": [
            "what is tokenization", "explain tokenization",
            "what are tokens", "how does tokenization work",
            "word tokenization", "sentence tokenization",
        ],
        "responses": [
            ("Tokenization is the process of splitting text into smaller units "
             "called tokens. Word tokenization splits text into individual words, "
             "while sentence tokenization splits a paragraph into sentences. "
             "Example: 'NLP is great' → ['NLP', 'is', 'great']."),
        ],
    },
    {
        "tag": "stemming_lemmatization",
        "patterns": [
            "what is stemming", "what is lemmatization",
            "difference between stemming and lemmatization",
            "explain stemming", "explain lemmatization",
            "stemming vs lemmatization",
        ],
        "responses": [
            ("Stemming reduces a word to its root form by removing suffixes "
             "(e.g., 'running' → 'run', 'better' → 'bett'). "
             "Lemmatization uses vocabulary analysis to return the dictionary "
             "base form (e.g., 'better' → 'good'). Lemmatization is more "
             "accurate but slower than stemming."),
        ],
    },
    {
        "tag": "stopwords",
        "patterns": [
            "what are stopwords", "explain stop words",
            "what is a stop word", "why remove stopwords",
            "common stopwords examples",
        ],
        "responses": [
            ("Stop words are common words (e.g., 'the', 'is', 'in', 'at') that "
             "carry little semantic meaning and are typically removed during text "
             "preprocessing. Removing them reduces noise and improves the "
             "efficiency and accuracy of NLP models."),
        ],
    },
    {
        "tag": "tfidf",
        "patterns": [
            "what is tfidf", "explain tfidf", "what is tf-idf",
            "term frequency inverse document frequency",
            "how does tfidf work", "tfidf formula",
        ],
        "responses": [
            ("TF-IDF stands for Term Frequency–Inverse Document Frequency. "
             "TF measures how often a word appears in a document; "
             "IDF penalises words that appear in many documents. "
             "TF-IDF = TF × IDF. High TF-IDF score means the word is "
             "important to that specific document but rare overall."),
        ],
    },
    {
        "tag": "bag_of_words",
        "patterns": [
            "what is bag of words", "explain bow", "bag of words model",
            "what is bow in nlp", "how does bag of words work",
        ],
        "responses": [
            ("Bag of Words (BoW) is a text representation technique that "
             "converts text into a vector of word frequency counts, "
             "ignoring grammar and word order. "
             "Example: 'I love NLP' → {'I':1, 'love':1, 'NLP':1}. "
             "It's simple but loses context and sequence information."),
        ],
    },
    {
        "tag": "word_embeddings",
        "patterns": [
            "what are word embeddings", "explain word2vec", "what is glove",
            "word vectors", "explain embeddings", "dense word representations",
            "semantic word vectors", "what is fasttext",
        ],
        "responses": [
            ("Word embeddings are dense vector representations of words where "
             "semantically similar words have similar vectors. "
             "Word2Vec (Google, 2013) uses a shallow neural network. "
             "GloVe (Stanford) uses global co-occurrence statistics. "
             "These capture semantic relationships: king − man + woman ≈ queen."),
        ],
    },
    {
        "tag": "naive_bayes",
        "patterns": [
            "what is naive bayes", "explain naive bayes", "naive bayes classifier",
            "how does naive bayes work", "bayes theorem nlp",
            "naive bayes text classification",
        ],
        "responses": [
            ("Naïve Bayes is a probabilistic classifier based on Bayes' theorem. "
             "It assumes all features (words) are independent — the 'naïve' "
             "assumption. Despite this simplification, it works remarkably well "
             "for text classification like spam detection and sentiment analysis."),
        ],
    },
    {
        "tag": "sentiment_analysis",
        "patterns": [
            "what is sentiment analysis", "explain sentiment analysis",
            "opinion mining", "how to detect sentiment",
            "positive negative classification text",
        ],
        "responses": [
            ("Sentiment Analysis (Opinion Mining) classifies text into sentiment "
             "categories such as Positive, Negative, or Neutral. "
             "Applications include product review analysis, social media "
             "monitoring, and customer feedback systems. "
             "Common approaches: Naïve Bayes, LSTM, BERT fine-tuning."),
        ],
    },
    {
        "tag": "ner",
        "patterns": [
            "what is named entity recognition", "explain ner",
            "what is ner", "entity extraction", "how to find entities in text",
            "person organization location recognition",
        ],
        "responses": [
            ("Named Entity Recognition (NER) identifies and classifies named "
             "entities in text into categories such as Person, Organization, "
             "Location, Date, and Money. "
             "Example: 'Elon Musk founded SpaceX in California' → "
             "Person: Elon Musk | Org: SpaceX | Loc: California."),
        ],
    },
    {
        "tag": "ai_applications",
        "patterns": [
            "applications of ai", "uses of artificial intelligence",
            "what can ai do", "real world ai applications",
            "examples of ai in daily life", "where is ai used",
        ],
        "responses": [
            ("AI is used across countless domains: "
             "Healthcare (diagnosis, drug discovery), Finance (fraud detection, "
             "algorithmic trading), Transportation (self-driving cars, route "
             "optimization), Entertainment (recommendations, game AI), "
             "Education (personalised learning), and more."),
        ],
    },
    {
        "tag": "machine_learning",
        "patterns": [
            "what is machine learning", "explain ml", "supervised learning",
            "unsupervised learning", "reinforcement learning",
            "machine learning types", "how does machine learning work",
        ],
        "responses": [
            ("Machine Learning is a subset of AI where algorithms learn patterns "
             "from data without being explicitly programmed. "
             "Supervised learning uses labelled data (classification, regression). "
             "Unsupervised learning finds hidden patterns (clustering). "
             "Reinforcement learning learns through rewards and penalties."),
        ],
    },
    {
        "tag": "deep_learning",
        "patterns": [
            "what is deep learning", "explain neural networks",
            "what are transformers", "what is bert", "what is gpt",
            "large language models", "llm",
        ],
        "responses": [
            ("Deep Learning uses multi-layered neural networks to learn "
             "complex representations from data. Transformers (2017) "
             "revolutionised NLP with self-attention mechanisms. "
             "BERT (Google) excels at understanding context. "
             "GPT models (OpenAI) are powerful text generators. "
             "These form the basis of modern Large Language Models (LLMs)."),
        ],
    },
    {
        "tag": "chatbot",
        "patterns": [
            "how do chatbots work", "what is a chatbot",
            "types of chatbots", "rule based chatbot",
            "ai chatbot", "how are you built",
        ],
        "responses": [
            ("Chatbots are AI systems that simulate human conversation. "
             "Rule-based chatbots match patterns to predefined responses. "
             "Retrieval-based chatbots use TF-IDF/embedding similarity to "
             "find the best answer. Generative chatbots (like GPT) produce "
             "novel responses. I am a retrieval-based intent chatbot using TF-IDF!"),
        ],
    },
    {
        "tag": "thanks",
        "patterns": [
            "thank you", "thanks", "that was helpful", "great answer",
            "awesome", "perfect", "good job",
        ],
        "responses": [
            "You're welcome! Feel free to ask more NLP questions.",
            "Glad I could help! Any other AI topics you'd like to explore?",
            "Happy to help! Keep learning about NLP and AI. 🚀",
        ],
    },
]

# ─────────────────────────────────────────────
# 2. CHATBOT ENGINE
# ─────────────────────────────────────────────
STOPWORDS_CHAT = {
    "i","me","my","we","our","you","your","he","his","she","her","it",
    "is","are","was","be","been","have","has","do","does","a","an","the",
    "and","but","or","for","to","in","of","on","at","as",
}

def preprocess_chat(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS_CHAT and len(w) > 1]
    return " ".join(tokens) if tokens else text.lower()

class NLPChatbot:
    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self):
        self.intents = INTENTS
        self._build_index()

    def _build_index(self):
        """Fit TF-IDF vectoriser on all training patterns."""
        self.all_patterns   = []
        self.pattern_labels = []        # intent tag for each pattern
        for intent in self.intents:
            for p in intent["patterns"]:
                self.all_patterns.append(preprocess_chat(p))
                self.pattern_labels.append(intent["tag"])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_patterns)

    def predict_intent(self, user_input: str):
        """Return (tag, confidence) for the best-matching intent."""
        processed = preprocess_chat(user_input)
        query_vec = self.vectorizer.transform([processed])
        sims      = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx  = int(np.argmax(sims))
        confidence = float(sims[best_idx])
        tag = self.pattern_labels[best_idx]
        return tag, confidence

    def get_response(self, tag: str) -> str:
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm not sure how to answer that."

    def chat(self, user_input: str) -> dict:
        tag, confidence = self.predict_intent(user_input)
        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "tag":        "unknown",
                "confidence": confidence,
                "response":   ("I'm not sure I understand that. Try asking about "
                               "NLP (tokenization, TF-IDF, embeddings, NER, sentiment) "
                               "or AI applications."),
            }
        return {
            "tag":        tag,
            "confidence": confidence,
            "response":   self.get_response(tag),
        }

# ─────────────────────────────────────────────
# 3. SAMPLE CONVERSATION DEMO
# ─────────────────────────────────────────────
SAMPLE_CONVERSATIONS = [
    "Hello!",
    "What is NLP?",
    "Can you explain tokenization?",
    "What is the difference between stemming and lemmatization?",
    "How does TF-IDF work?",
    "What are word embeddings?",
    "Tell me about sentiment analysis.",
    "What are some applications of AI?",
    "What is deep learning?",
    "How do chatbots work?",
    "Thank you!",
    "Goodbye.",
]

def run_demo():
    bot = NLPChatbot()
    print("=" * 65)
    print("NLP CHATBOT  –  SAMPLE CONVERSATION DEMO")
    print("=" * 65)
    for user_msg in SAMPLE_CONVERSATIONS:
        result = bot.chat(user_msg)
        print(f"\n🧑  User    : {user_msg}")
        print(f"🤖  Bot     : {result['response']}")
        print(f"   [Intent: {result['tag']:30s}  Confidence: {result['confidence']:.2f}]")
    print("\n" + "=" * 65)

# ─────────────────────────────────────────────
# 4. INTERACTIVE MODE
# ─────────────────────────────────────────────
def interactive_mode():
    bot = NLPChatbot()
    print("\n" + "=" * 65)
    print("NLP CHATBOT  –  INTERACTIVE MODE")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 65 + "\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋")
            break
        if not user_input:
            continue
        result = bot.chat(user_input)
        print(f"Bot: {result['response']}")
        print(f"     [Intent: {result['tag']}, Confidence: {result['confidence']:.2f}]\n")
        if result["tag"] == "farewell":
            print("Session ended.")
            break

# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_demo()

    # Run interactive mode if '--interactive' flag passed
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print("\nTo start interactive chat, run:")
        print("  python chatbot.py --interactive\n")
