"""
Q1 – Sentiment Analysis using Naïve Bayes
=========================================
Pipeline:
  1. 25 product reviews (positive / negative)
  2. Text preprocessing (lowercase, punctuation removal, stopword removal)
  3. Feature extraction: BoW (CountVectorizer) + TF-IDF
  4. Train MultinomialNB classifier
  5. Accuracy + Confusion matrix
  6. Detailed result explanation
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. DATASET  (25 reviews, balanced 13 pos / 12 neg)
# ─────────────────────────────────────────────
reviews = [
    # Positive (label = 1)
    "This product is absolutely amazing, I love it so much!",
    "Excellent quality and fast shipping. Will definitely buy again.",
    "Great value for money, highly recommend to everyone.",
    "The build quality is outstanding, very durable and well made.",
    "Fantastic product! Exceeded all my expectations completely.",
    "Works perfectly right out of the box. Very satisfied customer.",
    "Superb performance and beautiful design. Five stars easily.",
    "Best purchase I have made this year. Totally worth every penny.",
    "Incredible product with amazing features at a great price.",
    "Very happy with this purchase. Delivery was quick and packaging great.",
    "Exactly as described, premium quality and excellent finish.",
    "Brilliant! My whole family loves this product. Highly recommended.",
    "Outstanding customer service and a wonderful product overall.",
    # Negative (label = 0)
    "Terrible product, broke after just two days of use.",
    "Very disappointed. The quality is awful and not worth the price.",
    "Waste of money. Does not work as advertised at all.",
    "Poor build quality, feels very cheap and flimsy.",
    "Absolutely horrible experience. Would not recommend to anyone.",
    "Stopped working within a week. Totally unreliable product.",
    "The worst purchase I have ever made. Complete garbage.",
    "Defective item received. Customer service was also very unhelpful.",
    "Extremely poor quality. Broke on first use. Very frustrating.",
    "False advertising. Product looks nothing like the pictures shown.",
    "Cheap materials and bad smell. Returning this immediately.",
    "Disappointed with the product. Too many missing features.",
]

labels = [1]*13 + [0]*12   # 1 = Positive, 0 = Negative
label_names = {0: "Negative", 1: "Positive"}


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","we","our","you","your","he","his","she","her","it",
    "its","they","them","their","this","that","these","those","am","is",
    "are","was","were","be","been","being","have","has","had","do","does",
    "did","a","an","the","and","but","or","nor","so","for","yet","both",
    "either","neither","not","no","nor","at","by","in","of","on","to","up",
    "with","about","after","before","than","then","too","very","s","t","just",
    "will","would","also","all","each","more","most","out","as","into","from",
}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

clean_reviews = [preprocess(r) for r in reviews]

# Show before / after for first 3 reviews
print("=" * 65)
print("TEXT PREPROCESSING SAMPLES")
print("=" * 65)
for i in range(3):
    print(f"\nOriginal : {reviews[i]}")
    print(f"Cleaned  : {clean_reviews[i]}")

# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    clean_reviews, labels, test_size=0.3, random_state=42, stratify=labels
)

# ---------- BoW ----------
bow_vec  = CountVectorizer()
X_train_bow = bow_vec.fit_transform(X_train_raw)
X_test_bow  = bow_vec.transform(X_test_raw)

# ---------- TF-IDF ----------
tfidf_vec = TfidfVectorizer()
X_train_tfidf = tfidf_vec.fit_transform(X_train_raw)
X_test_tfidf  = tfidf_vec.transform(X_test_raw)

print(f"\n{'='*65}")
print("FEATURE EXTRACTION SUMMARY")
print(f"{'='*65}")
print(f"Training samples  : {len(X_train_raw)}")
print(f"Test samples      : {len(X_test_raw)}")
print(f"BoW  vocabulary   : {len(bow_vec.vocabulary_)} unique tokens")
print(f"TF-IDF vocabulary : {len(tfidf_vec.vocabulary_)} unique tokens")

# ─────────────────────────────────────────────
# 4. TRAIN NAÏVE BAYES
# ─────────────────────────────────────────────
nb_bow   = MultinomialNB(alpha=1.0)
nb_tfidf = MultinomialNB(alpha=1.0)

nb_bow.fit(X_train_bow,   y_train)
nb_tfidf.fit(X_train_tfidf, y_train)

y_pred_bow   = nb_bow.predict(X_test_bow)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

# ─────────────────────────────────────────────
# 5. RESULTS
# ─────────────────────────────────────────────
acc_bow   = accuracy_score(y_test, y_pred_bow)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

cv_bow   = cross_val_score(nb_bow,   X_train_bow,   y_train, cv=3).mean()
cv_tfidf = cross_val_score(nb_tfidf, X_train_tfidf, y_train, cv=3).mean()

print(f"\n{'='*65}")
print("MODEL PERFORMANCE")
print(f"{'='*65}")
print(f"{'Model':<30} {'Test Accuracy':>15} {'CV Accuracy':>13}")
print("-"*60)
print(f"{'Naïve Bayes (BoW)':<30} {acc_bow:>14.1%}  {cv_bow:>12.1%}")
print(f"{'Naïve Bayes (TF-IDF)':<30} {acc_tfidf:>14.1%}  {cv_tfidf:>12.1%}")

print(f"\n--- Classification Report (TF-IDF) ---")
print(classification_report(y_test, y_pred_tfidf,
                             target_names=["Negative","Positive"]))

# ─────────────────────────────────────────────
# 6. VISUALISATIONS  (saved to PNG)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle("Q1 – Sentiment Analysis: Naïve Bayes Results", fontsize=16, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# --- (A) Label Distribution ---
ax0 = fig.add_subplot(gs[0, 0])
counts = pd.Series(labels).map(label_names).value_counts()
bars = ax0.bar(counts.index, counts.values,
               color=["#e74c3c","#2ecc71"], edgecolor="white", linewidth=1.4)
ax0.set_title("Dataset: Label Distribution", fontweight='bold')
ax0.set_ylabel("Count")
for b in bars:
    ax0.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             str(int(b.get_height())), ha='center', fontweight='bold')
ax0.set_ylim(0, max(counts)+3)

# --- (B) BoW Confusion Matrix ---
ax1 = fig.add_subplot(gs[0, 1])
cm_bow = confusion_matrix(y_test, y_pred_bow)
sns.heatmap(cm_bow, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"],
            linewidths=1, linecolor='white', cbar=False)
ax1.set_title(f"BoW Confusion Matrix\n(Accuracy: {acc_bow:.1%})", fontweight='bold')
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

# --- (C) TF-IDF Confusion Matrix ---
ax2 = fig.add_subplot(gs[0, 2])
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"],
            linewidths=1, linecolor='white', cbar=False)
ax2.set_title(f"TF-IDF Confusion Matrix\n(Accuracy: {acc_tfidf:.1%})", fontweight='bold')
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

# --- (D) Top positive / negative BoW features ---
ax3 = fig.add_subplot(gs[1, :2])
feature_names = np.array(bow_vec.get_feature_names_out())
# log-prob difference: positive class (1) vs negative class (0)
log_diff = nb_bow.feature_log_prob_[1] - nb_bow.feature_log_prob_[0]
top_pos_idx = np.argsort(log_diff)[-10:][::-1]
top_neg_idx = np.argsort(log_diff)[:10]

top_words  = list(feature_names[top_pos_idx]) + list(feature_names[top_neg_idx])
top_scores = list(log_diff[top_pos_idx])       + list(log_diff[top_neg_idx])
colors     = ["#2ecc71"]*10 + ["#e74c3c"]*10

y_pos = range(len(top_words))
ax3.barh(list(y_pos), top_scores, color=colors, edgecolor='white', linewidth=0.8)
ax3.set_yticks(list(y_pos)); ax3.set_yticklabels(top_words, fontsize=9)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_title("Top 10 Positive (green) vs Negative (red) Features (BoW log-prob diff)",
              fontweight='bold')
ax3.set_xlabel("Log-Probability Difference (pos − neg)")

# --- (E) Accuracy Comparison ---
ax4 = fig.add_subplot(gs[1, 2])
models = ['BoW\nTest', 'BoW\nCV', 'TF-IDF\nTest', 'TF-IDF\nCV']
scores = [acc_bow, cv_bow, acc_tfidf, cv_tfidf]
bar_colors = ['#3498db','#85c1e9','#27ae60','#82e0aa']
bars2 = ax4.bar(models, scores, color=bar_colors, edgecolor='white', linewidth=1.3)
ax4.set_ylim(0, 1.15)
ax4.set_title("Accuracy Comparison", fontweight='bold')
ax4.set_ylabel("Accuracy")
ax4.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
for b in bars2:
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
             f"{b.get_height():.1%}", ha='center', fontsize=9, fontweight='bold')

plt.savefig("/home/prayash/Music/nlp_projects/Q1_sentiment_analysis/results.png",
            dpi=140, bbox_inches='tight')
print("\nChart saved → results.png")

# ─────────────────────────────────────────────
# 7. EXPLAIN RESULTS
# ─────────────────────────────────────────────
best_model = "TF-IDF" if acc_tfidf >= acc_bow else "BoW"
print(f"""
{'='*65}
RESULT EXPLANATION
{'='*65}
Dataset  : 25 product reviews  (13 positive, 12 negative)
Split    : 70 % train / 30 % test  →  {len(X_train_raw)} / {len(X_test_raw)} samples

Preprocessing steps performed:
  • Lowercase conversion
  • Punctuation / digit removal
  • Custom stopword removal ({len(STOPWORDS)} stopwords)

Feature Extraction:
  • Bag-of-Words (BoW) — counts raw token frequencies.
  • TF-IDF — weights tokens by term frequency × inverse
    document frequency, down-weighting common words.

BoW  Test Accuracy : {acc_bow:.1%}
BoW  CV  Accuracy  : {cv_bow:.1%}
TF-IDF Test Acc.   : {acc_tfidf:.1%}
TF-IDF CV  Acc.    : {cv_tfidf:.1%}

Winner: {best_model}

Why Naïve Bayes works well here:
  • Assumes feature (word) independence → fast & effective on text.
  • Laplace smoothing (alpha=1) handles unseen tokens gracefully.
  • With a clean, balanced dataset the model distinguishes
    clearly opinionated words (amazing, terrible, horrible,
    fantastic, garbage) → high accuracy despite small dataset.

Confusion Matrix interpretation:
  • True Positives / True Negatives = correctly classified.
  • False Positives = negative reviews predicted as positive.
  • False Negatives = positive reviews predicted as negative.
""")
