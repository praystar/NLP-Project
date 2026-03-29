# NLP Assignment Projects

This repository is now aligned with your assignment topics.

## Topic Mapping

1. Q1 Sentiment Analysis -> Topic 1 / Topic 3
   - Movie/Product Review Sentiment Analysis
2. Q2 Word Embeddings (converted) -> Topic 7
   - Emotion Detection from Text
3. Q3 NER (converted) -> Topic 13
   - Customer Feedback Analysis
4. Q4 Chatbot (updated) -> Topic 15
   - Chatbot Emotion Response System

## Project Structure

- Q1_sentiment_analysis/sentiment_analysis.py
- Q2_word_embeddings/word_similarity.py
- Q3_named_entity_recognition/ner.py
- Q3_named_entity_recognition/annotated_article.html
- Q4_chatbot/chatbot.py

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run Each Project

### Q1 - Sentiment Analysis (Topic 1 / 3)

```bash
cd Q1_sentiment_analysis
python sentiment_analysis.py
```

Output:
- Console metrics (accuracy, classification report)
- Plot file: Q1_sentiment_analysis/results.png

### Q2 - Emotion Detection from Text (Topic 7)

```bash
cd Q2_word_embeddings
python word_similarity.py
```

Output:
- Console metrics and sample predictions
- Plot file: Q2_word_embeddings/results.png

### Q3 - Customer Feedback Analysis (Topic 13)

```bash
cd Q3_named_entity_recognition
python ner.py
```

Output:
- Console extraction summary (products, issues, sentiment)
- Plot file: Q3_named_entity_recognition/results.png
- HTML report: Q3_named_entity_recognition/annotated_article.html

### Q4 - Chatbot Emotion Response System (Topic 15)

Demo mode:

```bash
cd Q4_chatbot
python chatbot.py
```

Interactive mode:

```bash
cd Q4_chatbot
python chatbot.py --interactive
```

Output:
- Emotion-aware chatbot responses
- Predicted emotion + confidence per user message

## Quick Validation Run

From repository root:

```bash
python Q1_sentiment_analysis/sentiment_analysis.py
python Q2_word_embeddings/word_similarity.py
python Q3_named_entity_recognition/ner.py
python Q4_chatbot/chatbot.py
```

## Notes

- All scripts are self-contained and do not require internet access.
- Output files are saved in their respective project folders.
