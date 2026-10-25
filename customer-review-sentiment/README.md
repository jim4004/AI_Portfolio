# Customer Review Sentiment Analyzer

Portfolio project for a Saturday/Sunday Python AI class.

**Question:** Can we automatically classify e-commerce customer reviews into Positive, Negative, Neutral, or Mixed?

**Method:** Simple text cleaning, TF-IDF (unigrams + bigrams), Logistic Regression inside a scikit-learn Pipeline.

This is **supervised**. The target is `sentiment`.

---

## Dataset

File: `customer_reviews_sentiment_practice.csv`  
Size: **300 rows × 4 columns**

| Column | Description |
| --- | --- |
| review_id | Unique review id |
| review_text | Raw customer review |
| sentiment | Target: Positive / Negative / Neutral / Mixed |
| category | Product category (Fashion, App, Food, etc.) |

Sentiment is roughly balanced: Positive 80, Negative 80, Neutral 70, Mixed 70.

---

## Pipeline (assignment rubric)

1. Load the CSV with Pandas
2. Basic EDA and check sentiment distribution
3. Clean the review text (lowercase + normalize whitespace)
4. Convert text to numbers with TF-IDF (`ngram_range=(1, 2)`)
5. Train Logistic Regression inside a `Pipeline`
6. Evaluate with accuracy, classification report (precision / recall / F1), and confusion matrix
7. Test the model on new customer reviews

Fit the vectorizer only on the training data (the Pipeline does this automatically).

---

## How to run

```bash
cd customer-review-sentiment
python -m pip install -r requirements.txt
python customer_review_sentiment.py
```

The CSV is already in the folder.

---

## What this project shows from class

- Traditional NLP pipeline: text → numbers → model
- Why TF-IDF + n-grams help with short reviews
- `make_pipeline` so the vectorizer is fit only on train data
- Multi-class evaluation (accuracy is not enough — look at per-class F1)
- How a small 300-row dataset still produces a usable baseline

---

## Project structure

```
customer-review-sentiment/
├── README.md
├── requirements.txt
├── customer_review_sentiment.py
└── customer_reviews_sentiment_practice.csv
```
