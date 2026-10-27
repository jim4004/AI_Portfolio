# Customer Review Sentiment — DistilBERT Walkthrough

Portfolio project for the current week of the Python AI class (Transformers, pre-trained models, pipelines, tokens).

**Question:** Can a pre-trained DistilBERT model classify customer reviews as Positive or Negative without any training on our data?

**Method:** Clean a small sample of reviews, tokenize them, run the Hugging Face DistilBERT SST-2 pipeline, and inspect every step.

This is the Transformer version of the earlier TF-IDF + Logistic Regression project in the same portfolio.

---

## Dataset

File: `sample_customer_reviews.csv`  
Size: 30 rows (small on purpose so you can see every row)

| Column | Description |
| --- | --- |
| review_id | Unique id |
| review_text | Raw customer review (some have extra spaces or are empty) |
| sentiment | Positive / Negative / Neutral / Mixed |
| category | Product category |

The file contains a few deliberate issues (whitespace, one missing text, Mixed/Neutral labels) so you can practice the repair steps.

---

## How to run (Windows + GitHub Desktop)

1. Pull the latest from `jim4004/AI_Portfolio` in GitHub Desktop.
2. Open the folder `customer-review-sentiment-transformers` in VS Code or a terminal.
3. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

4. Run the walkthrough:

```bash
python walkthrough_distilbert.py
```

The script prints every step so you can follow along. Open the CSV in Excel or VS Code while it runs if you want to compare.

---

## What this project shows from this week’s class

- Loading and repairing a real-looking review file
- Tokenization (what the model actually sees)
- Using a pre-trained DistilBERT pipeline (no training)
- Confidence scores
- Evaluating only on the labels the model understands
- How the pieces fit: tokens → attention → classification head → prediction

---

## Project structure

```
customer-review-sentiment-transformers/
├── README.md
├── requirements.txt
├── sample_customer_reviews.csv
└── walkthrough_distilbert.py
```
