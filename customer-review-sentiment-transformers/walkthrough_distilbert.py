# Customer Review Sentiment Walkthrough with DistilBERT
# This week's focus: Pre-trained Transformers, pipelines, tokens, and how the pieces fit.
# Run this step-by-step in VS Code, Jupyter, or the terminal on Windows.
# Every major step prints what is happening so you can see the data and the model at work.

import pandas as pd
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 1: LOAD THE SAMPLE DATA")
print("=" * 70)
# The sample CSV matches the structure of the class practice file.
# It has a few real-world problems on purpose so we can repair them.
df = pd.read_csv("sample_customer_reviews.csv")
print(f"Shape: {df.shape}")
print("\nFirst 8 rows:")
print(df.head(8))
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

print("\n" + "=" * 70)
print("STEP 2: INSPECT AND REPAIR THE DATA")
print("=" * 70)
# Common issues you will see in real review data:
# - Extra whitespace
# - Empty / null review text
# - Inconsistent capitalization in labels
# - Mixed / Neutral labels when the model is binary

print("\n--- Before cleaning ---")
print("Unique sentiments:", df["sentiment"].unique())
print("Null review_text count:", df["review_text"].isnull().sum())

# Repair 1: Drop rows with missing review text
df = df.dropna(subset=["review_text"]).copy()

# Repair 2: Strip extra whitespace and normalize
df["review_text"] = df["review_text"].astype(str).str.strip()

# Repair 3: Standardize labels (title case)
df["sentiment"] = df["sentiment"].str.strip().str.title()

# Repair 4: For this walkthrough we focus on the binary DistilBERT model
# (the one commonly shown in class). We map Mixed/Neutral so we can still
# evaluate, but we will also show how the model sees them.
print("\n--- After basic cleaning ---")
print(df[["review_id", "review_text", "sentiment"]].head(10))
print("\nSentiment counts after cleaning:")
print(df["sentiment"].value_counts())

print("\n" + "=" * 70)
print("STEP 3: TOKENIZATION — WHAT THE MODEL ACTUALLY SEES")
print("=" * 70)
# DistilBERT uses the same WordPiece tokenizer as BERT.
# Tokens are not always whole words. This is why API costs and context
# limits are measured in tokens, not words.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sample_text = df.iloc[0]["review_text"]
print(f"\nOriginal text:\n{sample_text}")

tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)
print(f"\nTokens ({len(tokens)}):")
print(tokens)
print(f"\nToken IDs ({len(token_ids)} including special tokens):")
print(token_ids)
print(f"Decoded back: {tokenizer.decode(token_ids)}")

print("\n" + "=" * 70)
print("STEP 4: LOAD THE PRE-TRAINED DISTILBERT PIPELINE")
print("=" * 70)
# This is the model the instructor demonstrated.
# It was already fine-tuned on millions of examples (SST-2).
# We do not train anything. We just use it.
# Labels: POSITIVE / NEGATIVE  (binary)
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=128
)
print("Pipeline loaded. Model is ready.")

print("\n" + "=" * 70)
print("STEP 5: RUN PREDICTIONS ON THE WHOLE DATASET")
print("=" * 70)
# The pipeline handles tokenization, attention, and the classification head
# for us. We get back a label and a confidence score for every review.
results = classifier(df["review_text"].tolist())

df["predicted_label"] = [r["label"] for r in results]
df["confidence"] = [round(r["score"], 4) for r in results]

print("\nPredictions (first 12 rows):")
print(df[["review_text", "sentiment", "predicted_label", "confidence"]].head(12).to_string(index=False))

print("\n" + "=" * 70)
print("STEP 6: EVALUATE (ONLY ON BINARY LABELS)")
print("=" * 70)
# The DistilBERT model only knows POSITIVE and NEGATIVE.
# We evaluate only on the rows that already have those labels.
binary_mask = df["sentiment"].isin(["Positive", "Negative"])
binary_df = df[binary_mask].copy()

# Map our labels to match the model's output
binary_df["true_label"] = binary_df["sentiment"].map({"Positive": "POSITIVE", "Negative": "NEGATIVE"})

acc = accuracy_score(binary_df["true_label"], binary_df["predicted_label"])
print(f"\nAccuracy on Positive/Negative reviews: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(binary_df["true_label"], binary_df["predicted_label"]))

print("\n" + "=" * 70)
print("STEP 7: TEST ON BRAND-NEW REVIEWS")
print("=" * 70)
new_reviews = [
    "The shoes are comfortable and look exactly like the pictures.",
    "The app keeps crashing whenever I try to pay.",
    "Delivery was on time. The packaging was ordinary.",
    "I absolutely love this service. The experience was excellent.",
    "The product quality is terrible. Do not buy."
]

print("\nNew review predictions:")
for text in new_reviews:
    result = classifier(text)[0]
    print(f"\n{text}")
    print(f"  → {result['label']}  (confidence {result['score']:.4f})")

print("\n" + "=" * 70)
print("STEP 8: HOW THE PIECES FIT TOGETHER")
print("=" * 70)
print("""
1. Raw text  →  Tokenizer turns it into token IDs
2. Token IDs + positional encodings  →  Transformer encoder layers
3. Self-attention lets every token look at every other token
4. The final representation of the [CLS] token goes to a small classification head
5. The head outputs two numbers (logits). Softmax turns them into probabilities.
6. The higher probability becomes the predicted label + confidence score.

You did not train any of this. The model was pre-trained and then fine-tuned
on a large sentiment dataset. That is why a few lines of code can do what
used to require a lot of feature engineering (TF-IDF, n-grams, etc.).

Compare this to your earlier TF-IDF + Logistic Regression project:
- Classical: you build the features yourself.
- Transformer: the model learns the features during pre-training.
""")

print("\nDone. You can now open sample_customer_reviews.csv, change a few rows,")
print("and re-run this script to see exactly how the model reacts.")
