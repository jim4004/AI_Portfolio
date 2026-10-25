# Customer Review Sentiment Analyzer
# Portfolio project for Saturday/Sunday Python AI class
# Pipeline: load -> EDA -> simple clean -> TF-IDF -> Logistic Regression -> evaluate

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load
df = pd.read_csv("customer_reviews_sentiment_practice.csv")
print("Shape:", df.shape)
print(df.head())

# 2. EDA
print("\nSentiment counts:")
print(df["sentiment"].value_counts())

df["sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# 3. Simple cleaning (class notes: .lower() + whitespace)
def clean_text(text):
    text = str(text).lower()
    text = " ".join(text.split())
    return text

df["clean_text"] = df["review_text"].apply(clean_text)

print("\nExample cleaning:")
print("Original:", df.loc[0, "review_text"])
print("Cleaned :", df.loc[0, "clean_text"])

# 4-5. TF-IDF + Logistic Regression (class pipeline style)
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))

# 7. Test on new reviews
new_reviews = [
    "The shoes are comfortable and look exactly like the pictures.",
    "The app keeps crashing whenever I try to pay.",
    "The food arrived hot, but the portion was too small.",
    "Delivery was on time. The packaging was ordinary.",
    "Customer support solved my issue within ten minutes. Great service!"
]

print("\n--- New review predictions ---")
for review in new_reviews:
    cleaned = clean_text(review)
    pred = model.predict([cleaned])[0]
    print(f"{review}")
    print(f"  -> {pred}\n")
