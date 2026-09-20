# AI Portfolio
Student: Jim  
GitHub: [jim4004](https://github.com/jim4004)

Python AI class work (Sat/Sun). Each folder is one assignment.

| Project | Method | Folder |
| --- | --- | --- |
| H1N1 vaccine uptake | Logistic regression + SMOTE | [`h1n1-vaccine-prediction/`](h1n1-vaccine-prediction/) |
| Ecommerce customer segments | K-Means clustering | [`ecommerce-customer-clustering/`](ecommerce-customer-clustering/) |

---

# Ecommerce Customer Clustering

**Question:** Can we group 12,000 shoppers into a few useful types?

**Method:** Median fill → IQR cap → encode → `StandardScaler` → elbow + silhouette → K-Means (K=4) → profile means.

Unsupervised: there is no `y`. Label numbers 0/1/2/3 mean nothing until you read the group averages.

Full write-up and notebook: [`ecommerce-customer-clustering/`](ecommerce-customer-clustering/).

Put `ecommerce_customer_clustering_12000.csv` next to the notebook (class file, ~1.4 MB).

---

# H1N1 Vaccine Usage Prediction

Portfolio project for a Python AI class.

**Question:** How likely is a person to take the H1N1 flu vaccine?

**Method:** Logistic Regression (Maximum Likelihood), with missing-value cleanup and SMOTE to handle class imbalance.

This matches the course assignment *Vaccine Usage Prediction* (logistic regression on the National 2009 H1N1 Flu Survey).

---

## Dataset

File: `data/h1n1_vaccine_prediction.csv`  
Size: **26,707 rows × 34 columns**

| Role | Column | Meaning |
| --- | --- | --- |
| Target | `h1n1_vaccine` | 1 = received H1N1 vaccine, 0 = did not |
| ID | `unique_id` | Dropped before modeling |
| Features | 32 survey questions | Worry, awareness, doctor recommendation, opinions about risk/effectiveness, demographics |

Target is **imbalanced**:

- Did not vaccinate: about **78.8%**
- Vaccinated: about **21.2%**

If we always guess “no vaccine,” accuracy would already look high (~79%). That is why we also report **precision, recall, F1, and ROC-AUC**, and why we use **SMOTE on the training set only**.

The full data dictionary is in `data/Problem_Statement_Logistic_Regression.pdf`.

---

## How missing values were fixed

| Column | Missing | What I did |
| --- | ---: | --- |
| `has_health_insur` | 46% | Added `has_health_insur_missing` flag, then filled remaining blanks with the mode |
| `income_level` | 17% | Filled with `"Unknown"` |
| Doctor recommendation columns | 8% | Filled with the mode (0 or 1) |
| Other Likert / yes-no items | 0–5% | Filled with the **mode** |
| Text categories (`qualification`, `marital_status`, `housing_status`, `employment`) | 5–8% | Filled with `"Unknown"` |
| `unique_id` | 0 | Dropped — it is not a predictor |

After cleaning there are **0 missing values**.

---

## Model pipeline

1. Load CSV
2. EDA
3. Clean missing values
4. One-hot encode text columns
5. Train / test split: **80 / 20**, `stratify=y`, `random_state=42`
6. **SMOTE** on the training set only
7. `LogisticRegression(max_iter=2000)`
8. Evaluate on the held-out test set

## Results (test set)

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8034 |
| Precision (got vaccine) | 0.5303 |
| Recall (got vaccine) | 0.6546 |
| F1 (got vaccine) | 0.5860 |
| ROC-AUC | 0.8268 |
