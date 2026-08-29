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

The class notes say to check missing values and fix them (same idea as the Netflix project).

| Column | Missing | What I did |
| --- | ---: | --- |
| `has_health_insur` | 46% | Added `has_health_insur_missing` flag, then filled remaining blanks with the mode |
| `income_level` | 17% | Filled with `"Unknown"` |
| Doctor recommendation columns | 8% | Filled with the mode (0 or 1) |
| Other Likert / yes-no items | 0–5% | Filled with the **mode** |
| Text categories (`qualification`, `marital_status`, `housing_status`, `employment`) | 5–8% | Filled with `"Unknown"` |
| `unique_id` | 0 | Dropped — it is not a predictor |

After cleaning there are **0 missing values**. The cleaned table is saved as `outputs/h1n1_cleaned.csv`.

Why `"Unknown"` instead of mode for text fields? Mode would pretend we know someone’s income or housing. Keeping `"Unknown"` lets the model learn that “did not answer” can itself be a signal.

---

## Model pipeline

1. Load CSV
2. EDA (class balance, missingness, vaccine rate by doctor recommendation and age)
3. Clean missing values
4. One-hot encode text columns (`age_bracket`, `qualification`, `race`, `sex`, `income_level`, `marital_status`, `housing_status`, `employment`, `census_msa`)
5. Train / test split: **80 / 20**, `stratify=y`, `random_state=42`
6. **SMOTE** on the training set only (never on the test set)
7. `LogisticRegression(max_iter=2000)`
8. Evaluate on the held-out test set

SMOTE is applied **after** the split so the test set still looks like real survey data.

---

## How to run

```bash
cd h1n1-vaccine-prediction
python -m pip install -r requirements.txt
python h1n1_vaccine_prediction.py
```

Outputs land in `outputs/`:

| File | What it is |
| --- | --- |
| `01_target_balance.png` | How many people got the vaccine |
| `02_missing_values.png` | Percent missing by column |
| `03_vaccine_by_doctor_rec.png` | Vaccine rate if a doctor recommended it |
| `04_vaccine_by_age.png` | Vaccine rate by age group |
| `05_confusion_matrix.png` | Test-set confusion matrix |
| `06_roc_curve.png` | ROC curve |
| `07_top_coefficients.png` | Largest logistic coefficients |
| `h1n1_cleaned.csv` | Data after missing-value repair |
| `model_metrics.csv` | Accuracy, precision, recall, F1, ROC-AUC |
| `feature_coefficients.csv` | Every feature’s log-odds coefficient |
| `classification_report.txt` | Full sklearn report |

---

## Results (test set)

20% holdout, stratified split, SMOTE only on training data.

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8034 |
| Precision (got vaccine) | 0.5303 |
| Recall (got vaccine) | 0.6546 |
| F1 (got vaccine) | 0.5860 |
| ROC-AUC | 0.8268 |

Confusion matrix:

|  | Predicted no | Predicted yes |
| --- | ---: | ---: |
| Actually no | 3,549 | 658 |
| Actually yes | 392 | 743 |

A dummy model that always says “no vaccine” would get ~79% accuracy but **0 recall** on people who vaccinated. This model keeps similar accuracy and finds about **65% of the people who actually got the shot**.

Doctor recommendation (`dr_recc_h1n1_vacc`) and belief items (`is_h1n1_vacc_effective`, `is_h1n1_risky`) are among the largest coefficients. Full list: `outputs/feature_coefficients.csv`.

---

## Project structure

```
h1n1-vaccine-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── h1n1_vaccine_prediction.py
├── data/
│   ├── h1n1_vaccine_prediction.csv
│   └── Problem_Statement_Logistic_Regression.pdf
└── outputs/                  # created when you run the script
```

---

## Upload this folder to GitHub

You need a free GitHub account. Then do this **on your computer**, inside this project folder.

### One-time: install Git if needed

Windows: https://git-scm.com/download/win  
Mac: `xcode-select --install` or install Git from git-scm.com

### Create the empty repo on GitHub

1. Open https://github.com/new
2. Repository name: `h1n1-vaccine-prediction`
3. Keep it **Public** (portfolio) unless your instructor said private
4. Do **not** add a README, .gitignore, or license on the website (this folder already has them)
5. Click **Create repository**

### Push from your laptop

```bash
cd h1n1-vaccine-prediction

git init
git add .
git commit -m "Initial commit: H1N1 vaccine logistic regression project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/h1n1-vaccine-prediction.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your real GitHub username.

GitHub will ask you to sign in. Use a **Personal Access Token** as the password if it rejects your normal password:

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a token with the `repo` scope
3. Paste that token when `git push` asks for a password

### After it uploads

Repo URL will look like:

`https://github.com/YOUR_GITHUB_USERNAME/h1n1-vaccine-prediction`

Put that link in your class submission / portfolio.

---

## What this project shows from class

- Conditionals and loops are inside the cleaning / EDA steps
- Lists, dictionaries, and functions (`load_data`, `clean_data`, `train_and_evaluate`)
- Pandas for tables (same family of work as the Netflix missing-value lab)
- Train / test split so we do not grade the model on data it already saw
- SMOTE for imbalance (same pattern as the medical logistic-regression lecture)
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report
