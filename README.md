# AI Class Portfolio

Student: Jim  
GitHub: [jim4004/AI_Portfolio](https://github.com/jim4004/AI_Portfolio)

Saturday / Sunday Python AI class. One repo, one folder per assignment.

| Project | Kind | Folder |
| --- | --- | --- |
| H1N1 vaccine uptake | Supervised -- logistic regression + SMOTE | [`h1n1-vaccine-prediction/`](h1n1-vaccine-prediction/) |
| Ecommerce customer segments | Unsupervised -- K-Means | [`ecommerce-customer-clustering/`](ecommerce-customer-clustering/) |
| Travel review segmentation | Unsupervised -- hierarchical clustering | [`travel-review-segmentation/`](travel-review-segmentation/) |

Class 6-step recipe (supervised): load -> inspect -> fill NaNs -> X/y -> train_test_split -> fit / predict / score.

Clustering skips the split. There is no `y`. Label numbers 0/1/2/3 mean nothing until you read the group means.

---

## 1. H1N1 Vaccine Usage Prediction

**Question:** How likely is a person to take the H1N1 flu vaccine?

**Method:** Clean the National 2009 H1N1 Flu Survey (26,707 rows), one-hot encode text, 80/20 stratified split, SMOTE on **train only**, `LogisticRegression`.

| Metric (test) | Score |
| --- | ---: |
| Accuracy | 0.8034 |
| Precision (vaccinated) | 0.5303 |
| Recall (vaccinated) | 0.6546 |
| F1 | 0.5860 |
| ROC-AUC | 0.8268 |

```bash
cd h1n1-vaccine-prediction
python -m pip install -r requirements.txt
python h1n1_vaccine_prediction.py
```

Put `data/h1n1_vaccine_prediction.csv` in that folder before you run it.

---

## 2. Ecommerce Customer Clustering

**Question:** Can we group 12,000 shoppers into a few useful types?

**Method:** Median fill -> IQR cap -> encode -> `StandardScaler` -> elbow + silhouette -> K-Means (K=4) -> profile means.

| Cluster | About | Move |
| --- | --- | --- |
| VIP (~14%) | Highest income, orders, spend, engagement | Perks, not extra coupons |
| Frequent mid (~29%) | Mid income, lots of orders | Bundles / subscription |
| High-ticket rare (~25%) | High income, few orders, high AOV | Reminders, not discounts |
| Deal / at-risk (~32%) | Lowest spend and engagement | Win-back only if margin allows |

Membership tier mix is almost the same in every cluster. Behavior separates people, not the badge.

```bash
cd ecommerce-customer-clustering
python -m pip install -r requirements.txt
jupyter notebook ecommerce_customer_clustering.ipynb
```

Put `ecommerce_customer_clustering_12000.csv` next to the notebook (class file, ~1.4 MB).

---

## 3. Travel Review Segmentation

**Question:** Can we group Google reviewers from 24 attraction-category ratings?

**Method:** Drop User + junk column, coerce Category 11, median fill, scale, Ward dendrogram on a sample, `AgglomerativeClustering` (k=4), profile means.

| Cluster | About |
| --- | --- |
| Culture / sightseeing (~47%) | Parks, theatres, museums, beaches |
| Outdoor / quiet (~19%) | Viewpoints, monuments, gardens |
| Food + night (~25%) | Malls, restaurants, pubs |
| Lodging / convenience (~9%) | Hotels, burger/pizza, juice bars |

```bash
cd travel-review-segmentation
python -m pip install -r requirements.txt
jupyter notebook travel_review_segmentation.ipynb
```

Put `google_review_ratings.csv` next to the notebook.

---

## Repo layout

```
AI_Portfolio/
├── README.md
├── START_HERE.txt
├── h1n1-vaccine-prediction/
├── ecommerce-customer-clustering/
└── travel-review-segmentation/
```
