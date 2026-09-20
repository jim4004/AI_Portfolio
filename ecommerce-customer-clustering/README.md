# Ecommerce Customer Clustering

Portfolio project for a Saturday/Sunday Python AI class.

**Question:** Can we group 12,000 shoppers into a few useful types using K-Means?

**Method:** Clean missing values, encode categories, cap IQR outliers, scale numeric features, pick K with elbow + silhouette, then profile the clusters for a business story.

This is **unsupervised**. There is no `y` and no train/test split. The label numbers `0, 1, 2, 3` have no meaning until you look at the group means.

---

## Dataset

File: `ecommerce_customer_clustering_12000.csv`  
Size: **12,000 rows × 20 columns**

| Role | Column | Notes |
| --- | --- | --- |
| ID | `Customer_ID` | Unique. Drop before clustering. |
| Money | `Annual_Income`, `Average_Order_Value`, `Annual_Spend` | Heavy right tail. Cap with IQR. |
| Activity | `Tenure_Months`, `App_Visits_Per_Month`, `Orders_Per_Month`, `Days_Since_Last_Order` | Recency + frequency |
| Behavior | `Discount_Usage_Percentage`, `Returns_Count`, `Support_Tickets`, `Engagement_Score` | 20 negative engagement scores → clip to 0 |
| Categories | `City`, `Membership_Tier`, `Device_Type`, `Preferred_Payment`, `Favorite_Category` | Encode. Do not dump 10 cities into K-Means. |

180 missing values sit on: income, visits, orders, AOV, discount %, rating.

---

## Pipeline (assignment rubric)

1. Data understanding — `shape`, `info`, `describe`, `value_counts`
2. Missing values — median fill on the six numeric columns
3. Categorical encoding — `Membership_Tier` as 1–4; `get_dummies` for device / payment / category
4. Outlier analysis — IQR fences, then `.clip` (do not delete half the table)
5. Feature selection — drop `Customer_ID`; cluster on behavior + money
6. Feature scaling — `StandardScaler` **before** K-Means (income would crush rating)
7. Elbow method — inertia vs K
8. Silhouette score — overlap is real on this file (~0.10)
9. K-Means — `n_clusters=4`, `random_state=42`, `n_init=10`
10. Cluster profiling — `groupby("cluster").mean()`
11. Business interpretation — table below

Class demo used two columns (`Annual_Income`, `Spending_Score`) on a toy mall file. This project has no `Spending_Score`. Use `Annual_Spend` or `Engagement_Score` for the 2-column live demo; use the 11-feature set for the graded notebook.

---

## How to run

```bash
cd ecommerce-customer-clustering
python -m pip install -r requirements.txt
jupyter notebook ecommerce_customer_clustering.ipynb
```

Put the CSV next to the notebook. Run every cell top to bottom.

---

## Results (K = 4, random_state = 42)

Silhouette is modest. Clusters overlap. Names come from the **means**, not from the label integers.

| Cluster | Size | What they look like | Business move |
| --- | --- | --- | --- |
| 1 VIP | ~14% | Highest income, visits, orders, spend (~147k), engagement | Perks, not extra coupons |
| 0 Frequent mid | ~29% | Mid income, high order count, mid spend | Bundles / subscription |
| 2 High-ticket rare | ~25% | High income, few orders, high AOV, cooler engagement | Reminders, not discounts |
| 3 Deal / at-risk | ~32% | Lowest spend and engagement, highest discount use | Win-back only if margin allows |

`Membership_Tier` mix is ~42% Basic in **every** cluster. The badge does not separate people. Behavior does.

If the instructor locks `n_clusters=3`, change that one argument and rename groups from the new `groupby` means.

---

## Project structure

```
ecommerce-customer-clustering/
├── README.md
├── requirements.txt
├── ecommerce_customer_clustering.ipynb
└── ecommerce_customer_clustering_12000.csv
```

---

## What this project shows from class

- Pandas inspect / fillna / groupby
- IQR outlier fences (upper = Q3 + 1.5×IQR, not Q1)
- Encoding ordered vs unordered categories
- Why K-Means needs scaling
- Elbow + silhouette to pick K
- Cluster labels are arbitrary integers until you profile them
