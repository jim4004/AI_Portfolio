# Travel Review Segmentation

Portfolio project for a Saturday/Sunday Python AI class.

**Question:** Can we group Google reviewers into a few useful types from 24 attraction-category ratings?

**Method:** Drop the user id and junk column, coerce the one broken text rating, median-fill three NaNs, `StandardScaler`, Ward dendrogram on a sample to pick K, `AgglomerativeClustering` on the full file, then profile cluster means.

This is **unsupervised**. There is no `y` and no train/test split. The label numbers `0, 1, 2, 3` have no meaning until you look at the group means.

---

## Dataset

File: `google_review_ratings.csv`  
Size: **5,456 rows x 26 columns** (24 rating categories + `User` + a junk `Unnamed: 25` column)

Google user ratings (1-5) averaged by attraction category across Europe.

| Column | Description |
| --- | --- |
| User | Unique user id. Drop before clustering. |
| Category 1 | Churches |
| Category 2 | Resorts |
| Category 3 | Beaches |
| Category 4 | Parks |
| Category 5 | Theatres |
| Category 6 | Museums |
| Category 7 | Malls |
| Category 8 | Zoo |
| Category 9 | Restaurants |
| Category 10 | Pubs / bars |
| Category 11 | Local services (stored as text -- one garbled cell) |
| Category 12 | Burger / pizza shops |
| Category 13 | Hotels / other lodgings |
| Category 14 | Juice bars |
| Category 15 | Art galleries |
| Category 16 | Dance clubs |
| Category 17 | Swimming pools |
| Category 18 | Gyms |
| Category 19 | Bakeries |
| Category 20 | Beauty & spas |
| Category 21 | Cafes |
| Category 22 | Viewpoints |
| Category 23 | Monuments |
| Category 24 | Gardens |
| Unnamed: 25 | Trailing-comma leftover. Drop it. |

Two dirty rows after coerce: one tab in Category 11 (`2\t2.`), one shifted Category 24. Median fill those three numeric holes.

---

## Pipeline (assignment rubric)

1. Data understanding -- `shape`, `info`, `isna`, `describe`
2. Missing values -- coerce Category 11, median fill
3. Drop IDs -- `User`, `Unnamed: 25`
4. Feature scaling -- `StandardScaler` **before** hierarchical clustering
5. Dendrogram -- Ward linkage on an 800-row sample; cut the tallest vertical gap
6. Cluster -- `AgglomerativeClustering(n_clusters=k, linkage="ward")` on the full file
7. Profile -- `groupby("Cluster").mean()`
8. Business interpretation -- table below

K-Means is only a comparison. The assignment asks for hierarchical clustering.

---

## How to run

```bash
cd travel-review-segmentation
python -m pip install -r requirements.txt
jupyter notebook travel_review_segmentation.ipynb
```

Put the CSV next to the notebook. Run every cell top to bottom.

---

## Results (Ward, k = 4)

Silhouette is modest (~0.13). Clusters overlap. Names come from the **means**, not from the label integers.

| Cluster | Size | What they look like |
| --- | --- | --- |
| 0 Culture / sightseeing | ~2,585 | High parks, theatres, museums, beaches |
| 1 Outdoor / quiet | ~1,024 | Higher viewpoints, monuments, gardens, bakeries |
| 2 Food + night | ~1,371 | Malls ~4.1, restaurants ~4.6, pubs ~4.1 |
| 3 Lodging / convenience | ~476 | Hotels ~4.6, burger/pizza ~4.3, juice bars ~4.9 |

The tallest dendrogram cuts are k=2 or k=3. k=4 is the more useful review-types cut. If the instructor locks k=3, change that one argument and rename groups from the new `groupby` means.

---

## Project structure

```
travel-review-segmentation/
├── README.md
├── requirements.txt
├── DATA.txt
├── travel_review_segmentation.ipynb
└── google_review_ratings.csv
```

---

## What this project shows from class

- Pandas inspect / fillna / groupby
- Why a column can be `object` when it looks numeric
- Why hierarchical clustering needs scaling
- Dendrogram first, then cut K (opposite of K-Means)
- Sample the tree so Jupyter does not freeze
- Cluster labels are arbitrary integers until you profile them
