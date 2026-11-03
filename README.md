# Customer Segmentation using K-Means Clustering & Pipeline

This project demonstrates an **end-to-end unsupervised machine learning workflow** that groups mall customers by **annual income and spending behaviour**, so marketing can treat each group differently. It chooses the number of groups with a written rule, checks that the groups are stable, compares K-Means with two other algorithms, names each segment, and attaches a marketing action to it.

It follows **industrial best practices** by:

- Putting **scaling inside a Pipeline**, so new customers are scaled exactly like the training customers
- Choosing the number of clusters with **several measures together** (elbow, silhouette, Davies-Bouldin) plus a **stability check**, not by eye alone
- Writing the selection rule down in the code, so the choice can be reviewed
- Comparing K-Means with **Hierarchical clustering and a Gaussian Mixture** at the same k
- Turning clusters into **named segments with marketing actions**
- Saving the trained pipeline with **Joblib** and reusing it from a separate **prediction script**
- Saving plots, tables and metrics to files for **reproducibility**
- Being **honest about limits**: clustering has no answer key, and this dataset is small

---

## Dependencies

Requires **Python 3.9+** (tested on Python 3.12). Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## Dataset Information

**Source:** `Mall_Customers.csv` (100 customers, 4 columns, no missing values, no duplicate customers)

**Columns:**

| # | Column | Description | Used for |
|---|--------|-------------|----------|
| 1 | CustomerID | Customer number | Label only |
| 2 | Age | Customer age (18 to 70) | Describing segments |
| 3 | AnnualIncome | Annual income in thousands (15 to 137) | **Clustering** |
| 4 | SpendingScore | Score from 1 to 100 given by the mall | **Clustering** |

**Things to know about this data:**

- `CustomerID` correlates **0.97 with AnnualIncome**, because the file is sorted by income. It is only a label, so it is **not** used as a feature.
- `SpendingScore` is a ready-made score, not real purchase records, so the segments describe the score and not actual transactions.
- There is **no target column**. Clustering is unsupervised, so there is no accuracy to report.

---

## Workflow

### Data Preparation

- Load the CSV and check columns, duplicate IDs, missing values, and valid ranges (SpendingScore 1 to 100, Age 18 or above)
- Explore distributions and correlations

### Pipeline Construction

- **Step 1:** `StandardScaler`, so income (15 to 137) does not outweigh spending score (1 to 100) in the distance calculation
- **Step 2:** `KMeans` with 20 restarts (`n_init=20`) and a fixed `random_state=42`

### Choosing the Number of Clusters (k)

For every k from 2 to 8 the script measures:

| Measure | Meaning |
|---------|---------|
| Inertia (elbow) | How tight the groups are (always falls as k grows) |
| Silhouette | How well customers fit their own group versus the nearest other group (higher is better) |
| Davies-Bouldin | Group overlap (lower is better) |
| Stability | Refit on 100 random 80% samples and compare with the full-data groups using the Adjusted Rand Index (1.0 = identical groups) |
| Smallest segment | Share of customers in the smallest group |

**Selection rule** (written in `choose_k`):

1. Keep only k with stability of **0.90 or more** and no segment smaller than **8%** of customers
2. Find the best silhouette among those k
3. Take the **largest** k whose silhouette is within **0.03** of the best, because finer segments are more useful for marketing when the evidence does not get clearly worse

> The rule was written after exploring the data. Treat k = 5 as a judgment call that the metrics support, not as a proof.

### Segment Profiling

- Describe each cluster in original units (average Age, income, spending)
- Name the level of income and spending (Very low to Very high), measured in standard deviations from the average customer
- Give each segment a persona name and a marketing action
- Number segments S1 to S5 by income, so numbering does not depend on K-Means' random cluster order

### Model Saving & Loading

- Save the full pipeline with **Joblib** (`joblib.dump`)
- Save a **metadata file** (expected columns, training ranges, what each cluster means, library versions, quality metrics)
- Load both in a separate script (`joblib.load`) and assign new customers without retraining

---

## Running the Project

```bash
python customer_segmentation.py
```

The run takes about 15 seconds. Expected output (shortened):

```text
Dataset Shape :: (100, 4)
Total missing values :: 0
  k=2  silhouette 0.432 | Davies-Bouldin 0.787 | stability 0.852 +/- 0.362 | smallest segment 24%
  k=3  silhouette 0.556 | Davies-Bouldin 0.618 | stability 0.972 +/- 0.032 | smallest segment 25%
  ...
Chosen k :: 5
silhouette               :: 0.534
stability_ari_mean       :: 0.974
...
Model saved to outputs/customer_segmentation_model.joblib
Loaded model: first customer is in S3 (Value seekers)
```

**Generated files** (in `outputs/`):

```text
outputs/
├── figures/
│   ├── eda_overview.png
│   ├── k_selection.png
│   ├── customer_segments.png
│   ├── segment_profiles.png
│   └── silhouette_plot.png
├── k_selection.csv
├── algorithm_comparison.csv
├── segment_profiles.csv
├── customer_segments.csv        # every customer with their segment
├── metrics.json
├── model_metadata.json
└── customer_segmentation_model.joblib
```

---

## Results

**Choosing k:**

| k | Silhouette | Davies-Bouldin | Stability (ARI) | Smallest segment | Eligible |
|---|-----------|----------------|-----------------|------------------|----------|
| 2 | 0.432 | 0.787 | 0.852 | 24% | No (unstable) |
| 3 | **0.556** | 0.618 | 0.972 | 25% | Yes |
| 4 | 0.528 | 0.639 | 0.951 | 15% | Yes |
| **5** | **0.534** | **0.587** | **0.974** | **12%** | **Yes (chosen)** |
| 6 | 0.545 | 0.551 | 0.899 | 2% | No (tiny segment, stability just under 0.90) |
| 7 | 0.529 | 0.576 | 0.784 | 2% | No |
| 8 | 0.452 | 0.665 | 0.801 | 2% | No |

Silhouette alone prefers k = 3 (0.556). The rule accepts k = 5 because it is just as stable (0.974 against 0.972), only 0.022 lower in silhouette, and splits the customers into five more useful groups.

**The five segments:**

| Segment | Persona | Customers | Avg Age | Avg Income | Avg Spending | Marketing action |
|---------|---------|-----------|---------|-----------|--------------|------------------|
| S1 | Disengaged low spenders | 12 (12%) | 48.8 | 24.9 | 9.6 | Low-cost reactivation by email or SMS; avoid expensive campaigns |
| S2 | Enthusiastic spenders | 24 (24%) | 26.7 | 28.5 | 78.9 | Reward loyalty, push new arrivals and trend offers |
| S3 | Value seekers | 16 (16%) | 40.1 | 33.7 | 37.6 | Discounts, coupons and bundle deals at low campaign cost |
| S4 | Core mainstream | 36 (36%) | 45.9 | 75.8 | 54.1 | Tiered loyalty program, cross-selling, seasonal campaigns |
| S5 | Affluent under-spenders | 12 (12%) | 38.2 | 122.7 | 50.6 | Personalized premium offers and events to raise spending |

**Quality of the final model:** silhouette 0.534, Davies-Bouldin 0.587, Calinski-Harabasz 163.3, stability 0.974 (+/- 0.066).

**Other algorithms at k = 5:**

| Algorithm | Silhouette | Davies-Bouldin | Agreement with K-Means (ARI) |
|-----------|-----------|----------------|------------------------------|
| K-Means | 0.534 | 0.587 | 1.000 |
| Hierarchical (Ward) | 0.492 | 0.676 | 0.786 |
| Gaussian Mixture | 0.555 | 0.538 | 0.766 |

**Key findings:**

1. **The groups are real and stable.** Refitting on random 80% samples gives almost the same groups (ARI 0.974), and the other two algorithms find similar groups (ARI 0.77 to 0.79).
2. **K-Means is not the clear winner.** The Gaussian Mixture scores slightly better on silhouette and Davies-Bouldin. K-Means is kept because it is simple, fast, and gives a clear center for each segment, which makes new customers easy to assign and explain.
3. **There is no high-income, high-spending group in this file.** The clustering found no "premium spenders", so the biggest opportunity is **S5**: 12 customers with the highest incomes (average 123) who spend only a mid-level amount.
4. **Younger customers spend more.** Age and SpendingScore correlate at -0.43. The high-spending S2 segment averages 26.7 years, and the disengaged S1 segment averages 48.8.
5. **Age was left out of the clustering on purpose.** Adding it lowers the silhouette from 0.534 to 0.404 and changes the groups (ARI 0.50 against the 2-feature result). The scores come from different feature spaces, so they are not strictly comparable, which is why Age is used to describe segments rather than to build them.
6. **The marketing actions are hypotheses.** They follow from each segment's profile, but nobody has tested them. A real project would A/B test them before relying on them.

**Limits:** only 100 customers and 2 behavioural features, so these results are exploratory. A different sample could move the borders between segments.

---

## Visualizations

- **Exploratory overview:** `outputs/figures/eda_overview.png`
- **Choosing k (elbow, silhouette, Davies-Bouldin, stability):** `outputs/figures/k_selection.png`
- **Segments with centers:** `outputs/figures/customer_segments.png`
- **Segment profile heatmap:** `outputs/figures/segment_profiles.png`
- **Silhouette plot:** `outputs/figures/silhouette_plot.png`

Set `SHOW_PLOTS = True` at the top of the script to also open the plots in a window.

---

## Model Persistence with Joblib (Train Once, Use Many Times)

**The idea:** fitting the model needs the full dataset. Assigning a new customer to a segment should take milliseconds and should not need the dataset. So the trained model is written to a file with `joblib.dump`, and any other program can bring it back with `joblib.load` and call `predict`.

```text
 TRAINING (run once)                          PREDICTION (run any time)
 customer_segmentation.py                     predict_segment.py
 ┌──────────────────────┐                     ┌──────────────────────────┐
 │ Mall_Customers.csv   │                     │ new_customers.csv        │
 │ choose k, fit, name  │                     │ load model + metadata    │
 │ joblib.dump(model)   │ ──► .joblib  ──►    │ check columns and ranges │
 │ write metadata.json  │ ──► .json    ──►    │ assign segment           │
 └──────────────────────┘                     └──────────────────────────┘
```

**What is stored:**

- `outputs/customer_segmentation_model.joblib`: the **whole pipeline** (fitted scaler and K-Means). The scaling travels with the model, so new customers are measured on the same scale as the training customers.
- `outputs/model_metadata.json`: the expected columns, the value ranges seen in training, the meaning of each cluster (segment, persona, action), a distance limit for flagging unusual customers, library versions, and the quality metrics.

**Assign new customers (no training, no dataset needed):**

```bash
python predict_segment.py
```

The file name is optional: with no argument it scores `new_customers.csv`. Train first (`python customer_segmentation.py`) so the `outputs/` folder with the model exists.

```text
  Segment                  Persona  DistanceToCentroid  Unusual
0      S2    Enthusiastic spenders                0.33    False
1      S5  Affluent under-spenders                0.26    False
2      S4          Core mainstream                0.08    False
3      S1  Disengaged low spenders                0.07    False
4      S5  Affluent under-spenders                2.04     True
```

Customer 4 has an income of 140, above the highest training income of 137. It is far from every segment center, so it is flagged as **unusual**, and the script also prints a warning. Distances are in standard-deviation units, and the limit (0.856) is the 95th percentile of the training customers' distances.

**What the prediction script checks:** missing columns, non-numeric values, empty values, values outside the training range (warning), and a scikit-learn version different from the one that trained the model (warning).

**Rules for using saved models:**

1. **Same library version:** load the model with the scikit-learn version that trained it.
2. **Trusted files only:** Joblib files can run code when opened, so load only files you created yourself.
3. **Refit when the customers change:** a saved model does not learn. When the customer base changes, rerun `customer_segmentation.py` and check that the segments still make sense.
4. **Git:** `.joblib` files are in `.gitignore`. After cloning, run `python customer_segmentation.py` once to create the model.

---

## Sample Prediction (from Python code)

```python
import pandas as pd
from customer_segmentation import load_model, FEATURES

model = load_model("outputs/customer_segmentation_model.joblib")

customer = pd.DataFrame([[24, 22, 85]], columns=["Age", "AnnualIncome", "SpendingScore"])
cluster = int(model.predict(customer[FEATURES])[0])
print("Cluster number:", cluster)  # look it up in outputs/model_metadata.json -> segments
```

> **Note:** This is an educational project on a small dataset. The segments and actions are a starting point for analysis, not tested business advice.

---

## Author

**Prathmesh Girme**
GitHub: [@Prathmesh-Girme](https://github.com/Prathmesh-Girme)
**Date:** 02/11/2026
