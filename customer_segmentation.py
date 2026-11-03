########################################################
# Customer Segmentation using K-Means Clustering
#
# Pipeline : StandardScaler -> KMeans
# Features : AnnualIncome, SpendingScore (Age is used to describe segments)
# Goal     : Group mall customers by income and spending behaviour so
#            marketing can treat each group differently
#
# Clustering has no "correct answer" to score against, so the work is
# judged with internal metrics, stability checks and business sense.
########################################################

########################################################
# Required Python Packages
########################################################
import json
import os
import platform
import warnings
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

########################################################
# Settings
########################################################

warnings.filterwarnings("ignore", category=FutureWarning)

SHOW_PLOTS = False  # False: only save plots to files (works without a display)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must come after backend selection

########################################################
# File Paths
########################################################

DATA_PATH = "Mall_Customers.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_PATH = os.path.join(OUTPUT_DIR, "customer_segmentation_model.joblib")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")
K_SELECTION_PATH = os.path.join(OUTPUT_DIR, "k_selection.csv")
ALGORITHMS_PATH = os.path.join(OUTPUT_DIR, "algorithm_comparison.csv")
PROFILES_PATH = os.path.join(OUTPUT_DIR, "segment_profiles.csv")
SEGMENTS_PATH = os.path.join(OUTPUT_DIR, "customer_segments.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

########################################################
# Columns
########################################################

ID_COLUMN = "CustomerID"
FEATURES = ["AnnualIncome", "SpendingScore"]  # used for clustering
PROFILE_COLUMNS = ["Age", "AnnualIncome", "SpendingScore"]  # used to describe segments

########################################################
# Clustering Settings
########################################################

RANDOM_STATE = 42
N_INIT = 20
K_RANGE = range(2, 9)

# Stability: refit on random 80% samples and check the groups stay the same
BOOTSTRAP_RUNS = 100
BOOTSTRAP_FRACTION = 0.8

# Rule for choosing k (see choose_k)
MIN_STABILITY = 0.90          # mean Adjusted Rand Index across resamples
MIN_CLUSTER_SHARE = 0.08      # smallest segment must hold at least 8% of customers
MAX_SILHOUETTE_GAP = 0.03     # accept a larger k if silhouette is within this of the best

# Level names for a segment's average income / spending, measured in standard
# deviations from the average customer
LEVEL_CUTS = [-1.5, -0.5, 0.5, 1.5]
LEVEL_NAMES = ["Very low", "Low", "Mid", "High", "Very high"]

# Marketing action for each persona
ACTIONS = {
    "Enthusiastic spenders": "Reward loyalty and push new arrivals and trend offers; this group already spends "
                             "heavily despite lower income, so protect the relationship.",
    "Premium spenders": "Offer VIP programs, early access and premium bundles.",
    "Affluent under-spenders": "Highest upside: personalized premium offers, curated recommendations and "
                               "invitations to events to raise spending.",
    "Core mainstream": "Retain with a tiered loyalty program, cross-selling and seasonal campaigns.",
    "Value seekers": "Use discounts, coupons and bundle deals, keeping campaign cost low.",
    "Disengaged low spenders": "Run low-cost reactivation (email or SMS offers) and avoid expensive campaigns.",
    "Average customers": "Use general promotions and test offers to see which way they move.",
    "Mixed segment": "Review this segment manually before choosing a campaign.",
}


########################################################
# Function name : read_data
# Description   : Read the customer dataset into a pandas dataframe
# Input         : Path of CSV file
# Output        : Dataframe
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def read_data(path):
    """Read the data into pandas dataframe"""
    return pd.read_csv(path)


########################################################
# Function name : validate_dataset
# Description   : Check columns, value ranges and duplicate customers
# Input         : Dataframe
# Output        : Nothing (raises ValueError if the data is not usable)
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def validate_dataset(dataset):
    """Stop early with a clear message if the dataset is not as expected"""
    needed = [ID_COLUMN] + PROFILE_COLUMNS
    missing_columns = [c for c in needed if c not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    if dataset[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} contains duplicates")

    if dataset[PROFILE_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")

    if not dataset["SpendingScore"].between(1, 100).all():
        raise ValueError("SpendingScore must be between 1 and 100")

    if (dataset["Age"] < 18).any():
        raise ValueError("Age must be 18 or above")


########################################################
# Function name : dataset_statistics
# Description   : Display shape, statistics and correlations
# Input         : Dataframe
# Output        : Nothing (prints)
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def dataset_statistics(dataset):
    """Print basic stats. CustomerID is shown only to explain why it is excluded."""
    print("Dataset Shape ::", dataset.shape)
    print("\nStatistics:\n", dataset.describe().T.round(2))
    print("\nCorrelation matrix:\n", dataset.corr().round(2))

    id_income = dataset[ID_COLUMN].corr(dataset["AnnualIncome"])
    print(f"\nNote: {ID_COLUMN} correlates {id_income:.2f} with AnnualIncome because the file is "
          f"sorted by income. It is only a label, so it is NOT used as a feature.")


########################################################
# Function name : check_missing_values
# Description   : Count missing values and duplicate rows
# Input         : Dataframe
# Output        : Total missing values
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def check_missing_values(dataset):
    """Report missing values per column and duplicate rows"""
    missing = dataset.isnull().sum()
    print("Missing values per column:\n", missing)
    print("Total missing values ::", int(missing.sum()))
    print("Duplicate rows       ::", int(dataset.duplicated().sum()))
    return int(missing.sum())


########################################################
# Function name : finish_plot
# Description   : Save the current figure (and show it if SHOW_PLOTS is True)
# Input         : File name
# Output        : Nothing (writes a PNG)
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def finish_plot(file_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, file_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"Plot saved to {path}")


########################################################
# Function name : plot_eda
# Description   : Exploratory analysis plots
# Input         : Dataframe
# Output        : outputs/figures/eda_overview.png
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def plot_eda(dataset):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, column in zip(axes.ravel()[:3], PROFILE_COLUMNS):
        ax.hist(dataset[column], bins=15, color="#4c78a8", edgecolor="white")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Customers")

    scatter = axes[1, 1].scatter(dataset["AnnualIncome"], dataset["SpendingScore"],
                                 c=dataset["Age"], cmap="viridis", edgecolor="white")
    fig.colorbar(scatter, ax=axes[1, 1], label="Age")
    axes[1, 1].set_title("Income vs Spending (colored by Age)")
    axes[1, 1].set_xlabel("AnnualIncome")
    axes[1, 1].set_ylabel("SpendingScore")

    finish_plot("eda_overview.png")


########################################################
# Function name : build_pipeline
# Description   : Build a Pipeline:
#                 StandardScaler : puts income and spending on the same scale
#                 KMeans         : groups customers into k segments
# Input         : Number of clusters, random state, number of restarts
# Output        : Untrained pipeline
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def build_pipeline(k, random_state=RANDOM_STATE, n_init=N_INIT):
    """Scaling is part of the pipeline so new customers are scaled exactly
    like the training customers. Without it, income (15 to 137) would outweigh
    spending score (1 to 100) in the distance calculation."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=random_state)),
    ])


########################################################
# Function name : bootstrap_stability
# Description   : Refit on random 80% samples and compare with the full-data
#                 grouping using the Adjusted Rand Index (1.0 = identical groups)
# Input         : Dataframe, number of clusters, full-data labels
# Output        : Mean and standard deviation of the ARI
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def bootstrap_stability(dataset, k, reference_labels):
    rng = np.random.RandomState(RANDOM_STATE)
    size = int(BOOTSTRAP_FRACTION * len(dataset))
    scores = []

    for run in range(BOOTSTRAP_RUNS):
        rows = rng.choice(len(dataset), size=size, replace=False)
        sample = dataset.iloc[rows][FEATURES]
        model = build_pipeline(k, random_state=run, n_init=10).fit(sample)
        scores.append(adjusted_rand_score(reference_labels[rows], model.predict(sample)))

    return float(np.mean(scores)), float(np.std(scores))


########################################################
# Function name : evaluate_k_range
# Description   : For each k: inertia (elbow), silhouette, Davies-Bouldin,
#                 Calinski-Harabasz, bootstrap stability, smallest segment
# Input         : Dataframe
# Output        : Dataframe with one row per k
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def evaluate_k_range(dataset):
    rows = []
    for k in K_RANGE:
        pipeline = build_pipeline(k).fit(dataset[FEATURES])
        scaled = pipeline.named_steps["scaler"].transform(dataset[FEATURES])
        labels = pipeline.named_steps["kmeans"].labels_
        stability_mean, stability_std = bootstrap_stability(dataset, k, labels)

        rows.append({
            "k": k,
            "Inertia": pipeline.named_steps["kmeans"].inertia_,
            "Silhouette": silhouette_score(scaled, labels),
            "Davies_Bouldin": davies_bouldin_score(scaled, labels),
            "Calinski_Harabasz": calinski_harabasz_score(scaled, labels),
            "Stability_ARI_Mean": stability_mean,
            "Stability_ARI_Std": stability_std,
            "Smallest_Segment_Share": float(np.bincount(labels).min() / len(labels)),
        })
        print(f"  k={k}  silhouette {rows[-1]['Silhouette']:.3f} | "
              f"Davies-Bouldin {rows[-1]['Davies_Bouldin']:.3f} | "
              f"stability {stability_mean:.3f} +/- {stability_std:.3f} | "
              f"smallest segment {rows[-1]['Smallest_Segment_Share']:.0%}")

    return pd.DataFrame(rows)


########################################################
# Function name : choose_k
# Description   : Pick the number of clusters with a written rule:
#                 1) keep only k that are stable (mean ARI >= 0.90) and have no
#                    tiny segment (smallest >= 8% of customers)
#                 2) among those, find the best silhouette
#                 3) take the LARGEST k whose silhouette is within 0.03 of the
#                    best, because finer segments are more useful to marketing
#                    when the evidence does not get clearly worse
# Input         : k-selection dataframe
# Output        : Chosen k
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def choose_k(table):
    eligible = table[
        (table["Stability_ARI_Mean"] >= MIN_STABILITY)
        & (table["Smallest_Segment_Share"] >= MIN_CLUSTER_SHARE)
    ]
    if eligible.empty:
        return int(table.sort_values("Silhouette", ascending=False).iloc[0]["k"])

    best_silhouette = eligible["Silhouette"].max()
    close = eligible[eligible["Silhouette"] >= best_silhouette - MAX_SILHOUETTE_GAP]
    return int(close["k"].max())


########################################################
# Function name : plot_k_selection
# Description   : Elbow, silhouette, Davies-Bouldin and stability per k
# Input         : k-selection dataframe, chosen k
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def plot_k_selection(table, chosen_k):
    panels = [
        ("Inertia", "Elbow method: inertia (lower is tighter)"),
        ("Silhouette", "Silhouette (higher is better)"),
        ("Davies_Bouldin", "Davies-Bouldin (lower is better)"),
        ("Stability_ARI_Mean", "Stability across resamples (higher is better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, (column, title) in zip(axes.ravel(), panels):
        ax.plot(table["k"], table[column], marker="o")
        ax.axvline(chosen_k, linestyle="--", color="#e45756", label=f"chosen k = {chosen_k}")
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
        ax.legend()

    finish_plot("k_selection.png")


########################################################
# Function name : compare_algorithms
# Description   : Compare K-Means with Hierarchical (Ward) and Gaussian Mixture
#                 at the same k. ARI shows how much the groupings agree.
# Input         : Dataframe, k, K-Means labels
# Output        : Dataframe with one row per algorithm
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def compare_algorithms(dataset, k, kmeans_labels):
    scaled = StandardScaler().fit_transform(dataset[FEATURES])
    candidates = {
        "K-Means": kmeans_labels,
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled),
        "Gaussian Mixture": GaussianMixture(n_components=k, n_init=10,
                                            random_state=RANDOM_STATE).fit_predict(scaled),
    }
    rows = []
    for name, labels in candidates.items():
        rows.append({
            "Algorithm": name,
            "Silhouette": silhouette_score(scaled, labels),
            "Davies_Bouldin": davies_bouldin_score(scaled, labels),
            "Calinski_Harabasz": calinski_harabasz_score(scaled, labels),
            "ARI_vs_KMeans": adjusted_rand_score(kmeans_labels, labels),
        })
    return pd.DataFrame(rows)


########################################################
# Function name : check_age_as_feature
# Description   : See what happens if Age is added to the clustering features
# Input         : Dataframe, k, labels from the 2-feature model
# Output        : Dictionary of results
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def check_age_as_feature(dataset, k, labels_two_features):
    columns = ["Age"] + FEATURES
    scaled = StandardScaler().fit_transform(dataset[columns])
    labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(scaled)
    return {
        "silhouette_with_age": float(silhouette_score(scaled, labels)),
        "ari_vs_two_feature_segments": float(adjusted_rand_score(labels_two_features, labels)),
        "age_spending_correlation": float(dataset["Age"].corr(dataset["SpendingScore"])),
    }


########################################################
# Function name : level_name
# Description   : Turn a z-score into Very low / Low / Mid / High / Very high
# Input         : z-score
# Output        : Level name
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def level_name(z_score):
    return LEVEL_NAMES[int(np.searchsorted(LEVEL_CUTS, z_score))]


########################################################
# Function name : persona_for
# Description   : Give a segment a business-friendly name from its income and
#                 spending levels (first matching rule wins)
# Input         : Income level, spending level
# Output        : Persona name
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def persona_for(income_level, spending_level):
    income_low = income_level in ("Very low", "Low")
    income_high = income_level in ("High", "Very high")
    spending_low = spending_level in ("Very low", "Low")
    spending_high = spending_level in ("High", "Very high")

    if spending_level == "Very low":
        return "Disengaged low spenders"
    if income_low and spending_high:
        return "Enthusiastic spenders"
    if income_low and spending_low:
        return "Value seekers"
    if income_level == "Very high" and not spending_high:
        return "Affluent under-spenders"
    if income_high and spending_high:
        return "Premium spenders"
    if income_high:
        return "Core mainstream"
    if income_level == "Mid":
        return "Average customers"
    return "Mixed segment"


########################################################
# Function name : describe_segments
# Description   : Profile every cluster in original units (Age, income,
#                 spending), name it, and attach a marketing action.
#                 Segments are numbered S1, S2, ... by income then spending,
#                 so numbering does not depend on K-Means' random cluster order.
# Input         : Fitted pipeline, dataframe
# Output        : Profile dataframe, dictionary {raw cluster id: segment info}
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def describe_segments(pipeline, dataset):
    kmeans = pipeline.named_steps["kmeans"]
    labels = kmeans.labels_
    centers_z = kmeans.cluster_centers_  # already in standard-deviation units

    rows = []
    for cluster in range(kmeans.n_clusters):
        members = dataset[labels == cluster]
        income_level = level_name(centers_z[cluster][FEATURES.index("AnnualIncome")])
        spending_level = level_name(centers_z[cluster][FEATURES.index("SpendingScore")])
        persona = persona_for(income_level, spending_level)
        rows.append({
            "Cluster": cluster,
            "Customers": len(members),
            "Share": len(members) / len(dataset),
            "Avg_Age": members["Age"].mean(),
            "Avg_AnnualIncome": members["AnnualIncome"].mean(),
            "Avg_SpendingScore": members["SpendingScore"].mean(),
            "Income_Level": income_level,
            "Spending_Level": spending_level,
            "Persona": persona,
            "Action": ACTIONS[persona],
        })

    profiles = pd.DataFrame(rows).sort_values(["Avg_AnnualIncome", "Avg_SpendingScore"]).reset_index(drop=True)
    profiles.insert(0, "Segment", [f"S{i + 1}" for i in range(len(profiles))])

    segment_info = {
        int(row["Cluster"]): {
            "segment": row["Segment"],
            "persona": row["Persona"],
            "action": row["Action"],
            "customers": int(row["Customers"]),
            "avg_annual_income": round(float(row["Avg_AnnualIncome"]), 1),
            "avg_spending_score": round(float(row["Avg_SpendingScore"]), 1),
            "avg_age": round(float(row["Avg_Age"]), 1),
        }
        for _, row in profiles.iterrows()
    }
    return profiles, segment_info


########################################################
# Function name : plot_clusters
# Description   : Scatter plot of customers colored by segment, with centers
# Input         : Fitted pipeline, dataframe, profile dataframe
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def plot_clusters(pipeline, dataset, profiles):
    labels = pipeline.named_steps["kmeans"].labels_
    centers = pipeline.named_steps["scaler"].inverse_transform(pipeline.named_steps["kmeans"].cluster_centers_)
    colors = plt.cm.tab10(np.arange(len(profiles)))

    plt.figure(figsize=(9, 6.5))
    for position, row in profiles.iterrows():
        cluster = int(row["Cluster"])
        members = dataset[labels == cluster]
        plt.scatter(members["AnnualIncome"], members["SpendingScore"], color=colors[position],
                    edgecolor="white", s=60,
                    label=f"{row['Segment']} {row['Persona']} (n={row['Customers']})")
        plt.scatter(*centers[cluster], color=colors[position], marker="X", s=260, edgecolor="black")
        plt.annotate(row["Segment"], centers[cluster], textcoords="offset points", xytext=(8, 8),
                     fontweight="bold")

    plt.xlabel("AnnualIncome")
    plt.ylabel("SpendingScore")
    plt.title("Customer Segments (X = segment center)")
    plt.legend(fontsize=8, loc="best")
    finish_plot("customer_segments.png")


########################################################
# Function name : plot_segment_profiles
# Description   : Heatmap of each segment's average Age, income and spending
#                 compared with the average customer (in standard deviations)
# Input         : Dataframe, profile dataframe
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def plot_segment_profiles(dataset, profiles):
    columns = {"Avg_Age": "Age", "Avg_AnnualIncome": "AnnualIncome", "Avg_SpendingScore": "SpendingScore"}
    z_scores = np.column_stack([
        (profiles[column] - dataset[original].mean()) / dataset[original].std(ddof=0)
        for column, original in columns.items()
    ])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(z_scores, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    fig.colorbar(image, label="Standard deviations from the average customer")
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(columns.values()))
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels([f"{r['Segment']} {r['Persona']} (n={r['Customers']})" for _, r in profiles.iterrows()],
                       fontsize=8)
    for i in range(len(profiles)):
        for j, column in enumerate(columns):
            ax.text(j, i, f"{profiles.iloc[i][column]:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(z_scores[i, j]) > 1.2 else "black")
    ax.set_title("Segment Profiles (numbers are averages)")
    finish_plot("segment_profiles.png")


########################################################
# Function name : plot_silhouette
# Description   : Silhouette value of every customer, grouped by segment.
#                 Values near 1 fit their segment well; negative values may
#                 belong to another segment.
# Input         : Fitted pipeline, dataframe, profile dataframe
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def plot_silhouette(pipeline, dataset, profiles):
    scaled = pipeline.named_steps["scaler"].transform(dataset[FEATURES])
    labels = pipeline.named_steps["kmeans"].labels_
    values = silhouette_samples(scaled, labels)
    colors = plt.cm.tab10(np.arange(len(profiles)))

    plt.figure(figsize=(8, 6))
    start = 0
    for position, row in profiles.iterrows():
        cluster_values = np.sort(values[labels == int(row["Cluster"])])
        end = start + len(cluster_values)
        plt.fill_betweenx(np.arange(start, end), 0, cluster_values, color=colors[position])
        plt.text(-0.08, start + len(cluster_values) / 2, row["Segment"], va="center", fontweight="bold")
        start = end + 3

    plt.axvline(values.mean(), linestyle="--", color="red", label=f"average = {values.mean():.3f}")
    plt.xlabel("Silhouette value")
    plt.yticks([])
    plt.title("Silhouette Plot by Segment")
    plt.legend()
    finish_plot("silhouette_plot.png")


########################################################
# Function name : predict_new_customers
# Description   : Assign new customers to a segment. DistanceToCentroid is in
#                 standard-deviation units; customers farther than the
#                 training 95th percentile are flagged as unusual.
# Input         : Trained pipeline, new customers, segment info, distance limit
# Output        : Dataframe with segment, persona and distance
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def predict_new_customers(model, customers, segment_info, distance_limit):
    clusters = model.predict(customers[FEATURES])
    distances = model.transform(customers[FEATURES]).min(axis=1)

    result = customers.copy()
    result["Segment"] = [segment_info[int(c)]["segment"] for c in clusters]
    result["Persona"] = [segment_info[int(c)]["persona"] for c in clusters]
    result["DistanceToCentroid"] = distances.round(2)
    result["Unusual"] = distances > distance_limit
    return result


########################################################
# Function name : save_model / load_model
# Description   : Save or load the trained Pipeline with joblib.
#                 Only load model files you created yourself.
# Input         : Model and path / path
# Output        : Nothing / model
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


########################################################
# Function name : json_default
# Description   : Let json.dump handle numpy numbers
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot save {type(value)} to JSON")


########################################################
# Function name : save_model_metadata
# Description   : Save a "model card" next to the .joblib file: the columns the
#                 model expects, the value ranges it was trained on, what each
#                 cluster means, library versions and quality metrics.
# Output        : outputs/model_metadata.json
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def save_model_metadata(model, dataset, k, segment_info, quality, distance_limit, path=METADATA_PATH):
    metadata = {
        "model_name": "Customer Segmentation K-Means Pipeline",
        "model_class": "Pipeline(StandardScaler, KMeans)",
        "n_clusters": k,
        "n_init": N_INIT,
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "training_customers": int(len(dataset)),
        "features": FEATURES,
        "training_ranges": {c: [float(dataset[c].min()), float(dataset[c].max())] for c in FEATURES},
        "unusual_distance_limit": round(float(distance_limit), 3),
        "segments": {str(cluster): info for cluster, info in segment_info.items()},
        "quality": quality,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=json_default)
    print(f"Model metadata saved to {path}")


########################################################
# Function name : main
# Description   : Main function from where execution starts
# Author        : Prathmesh Sunil Girme
# Date          : 02/11/2026
########################################################
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and understand the dataset
    dataset = read_data(DATA_PATH)
    validate_dataset(dataset)
    print("\n--- 1) Dataset ---")
    dataset_statistics(dataset)

    # 2) Missing values
    print("\n--- 2) Missing values ---")
    check_missing_values(dataset)

    # 3) Exploratory plots
    print("\n--- 3) Exploratory analysis ---")
    plot_eda(dataset)

    # 4) Choose the number of clusters
    print("\n--- 4) Choosing k (elbow, silhouette, Davies-Bouldin, stability) ---")
    k_table = evaluate_k_range(dataset)
    k_table.round(4).to_csv(K_SELECTION_PATH, index=False)
    chosen_k = choose_k(k_table)
    plot_k_selection(k_table, chosen_k)
    print(f"\nChosen k :: {chosen_k}")
    best = k_table.sort_values("Silhouette", ascending=False).iloc[0]
    print(f"Best silhouette alone would be k = {int(best['k'])} ({best['Silhouette']:.3f}); "
          f"k = {chosen_k} is accepted by the written rule (stable, no tiny segment, "
          f"silhouette within {MAX_SILHOUETTE_GAP} of the best).")

    # 5) Train the final pipeline
    print("\n--- 5) Final K-Means pipeline ---")
    pipeline = build_pipeline(chosen_k).fit(dataset[FEATURES])
    scaled = pipeline.named_steps["scaler"].transform(dataset[FEATURES])
    labels = pipeline.named_steps["kmeans"].labels_

    quality = {
        "silhouette": float(silhouette_score(scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
        "stability_ari_mean": float(k_table.loc[k_table["k"] == chosen_k, "Stability_ARI_Mean"].iloc[0]),
        "stability_ari_std": float(k_table.loc[k_table["k"] == chosen_k, "Stability_ARI_Std"].iloc[0]),
        "smallest_segment_share": float(np.bincount(labels).min() / len(labels)),
    }
    for name, value in quality.items():
        print(f"{name:<24} :: {value:.3f}")

    # 6) Compare with other algorithms and check Age as a feature
    print("\n--- 6) Algorithm comparison (same k, same scaled data) ---")
    algorithms = compare_algorithms(dataset, chosen_k, labels)
    algorithms.round(4).to_csv(ALGORITHMS_PATH, index=False)
    print(algorithms.round(3).to_string(index=False))

    age_check = check_age_as_feature(dataset, chosen_k, labels)
    print(f"\nAge as a third feature: silhouette {age_check['silhouette_with_age']:.3f} "
          f"(vs {quality['silhouette']:.3f} without Age), segments agree with the "
          f"2-feature result at ARI {age_check['ari_vs_two_feature_segments']:.3f}.")
    print(f"Age vs SpendingScore correlation :: {age_check['age_spending_correlation']:.2f} "
          f"(Age is kept for describing segments).")
    quality["age_check"] = age_check

    # 7) Profile and name the segments
    print("\n--- 7) Segment profiles ---")
    profiles, segment_info = describe_segments(pipeline, dataset)
    profiles.round(3).to_csv(PROFILES_PATH, index=False)
    print(profiles[["Segment", "Persona", "Customers", "Share", "Avg_Age", "Avg_AnnualIncome",
                    "Avg_SpendingScore"]].round(1).to_string(index=False))
    print("\nMarketing actions:")
    for _, row in profiles.iterrows():
        print(f"  {row['Segment']} {row['Persona']}: {row['Action']}")

    assigned = dataset.copy()
    cluster_to_segment = {c: i["segment"] for c, i in segment_info.items()}
    cluster_to_persona = {c: i["persona"] for c, i in segment_info.items()}
    assigned["Segment"] = [cluster_to_segment[int(c)] for c in labels]
    assigned["Persona"] = [cluster_to_persona[int(c)] for c in labels]
    assigned.to_csv(SEGMENTS_PATH, index=False)
    print(f"\nCustomer segments saved to {SEGMENTS_PATH}")

    # 8) Plots
    print("\n--- 8) Plots ---")
    plot_clusters(pipeline, dataset, profiles)
    plot_segment_profiles(dataset, profiles)
    plot_silhouette(pipeline, dataset, profiles)

    # 9) Segment new customers (illustrative examples)
    print("\n--- 9) New customers (illustrative examples) ---")
    distances = pipeline.transform(dataset[FEATURES]).min(axis=1)
    distance_limit = float(np.percentile(distances, 95))
    new_customers = pd.DataFrame(
        [[24, 22, 85], [50, 120, 45], [45, 78, 55], [52, 25, 8], [33, 140, 95]],
        columns=PROFILE_COLUMNS,
        index=["Customer A", "Customer B", "Customer C", "Customer D", "Customer E (unusual)"],
    )
    print(predict_new_customers(pipeline, new_customers, segment_info, distance_limit).to_string())

    # 10) Save the model and metadata, load the model, and test a sample
    print("\n--- 10) Save and load ---")
    save_model(pipeline, MODEL_PATH)
    save_model_metadata(pipeline, dataset, chosen_k, segment_info, quality, distance_limit)
    loaded = load_model(MODEL_PATH)
    sample = dataset[FEATURES].iloc[[0]]
    cluster = int(loaded.predict(sample)[0])
    print(f"Loaded model: first customer is in {segment_info[cluster]['segment']} "
          f"({segment_info[cluster]['persona']})")

    summary = {
        "chosen_k": chosen_k,
        "quality": quality,
        "k_selection": k_table.to_dict(orient="records"),
        "algorithm_comparison": algorithms.to_dict(orient="records"),
        "segments": segment_info,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)
    print(f"Metrics saved to {METRICS_PATH}")


########################################################
# Application starter
########################################################

if __name__ == "__main__":
    main()
