from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd


def _cluster_counts(labels):
    counts = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1

    return [
        {"cluster": k, "count": v}
        for k, v in sorted(counts.items())
    ]


def _safe_silhouette_score(X, labels):
    unique = set(labels)

    if len(unique) < 2 or len(unique) >= len(labels):
        return -1

    try:
        return float(silhouette_score(X, labels))
    except:
        return -1


def find_best_k(X, max_k=10):
    best_k = 2
    best_score = -1

    max_k = min(max_k, len(X) - 1)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = _safe_silhouette_score(X, labels)

        print(f"K={k}, Silhouette Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def auto_describe_clusters(df: pd.DataFrame, labels):
    df_copy = df.copy()

    df_copy["Cluster"] = labels

    numeric_cols = df_copy.select_dtypes(include="number").columns.tolist()

    if "Cluster" in numeric_cols:
        numeric_cols.remove("Cluster")

    overall_mean = df[numeric_cols].mean()
    cluster_mean = df_copy.groupby("Cluster")[numeric_cols].mean()

    descriptions = {}

    for cluster_id, row in cluster_mean.iterrows():
        desc = []

        for col in numeric_cols:
            if overall_mean[col] == 0:
                diff = 0
            else:
                diff = (row[col] - overall_mean[col]) / overall_mean[col]

            if diff > 0.15:
                desc.append(f"high {col}")
            elif diff < -0.15:
                desc.append(f"low {col}")
            else:
                desc.append(f"average {col}")

        descriptions[int(cluster_id)] = desc

    return descriptions


def clustering_models(X, original_df, feature_names=None):
    if len(X) < 2:
        raise ValueError("Need at least 2 rows")

    # Find best K
    best_k, best_k_score = find_best_k(X)

    print(f"\nBest K = {best_k} (score={best_k_score:.4f})\n")

    models = {
        "KMeans": KMeans(n_clusters=best_k, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
    }

    results = {}

    for name, model in models.items():
        model.fit(X)
        labels = model.labels_

        score = _safe_silhouette_score(X, labels)

        results[name] = {
            "model": model,
            "silhouette_score": score,
            "labels": labels.tolist(),
            "cluster_counts": _cluster_counts(labels),
        }

    print("\nModel Comparison:")
    for name, r in results.items():
        print(f"{name}: {r['silhouette_score']:.4f}")

    # Best model
    best_model_name = max(
        results,
        key=lambda x: results[x]["silhouette_score"]
    )

    best_model = results[best_model_name]

    # Cluster interpretation (IMPORTANT FIX HERE)
    cluster_descriptions = auto_describe_clusters(
        original_df,
        best_model["labels"]
    )

    return {
        "best_model_name": best_model_name,
        "best_model": best_model["model"],
        "best_metrics": {
            k: v for k, v in best_model.items()
            if k != "model"
        },
        "cluster_descriptions": cluster_descriptions,
        "model_runs": [
            {
                "name": name,
                "metrics": {
                    k: v for k, v in r.items()
                    if k != "model"
                }
            }
            for name, r in results.items()
        ],
    }