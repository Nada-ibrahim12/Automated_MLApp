from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def _cluster_counts(labels):
    counts = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return [{"cluster": cluster, "count": count} for cluster, count in sorted(counts.items())]


def _safe_silhouette_score(X, labels):
    unique_labels = set(labels)
    n_labels = len(unique_labels)
    n_samples = len(labels)

    # Silhouette score is only valid when 2 <= n_labels <= n_samples - 1.
    if n_labels < 2 or n_labels >= n_samples:
        return -1

    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1


def clustering_models(X, n_clusters=3, feature_names=None):
    if len(X) < 2:
        raise ValueError("Clustering requires at least 2 rows.")

    safe_n_clusters = max(2, min(int(n_clusters), len(X)))

    models = {
        "K Means": KMeans(n_clusters=safe_n_clusters, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
    }

    results = {}

    for name, model in models.items():
        try:
            model.fit(X)
            labels = model.labels_
            score = _safe_silhouette_score(X, labels)
        except Exception:
            # Skip models that fail on edge-case datasets and keep evaluation going.
            continue

        results[name] = {
            "model": model,
            "silhouette_score": score,
            "labels": labels.tolist(),
            "cluster_counts": _cluster_counts(labels),
        }
    
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print("\n")
        
    print("Model Comparison Summary:")
    for name, acc in sorted(results.items(), key=lambda item: item[1]["silhouette_score"], reverse=True):
        print(f"{name}: Silhouette Score = {acc['silhouette_score']:.4f}")
        
        
    if not results:
        raise ValueError("No clustering model could be trained on the provided dataset.")

    best_model_name = max(results, key=lambda x: results[x]["silhouette_score"])
    best_model = results[best_model_name]

    return {
        "best_model_name": best_model_name,
        "best_model": best_model["model"],
        "best_metrics": {k: v for k, v in best_model.items() if k != "model"},
        "model_runs": [
            {"name": name, "metrics": {k: v for k, v in metrics.items() if k != "model"}}
            for name, metrics in results.items()
        ],
    }