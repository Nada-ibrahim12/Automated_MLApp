from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

from backend.models.reporting import extract_feature_importance


def classification_models(X_train, y_train, X_test, y_test, feature_names=None):
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred), 
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": extract_feature_importance(model, feature_names),
        }
        
    print("Model Comparison Summary:")
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["accuracy"], reverse=True):
        print(f"{name}: {metrics['accuracy']:.4f}")

    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
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