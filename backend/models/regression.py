from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.models.reporting import extract_feature_importance


def regression_models(X_train, y_train, X_test, y_test, feature_names=None):
    
    models = {
        "linear_regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "actual_values": list(y_test),
            "predicted_values": list(y_pred),
            "feature_importance": extract_feature_importance(model, feature_names),
        }
        
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"Mean Squared Error: {metrics['mse']:.4f}")
        print(f"R2 Score: {metrics['r2']:.4f}")
        print("\n")
        
    print("Model Comparison Summary:")
    for name, acc in sorted(results.items(), key=lambda item: item[1]["r2"], reverse=True):
        print(f"{name}: R2 Score = {acc['r2']:.4f}")
        
    
    best_model_name = max(results, key=lambda x: results[x]["r2"])
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