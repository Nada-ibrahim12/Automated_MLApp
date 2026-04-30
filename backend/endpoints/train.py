from fastapi import APIRouter
import pandas as pd

from backend.pipelines.preprocessing import build_preprocessing_pipeline, preprocess_data
from backend.models.model_selector import select_and_train_model
from backend.utils.serializer import save_model

router = APIRouter()


@router.post("/train")
def train_model(request: dict):

    data = request.get("data")
    task_type = request["task_type"]
    target = request.get("target")

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return {"error": "'data' must be a list of records or a DataFrame."}

    if df.empty:
        return {"error": "Dataset is empty."}
    
    if task_type in ["classification", "regression"] and not target:
        return {"error": "Target variable is required for classification and regression tasks."}
    elif task_type == "clustering":
        target = None  
        
    try:
        X_train, X_test, y_train, y_test, preprocessing_report = preprocess_data(df, target, task_type=task_type)

        result = select_and_train_model(
            task_type = task_type,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            feature_names = list(X_train.columns),
            original_df=df 
        )
    except ValueError as exc:
        return {"error": str(exc)}

    preprocessing_pipeline = build_preprocessing_pipeline(df, target)
    save_model(
        {
            "model": result["best_model"],
            "preprocessing_pipeline": preprocessing_pipeline,
            "best_model_name": result["best_model_name"],
        }
    )

    return {
    "best_model_name": result["best_model_name"],
    "best_model": result["best_model"].__class__.__name__,
    "best_metrics": result["best_metrics"],
    "metrics": result["best_metrics"],
    "model_runs": result["model_runs"],
    "cluster_descriptions": result.get("cluster_descriptions", {}),
    "preprocessing_report": preprocessing_report,
    "message": "Training completed successfully"
}