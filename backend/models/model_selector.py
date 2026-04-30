from backend.models.clustering import clustering_models
from backend.models.classification import classification_models
from backend.models.regression import regression_models

def select_and_train_model(task_type, X_train, X_test, y_train, y_test,  original_df, feature_names=None):
    
    if task_type == "classification":
        return classification_models(X_train, y_train, X_test, y_test, feature_names=feature_names)
    
    elif task_type == "regression":
        return regression_models(X_train, y_train, X_test, y_test, feature_names=feature_names)
    
    elif task_type == "clustering":
        return clustering_models(X_train, original_df=original_df, feature_names=feature_names) 
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")