import joblib
import os

def save_model(model_artifact):
    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(model_artifact, "models_saved/trained_model.joblib")