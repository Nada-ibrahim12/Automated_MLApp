"""GET /download-model endpoint for exporting saved model artifacts."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/download-model")
def download_model() -> FileResponse:
	model_path = Path("models_saved") / "trained_model.joblib"
	if not model_path.exists():
		raise HTTPException(status_code=404, detail="No trained model found. Train a model first.")

	return FileResponse(
		path=model_path,
		media_type="application/octet-stream",
		filename="trained_model.joblib",
	)
