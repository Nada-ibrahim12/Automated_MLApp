"""FastAPI backend for uploading data and training a model."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.endpoints.download import router as download_router
from backend.endpoints.train import router as train_router
from backend.utils.session_manager import create_session, get_session, session_snapshot, configure_task_logic

app = FastAPI(title="Automated ML App")
app.include_router(train_router)
app.include_router(download_router)


@app.get("/")
def health() -> dict[str, str]:
	return {"status": "ok", "message": "backend is running"}


def _read_dataframe(file: UploadFile) -> pd.DataFrame:
	extension = Path(file.filename or "").suffix.lower()
	raw_bytes = file.file.read()

	if extension == ".csv":
		return pd.read_csv(BytesIO(raw_bytes))
	if extension in {".tsv", ".txt"}:
		return pd.read_csv(BytesIO(raw_bytes), sep="\t")
	if extension == ".json":
		return pd.read_json(BytesIO(raw_bytes))
	if extension == ".xlsx":
		return pd.read_excel(BytesIO(raw_bytes), engine="openpyxl")
	if extension == ".xls":
		try:
			return pd.read_excel(BytesIO(raw_bytes))
		except ImportError as exc:
			raise HTTPException(
				status_code=400,
				detail=".xls requires an additional Excel engine. Install 'xlrd' to enable this format.",
			) from exc

	raise HTTPException(
		status_code=400,
		detail="Unsupported file type. Supported extensions: .csv, .tsv, .txt, .json, .xlsx, .xls",
	)


@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict:
	df = _read_dataframe(file)
	if df.empty:
		raise HTTPException(status_code=400, detail="Uploaded file is empty")

	session_id = create_session(filename=file.filename or "uploaded_file", dataframe=df)
	preview = df.head(5).fillna("").to_dict(orient="records")

	return {
		"session_id": session_id,
		"filename": file.filename,
		"rows": len(df),
		"columns": df.columns.tolist(),
		"preview": preview,
	}


@app.get("/session/{session_id}")
def get_uploaded_session(session_id: str) -> dict:
	session = get_session(session_id)
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")
	return session_snapshot(session)

@app.post("/configure-task")
def configure_task(request: dict):
    session_id = request.get("session_id")
    task_type = request.get("task_type")
    target = request.get("target")

    try:
        config = configure_task_logic(session_id, task_type, target)
        session = get_session(session_id)
        return {
            "message": "Configuration saved",
            "configuration": config,
            "columns": session["columns"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

