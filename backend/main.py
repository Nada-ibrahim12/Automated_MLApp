"""Minimal FastAPI backend used only to test the project structure."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="ML AutoApp Structure Test")

SESSIONS: dict[str, dict] = {}


@app.get("/")
def health() -> dict[str, str]:
	return {"status": "ok", "message": "backend is running"}


def _read_dataframe(file: UploadFile) -> pd.DataFrame:
	extension = Path(file.filename or "").suffix.lower()
	raw_bytes = file.file.read()

	if extension == ".csv":
		return pd.read_csv(BytesIO(raw_bytes))
	if extension == ".xlsx":
		return pd.read_excel(BytesIO(raw_bytes), engine="openpyxl")
	raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported")


@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict:
	df = _read_dataframe(file)
	if df.empty:
		raise HTTPException(status_code=400, detail="Uploaded file is empty")

	session_id = str(uuid4())
	preview = df.head(5).fillna("").to_dict(orient="records")

	SESSIONS[session_id] = {
		"filename": file.filename,
		"rows": len(df),
		"columns": df.columns.tolist(),
		"preview": preview,
	}

	return {
		"session_id": session_id,
		"filename": file.filename,
		"rows": len(df),
		"columns": df.columns.tolist(),
		"preview": preview,
	}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
	session = SESSIONS.get(session_id)
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")
	return session
