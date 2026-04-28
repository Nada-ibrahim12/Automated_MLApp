"""temp in-memory storage for uploaded datasets and trained models."""
from __future__ import annotations

from uuid import uuid4

import pandas as pd

SESSIONS: dict[str, dict] = {}


def create_session(filename: str, dataframe: pd.DataFrame) -> str:
	session_id = str(uuid4())
	SESSIONS[session_id] = {
		"filename": filename,
		"rows": len(dataframe),
		"columns": dataframe.columns.tolist(),
		"preview": dataframe.head(5).fillna("").to_dict(orient="records"),
		"dataframe": dataframe,
		"configuration": {},
		"training": {},
	}
	return session_id


def get_session(session_id: str) -> dict | None:
	return SESSIONS.get(session_id)


def update_session(session_id: str, **updates: object) -> dict:
	session = SESSIONS.get(session_id)
	if session is None:
		raise KeyError(session_id)
	session.update(updates)
	return session


def session_snapshot(session: dict) -> dict:
	return {
		"filename": session.get("filename"),
		"rows": session.get("rows"),
		"columns": session.get("columns", []),
		"preview": session.get("preview", []),
		"configuration": session.get("configuration", {}),
		"training": session.get("training", {}),
	}


def configure_task_logic(session_id: str, task_type: str, target: str | None) -> dict:
    session = get_session(session_id)
    if not session:
        raise KeyError("Session not found")

    df = session["dataframe"]
    if target and target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the dataset.")

    session["configuration"] = {
        "task_type": task_type,
        "target": target
    }
    return session["configuration"]
