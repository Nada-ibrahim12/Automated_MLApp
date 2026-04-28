from __future__ import annotations

import streamlit as st


DEFAULT_STATE = {
	"backend_url": "http://127.0.0.1:8000",
	"session_id": None,
	"filename": None,
	"rows": 0,
	"columns": [],
	"preview": [],
	"data_records": [],
	"task_type": "classification",
	"target": None,
	"n_clusters": 3,
	"training_result": None,
	"model_artifact_bytes": None,
	"model_artifact_name": "trained_model.pkl",
	"backend_config": None,
}


def init_state() -> None:
	for key, value in DEFAULT_STATE.items():
		if key not in st.session_state:
			st.session_state[key] = value


def has_dataset() -> bool:
	return bool(st.session_state.get("session_id"))


def clear_training_result() -> None:
	st.session_state.training_result = None
