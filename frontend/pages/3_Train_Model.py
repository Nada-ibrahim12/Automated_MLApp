from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.api_client import APIClientError, get_session_data, train_model
from utils.session_state import has_dataset, init_state

st.set_page_config(page_title="Train Model", layout="wide")
init_state()

st.title("Train Model")
st.caption("This page sends HTTP requests to GET /session/{session_id} and POST /train")

if not has_dataset():
	st.warning("Upload a dataset first on the Upload Data page.")
	st.stop()

st.write(f"Session: {st.session_state.session_id}")
# st.write(f"Task: {st.session_state.task_type}")
config = st.session_state.get("backend_config")
if config is not None:
    st.write(f"Task: {config.get('task_type')}")
else:
    st.error("No config returned from session.")

if st.session_state.task_type in {"classification", "regression"}:
	if config:
		st.write(f"Target: {config.get('target')}")
	else:
		st.error("No configuration found. Please configure task first.")
		st.stop()
	# st.write(f"Target: {st.session_state.target}")
# else:
# 	st.write(f"n_clusters: {st.session_state.n_clusters}")

if st.button("Start Training", type="primary"):
	try:
		if not st.session_state.get("backend_config"):
			st.warning("Please save configuration to backend first.")
			st.stop()
		session_payload = {}
		preview = []
		with st.spinner("Checking backend session..."):
			try:
				session_payload = get_session_data(
					backend_url=st.session_state.backend_url,
					session_id=st.session_state.session_id,
				)
				preview = session_payload.get("preview", [])
				if preview:
					st.info("Session data is accessible from backend.")
			except APIClientError as exc:
				st.error("Backend session is unavailable. Please re-upload your dataset.")
				st.stop()
				# st.warning(
				# 	f"Backend session is unavailable ({exc}). Continuing with local uploaded data."
				# )

		records = st.session_state.get("data_records") or preview
		if not records:
			raise APIClientError("No records available for training; please re-upload your dataset.")

		with st.spinner("Training model..."):
			result = train_model(
				backend_url=st.session_state.backend_url,
				data_records=records,
				task_type=st.session_state.task_type,
				target=st.session_state.target,
				n_clusters=int(st.session_state.n_clusters),
			)

		if result.get("error"):
			st.error(result["error"])
		else:
			st.session_state.training_result = result
			st.success("Training completed")

			st.subheader("Model Comparison")
			model_runs = result.get("model_runs", [])
			if not model_runs:
				st.info("No model comparison data returned from backend.")
			else:
				comparison_rows = []
				for idx, run in enumerate(model_runs, start=1):
					metrics = run.get("metrics", {})
					comparison_rows.append({"model": run.get("name", "model"), **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
					with st.expander(f"{idx}. {run.get('name', 'model')}", expanded=idx == 1):
						st.write("Details")
						st.json(metrics)
						if metrics.get("feature_importance"):
							st.write("Top features")
							st.dataframe(pd.DataFrame(metrics["feature_importance"]), use_container_width=True)
						if metrics.get("classification_report"):
							st.write("Classification report")
							st.json(metrics["classification_report"])
						st.caption("This section lists every model trained for the selected task.")

				if comparison_rows:
					st.write("Comparison table")
					st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

	except APIClientError as exc:
		st.error(str(exc))
