"""Page 2: machine learning task and target column configuration."""

from __future__ import annotations

import streamlit as st

from utils.session_state import has_dataset, init_state
from utils.api_client import configure_task, APIClientError

st.set_page_config(page_title="Configure Task", layout="wide")
init_state()

st.title("Configure Task")
st.caption("Choose the ML task and parameters that will be sent to POST /train")

if not has_dataset():
	st.warning("Upload a dataset first on the Upload Data page.")
	st.stop()

task_type = st.selectbox(
	"Task type",
	options=["classification", "regression", "clustering"],
	index=["classification", "regression", "clustering"].index(st.session_state.task_type),
)
st.session_state.task_type = task_type


if task_type in {"classification", "regression"}:
	default_target = st.session_state.target if st.session_state.target in st.session_state.columns else None
	target_index = st.session_state.columns.index(default_target) if default_target in st.session_state.columns else 0
	st.session_state.target = st.selectbox("Target column", options=st.session_state.columns, index=target_index)
	#st.session_state.n_clusters = 3
# else:
# 	st.session_state.target = None
# 	st.session_state.n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=int(st.session_state.n_clusters))

# st.subheader("Current Configuration")
# config_preview = {
# 	"task_type": st.session_state.task_type,
# 	"target": st.session_state.target,
# 	"n_clusters": st.session_state.n_clusters,
# }
# st.json(config_preview)

if st.button("Save Configuration", type="primary"):
	try:
		result = configure_task(
			backend_url=st.session_state.backend_url,
			session_id=st.session_state.session_id,
			task_type=st.session_state.task_type,
			target=st.session_state.target,
		)
		config = result.get("configuration", {})
		st.session_state.backend_config = config
		st.session_state.task_type = config.get("task_type", st.session_state.task_type)
		st.session_state.target = config.get("target", st.session_state.target)
		st.success("Configuration saved successfully in backend")

	except APIClientError as exc:
		st.error(str(exc))
