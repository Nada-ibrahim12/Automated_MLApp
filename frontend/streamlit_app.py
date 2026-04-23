from __future__ import annotations

import streamlit as st

from utils.api_client import APIClientError, health_check
from utils.session_state import has_dataset, init_state

st.set_page_config(page_title="Automated ML App", layout="wide")
init_state()

st.title("Automated ML App")
st.caption("Upload data, configure your task, train a model, and inspect results.")

with st.sidebar:
	st.subheader("Backend")
	st.session_state.backend_url = st.text_input(
		"Base URL",
		value=st.session_state.backend_url,
		help="Example: http://127.0.0.1:8000",
	)

	if st.button("Check Connection", use_container_width=True):
		try:
			status = health_check(st.session_state.backend_url)
			st.success(f"Connected: {status.get('message', 'ok')}")
		except APIClientError as exc:
			st.error(str(exc))

	st.divider()
	if st.session_state.session_id:
		st.write(f"Session: {st.session_state.session_id[:8]}...")
	else:
		st.write("Session: not started")

col1, col2 = st.columns([1.3, 1])
with col1:
	st.subheader("Workflow")
	st.markdown(
		"""
1. Go to **Upload Data** and send your CSV/XLSX to the backend `/upload` endpoint.
2. Use **Configure Task** to choose ML task and target settings.
3. Run **Train Model** to call backend `/train`.
4. Review outcomes in **Results**.
"""
	)

with col2:
	st.subheader("Current Dataset")
	if has_dataset():
		st.success("Dataset uploaded")
		st.write(f"File: {st.session_state.filename}")
		st.write(f"Rows: {st.session_state.rows}")
		st.write(f"Columns: {len(st.session_state.columns)}")
	else:
		st.info("No dataset uploaded yet")

if st.session_state.preview:
	st.subheader("Preview")
	st.dataframe(st.session_state.preview, use_container_width=True)

st.caption("Use the pages in the left sidebar to continue.")
