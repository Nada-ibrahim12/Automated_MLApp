from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from utils.api_client import APIClientError, upload_file
from utils.session_state import clear_training_result, init_state

st.set_page_config(page_title="Upload Data", layout="wide")
init_state()

st.title("Upload Dataset")
st.caption("This page sends your selected file to backend endpoint: POST /upload")

with st.sidebar:
	st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
	st.info(f"Current backend: {st.session_state.backend_url}")

uploaded_file = st.file_uploader(
	"Choose a dataset file",
	type=["csv", "tsv", "txt", "json", "xlsx", "xls"],
)

if st.button("Upload to Backend", type="primary", disabled=uploaded_file is None):
	try:
		file_bytes = uploaded_file.getvalue()
		file_name = uploaded_file.name.lower()
		if file_name.endswith(".csv"):
			dataframe = pd.read_csv(BytesIO(file_bytes))
		elif file_name.endswith((".tsv", ".txt")):
			dataframe = pd.read_csv(BytesIO(file_bytes), sep="\t")
		elif file_name.endswith(".json"):
			dataframe = pd.read_json(BytesIO(file_bytes))
		elif file_name.endswith(".xls"):
			dataframe = pd.read_excel(BytesIO(file_bytes))
		else:
			dataframe = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

		result = upload_file(
			backend_url=st.session_state.backend_url,
			file_name=uploaded_file.name,
			file_bytes=file_bytes,
			mime_type=uploaded_file.type or "application/octet-stream",
		)
		st.session_state.session_id = result.get("session_id")
		st.session_state.filename = result.get("filename")
		st.session_state.rows = result.get("rows", 0)
		st.session_state.columns = result.get("columns", [])
		st.session_state.preview = result.get("preview", [])
		st.session_state.data_records = dataframe.to_dict(orient="records")
		clear_training_result()

		st.success("Upload completed successfully")
	except APIClientError as exc:
		st.error(str(exc))
	except Exception as exc:
		st.error(f"Failed to read the selected file: {exc}")

if st.session_state.session_id:
	c1, c2, c3 = st.columns(3)
	c1.metric("Rows", st.session_state.rows)
	c2.metric("Columns", len(st.session_state.columns))
	c3.metric("Session", st.session_state.session_id[:8])

	st.subheader("Columns")
	st.write(st.session_state.columns)

	if st.session_state.preview:
		st.subheader("Data Preview")
		st.dataframe(st.session_state.preview, use_container_width=True)
