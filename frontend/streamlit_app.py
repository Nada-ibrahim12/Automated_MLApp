"""Minimal Streamlit frontend used only to test the project structure."""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="ML AutoApp Structure Test", layout="wide")
st.title("ML AutoApp - Structure Check")

if "session_id" not in st.session_state:
	st.session_state.session_id = None
if "preview" not in st.session_state:
	st.session_state.preview = []

backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

st.subheader("Backend status")
if st.button("Ping Backend"):
	try:
		response = requests.get(f"{backend_url}/", timeout=20)
		st.write(response.json())
	except Exception as ex:
		st.error(str(ex))

st.subheader("Upload a file")
uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

if st.button("Send File", disabled=uploaded_file is None):
	files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
	try:
		response = requests.post(f"{backend_url}/upload", files=files, timeout=60)
		if response.ok:
			data = response.json()
			st.session_state.session_id = data["session_id"]
			st.session_state.preview = data["preview"]
			st.success(f"Uploaded successfully. Session: {data['session_id']}")
			st.write("Columns:", data["columns"])
			st.write("Rows:", data["rows"])
		else:
			st.error(response.text)
	except Exception as ex:
		st.error(str(ex))

if st.session_state.preview:
	st.subheader("Preview")
	st.dataframe(st.session_state.preview, use_container_width=True)

if st.session_state.session_id:
	st.info(f"Current session: {st.session_state.session_id}")

st.caption("This is only a structure test app, not a full ML project.")
