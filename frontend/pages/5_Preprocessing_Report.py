from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.session_state import init_state

st.set_page_config(page_title="Preprocessing Report", layout="wide")
init_state()

st.title("Preprocessing Report")
st.caption("Shows what preprocessing was applied and which cells/columns were affected.")

result = st.session_state.get("training_result")
if not result:
	st.info("No training result yet. Train a model first on the Train Model page.")
	st.stop()

report = result.get("preprocessing_report")
if not report:
	st.warning("No preprocessing report returned by backend. Re-run training to generate it.")
	st.stop()

imputation = report.get("imputation", {})
encoding = report.get("encoding", {})
scaling = report.get("scaling", {})
resampling = report.get("resampling", {})

st.subheader("1. Missing Values Handling")
column_rules = imputation.get("column_rules", [])
affected_cells = imputation.get("affected_cells", [])

if column_rules:
	st.write("Column-level strategy")
	st.dataframe(pd.DataFrame(column_rules), use_container_width=True)
else:
	st.info("No imputation rules were generated.")

if affected_cells:
	st.write("Affected cells (row and column)")
	st.dataframe(pd.DataFrame(affected_cells), use_container_width=True)
else:
	st.success("No missing cells required imputation.")

st.subheader("2. Categorical Encoding")
encoding_summary = encoding.get("column_summaries", [])
if encoding_summary:
	st.dataframe(pd.DataFrame(encoding_summary), use_container_width=True)
else:
	st.info("No categorical columns were encoded.")

generated_features = encoding.get("generated_features", [])
if generated_features:
	with st.expander("Generated encoded features"):
		st.write(generated_features)

st.subheader("3. Numeric Scaling")
scaling_summary = scaling.get("column_summaries", [])
if scaling_summary:
	st.dataframe(pd.DataFrame(scaling_summary), use_container_width=True)
else:
	st.info("No numeric columns were scaled.")

st.subheader("4. Target Imbalance Resampling")
if not resampling:
	st.info("No resampling information returned.")
else:
	st.json(resampling)
	if resampling.get("applied"):
		st.success("Resampling was applied to training target classes.")
	else:
		st.info("Resampling was not applied.")
