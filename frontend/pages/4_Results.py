from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils.api_client import APIClientError, download_model_artifact
from utils.session_state import init_state

st.set_page_config(page_title="Results", layout="wide")
init_state()

st.title("Results")
st.caption("Displays the final best-model output returned by backend endpoint: POST /train")

result = st.session_state.training_result
if not result:
	st.info("No training result yet. Complete training on the Train Model page.")
	st.stop()

st.success(result.get("message", "Training completed"))
st.subheader("Best Model")
st.write(result.get("best_model_name", result.get("best_model", "Unknown")))

st.subheader("Save Model")
if st.button("Save Model"):
	try:
		artifact_bytes, artifact_name = download_model_artifact(st.session_state.backend_url)
		st.session_state.model_artifact_bytes = artifact_bytes
		st.session_state.model_artifact_name = artifact_name
		st.success("Model artifact fetched successfully")
	except APIClientError as exc:
		st.error(str(exc))

if st.session_state.model_artifact_bytes:
	st.download_button(
		label="Download .joblib File",
		data=st.session_state.model_artifact_bytes,
		file_name=st.session_state.model_artifact_name,
		mime="application/octet-stream",
	)

metrics = result.get("best_metrics") or result.get("metrics", {})
if not metrics:
	st.warning("No metrics returned from backend.")
	st.stop()

scalar_metrics = {}
other_metrics = {}

for key, value in metrics.items():
	if isinstance(value, (int, float)):
		scalar_metrics[key] = value
	else:
		other_metrics[key] = value

if scalar_metrics:
	st.subheader("Main Metrics")
	cols = st.columns(min(4, len(scalar_metrics)))
	for idx, (name, value) in enumerate(scalar_metrics.items()):
		cols[idx % len(cols)].metric(name.upper(), f"{value:.4f}")

def _render_confusion_matrix(matrix):
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.heatmap(pd.DataFrame(matrix), annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
	ax.set_title("Confusion Matrix")
	ax.set_xlabel("Predicted")
	ax.set_ylabel("Actual")
	st.pyplot(fig, clear_figure=True)


def _render_feature_importance(dataframe):
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.barplot(data=dataframe, x="importance", y="feature", ax=ax, color="#4C78A8")
	ax.set_title("Feature Importance")
	ax.set_xlabel("Importance")
	ax.set_ylabel("Feature")
	st.pyplot(fig, clear_figure=True)


def _render_regression_scatter(actual_values, predicted_values):
	fig, ax = plt.subplots(figsize=(5, 4))
	ax.scatter(actual_values, predicted_values, alpha=0.75, color="#F58518")
	min_value = min(min(actual_values), min(predicted_values))
	max_value = max(max(actual_values), max(predicted_values))
	ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
	ax.set_title("Actual vs Predicted")
	ax.set_xlabel("Actual")
	ax.set_ylabel("Predicted")
	st.pyplot(fig, clear_figure=True)


def _render_cluster_counts(cluster_counts):
	fig, ax = plt.subplots(figsize=(5, 4))
	cluster_df = pd.DataFrame(cluster_counts)
	sns.barplot(data=cluster_df, x="cluster", y="count", ax=ax, color="#54A24B")
	ax.set_title("Cluster Counts")
	ax.set_xlabel("Cluster")
	ax.set_ylabel("Count")
	st.pyplot(fig, clear_figure=True)

if other_metrics:
	st.subheader("Detailed Metrics")
	for key, value in other_metrics.items():
		st.write(f"{key}:")
		if isinstance(value, list):
			try:
				st.dataframe(pd.DataFrame(value), use_container_width=True)
			except Exception:
				st.write(value)
		else:
			st.json(value)

if metrics.get("confusion_matrix"):
	st.subheader("Classification Visual")
	_render_confusion_matrix(metrics["confusion_matrix"])

if metrics.get("feature_importance"):
	st.subheader("Top Features")
	_render_feature_importance(pd.DataFrame(metrics["feature_importance"]))

if metrics.get("actual_values") and metrics.get("predicted_values"):
	st.subheader("Regression Visual")
	_render_regression_scatter(metrics["actual_values"], metrics["predicted_values"])

if metrics.get("cluster_counts"):
	st.subheader("Clustering Visual")
	_render_cluster_counts(metrics["cluster_counts"])
