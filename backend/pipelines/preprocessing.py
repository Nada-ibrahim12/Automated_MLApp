from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.pipelines.imbalance import balance_training_data
from backend.pipelines.pipeline_builder import build_feature_transformer


@dataclass
class FeaturePreprocessor:

	target_column: str | None = None
	feature_columns: list[str] | None = None
	transformer: object | None = None

	def fit(self, dataframe: pd.DataFrame, target: str | None = None) -> "FeaturePreprocessor":
		self.target_column = target
		features = self._get_features(dataframe, target)
		self.transformer = build_feature_transformer(dataframe, target)
		self.transformer.fit(features)
		self.feature_columns = self.transformer.get_feature_names_out().tolist() if self.transformer is not None else []
		return self

	def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		if self.feature_columns is None or self.transformer is None:
			raise ValueError("Preprocessor has not been fitted yet.")

		features = self._get_features(dataframe, self.target_column)
		transformed = self.transformer.transform(features)
		return pd.DataFrame(transformed, columns=self.feature_columns, index=features.index)

	def fit_transform(self, dataframe: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
		return self.fit(dataframe, target).transform(dataframe)

	@staticmethod
	def _get_features(dataframe: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
		if target and target in dataframe.columns:
			return dataframe.drop(columns=[target])
		return dataframe.copy()


def _to_python_scalar(value):
	if isinstance(value, (np.generic,)):
		return value.item()
	return value

# reporting functions for ui
def _build_imputation_report(features: pd.DataFrame) -> dict:
	numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
	categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()

	column_rules: list[dict] = []
	affected_cells: list[dict] = []

	for column in numeric_columns:
		missing_mask = features[column].isna()
		median_value = _to_python_scalar(features[column].median())
		column_rules.append(
			{
				"column": column,
				"strategy": "median",
				"imputed_value": median_value,
				"missing_count": int(missing_mask.sum()),
			}
		)
		for row_idx in features.index[missing_mask]:
			affected_cells.append(
				{
					"row": int(row_idx) if isinstance(row_idx, (int, np.integer)) else str(row_idx),
					"column": column,
					"step": "missing_value_imputation",
					"strategy": "median",
					"new_value": median_value,
				}
			)

	for column in categorical_columns:
		missing_mask = features[column].isna()
		mode = features[column].mode(dropna=True)
		mode_value = _to_python_scalar(mode.iloc[0] if not mode.empty else "missing")
		column_rules.append(
			{
				"column": column,
				"strategy": "most_frequent",
				"imputed_value": mode_value,
				"missing_count": int(missing_mask.sum()),
			}
		)
		for row_idx in features.index[missing_mask]:
			affected_cells.append(
				{
					"row": int(row_idx) if isinstance(row_idx, (int, np.integer)) else str(row_idx),
					"column": column,
					"step": "missing_value_imputation",
					"strategy": "most_frequent",
					"new_value": mode_value,
				}
			)

	return {
		"column_rules": column_rules,
		"affected_cells": affected_cells,
	}


def _build_encoding_report(features: pd.DataFrame, transformed_columns: list[str]) -> dict:
	categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
	encoded_columns = [name for name in transformed_columns if name.startswith("cat__")]
	column_summaries = []

	for column in categorical_columns:
		categories = sorted(features[column].fillna(features[column].mode(dropna=True).iloc[0] if not features[column].mode(dropna=True).empty else "missing").astype(str).unique().tolist())
		column_summaries.append(
			{
				"column": column,
				"technique": "one_hot_encoding",
				"categories": categories,
				"affected_rows": int(len(features)),
			}
		)

	return {
		"column_summaries": column_summaries,
		"generated_features": encoded_columns,
	}


def _build_scaling_report(features: pd.DataFrame) -> dict:
	numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
	return {
		"column_summaries": [
			{
				"column": column,
				"technique": "standard_scaler",
				"affected_rows": int(len(features)),
			}
			for column in numeric_columns
		]
	}

# actual preprocessing
def preprocess_data(
	df: pd.DataFrame,
	target: str | None = None,
	task_type: str | None = None,
	return_preprocessor: bool = False,
):
	preprocessor = FeaturePreprocessor()

	if target is None:
		features = preprocessor._get_features(df, target)
		imputation_report = _build_imputation_report(features)
		X = preprocessor.fit_transform(df, target)
		result = (
			X,
			X,
			None,
			None,
			{
			"imputation": imputation_report,
			"encoding": _build_encoding_report(features, list(X.columns)),
			"scaling": _build_scaling_report(features),
			"resampling": {
				"applied": False,
				"reason": "No target column provided for resampling.",
			},
			},
		)
		if return_preprocessor:
			return result + (preprocessor,)
		return result

	if target not in df.columns:
		raise ValueError(f"Target column '{target}' not found in dataframe")

	features = preprocessor._get_features(df, target)
	y = df[target]

	stratify_target = y if task_type == "classification" and y.nunique() > 1 else None
	X_train_raw, X_test_raw, y_train, y_test = train_test_split(
		features, y, test_size=0.2, random_state=42, stratify=stratify_target
	)

	preprocessor.fit(X_train_raw)
	X_train = preprocessor.transform(X_train_raw)
	X_test = preprocessor.transform(X_test_raw)

	imputation_report = _build_imputation_report(X_train_raw)

	class_distribution_before = y_train.value_counts(dropna=False).to_dict() if task_type == "classification" else {}
	if task_type == "classification":
		X_train, y_train = balance_training_data(X_train, y_train)
	class_distribution_after = y_train.value_counts(dropna=False).to_dict() if task_type == "classification" else {}

	resampling_applied = bool(class_distribution_before and class_distribution_before != class_distribution_after)
	preprocessing_report = {
		"imputation": imputation_report,
		"encoding": _build_encoding_report(X_train_raw, list(X_train.columns)),
		"scaling": _build_scaling_report(X_train_raw),
		"resampling": {
			"applied": resampling_applied,
			"technique": "random_oversampling" if resampling_applied else None,
			"before_class_counts": {str(k): int(v) for k, v in class_distribution_before.items()},
			"after_class_counts": {str(k): int(v) for k, v in class_distribution_after.items()},
		},
	}

	result = (X_train, X_test, y_train, y_test, preprocessing_report)
	if return_preprocessor:
		return result + (preprocessor,)
	return result


def build_preprocessing_pipeline(df: pd.DataFrame, target: str | None = None) -> FeaturePreprocessor:
	"""Create the fitted preprocessing bundle used for serialization."""
	return FeaturePreprocessor().fit(df, target)
