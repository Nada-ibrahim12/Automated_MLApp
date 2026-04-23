from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(dataframe: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
	features = dataframe.drop(columns=[target_column]) if target_column and target_column in dataframe.columns else dataframe.copy()

	numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
	categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()

	transformers: list[tuple[str, Pipeline, list[str]]] = []
	if numeric_columns:
		transformers.append(
			(
				"num",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				numeric_columns,
			)
		)
	if categorical_columns:
		transformers.append(
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
					]
				),
				categorical_columns,
			)
		)

	if not transformers:
		return pd.DataFrame(index=features.index)

	preprocessor = ColumnTransformer(transformers=transformers)
	transformed = preprocessor.fit_transform(features)
	feature_names = preprocessor.get_feature_names_out().tolist()
	return pd.DataFrame(transformed, columns=feature_names, index=features.index)


def build_feature_transformer(dataframe: pd.DataFrame, target_column: str | None = None) -> ColumnTransformer:
	features = dataframe.drop(columns=[target_column]) if target_column and target_column in dataframe.columns else dataframe.copy()

	numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
	categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()

	transformers: list[tuple[str, Pipeline, list[str]]] = []
	if numeric_columns:
		transformers.append(
			(
				"num",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				numeric_columns,
			)
		)
	if categorical_columns:
		transformers.append(
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
					]
				),
				categorical_columns,
			)
		)

	return ColumnTransformer(transformers=transformers)
