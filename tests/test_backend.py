import unittest
from unittest.mock import patch

import pandas as pd
from backend.endpoints.train import train_model


class DummyModel:
	pass


class TestTrainEndpoint(unittest.TestCase):
	def setUp(self):
		self.df = pd.DataFrame(
			{
				"feature_a": [1, 2, 3, 4],
				"feature_b": [10, 20, 30, 40],
				"target": [0, 1, 0, 1],
			}
		)
		self.xtr = pd.DataFrame({"f1": [0.1, 0.2], "f2": [1.0, 2.0]})
		self.xte = pd.DataFrame({"f1": [0.3], "f2": [3.0]})

	def test_train_classification_requires_target(self):
		response = train_model(
			{
				"data": self.df,
				"task_type": "classification",
			}
		)

		self.assertEqual(
			response["error"],
			"Target variable is required for classification and regression tasks.",
		)

	def test_train_clustering_success(self):
		dummy_model = DummyModel()
		with patch("backend.endpoints.train.preprocess_data", return_value=(self.xtr, self.xte, None, None, {})) as mocked_preprocess:
			with patch(
				"backend.endpoints.train.select_and_train_model",
				return_value={
					"best_model_name": "K Means",
					"best_model": dummy_model,
					"best_metrics": {
						"silhouette_score": 0.75,
						"labels": [0, 1, 0, 1],
					},
					"model_runs": [],
				},
			) as mocked_select:
				with patch("backend.endpoints.train.save_model") as mocked_save:
					response = train_model(
						{
							"data": self.df,
							"task_type": "clustering",
							"n_clusters": 4,
						}
					)

		payload = response
		self.assertEqual(payload["best_model"], "DummyModel")
		self.assertEqual(payload["message"], "Training completed successfully")
		self.assertIn("silhouette_score", payload["metrics"])
		self.assertNotIn("model", payload["metrics"])

		mocked_preprocess.assert_called_once()
		mocked_select.assert_called_once_with(
			task_type="clustering",
			X_train=self.xtr,
			X_test=self.xte,
			y_train=None,
			y_test=None,
			n_clusters=4,
			feature_names=["f1", "f2"],
		)
		saved_payload = mocked_save.call_args[0][0]
		self.assertEqual(saved_payload["model"], dummy_model)

	def test_train_classification_success(self):
		dummy_model = DummyModel()
		with patch("backend.endpoints.train.preprocess_data", return_value=(self.xtr, self.xte, "ytr", "yte", {})):
			with patch(
				"backend.endpoints.train.select_and_train_model",
				return_value={
					"best_model_name": "logistic_regression",
					"best_model": dummy_model,
					"best_metrics": {
						"accuracy": 0.9,
						"f1": 0.89,
					},
					"model_runs": [],
				},
			):
				with patch("backend.endpoints.train.save_model"):
					response = train_model(
						{
							"data": self.df,
							"task_type": "classification",
							"target": "target",
						}
					)

		payload = response
		self.assertEqual(payload["best_model"], "DummyModel")
		self.assertEqual(payload["metrics"]["accuracy"], 0.9)

	def test_train_accepts_list_payload(self):
		dummy_model = DummyModel()
		with patch("backend.endpoints.train.preprocess_data", return_value=(self.xtr, self.xte, "ytr", "yte", {})) as mocked_preprocess:
			with patch(
				"backend.endpoints.train.select_and_train_model",
				return_value={
					"best_model_name": "logistic_regression",
					"best_model": dummy_model,
					"best_metrics": {
						"accuracy": 0.8,
					},
					"model_runs": [],
				},
			):
				with patch("backend.endpoints.train.save_model"):
					response = train_model(
						{
							"data": self.df.to_dict(orient="records"),
							"task_type": "classification",
							"target": "target",
						}
					)

		self.assertEqual(response["message"], "Training completed successfully")
		called_df = mocked_preprocess.call_args[0][0]
		self.assertIsInstance(called_df, pd.DataFrame)


if __name__ == "__main__":
	unittest.main()
