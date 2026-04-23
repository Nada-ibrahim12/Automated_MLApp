import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_regression

from backend.models.classification import classification_models
from backend.models.clustering import clustering_models
from backend.models.model_selector import select_and_train_model
from backend.models.regression import regression_models


class TestModelSelector(unittest.TestCase):
	def test_dispatches_to_classification(self):
		with patch("backend.models.model_selector.classification_models", return_value={"ok": True}) as mocked:
			result = select_and_train_model("classification", "xtr", "xte", "ytr", "yte", 3)

		self.assertEqual(result, {"ok": True})
		mocked.assert_called_once_with("xtr", "ytr", "xte", "yte")

	def test_dispatches_to_regression(self):
		with patch("backend.models.model_selector.regression_models", return_value={"ok": True}) as mocked:
			result = select_and_train_model("regression", "xtr", "xte", "ytr", "yte", 3)

		self.assertEqual(result, {"ok": True})
		mocked.assert_called_once_with("xtr", "ytr", "xte", "yte")

	def test_dispatches_to_clustering(self):
		with patch("backend.models.model_selector.clustering_models", return_value={"ok": True}) as mocked:
			result = select_and_train_model("clustering", "xtr", "xte", "ytr", "yte", 5)

		self.assertEqual(result, {"ok": True})
		mocked.assert_called_once_with("xtr", 5)

	def test_raises_for_unsupported_task(self):
		with self.assertRaises(ValueError):
			select_and_train_model("time_series", None, None, None, None, 3)


class TestModelImplementations(unittest.TestCase):
	def test_classification_models_returns_metrics(self):
		X, y = make_classification(
			n_samples=80,
			n_features=5,
			n_informative=3,
			n_redundant=0,
			random_state=42,
		)
		X_train = pd.DataFrame(X[:60])
		X_test = pd.DataFrame(X[60:])
		y_train = pd.Series(y[:60])
		y_test = pd.Series(y[60:])

		result = classification_models(X_train, y_train, X_test, y_test)

		self.assertIn("model", result)
		self.assertIn("accuracy", result)
		self.assertIn("f1", result)
		self.assertIn("precision", result)
		self.assertIn("recall", result)
		self.assertIn("confusion_matrix", result)
		self.assertIn("classification_report", result)

	def test_regression_models_returns_metrics(self):
		X, y = make_regression(n_samples=80, n_features=4, noise=0.1, random_state=42)
		X_train = pd.DataFrame(X[:60])
		X_test = pd.DataFrame(X[60:])
		y_train = pd.Series(y[:60])
		y_test = pd.Series(y[60:])

		result = regression_models(X_train, y_train, X_test, y_test)

		self.assertIn("model", result)
		self.assertIn("mae", result)
		self.assertIn("mse", result)
		self.assertIn("r2", result)

	def test_clustering_models_returns_metrics(self):
		X, _ = make_blobs(n_samples=60, centers=3, n_features=2, random_state=42)
		X_df = pd.DataFrame(X)

		result = clustering_models(X_df, n_clusters=3)

		self.assertIn("model", result)
		self.assertIn("silhouette_score", result)
		self.assertIn("labels", result)
		self.assertEqual(len(result["labels"]), 60)

	def test_clustering_models_handles_large_requested_clusters(self):
		X, _ = make_blobs(n_samples=8, centers=2, n_features=2, random_state=42)
		X_df = pd.DataFrame(X)

		result = clustering_models(X_df, n_clusters=50)

		self.assertIn("best_model", result)
		self.assertIn("best_metrics", result)


if __name__ == "__main__":
	unittest.main()
