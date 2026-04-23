import unittest

import numpy as np
import pandas as pd

from backend.pipelines.preprocessing import preprocess_data


class TestPreprocessingPipeline(unittest.TestCase):
	def test_handles_missing_encodes_and_scales(self):
		df = pd.DataFrame(
			{
				"num1": [1.0, 2.0, np.nan, 4.0, 5.0],
				"num2": [10.0, np.nan, 30.0, 40.0, 50.0],
				"city": ["A", "B", None, "A", "C"],
				"target": [0, 1, 0, 1, 0],
			}
		)

		X_train, X_test, y_train, y_test, report = preprocess_data(df, target="target", task_type="regression")

		self.assertFalse(X_train.isna().any().any())
		self.assertFalse(X_test.isna().any().any())
		self.assertGreaterEqual(len(X_train.columns), 4)
		self.assertTrue(any(col.startswith("cat__city_") for col in X_train.columns))

		num_cols = [col for col in X_train.columns if col.startswith("num__")]
		for col in num_cols:
			self.assertLess(abs(float(X_train[col].mean())), 1.0)
			self.assertGreater(float(X_train[col].std()), 0.0)

		self.assertIsNotNone(y_train)
		self.assertIsNotNone(y_test)
		self.assertIn("imputation", report)
		self.assertIn("encoding", report)
		self.assertIn("scaling", report)
		self.assertIn("resampling", report)

	def test_balances_imbalanced_target_for_classification(self):
		df = pd.DataFrame(
			{
				"num": list(range(1, 31)),
				"cat": ["x", "y"] * 15,
				"target": [0] * 27 + [1] * 3,
			}
		)

		X_train, _X_test, y_train, _y_test, report = preprocess_data(df, target="target", task_type="classification")

		counts = y_train.value_counts()
		self.assertEqual(counts.iloc[0], counts.iloc[1])
		self.assertEqual(len(X_train), len(y_train))
		self.assertTrue(report["resampling"]["applied"])


if __name__ == "__main__":
	unittest.main()
