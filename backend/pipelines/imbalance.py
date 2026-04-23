from __future__ import annotations

import pandas as pd


def is_imbalanced_target(target: pd.Series, threshold_ratio: float = 1.5) -> bool:
	class_counts = target.value_counts(dropna=False)
	if len(class_counts) < 2:
		return False

	majority = class_counts.max()
	minority = class_counts.min()
	if minority == 0:
		return False

	return (majority / minority) >= threshold_ratio


def balance_training_data(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	threshold_ratio: float = 1.5,
) -> tuple[pd.DataFrame, pd.Series]:
	if not is_imbalanced_target(y_train, threshold_ratio=threshold_ratio):
		return X_train, y_train

	combined = X_train.copy()
	combined["__target__"] = y_train.values
	class_counts = combined["__target__"].value_counts(dropna=False)
	max_count = int(class_counts.max())

	balanced_parts: list[pd.DataFrame] = []
	for class_value in class_counts.index:
		class_subset = combined[combined["__target__"] == class_value]
		if len(class_subset) < max_count:
			class_subset = class_subset.sample(n=max_count, replace=True, random_state=42)
		balanced_parts.append(class_subset)

	balanced = pd.concat(balanced_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
	y_balanced = balanced.pop("__target__")
	return balanced, y_balanced
