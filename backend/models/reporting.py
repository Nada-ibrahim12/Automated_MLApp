from __future__ import annotations


def extract_feature_importance(model, feature_names, top_n: int = 10):
	if not feature_names or (not hasattr(model, "coef_") and not hasattr(model, "feature_importances_")):
		return []

	if hasattr(model, "feature_importances_"):
		importances = list(model.feature_importances_)
	else:
		coefficients = model.coef_
		if len(getattr(coefficients, "shape", ())) > 1:
			importances = list(abs(coefficients).mean(axis=0))
		else:
			importances = list(abs(coefficients))

	ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
	return [{"feature": feature, "importance": float(score)} for feature, score in ranked[:top_n]]