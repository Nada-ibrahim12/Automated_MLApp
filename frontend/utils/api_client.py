from __future__ import annotations

from typing import Any

import requests


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class APIClientError(Exception):
	"""Raised when a backend request fails."""

def normalize_backend_url(url: str) -> str:
	return (url or DEFAULT_BACKEND_URL).strip().rstrip("/")


def _extract_error_message(response: requests.Response) -> str:
	try:
		payload = response.json()
	except Exception:
		return response.text or f"HTTP {response.status_code}"

	if isinstance(payload, dict):
		if "detail" in payload:
			return str(payload["detail"])
		if "error" in payload:
			return str(payload["error"])
	return str(payload)


def _handle_response(response: requests.Response) -> dict[str, Any]:
	if not response.ok:
		message = _extract_error_message(response)
		raise APIClientError(f"Request failed ({response.status_code}): {message}")

	try:
		return response.json()
	except Exception as exc:
		raise APIClientError("Backend returned a non-JSON response") from exc


def health_check(backend_url: str, timeout: int = 10) -> dict[str, Any]:
	response = requests.get(f"{normalize_backend_url(backend_url)}/", timeout=timeout)
	return _handle_response(response)


def upload_file(
	backend_url: str,
	file_name: str,
	file_bytes: bytes,
	mime_type: str = "application/octet-stream",
	timeout: int = 60,
) -> dict[str, Any]:
	files = {"file": (file_name, file_bytes, mime_type)}
	response = requests.post(
		f"{normalize_backend_url(backend_url)}/upload",
		files=files,
		timeout=timeout,
	)
	return _handle_response(response)


def get_session_data(backend_url: str, session_id: str, timeout: int = 15) -> dict[str, Any]:
	response = requests.get(
		f"{normalize_backend_url(backend_url)}/session/{session_id}",
		timeout=timeout,
	)
	return _handle_response(response)


def train_model(
	backend_url: str,
	data_records: list[dict[str, Any]],
	task_type: str,
	target: str | None = None,
	n_clusters: int = 3,
	timeout: int = 120,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"data": data_records,
		"task_type": task_type,
		"n_clusters": int(n_clusters),
	}
	if target:
		payload["target"] = target

	response = requests.post(
		f"{normalize_backend_url(backend_url)}/train",
		json=payload,
		timeout=timeout,
	)
	return _handle_response(response)


def download_model_artifact(backend_url: str, timeout: int = 30) -> tuple[bytes, str]:
	response = requests.get(
		f"{normalize_backend_url(backend_url)}/download-model",
		timeout=timeout,
	)
	if not response.ok:
		message = _extract_error_message(response)
		raise APIClientError(f"Request failed ({response.status_code}): {message}")

	content_disposition = response.headers.get("content-disposition", "")
	default_name = "trained_model.pkl"
	filename = default_name
	if "filename=" in content_disposition:
		filename = content_disposition.split("filename=", 1)[1].strip().strip('"') or default_name

	return response.content, filename
