#!/usr/bin/env bash
# Start backend and frontend servers for local development.

set -e
cd "$(dirname "$0")"

python -m uvicorn backend.main:app --reload --port 8000 &
streamlit run frontend/streamlit_app.py --server.port 8501

