#!/usr/bin/env bash
echo "🚀 Launching CampusCopilot Agentic RAG Platform..."
python3 -m venv venv 2>/dev/null
source venv/bin/activate 2>/dev/null
pip install -r requirements.txt
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
