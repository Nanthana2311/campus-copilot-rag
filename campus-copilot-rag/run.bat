@echo off
echo ==============================================
echo 🎓 Launching CampusCopilot Agentic RAG Platform
echo ==============================================

IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Launching FastAPI SSE Server on http://localhost:8000 ...
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

pause
