@echo off
set PYTHONIOENCODING=utf-8

echo =======================================================
echo   Starting ASTRA Pipeline...
echo =======================================================
python src\run_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo Pipeline execution failed!
    exit /b %ERRORLEVEL%
)

echo =======================================================
echo   Starting ASTRA FastAPI Server (Background)
echo =======================================================
start /B uvicorn src.api.main:app --reload --port 8000

echo =======================================================
echo   Starting ASTRA Dashboard...
echo =======================================================
streamlit run src\dashboard\app.py