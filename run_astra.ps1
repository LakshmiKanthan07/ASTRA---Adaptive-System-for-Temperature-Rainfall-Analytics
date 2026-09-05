$env:PYTHONIOENCODING="utf-8"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Starting ASTRA Pipeline..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
python src\run_pipeline.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline execution failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Starting ASTRA FastAPI Server (Background)..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "src.api.main:app --reload --port 8000"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Starting ASTRA Dashboard..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
streamlit run src\dashboard\app.py
