$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $ProjectRoot
$env:HF_HUB_OFFLINE = "1"

Write-Host ""
Write-Host "Starting SHL Assessment Recommender FastAPI service..." -ForegroundColor Cyan
Write-Host "UI: http://localhost:8000/" -ForegroundColor Green
Write-Host "API docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
