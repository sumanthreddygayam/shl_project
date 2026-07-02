$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $ProjectRoot
$env:HF_HUB_OFFLINE = "1"

Write-Host ""
Write-Host "Starting SHL Assessment Recommender Streamlit UI..." -ForegroundColor Cyan
Write-Host "UI: http://localhost:8501/" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

streamlit run "$ProjectRoot\streamlit_app.py"
