Param()

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Python not found at $python"
    Write-Host "Create the virtual environment or install dependencies first."
    exit 1
}

Write-Host "Starting backend on http://127.0.0.1:8000"
Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "backend.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"

Write-Host "Starting frontend on http://127.0.0.1:8501"
Start-Process -FilePath $python -ArgumentList "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.address", "127.0.0.1", "--server.port", "8501"

Write-Host "Both servers started. Open the Streamlit URL in your browser."