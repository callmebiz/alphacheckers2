# Run from project root with: .\run.ps1
# Requires conda activate alphacheckers2 first (or conda init powershell if first time)

$env_python = "$env:CONDA_PREFIX\python.exe"
if (-not (Test-Path $env_python)) {
    # Fallback: use whichever python is on PATH
    $env_python = "python"
}

Write-Host "Starting AlphaCheckers server at http://localhost:8000" -ForegroundColor Cyan
Start-Process "http://localhost:8000"
& $env_python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
