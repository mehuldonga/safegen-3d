# SAFEGEN 3D - Unified Development Launcher for Windows
# Classification: [ENGINEERING ADDITION]

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " SAFEGEN 3D - Safe Generative Planning & 3D Digital Twin" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 1. Check Python Virtual Environment
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[1/3] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Green
$BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "[2/3] Starting Vite + React + Three.js Frontend on http://localhost:5173..." -ForegroundColor Green
Set-Location (Join-Path $ProjectRoot "app\frontend")
$FrontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -PassThru -NoNewWindow

Set-Location $ProjectRoot
Write-Host "[3/3] System Online!" -ForegroundColor Cyan
Write-Host "   -> Frontend UI: http://localhost:5173" -ForegroundColor Yellow
Write-Host "   -> API Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "   -> Press Ctrl+C or run scripts/stop_dev.ps1 to terminate." -ForegroundColor Gray

# Save PIDs to .pids file for clean shutdown
@{
    BackendPid = $BackendProcess.Id
    FrontendPid = $FrontendProcess.Id
} | ConvertTo-Json | Set-Content (Join-Path $ProjectRoot ".pids")
