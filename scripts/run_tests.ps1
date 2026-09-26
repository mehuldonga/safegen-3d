# SAFEGEN 3D — Automated Test Suite Runner
# Classification: [ENGINEERING ADDITION]

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Running SAFEGEN 3D Test Suites" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "[1/3] Running ML Tests..." -ForegroundColor Yellow
& $PythonExe -m unittest discover -s tests/ml -p "test_*.py"

Write-Host "[2/3] Running Integration Tests..." -ForegroundColor Yellow
& $PythonExe -m unittest discover -s tests/integration -p "test_*.py"

Write-Host "[3/3] Running API Tests..." -ForegroundColor Yellow
& $PythonExe -m unittest discover -s tests/api -p "test_*.py"

Write-Host "Test runs completed." -ForegroundColor Green
