# SAFEGEN 3D — Autonomous Requirement Verification Runner
# Classification: [ENGINEERING ADDITION]

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " SAFEGEN 3D Verification Agent" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

& $PythonExe verification/verification_agent.py

Write-Host "Reports generated in verification/report.md and verification/results.json" -ForegroundColor Green
