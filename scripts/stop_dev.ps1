# SAFEGEN 3D — Stop Services Script
# Classification: [ENGINEERING ADDITION]

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ProjectRoot ".pids"

if (Test-Path $PidFile) {
    $pids = Get-Content $PidFile | ConvertFrom-Json
    if ($pids.BackendPid) {
        Write-Host "Stopping backend process $($pids.BackendPid)..." -ForegroundColor Yellow
        Stop-Process -Id $pids.BackendPid -Force -ErrorAction SilentlyContinue
    }
    if ($pids.FrontendPid) {
        Write-Host "Stopping frontend process $($pids.FrontendPid)..." -ForegroundColor Yellow
        Stop-Process -Id $pids.FrontendPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# Cleanup port 8000 and 5173 if still bound
Get-Process -Name "*uvicorn*", "*node*" -ErrorAction SilentlyContinue | Where-Object { $_.Path -match "Research paper code" } | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "All SAFEGEN 3D services stopped." -ForegroundColor Green
