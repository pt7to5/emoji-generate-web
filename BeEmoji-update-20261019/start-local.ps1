$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $projectRoot "backend"
$frontendRoot = Join-Path $projectRoot "frontend"
$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "logs"
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$backendLog = Join-Path $logRoot "backend-$runId.log"
$backendErrorLog = Join-Path $logRoot "backend-error-$runId.log"
$frontendLog = Join-Path $logRoot "frontend-$runId.log"
$frontendErrorLog = Join-Path $logRoot "frontend-error-$runId.log"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Stop-ProjectPort([int]$port) {
    $processIds = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    if (-not $processIds) {
        $processIds = @(netstat -ano -p tcp 2>$null |
            Select-String -Pattern "^\s*TCP\s+\S+:$port\s+\S+\s+LISTENING\s+\d+\s*$" |
            ForEach-Object { [int](($_.Line -split '\s+')[-1]) } |
            Select-Object -Unique)
    }
    foreach ($processId in $processIds) {
        if ($processId -and $processId -ne $PID) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
    $deadline = (Get-Date).AddSeconds(8)
    while ((Get-Date) -lt $deadline) {
        $occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if (-not $occupied) { return }
        Start-Sleep -Milliseconds 250
    }
    throw "Port $port is still occupied after stopping its previous listener."
}

function Wait-Url([string]$url, [int]$seconds = 25) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) { return $true }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    return $false
}

if (-not (Test-Path $python)) {
    throw "Backend Python environment is missing: $python"
}

# A copied virtual environment can keep an absolute launcher path to a Python
# installation that no longer exists. Probe it before startup and fall back to
# an available Codex Python while continuing to use this project's packages.
$pythonWorks = $false
try {
    & $python -c "import sys" 2>$null
    $pythonWorks = ($LASTEXITCODE -eq 0)
} catch {}
if (-not $pythonWorks) {
    $runtimeRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes"
    $fallback = Get-ChildItem -LiteralPath $runtimeRoot -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\dependencies\python\python.exe" } |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $fallback) {
        throw "The project virtual environment points to a missing Python installation. Recreate backend\.venv with Python 3.12."
    }
    $python = $fallback
    $env:PYTHONPATH = Join-Path $backendRoot ".venv\Lib\site-packages"
    Write-Host "Using fallback Python because backend\.venv launcher is stale." -ForegroundColor Yellow
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js first."
}

Write-Host "Stopping existing services..." -ForegroundColor Cyan
Stop-ProjectPort 8000
Stop-ProjectPort 5173
Start-Sleep -Milliseconds 700

Write-Host "Starting backend..." -ForegroundColor Cyan
Start-Process -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $backendRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError $backendErrorLog

$backendReady = Wait-Url "http://127.0.0.1:8000/api/health" 40
if (-not $backendReady) {
    throw "Backend did not become ready. See $backendErrorLog"
}

Write-Host "Starting frontend after backend is ready..." -ForegroundColor Cyan
Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host=127.0.0.1", "--port=5173", "--strictPort") `
    -WorkingDirectory $frontendRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $frontendLog `
    -RedirectStandardError $frontendErrorLog

$frontendReady = Wait-Url "http://127.0.0.1:5173/"
$providersReady = $false
if ($backendReady) {
    try {
        $providerHealth = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health/providers" -TimeoutSec 25
        $providersReady = [bool]$providerHealth.ok
    } catch {}
}

Write-Host ""
if ($backendReady -and $frontendReady -and $providersReady) {
    Write-Host "Ready: frontend, backend, and image providers are available." -ForegroundColor Green
    Write-Host "Open: http://127.0.0.1:5173/"
    Start-Process "http://127.0.0.1:5173/"
} else {
    Write-Host "Startup did not fully complete:" -ForegroundColor Red
    Write-Host "Frontend: $frontendReady  Backend: $backendReady  Providers: $providersReady"
    Write-Host "Backend log: $backendErrorLog"
    Write-Host "Frontend log: $frontendErrorLog"
}

Write-Host "Press Enter to close this window. Services will keep running."
Read-Host | Out-Null
