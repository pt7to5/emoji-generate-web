$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $projectRoot "backend"
$frontendRoot = Join-Path $projectRoot "frontend"
$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "logs"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Stop-ProjectPort([int]$port) {
    $listeners = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        if ($listener.OwningProcess -and $listener.OwningProcess -ne $PID) {
            Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
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
    throw "后端 Python 环境不存在：$python"
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "没有找到 npm，请先安装 Node.js。"
}

Write-Host "正在关闭旧服务..." -ForegroundColor Cyan
Stop-ProjectPort 8000
Stop-ProjectPort 5173
Start-Sleep -Milliseconds 700

Write-Host "正在启动后端..." -ForegroundColor Cyan
Start-Process -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $backendRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logRoot "backend.log") `
    -RedirectStandardError (Join-Path $logRoot "backend-error.log")

Write-Host "正在启动网页..." -ForegroundColor Cyan
Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") `
    -WorkingDirectory $frontendRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logRoot "frontend.log") `
    -RedirectStandardError (Join-Path $logRoot "frontend-error.log")

$backendReady = Wait-Url "http://127.0.0.1:8000/api/health"
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
    Write-Host "启动成功：网页、后端和图片服务均可用。" -ForegroundColor Green
    Write-Host "打开：http://127.0.0.1:5173/"
    Start-Process "http://127.0.0.1:5173/"
} else {
    Write-Host "启动未完全成功：" -ForegroundColor Red
    Write-Host "网页：$frontendReady  后端：$backendReady  第三方图片服务：$providersReady"
    Write-Host "日志目录：$logRoot"
}

Write-Host "按回车关闭此窗口；服务会继续在后台运行。"
Read-Host | Out-Null
