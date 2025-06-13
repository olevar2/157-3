# Platform3 Communication Bridge Startup Script
# Starts both Python AI Engine API Server and TypeScript Analytics Service

# Get the project root directory (parent of scripts folder)
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot

param(
    [switch]$Development,
    [switch]$Production,
    [string]$PythonPort = "8000",
    [string]$AnalyticsPort = "3007"
)

Write-Host "🚀 Starting Platform3 TypeScript-Python Communication Bridge..." -ForegroundColor Green

# Function to check if port is available
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Check if ports are available
if (Test-Port -Port $PythonPort) {
    Write-Host "⚠️ Port $PythonPort is already in use" -ForegroundColor Yellow
}

if (Test-Port -Port $AnalyticsPort) {
    Write-Host "⚠️ Port $AnalyticsPort is already in use" -ForegroundColor Yellow
}

# Set environment variables
$env:PYTHON_ENGINE_URL = "http://localhost:$PythonPort"
$env:PYTHON_WS_URL = "ws://localhost:$PythonPort/ws"
$env:ANALYTICS_PORT = $AnalyticsPort

if ($Development) {
    $env:NODE_ENV = "development"
    $env:LOG_LEVEL = "debug"
} elseif ($Production) {
    $env:NODE_ENV = "production"
    $env:LOG_LEVEL = "info"
}

Write-Host "📡 Starting Python AI Engine API Server on port $PythonPort..." -ForegroundColor Cyan

# Start Python API Server in background
$pythonJob = Start-Job -ScriptBlock {
    param($PythonPort)
    Set-Location "$PROJECT_ROOT\ai-platform\api-server"
    $env:PYTHON_ENGINE_PORT = $PythonPort
    python start.py
} -ArgumentList $PythonPort

# Wait a moment for Python server to start
Start-Sleep -Seconds 3

Write-Host "🔧 Starting TypeScript Analytics Service on port $AnalyticsPort..." -ForegroundColor Cyan

# Start TypeScript Analytics Service in background  
$analyticsJob = Start-Job -ScriptBlock {
    param($AnalyticsPort)
    Set-Location "$PROJECT_ROOT\services\analytics-service"
    $env:PORT = $AnalyticsPort
    npm start
} -ArgumentList $AnalyticsPort

# Start Trading Service as well
Write-Host "💰 Starting TypeScript Trading Service on port 3006..." -ForegroundColor Cyan

$tradingJob = Start-Job -ScriptBlock {
    Set-Location "$PROJECT_ROOT\services\trading-service"
    $env:PORT = "3006"
    npm start
}

Write-Host ""
Write-Host "✅ Platform3 Communication Bridge Started!" -ForegroundColor Green
Write-Host "📊 Analytics Service: http://localhost:$AnalyticsPort" -ForegroundColor White
Write-Host "💰 Trading Service:   http://localhost:3006" -ForegroundColor White
Write-Host "🧠 Python AI Server: http://localhost:$PythonPort" -ForegroundColor White
Write-Host "🌐 WebSocket:        ws://localhost:$PythonPort/ws" -ForegroundColor White
Write-Host ""
Write-Host "🔍 Monitoring logs... Press Ctrl+C to stop all services" -ForegroundColor Yellow

try {
    # Monitor jobs and display output
    while ($true) {
        # Check Python server job
        if ($pythonJob.State -eq "Failed") {
            Write-Host "❌ Python server failed" -ForegroundColor Red
            Receive-Job $pythonJob
            break
        }
        
        # Check Analytics service job
        if ($analyticsJob.State -eq "Failed") {
            Write-Host "❌ Analytics service failed" -ForegroundColor Red
            Receive-Job $analyticsJob
            break
        }
        
        # Check Trading service job
        if ($tradingJob.State -eq "Failed") {
            Write-Host "❌ Trading service failed" -ForegroundColor Red
            Receive-Job $tradingJob
            break
        }
        
        Start-Sleep -Seconds 2
    }
} catch {
    Write-Host "🛑 Stopping all services..." -ForegroundColor Yellow
} finally {
    # Clean up jobs
    Stop-Job $pythonJob, $analyticsJob, $tradingJob -ErrorAction SilentlyContinue
    Remove-Job $pythonJob, $analyticsJob, $tradingJob -Force -ErrorAction SilentlyContinue
    Write-Host "✅ All services stopped" -ForegroundColor Green
}