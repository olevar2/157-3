# Personal Forex Trading Platform - Development Setup Script (PowerShell)
# This script sets up the complete development environment for the server-based platform

param(
    [switch]$SkipDocker,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Cyan = "Cyan"
}

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Test-Docker {
    Write-Status "Checking Docker installation..."
    
    if (-not (Test-Command "docker")) {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    try {
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is not running. Please start Docker Desktop."
            exit 1
        }
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop."
        exit 1
    }
    
    Write-Success "Docker is installed and running"
}

function Test-DockerCompose {
    Write-Status "Checking Docker Compose..."
    
    $hasDockerCompose = Test-Command "docker-compose"
    $hasDockerComposeV2 = $false
    
    try {
        docker compose version 2>$null
        $hasDockerComposeV2 = $LASTEXITCODE -eq 0
    }
    catch {
        $hasDockerComposeV2 = $false
    }
    
    if (-not $hasDockerCompose -and -not $hasDockerComposeV2) {
        Write-Error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    }
    
    Write-Success "Docker Compose is available"
}

function New-Directories {
    Write-Status "Creating necessary directories..."
    
    $directories = @(
        "logs\gateway",
        "logs\user-service",
        "logs\market-data",
        "logs\trading-service",
        "logs\risk-service",
        "logs\analytics",
        "logs\payment",
        "logs\notifications",
        "data\postgres",
        "data\redis",
        "data\influxdb",
        "data\kafka",
        "data\prometheus",
        "data\grafana",
        "infrastructure\monitoring\grafana\dashboards",
        "infrastructure\monitoring\grafana\datasources",
        "infrastructure\database\schemas"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            if ($Verbose) {
                Write-Host "Created: $dir" -ForegroundColor $Colors.Cyan
            }
        }
    }
    
    Write-Success "Directories created successfully"
}

function Set-Environment {
    Write-Status "Setting up environment configuration..."
    
    if (-not (Test-Path ".env")) {
        Write-Warning ".env file not found. Please create one based on the template."
        return
    }
    
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "ForexSecure2025!") {
        Write-Warning "⚠️  Default passwords detected in .env file!"
        Write-Warning "⚠️  Please change all passwords before deploying to production!"
        Write-Host ""
        Write-Host "To generate secure passwords, you can use:" -ForegroundColor $Colors.Yellow
        Write-Host "  [System.Web.Security.Membership]::GeneratePassword(32, 8)" -ForegroundColor $Colors.Cyan
        Write-Host ""
    }
    
    Write-Success "Environment configuration checked"
}

function Set-PrometheusConfig {
    Write-Status "Setting up Prometheus configuration..."
    
    $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Platform services
  - job_name: 'forex-gateway'
    static_configs:
      - targets: ['gateway-service:3000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'forex-user-service'
    static_configs:
      - targets: ['user-service:3001']
    metrics_path: '/metrics'

  - job_name: 'forex-market-data'
    static_configs:
      - targets: ['market-data-service:3002']
    metrics_path: '/metrics'
    scrape_interval: 1s

  - job_name: 'forex-trading-engine'
    static_configs:
      - targets: ['trading-service:3003']
    metrics_path: '/metrics'
    scrape_interval: 1s

  - job_name: 'forex-risk-service'
    static_configs:
      - targets: ['risk-service:3004']
    metrics_path: '/metrics'
    scrape_interval: 1s

  - job_name: 'forex-analytics'
    static_configs:
      - targets: ['analytics-service:3005']
    metrics_path: '/metrics'

  # Infrastructure services
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']
"@

    $prometheusConfig | Out-File -FilePath "infrastructure\monitoring\prometheus.yml" -Encoding UTF8
    Write-Success "Prometheus configuration created"
}

function Set-GrafanaConfig {
    Write-Status "Setting up Grafana configuration..."
    
    # Datasources
    $datasourceConfig = @"
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"@

    $datasourceConfig | Out-File -FilePath "infrastructure\monitoring\grafana\datasources\prometheus.yml" -Encoding UTF8
    
    # Dashboard provisioning
    $dashboardConfig = @"
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
"@

    $dashboardConfig | Out-File -FilePath "infrastructure\monitoring\grafana\dashboards\dashboard.yml" -Encoding UTF8
    Write-Success "Grafana configuration created"
}

function Start-Platform {
    Write-Status "Starting the Personal Forex Trading Platform..."
    
    # Determine which docker-compose command to use
    $composeCmd = "docker-compose"
    if (Test-Command "docker") {
        try {
            docker compose version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $composeCmd = "docker compose"
            }
        }
        catch {
            # Use docker-compose
        }
    }
    
    # Pull latest images
    Write-Status "Pulling Docker images..."
    Invoke-Expression "$composeCmd pull"
    
    # Start infrastructure services first
    Write-Status "Starting infrastructure services..."
    Invoke-Expression "$composeCmd up -d postgres redis influxdb zookeeper kafka prometheus grafana"
    
    # Wait for databases to be ready
    Write-Status "Waiting for databases to be ready..."
    Start-Sleep -Seconds 30
    
    # Check PostgreSQL readiness
    Write-Status "Checking PostgreSQL readiness..."
    $maxAttempts = 30
    $attempt = 0
    
    do {
        $attempt++
        try {
            $result = Invoke-Expression "docker exec forex-postgres pg_isready -U forex_admin -d forex_trading" 2>$null
            if ($LASTEXITCODE -eq 0) {
                break
            }
        }
        catch {
            # Continue waiting
        }
        
        if ($attempt -lt $maxAttempts) {
            Write-Status "Waiting for PostgreSQL... (attempt $attempt/$maxAttempts)"
            Start-Sleep -Seconds 5
        }
    } while ($attempt -lt $maxAttempts)
    
    if ($attempt -ge $maxAttempts) {
        Write-Warning "PostgreSQL took longer than expected to start"
    } else {
        Write-Success "PostgreSQL is ready!"
    }
    
    Write-Success "Infrastructure services are running!"
    Write-Warning "Application services will be started after implementation in Phase 1F"
    
    Write-Host ""
    Write-Host "🎉 Development environment setup complete!" -ForegroundColor $Colors.Green
    Write-Host ""
    Write-Host "📊 Access Points:" -ForegroundColor $Colors.Yellow
    Write-Host "  - Grafana Dashboard: http://localhost:3010 (admin/GrafanaSecure2025!)" -ForegroundColor $Colors.Cyan
    Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor $Colors.Cyan
    Write-Host "  - InfluxDB: http://localhost:8086" -ForegroundColor $Colors.Cyan
    Write-Host "  - PostgreSQL: localhost:5432 (forex_admin/ForexSecure2025!)" -ForegroundColor $Colors.Cyan
    Write-Host ""
    Write-Host "📋 Next Steps:" -ForegroundColor $Colors.Yellow
    Write-Host "  1. Services will be implemented in upcoming phases" -ForegroundColor $Colors.Cyan
    Write-Host "  2. Web dashboard will be available at http://localhost:8080" -ForegroundColor $Colors.Cyan
    Write-Host "  3. API Gateway will be at http://localhost:3000" -ForegroundColor $Colors.Cyan
    Write-Host ""
    Write-Host "🔧 Useful Commands:" -ForegroundColor $Colors.Yellow
    Write-Host "  - View logs: $composeCmd logs -f [service-name]" -ForegroundColor $Colors.Cyan
    Write-Host "  - Stop platform: $composeCmd down" -ForegroundColor $Colors.Cyan
    Write-Host "  - Restart service: $composeCmd restart [service-name]" -ForegroundColor $Colors.Cyan
    Write-Host "  - View status: $composeCmd ps" -ForegroundColor $Colors.Cyan
}

function Main {
    Write-Host "🚀 Personal Forex Trading Platform - Development Setup" -ForegroundColor $Colors.Green
    Write-Host "======================================================" -ForegroundColor $Colors.Green
    Write-Host ""
    
    if (-not $SkipDocker) {
        Test-Docker
        Test-DockerCompose
    }
    
    New-Directories
    Set-Environment
    Set-PrometheusConfig
    Set-GrafanaConfig
    
    if (-not $SkipDocker) {
        Start-Platform
    }
    
    Write-Success "Personal Forex Trading Platform development environment is ready!"
}

# Execute main function
try {
    Main
}
catch {
    Write-Error "Setup failed: $($_.Exception.Message)"
    exit 1
}
