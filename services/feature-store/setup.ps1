# Feature Store Setup Script for Windows
# Automates deployment and configuration of AI Feature Store microservice

param(
    [switch]$Build,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Clean,
    [string]$Action = "deploy"
)

# Configuration
$ErrorActionPreference = "Stop"
$FeatureStorePort = 8080
$WebSocketPort = 8081
$RedisPort = 6379
$InfluxDBPort = 8086

Write-Host "🚀 Feature Store Management Script" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

function Test-Dependencies {
    Write-Host "🔍 Checking dependencies..." -ForegroundColor Yellow
    
    $dependencies = @("docker", "docker-compose", "curl")
    $missing = @()
    
    foreach ($dep in $dependencies) {
        try {
            $null = Get-Command $dep -ErrorAction Stop
            Write-Host "✅ $dep is available" -ForegroundColor Green
        }
        catch {
            $missing += $dep
            Write-Host "❌ $dep is not available" -ForegroundColor Red
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Host "❌ Missing dependencies: $($missing -join ', ')" -ForegroundColor Red
        Write-Host "Please install Docker Desktop and ensure it's in your PATH" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ All dependencies are available" -ForegroundColor Green
}

function New-Directories {
    Write-Host "📁 Creating directories..." -ForegroundColor Yellow
    
    $directories = @(
        "logs",
        "data\redis",
        "data\kafka",
        "data\influxdb",
        "config"
    )
    
    foreach ($directory in $directories) {
        if (!(Test-Path $directory)) {
            New-Item -Path $directory -ItemType Directory -Force | Out-Null
            Write-Host "✅ Created directory: $directory" -ForegroundColor Green
        } else {
            Write-Host "ℹ️  Directory exists: $directory" -ForegroundColor Blue
        }
    }
}

function Test-Configuration {
    Write-Host "⚙️  Validating configuration..." -ForegroundColor Yellow
    
    $requiredFiles = @(
        "feature-definitions.yaml",
        "package.json",
        "requirements.txt",
        "src\feature-pipeline.py",
        "src\feature-serving-api.ts",
        "Dockerfile",
        "docker-compose.yml"
    )
    
    $missingFiles = @()
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✅ Found: $file" -ForegroundColor Green
        } else {
            $missingFiles += $file
            Write-Host "❌ Missing: $file" -ForegroundColor Red
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Host "❌ Missing required files: $($missingFiles -join ', ')" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ All configuration files found" -ForegroundColor Green
}

function Build-Containers {
    Write-Host "🏗️  Building Docker containers..." -ForegroundColor Yellow
    
    try {
        # Build feature store container
        Write-Host "Building feature store container..." -ForegroundColor Blue
        docker-compose build feature-store
        if ($LASTEXITCODE -ne 0) { throw "Failed to build feature store container" }
        
        # Pull required images
        $images = @(
            "redis:7-alpine",
            "confluentinc/cp-kafka:7.4.0", 
            "confluentinc/cp-zookeeper:7.4.0",
            "influxdb:2.7"
        )
        
        foreach ($image in $images) {
            Write-Host "Pulling $image..." -ForegroundColor Blue
            docker pull $image
            if ($LASTEXITCODE -ne 0) { throw "Failed to pull $image" }
        }
        
        Write-Host "✅ All containers built successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error building containers: $_" -ForegroundColor Red
        exit 1
    }
}

function Start-Services {
    Write-Host "🚀 Starting services..." -ForegroundColor Yellow
    
    try {
        # Start infrastructure services first
        Write-Host "Starting infrastructure services..." -ForegroundColor Blue
        docker-compose up -d zookeeper redis influxdb
        if ($LASTEXITCODE -ne 0) { throw "Failed to start infrastructure services" }
        
        Write-Host "⏳ Waiting for infrastructure to initialize (30s)..." -ForegroundColor Blue
        Start-Sleep -Seconds 30
        
        # Start Kafka cluster
        Write-Host "Starting Kafka cluster..." -ForegroundColor Blue
        docker-compose up -d kafka1 kafka2 kafka3
        if ($LASTEXITCODE -ne 0) { throw "Failed to start Kafka cluster" }
        
        Write-Host "⏳ Waiting for Kafka to initialize (30s)..." -ForegroundColor Blue
        Start-Sleep -Seconds 30
        
        # Start feature store
        Write-Host "Starting feature store service..." -ForegroundColor Blue
        docker-compose up -d feature-store
        if ($LASTEXITCODE -ne 0) { throw "Failed to start feature store" }
        
        Write-Host "⏳ Waiting for feature store to initialize (20s)..." -ForegroundColor Blue
        Start-Sleep -Seconds 20
        
        Write-Host "✅ All services started" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error starting services: $_" -ForegroundColor Red
        exit 1
    }
}

function Test-Services {
    Write-Host "🔍 Verifying services..." -ForegroundColor Yellow
    
    $services = @(
        @{Name="Feature Store API"; Url="http://localhost:$FeatureStorePort/health"},
        @{Name="InfluxDB"; Url="http://localhost:$InfluxDBPort/health"}
    )
    
    foreach ($service in $services) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ $($service.Name) is healthy" -ForegroundColor Green
            } else {
                Write-Host "⚠️  $($service.Name) returned status $($response.StatusCode)" -ForegroundColor Yellow
            }
        }
        catch {
            Write-Host "❌ $($service.Name) is not responding: $_" -ForegroundColor Red
        }
    }
    
    # Test Redis connection
    try {
        $redisTest = docker exec forex-feature-redis redis-cli ping 2>$null
        if ($redisTest -eq "PONG") {
            Write-Host "✅ Redis is healthy" -ForegroundColor Green
        } else {
            Write-Host "❌ Redis is not responding" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ Cannot test Redis connection" -ForegroundColor Red
    }
    
    # Show container status
    Write-Host "`n📊 Container status:" -ForegroundColor Blue
    docker-compose ps
}

function Set-KafkaTopics {
    Write-Host "📝 Setting up Kafka topics..." -ForegroundColor Yellow
    
    $topics = @(
        "tick-data-stream",
        "feature-updates", 
        "order-book-updates",
        "session-events"
    )
    
    foreach ($topic in $topics) {
        try {
            $command = "kafka-topics --create --topic $topic --partitions 16 --replication-factor 3 --config compression.type=lz4 --config cleanup.policy=delete --config retention.ms=86400000 --bootstrap-server localhost:9092"
            docker-compose exec kafka1 $command 2>$null
            Write-Host "✅ Created topic: $topic" -ForegroundColor Green
        }
        catch {
            Write-Host "ℹ️  Topic $topic might already exist" -ForegroundColor Blue
        }
    }
}

function Show-Status {
    Write-Host "`n" + "="*60 -ForegroundColor Green
    Write-Host "🎉 FEATURE STORE STATUS" -ForegroundColor Green
    Write-Host "="*60 -ForegroundColor Green
    
    Write-Host "`n📊 Service Endpoints:" -ForegroundColor Blue
    Write-Host "  • Feature API:        http://localhost:$FeatureStorePort" -ForegroundColor White
    Write-Host "  • WebSocket:          ws://localhost:$WebSocketPort" -ForegroundColor White
    Write-Host "  • Health Check:       http://localhost:$FeatureStorePort/health" -ForegroundColor White
    Write-Host "  • Metrics:            http://localhost:$FeatureStorePort/metrics" -ForegroundColor White
    Write-Host "  • Redis:              localhost:$RedisPort" -ForegroundColor White
    Write-Host "  • InfluxDB:           http://localhost:$InfluxDBPort" -ForegroundColor White
    Write-Host "  • Kafka:              localhost:9092,9093,9094" -ForegroundColor White
    
    Write-Host "`n🔧 Management Commands:" -ForegroundColor Blue
    Write-Host "  • View logs:          docker-compose logs -f feature-store" -ForegroundColor White
    Write-Host "  • Stop services:      .\setup.ps1 -Action stop" -ForegroundColor White
    Write-Host "  • Restart services:   docker-compose restart feature-store" -ForegroundColor White
    Write-Host "  • Clean deployment:   .\setup.ps1 -Action clean" -ForegroundColor White
    
    Write-Host "`n🧪 API Test Examples:" -ForegroundColor Blue
    Write-Host "  • Get feature:        curl http://localhost:$FeatureStorePort/api/features/EURUSD/rsi_14" -ForegroundColor White
    Write-Host "  • Get all features:   curl http://localhost:$FeatureStorePort/api/features/EURUSD/all" -ForegroundColor White
    Write-Host "  • Get feature vector: curl http://localhost:$FeatureStorePort/api/features/EURUSD/vector" -ForegroundColor White
}

function Stop-Services {
    Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "✅ Services stopped" -ForegroundColor Green
}

function Clear-Deployment {
    Write-Host "🧹 Cleaning deployment..." -ForegroundColor Yellow
    docker-compose down -v --remove-orphans
    docker system prune -f
    Write-Host "✅ Deployment cleaned" -ForegroundColor Green
}

# Main execution logic
switch ($Action.ToLower()) {
    "deploy" {
        Test-Dependencies
        New-Directories
        Test-Configuration
        Build-Containers
        Start-Services
        Set-KafkaTopics
        Test-Services
        Show-Status
        Write-Host "`n✅ Feature Store deployment completed successfully!" -ForegroundColor Green
    }
    "build" {
        Test-Dependencies
        Test-Configuration
        Build-Containers
    }
    "start" {
        Start-Services
        Test-Services
        Show-Status
    }
    "stop" {
        Stop-Services
    }
    "status" {
        Test-Services
        Show-Status
    }
    "clean" {
        Clear-Deployment
    }
    default {
        Write-Host "Usage: .\setup.ps1 -Action [deploy|build|start|stop|status|clean]" -ForegroundColor Yellow
        Write-Host "  deploy - Full deployment (build + start + configure)" -ForegroundColor White
        Write-Host "  build  - Build containers only" -ForegroundColor White
        Write-Host "  start  - Start services" -ForegroundColor White
        Write-Host "  stop   - Stop services" -ForegroundColor White
        Write-Host "  status - Check service status" -ForegroundColor White
        Write-Host "  clean  - Clean deployment and remove volumes" -ForegroundColor White
    }
}
