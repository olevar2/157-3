# Kafka High-Frequency Trading Setup Script
# Initializes Kafka cluster for forex scalping operations
# Author: Platform3 Development Team

param(
    [switch]$StartOnly,
    [switch]$StopOnly,
    [switch]$Reset,
    [switch]$Monitor
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Color functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }
function Write-Info { Write-ColorOutput Cyan $args }

# Configuration
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$KAFKA_DIR = $SCRIPT_DIR
$COMPOSE_FILE = Join-Path $KAFKA_DIR "docker-compose.kafka-trading.yml"
$TOPICS_SCRIPT = Join-Path $KAFKA_DIR "scalping-topics.sh"

Write-Info "🚀 Kafka High-Frequency Trading Platform Setup"
Write-Info "=============================================="
Write-Info "Script Directory: $SCRIPT_DIR"
Write-Info "Compose File: $COMPOSE_FILE"
Write-Info ""

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker ps | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if Docker Compose is available
function Test-DockerCompose {
    try {
        docker-compose --version | Out-Null
        return $true
    }
    catch {
        try {
            docker compose version | Out-Null
            return $true
        }
        catch {
            return $false
        }
    }
}

# Function to get Docker Compose command
function Get-DockerComposeCmd {
    try {
        docker-compose --version | Out-Null
        return "docker-compose"
    }
    catch {
        return "docker compose"
    }
}

# Function to wait for Kafka cluster to be ready
function Wait-ForKafkaCluster {
    param([int]$TimeoutSeconds = 300)
    
    Write-Info "⏳ Waiting for Kafka cluster to be ready..."
    $startTime = Get-Date
    $timeout = $startTime.AddSeconds($TimeoutSeconds)
    
    while ((Get-Date) -lt $timeout) {
        try {
            # Check if all brokers are running
            $brokerCount = docker ps --filter "name=forex-kafka-broker" --filter "status=running" --format "table {{.Names}}" | Measure-Object -Line | Select-Object -ExpandProperty Lines
            
            if ($brokerCount -eq 4) { # 3 brokers + header line
                # Test if we can list topics
                $testResult = docker exec forex-kafka-broker-1 kafka-topics --bootstrap-server localhost:19092 --list 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "✅ Kafka cluster is ready!"
                    return $true
                }
            }
        }
        catch {
            # Continue waiting
        }
        
        Write-Info "   Still waiting... ($(((Get-Date) - $startTime).TotalSeconds) seconds elapsed)"
        Start-Sleep -Seconds 5
    }
    
    Write-Error "❌ Timeout waiting for Kafka cluster to be ready"
    return $false
}

# Function to create Kafka topics
function New-KafkaTopics {
    Write-Info "📋 Creating Kafka topics for forex trading..."
    
    # Make script executable and run it
    if (Test-Path $TOPICS_SCRIPT) {
        # Convert bash script to PowerShell commands
        Write-Info "Creating topics using Docker exec commands..."
        
        $topics = @(
            @{Name="forex.tick.m1"; Partitions=16; Retention=86400000; Segment=60000; Description="M1 tick data for scalping"},
            @{Name="forex.orderbook.updates"; Partitions=32; Retention=3600000; Segment=30000; Description="Real-time order book updates"},
            @{Name="forex.orderflow"; Partitions=16; Retention=21600000; Segment=60000; Description="Order flow analysis data"},
            @{Name="forex.bars.m5"; Partitions=8; Retention=604800000; Segment=300000; Description="M5 OHLCV bars"},
            @{Name="forex.signals.scalping"; Partitions=16; Retention=3600000; Segment=30000; Description="Scalping signals"},
            @{Name="forex.orders.requests"; Partitions=16; Retention=86400000; Segment=60000; Description="Order requests"},
            @{Name="forex.orders.executions"; Partitions=16; Retention=259200000; Segment=120000; Description="Order executions"},
            @{Name="forex.positions.updates"; Partitions=8; Retention=259200000; Segment=120000; Description="Position updates"},
            @{Name="forex.sessions.events"; Partitions=4; Retention=604800000; Segment=3600000; Description="Session events"},
            @{Name="forex.analytics.pnl"; Partitions=8; Retention=2592000000; Segment=900000; Description="P&L updates"}
        )
        
        foreach ($topic in $topics) {
            Write-Info "Creating topic: $($topic.Name)"
            $cmd = "kafka-topics --create --bootstrap-server localhost:19092 --topic $($topic.Name) --partitions $($topic.Partitions) --replication-factor 3 --config min.insync.replicas=2 --config retention.ms=$($topic.Retention) --config segment.ms=$($topic.Segment) --config compression.type=lz4 --config cleanup.policy=delete --config max.message.bytes=1000000 --config flush.ms=1000 --config unclean.leader.election.enable=false --config preallocate=true --if-not-exists"
            
            try {
                docker exec forex-kafka-broker-1 $cmd
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "  ✅ Created: $($topic.Name)"
                } else {
                    Write-Warning "  ⚠️ Topic may already exist: $($topic.Name)"
                }
            }
            catch {
                Write-Error "  ❌ Failed to create: $($topic.Name)"
            }
        }
        
        Write-Success "✅ Topic creation completed!"
    } else {
        Write-Error "❌ Topics script not found: $TOPICS_SCRIPT"
    }
}

# Function to start Kafka cluster
function Start-KafkaCluster {
    Write-Info "🚀 Starting Kafka cluster for forex trading..."
    
    if (-not (Test-Path $COMPOSE_FILE)) {
        Write-Error "❌ Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    }
    
    $dockerComposeCmd = Get-DockerComposeCmd
    
    try {
        # Start the cluster
        & $dockerComposeCmd -f $COMPOSE_FILE up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✅ Kafka cluster started successfully!"
            
            # Wait for cluster to be ready
            if (Wait-ForKafkaCluster) {
                # Create topics
                New-KafkaTopics
                
                Write-Success ""
                Write-Success "🎉 Kafka High-Frequency Trading Platform is ready!"
                Write-Success "📊 Control Center: http://localhost:9021"
                Write-Success "🔧 Schema Registry: http://localhost:8081"
                Write-Success "🔌 Kafka Connect: http://localhost:8083"
                Write-Success ""
                Write-Success "Broker endpoints:"
                Write-Success "  - localhost:9092 (Broker 1)"
                Write-Success "  - localhost:9093 (Broker 2)"
                Write-Success "  - localhost:9094 (Broker 3)"
            }
        } else {
            Write-Error "❌ Failed to start Kafka cluster"
            exit 1
        }
    }
    catch {
        Write-Error "❌ Error starting Kafka cluster: $($_.Exception.Message)"
        exit 1
    }
}

# Function to stop Kafka cluster
function Stop-KafkaCluster {
    Write-Info "🛑 Stopping Kafka cluster..."
    
    $dockerComposeCmd = Get-DockerComposeCmd
    
    try {
        & $dockerComposeCmd -f $COMPOSE_FILE down
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✅ Kafka cluster stopped successfully!"
        } else {
            Write-Error "❌ Failed to stop Kafka cluster"
        }
    }
    catch {
        Write-Error "❌ Error stopping Kafka cluster: $($_.Exception.Message)"
    }
}

# Function to reset Kafka cluster (remove data)
function Reset-KafkaCluster {
    Write-Warning "⚠️ This will remove all Kafka data and topics!"
    $confirmation = Read-Host "Are you sure you want to continue? (y/N)"
    
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        Write-Info "🧹 Resetting Kafka cluster..."
        
        $dockerComposeCmd = Get-DockerComposeCmd
        
        try {
            # Stop and remove containers, networks, and volumes
            & $dockerComposeCmd -f $COMPOSE_FILE down -v --remove-orphans
            
            # Remove any dangling volumes
            docker volume prune -f
            
            Write-Success "✅ Kafka cluster reset completed!"
            Write-Info "Run the script again to start fresh."
        }
        catch {
            Write-Error "❌ Error resetting Kafka cluster: $($_.Exception.Message)"
        }
    } else {
        Write-Info "Reset cancelled."
    }
}

# Function to monitor Kafka cluster
function Show-KafkaStatus {
    Write-Info "📊 Kafka Cluster Status"
    Write-Info "======================="
    
    # Check container status
    Write-Info ""
    Write-Info "Container Status:"
    docker ps --filter "name=forex-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # Check topics
    Write-Info ""
    Write-Info "Available Topics:"
    try {
        docker exec forex-kafka-broker-1 kafka-topics --bootstrap-server localhost:19092 --list 2>$null | Sort-Object
    }
    catch {
        Write-Warning "Could not retrieve topics. Cluster may not be ready."
    }
    
    # Show useful URLs
    Write-Info ""
    Write-Info "Management URLs:"
    Write-Info "  📊 Control Center: http://localhost:9021"
    Write-Info "  🔧 Schema Registry: http://localhost:8081"
    Write-Info "  🔌 Kafka Connect: http://localhost:8083"
}

# Main execution logic
try {
    # Check prerequisites
    if (-not (Test-DockerRunning)) {
        Write-Error "❌ Docker is not running. Please start Docker Desktop."
        exit 1
    }
    
    if (-not (Test-DockerCompose)) {
        Write-Error "❌ Docker Compose is not available. Please install Docker Compose."
        exit 1
    }
    
    # Execute based on parameters
    if ($StopOnly) {
        Stop-KafkaCluster
    }
    elseif ($Reset) {
        Reset-KafkaCluster
    }
    elseif ($Monitor) {
        Show-KafkaStatus
    }
    elseif ($StartOnly) {
        Start-KafkaCluster
    }
    else {
        # Default: start the cluster
        Start-KafkaCluster
    }
    
} catch {
    Write-Error "❌ Script execution failed: $($_.Exception.Message)"
    exit 1
}

Write-Info ""
Write-Info "Script execution completed."
