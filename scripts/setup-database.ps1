# Personal Forex Trading Platform - Database Setup Script
# This script sets up the database infrastructure for the trading platform

Write-Host "🚀 Setting up Database Infrastructure for Forex Trading Platform" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

# Check if Docker Desktop is running
function Test-DockerRunning {
    try {
        $null = docker ps 2>$null
        return $true
    }
    catch {
        return $false
    }
}

# Function to wait for Docker Desktop to start
function Wait-ForDocker {
    Write-Host "⏳ Waiting for Docker Desktop to start..." -ForegroundColor Yellow
    $timeout = 120 # 2 minutes timeout
    $elapsed = 0
    
    while (-not (Test-DockerRunning) -and $elapsed -lt $timeout) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        Write-Host "." -NoNewline -ForegroundColor Yellow
    }
    
    if (Test-DockerRunning) {
        Write-Host "`n✅ Docker Desktop is running!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "`n❌ Docker Desktop failed to start within timeout" -ForegroundColor Red
        return $false
    }
}

# Function to start database containers
function Start-DatabaseContainers {
    Write-Host "🐳 Starting database containers..." -ForegroundColor Cyan
    
    try {
        # Start PostgreSQL
        Write-Host "📊 Starting PostgreSQL container..." -ForegroundColor Yellow
        docker run -d `
            --name forex-postgres `
            -e POSTGRES_DB=forex_trading `
            -e POSTGRES_USER=forex_admin `
            -e POSTGRES_PASSWORD=ForexSecure2025! `
            -p 5432:5432 `
            -v "${PWD}/infrastructure/database/init.sql:/docker-entrypoint-initdb.d/init.sql" `
            postgres:15-alpine
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PostgreSQL container started successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to start PostgreSQL container" -ForegroundColor Red
            return $false
        }
        
        # Start Redis
        Write-Host "🔴 Starting Redis container..." -ForegroundColor Yellow
        docker run -d `
            --name forex-redis `
            --command "redis-server --appendonly yes --requirepass RedisSecure2025!" `
            -p 6379:6379 `
            redis:7-alpine
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Redis container started successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to start Redis container" -ForegroundColor Red
            return $false
        }
        
        return $true
    }
    catch {
        Write-Host "❌ Error starting containers: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to wait for databases to be ready
function Wait-ForDatabases {
    Write-Host "⏳ Waiting for databases to be ready..." -ForegroundColor Yellow
    
    # Wait for PostgreSQL
    $pgReady = $false
    $attempts = 0
    $maxAttempts = 30
    
    while (-not $pgReady -and $attempts -lt $maxAttempts) {
        try {
            $result = docker exec forex-postgres pg_isready -U forex_admin -d forex_trading 2>$null
            if ($LASTEXITCODE -eq 0) {
                $pgReady = $true
                Write-Host "✅ PostgreSQL is ready!" -ForegroundColor Green
            } else {
                Start-Sleep -Seconds 2
                $attempts++
                Write-Host "." -NoNewline -ForegroundColor Yellow
            }
        }
        catch {
            Start-Sleep -Seconds 2
            $attempts++
            Write-Host "." -NoNewline -ForegroundColor Yellow
        }
    }
    
    if (-not $pgReady) {
        Write-Host "`n❌ PostgreSQL failed to become ready" -ForegroundColor Red
        return $false
    }
    
    # Wait for Redis
    $redisReady = $false
    $attempts = 0
    
    while (-not $redisReady -and $attempts -lt $maxAttempts) {
        try {
            $result = docker exec forex-redis redis-cli -a RedisSecure2025! ping 2>$null
            if ($result -eq "PONG") {
                $redisReady = $true
                Write-Host "✅ Redis is ready!" -ForegroundColor Green
            } else {
                Start-Sleep -Seconds 2
                $attempts++
                Write-Host "." -NoNewline -ForegroundColor Yellow
            }
        }
        catch {
            Start-Sleep -Seconds 2
            $attempts++
            Write-Host "." -NoNewline -ForegroundColor Yellow
        }
    }
    
    if (-not $redisReady) {
        Write-Host "`n❌ Redis failed to become ready" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Function to update service configurations
function Update-ServiceConfigurations {
    Write-Host "⚙️ Updating service configurations..." -ForegroundColor Cyan
    
    # Create environment file for services
    $envContent = @"
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=forex_trading
DB_USER=forex_admin
DB_PASSWORD=ForexSecure2025!
DATABASE_URL=postgresql://forex_admin:ForexSecure2025!@localhost:5432/forex_trading

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=RedisSecure2025!

# JWT Configuration
JWT_SECRET=forex-jwt-secret-super-secure-2025

# Node Environment
NODE_ENV=development
"@
    
    # Write environment file
    $envContent | Out-File -FilePath ".env.local" -Encoding UTF8
    Write-Host "✅ Created .env.local configuration file" -ForegroundColor Green
    
    return $true
}

# Function to test database connections
function Test-DatabaseConnections {
    Write-Host "🔍 Testing database connections..." -ForegroundColor Cyan
    
    # Test PostgreSQL connection
    try {
        $pgTest = docker exec forex-postgres psql -U forex_admin -d forex_trading -c "SELECT version();" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PostgreSQL connection test passed" -ForegroundColor Green
        } else {
            Write-Host "❌ PostgreSQL connection test failed" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ PostgreSQL connection test failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    # Test Redis connection
    try {
        $redisTest = docker exec forex-redis redis-cli -a RedisSecure2025! info server 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Redis connection test passed" -ForegroundColor Green
        } else {
            Write-Host "❌ Redis connection test failed" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Redis connection test failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Main execution
try {
    # Check if Docker is running
    if (-not (Test-DockerRunning)) {
        Write-Host "🐳 Docker Desktop is not running. Attempting to start..." -ForegroundColor Yellow
        
        # Try to start Docker Desktop
        try {
            Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -WindowStyle Hidden
            if (-not (Wait-ForDocker)) {
                Write-Host "❌ Please start Docker Desktop manually and run this script again." -ForegroundColor Red
                exit 1
            }
        }
        catch {
            Write-Host "❌ Could not start Docker Desktop. Please start it manually and run this script again." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "✅ Docker Desktop is already running!" -ForegroundColor Green
    }
    
    # Clean up existing containers if they exist
    Write-Host "🧹 Cleaning up existing containers..." -ForegroundColor Yellow
    docker stop forex-postgres forex-redis 2>$null
    docker rm forex-postgres forex-redis 2>$null
    
    # Start database containers
    if (-not (Start-DatabaseContainers)) {
        Write-Host "❌ Failed to start database containers" -ForegroundColor Red
        exit 1
    }
    
    # Wait for databases to be ready
    if (-not (Wait-ForDatabases)) {
        Write-Host "❌ Databases failed to become ready" -ForegroundColor Red
        exit 1
    }
    
    # Update service configurations
    if (-not (Update-ServiceConfigurations)) {
        Write-Host "❌ Failed to update service configurations" -ForegroundColor Red
        exit 1
    }
    
    # Test database connections
    if (-not (Test-DatabaseConnections)) {
        Write-Host "❌ Database connection tests failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "🎉 Database infrastructure setup completed successfully!" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "📊 PostgreSQL: localhost:5432 (Database: forex_trading)" -ForegroundColor Cyan
    Write-Host "🔴 Redis: localhost:6379" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Restart your services to use the new database configuration" -ForegroundColor White
    Write-Host "2. Services will automatically connect to the databases" -ForegroundColor White
    Write-Host "3. Database schema has been initialized with the init.sql script" -ForegroundColor White
    Write-Host ""
    Write-Host "To stop the databases: docker stop forex-postgres forex-redis" -ForegroundColor Gray
    Write-Host "To start the databases: docker start forex-postgres forex-redis" -ForegroundColor Gray
    
}
catch {
    Write-Host "❌ Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
