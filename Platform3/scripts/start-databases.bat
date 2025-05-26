@echo off
echo Starting Database Infrastructure for Forex Trading Platform
echo ============================================================

echo Checking Docker status...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not available or not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo Docker is available!

echo Cleaning up existing containers...
docker stop forex-postgres forex-redis >nul 2>&1
docker rm forex-postgres forex-redis >nul 2>&1

echo Starting PostgreSQL container...
docker run -d ^
    --name forex-postgres ^
    -e POSTGRES_DB=forex_trading ^
    -e POSTGRES_USER=forex_admin ^
    -e POSTGRES_PASSWORD=ForexSecure2025! ^
    -p 5432:5432 ^
    -v "%cd%\infrastructure\database\init.sql:/docker-entrypoint-initdb.d/init.sql" ^
    postgres:15-alpine

if %errorlevel% neq 0 (
    echo ERROR: Failed to start PostgreSQL container
    pause
    exit /b 1
)

echo PostgreSQL container started successfully!

echo Starting Redis container...
docker run -d ^
    --name forex-redis ^
    -p 6379:6379 ^
    redis:7-alpine redis-server --appendonly yes --requirepass RedisSecure2025!

if %errorlevel% neq 0 (
    echo ERROR: Failed to start Redis container
    pause
    exit /b 1
)

echo Redis container started successfully!

echo Waiting for databases to be ready...
timeout /t 10 /nobreak >nul

echo Testing PostgreSQL connection...
docker exec forex-postgres pg_isready -U forex_admin -d forex_trading
if %errorlevel% neq 0 (
    echo WARNING: PostgreSQL may not be ready yet, but container is running
)

echo Testing Redis connection...
docker exec forex-redis redis-cli -a RedisSecure2025! ping
if %errorlevel% neq 0 (
    echo WARNING: Redis may not be ready yet, but container is running
)

echo.
echo ============================================================
echo Database infrastructure setup completed!
echo PostgreSQL: localhost:5432 (Database: forex_trading)
echo Redis: localhost:6379
echo.
echo Next: Restart your services to use the new database configuration
echo ============================================================
pause
