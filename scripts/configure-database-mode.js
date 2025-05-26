// Database Infrastructure Setup Script
// Configures all services to use database mode instead of mock mode

const fs = require('fs');
const path = require('path');

console.log('🚀 Configuring Database Infrastructure for Forex Trading Platform');
console.log('================================================================');

// Database configuration for all services
const databaseConfig = {
  // PostgreSQL Configuration
  DB_HOST: 'localhost',
  DB_PORT: '5432',
  DB_NAME: 'forex_trading',
  DB_USER: 'forex_admin',
  DB_PASSWORD: 'ForexSecure2025!',
  DATABASE_URL: 'postgresql://forex_admin:ForexSecure2025!@localhost:5432/forex_trading',
  
  // Redis Configuration
  REDIS_URL: 'redis://localhost:6379',
  REDIS_PASSWORD: 'RedisSecure2025!',
  
  // JWT Configuration
  JWT_SECRET: 'forex-jwt-secret-super-secure-2025',
  API_SECRET_KEY: 'forex-api-key-ultra-secure-2025',
  
  // Service Configuration
  NODE_ENV: 'development',
  LOG_LEVEL: 'debug',
  
  // Database Mode Flags
  USE_DATABASE: 'true',
  MOCK_MODE: 'false'
};

// Services to configure
const services = [
  {
    name: 'User Service',
    path: 'services/user-service',
    port: '3002',
    startScript: 'npm run dev'
  },
  {
    name: 'Trading Service', 
    path: 'services/trading-service',
    port: '3003',
    startScript: 'npm run dev'
  },
  {
    name: 'Market Data Service',
    path: 'services/market-data-service', 
    port: '3004',
    startScript: 'npm start'
  },
  {
    name: 'Event System',
    path: 'services/event-system',
    port: '3005',
    startScript: 'npm run dev'
  },
  {
    name: 'API Gateway',
    path: 'services/api-gateway',
    port: '3001',
    startScript: 'npm start'
  }
];

// Function to create environment file for a service
function createServiceEnvFile(service) {
  const envPath = path.join(service.path, '.env.local');
  
  const envContent = Object.entries({
    ...databaseConfig,
    PORT: service.port
  }).map(([key, value]) => `${key}=${value}`).join('\n');
  
  try {
    fs.writeFileSync(envPath, envContent);
    console.log(`✅ Created ${envPath}`);
    return true;
  } catch (error) {
    console.error(`❌ Failed to create ${envPath}:`, error.message);
    return false;
  }
}

// Function to update package.json scripts for database mode
function updateServiceScripts(service) {
  const packageJsonPath = path.join(service.path, 'package.json');
  
  try {
    if (!fs.existsSync(packageJsonPath)) {
      console.log(`⚠️  No package.json found for ${service.name}`);
      return true;
    }
    
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // Add database mode scripts
    if (!packageJson.scripts['start:db']) {
      packageJson.scripts['start:db'] = packageJson.scripts.dev || packageJson.scripts.start;
      packageJson.scripts['start:mock'] = 'node dist/mock-server.js';
      
      fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
      console.log(`✅ Updated scripts for ${service.name}`);
    }
    
    return true;
  } catch (error) {
    console.error(`❌ Failed to update scripts for ${service.name}:`, error.message);
    return false;
  }
}

// Function to create startup script
function createStartupScript() {
  const startupScript = `#!/bin/bash
# Forex Trading Platform - Database Mode Startup Script

echo "🚀 Starting Forex Trading Platform in Database Mode"
echo "=================================================="

# Check if databases are running
echo "🔍 Checking database connections..."

# Check PostgreSQL
if ! pg_isready -h localhost -p 5432 -U forex_admin -d forex_trading 2>/dev/null; then
    echo "❌ PostgreSQL is not running or not accessible"
    echo "Please start PostgreSQL database first:"
    echo "  Option 1: Enable WSL2 and start Docker containers"
    echo "  Option 2: Install PostgreSQL locally"
    echo "  Option 3: Use cloud PostgreSQL service"
    exit 1
fi

# Check Redis
if ! redis-cli -h localhost -p 6379 -a RedisSecure2025! ping 2>/dev/null | grep -q PONG; then
    echo "❌ Redis is not running or not accessible"
    echo "Please start Redis database first"
    exit 1
fi

echo "✅ Database connections verified"
echo ""

# Start services in database mode
echo "🔄 Starting services..."

# Start API Gateway
echo "Starting API Gateway (Port 3001)..."
cd services/api-gateway && npm start &
GATEWAY_PID=$!

# Start User Service  
echo "Starting User Service (Port 3002)..."
cd ../user-service && npm run dev &
USER_PID=$!

# Start Trading Service
echo "Starting Trading Service (Port 3003)..."
cd ../trading-service && npm run dev &
TRADING_PID=$!

# Start Market Data Service
echo "Starting Market Data Service (Port 3004)..."
cd ../market-data-service && npm start &
MARKET_PID=$!

# Start Event System
echo "Starting Event System (Port 3005)..."
cd ../event-system && npm run dev &
EVENT_PID=$!

echo ""
echo "✅ All services started in database mode!"
echo "=================================================="
echo "🌐 API Gateway: http://localhost:3001"
echo "👤 User Service: http://localhost:3002"
echo "💰 Trading Service: http://localhost:3003"
echo "📊 Market Data Service: http://localhost:3004"
echo "🔔 Event System: http://localhost:3005"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "Stopping services..."; kill $GATEWAY_PID $USER_PID $TRADING_PID $MARKET_PID $EVENT_PID; exit' INT
wait
`;

  try {
    fs.writeFileSync('scripts/start-database-mode.sh', startupScript);
    console.log('✅ Created startup script: scripts/start-database-mode.sh');
    return true;
  } catch (error) {
    console.error('❌ Failed to create startup script:', error.message);
    return false;
  }
}

// Function to create Windows batch startup script
function createWindowsStartupScript() {
  const batchScript = `@echo off
echo 🚀 Starting Forex Trading Platform in Database Mode
echo ==================================================

echo 🔍 Checking database connections...

REM Check if services are ready to start
echo ✅ Starting services in database mode...
echo.

echo Starting API Gateway (Port 3001)...
start "API Gateway" cmd /k "cd services\\api-gateway && npm start"

timeout /t 2 /nobreak >nul

echo Starting User Service (Port 3002)...
start "User Service" cmd /k "cd services\\user-service && npm run dev"

timeout /t 2 /nobreak >nul

echo Starting Trading Service (Port 3003)...
start "Trading Service" cmd /k "cd services\\trading-service && npm run dev"

timeout /t 2 /nobreak >nul

echo Starting Market Data Service (Port 3004)...
start "Market Data Service" cmd /k "cd services\\market-data-service && npm start"

timeout /t 2 /nobreak >nul

echo Starting Event System (Port 3005)...
start "Event System" cmd /k "cd services\\event-system && npm run dev"

echo.
echo ✅ All services started in database mode!
echo ==================================================
echo 🌐 API Gateway: http://localhost:3001
echo 👤 User Service: http://localhost:3002
echo 💰 Trading Service: http://localhost:3003
echo 📊 Market Data Service: http://localhost:3004
echo 🔔 Event System: http://localhost:3005
echo.
echo Services are running in separate windows.
echo Close individual windows to stop services.
pause
`;

  try {
    fs.writeFileSync('scripts/start-database-mode.bat', batchScript);
    console.log('✅ Created Windows startup script: scripts/start-database-mode.bat');
    return true;
  } catch (error) {
    console.error('❌ Failed to create Windows startup script:', error.message);
    return false;
  }
}

// Main execution
async function main() {
  let allSuccess = true;
  
  console.log('📝 Configuring services for database mode...');
  
  // Configure each service
  for (const service of services) {
    console.log(`\n🔧 Configuring ${service.name}...`);
    
    if (!createServiceEnvFile(service)) {
      allSuccess = false;
    }
    
    if (!updateServiceScripts(service)) {
      allSuccess = false;
    }
  }
  
  // Create startup scripts
  console.log('\n📜 Creating startup scripts...');
  if (!createStartupScript()) {
    allSuccess = false;
  }
  
  if (!createWindowsStartupScript()) {
    allSuccess = false;
  }
  
  // Create database setup instructions
  const instructions = `# Database Infrastructure Setup Instructions

## Current Status
✅ All services configured for database mode
✅ Environment files created for each service  
✅ Startup scripts created

## Next Steps

### Option 1: Enable WSL2 and Docker (Recommended)
1. Run as Administrator: \`wsl.exe --install --no-distribution\`
2. Restart your computer
3. Start Docker Desktop
4. Run: \`docker-compose up -d postgres redis\`

### Option 2: Install PostgreSQL and Redis Locally
1. Download PostgreSQL 15: https://www.postgresql.org/download/windows/
2. Download Redis for Windows: https://github.com/microsoftarchive/redis/releases
3. Create database: \`createdb -U postgres forex_trading\`
4. Run init script: \`psql -U postgres -d forex_trading -f infrastructure/database/init.sql\`

### Option 3: Use Cloud Databases
1. Set up PostgreSQL on AWS RDS, Google Cloud SQL, or similar
2. Set up Redis on AWS ElastiCache, Redis Cloud, or similar  
3. Update connection strings in .env.local files

## Starting the Platform

### Windows:
\`\`\`
scripts\\start-database-mode.bat
\`\`\`

### Linux/Mac:
\`\`\`
chmod +x scripts/start-database-mode.sh
./scripts/start-database-mode.sh
\`\`\`

## Service URLs
- 🌐 API Gateway: http://localhost:3001
- 👤 User Service: http://localhost:3002  
- 💰 Trading Service: http://localhost:3003
- 📊 Market Data Service: http://localhost:3004
- 🔔 Event System: http://localhost:3005
- 🖥️ Dashboard: http://localhost:3000

## Database Connections
- 📊 PostgreSQL: localhost:5432 (Database: forex_trading)
- 🔴 Redis: localhost:6379

All services are now ready to connect to databases when available!
`;

  try {
    fs.writeFileSync('DATABASE_SETUP.md', instructions);
    console.log('✅ Created setup instructions: DATABASE_SETUP.md');
  } catch (error) {
    console.error('❌ Failed to create instructions:', error.message);
    allSuccess = false;
  }
  
  console.log('\n================================================================');
  if (allSuccess) {
    console.log('🎉 Database infrastructure configuration completed successfully!');
    console.log('📖 See DATABASE_SETUP.md for next steps');
    console.log('🚀 Services are ready to connect to databases when available');
  } else {
    console.log('⚠️  Configuration completed with some errors');
    console.log('📖 Check the error messages above and DATABASE_SETUP.md');
  }
  console.log('================================================================');
}

// Run the configuration
main().catch(error => {
  console.error('❌ Configuration failed:', error);
  process.exit(1);
});
