@echo off
echo 🚀 Starting Forex Trading Platform in Database Mode
echo ==================================================

echo 🔍 Checking if services are ready...
echo.

echo ✅ Starting services in database mode...
echo.

echo Starting API Gateway (Port 3001)...
start "API Gateway" cmd /k "cd services\api-gateway && npm start"

timeout /t 3 /nobreak >nul

echo Starting User Service (Port 3002)...
start "User Service" cmd /k "cd services\user-service && npm run dev"

timeout /t 3 /nobreak >nul

echo Starting Trading Service (Port 3003)...
start "Trading Service" cmd /k "cd services\trading-service && npm run dev"

timeout /t 3 /nobreak >nul

echo Starting Market Data Service (Port 3004)...
start "Market Data Service" cmd /k "cd services\market-data-service && npm start"

timeout /t 3 /nobreak >nul

echo Starting Event System (Port 3005)...
start "Event System" cmd /k "cd services\event-system && npm run dev"

timeout /t 3 /nobreak >nul

echo Starting Dashboard (Port 3000)...
start "Dashboard" cmd /k "cd dashboard\frontend && npm start"

echo.
echo ✅ All services started in database mode!
echo ==================================================
echo 🌐 API Gateway: http://localhost:3001
echo 👤 User Service: http://localhost:3002
echo 💰 Trading Service: http://localhost:3003
echo 📊 Market Data Service: http://localhost:3004
echo 🔔 Event System: http://localhost:3005
echo 🖥️ Dashboard: http://localhost:3000
echo.
echo Services are running in separate windows.
echo Close individual windows to stop services.
echo.
echo 📖 See DATABASE_SETUP.md for database configuration
pause
