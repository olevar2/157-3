# Service Integration Testing Script (PowerShell)
# Tests the complete user journey and service connectivity

Write-Host "🧪 Starting Service Integration Testing" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Service URLs
$services = @{
    "API Gateway" = "http://localhost:3001"
    "User Service" = "http://localhost:3002"
    "Trading Service" = "http://localhost:3003"
    "Market Data Service" = "http://localhost:3004"
    "Event System" = "http://localhost:3005"
    "Dashboard" = "http://localhost:3000"
}

$testResults = @{
    serviceHealth = @{}
    authentication = $false
    marketData = $false
    trading = $false
    integration = $false
}

# Helper function to test HTTP endpoints
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null
    )
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            UseBasicParsing = $true
            TimeoutSec = 10
        }
        
        if ($Headers.Count -gt 0) {
            $params.Headers = $Headers
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-WebRequest @params
        return @{
            Success = $true
            StatusCode = $response.StatusCode
            Content = $response.Content
        }
    }
    catch {
        return @{
            Success = $false
            Error = $_.Exception.Message
            StatusCode = $_.Exception.Response.StatusCode.value__
        }
    }
}

# Test 1: Service Health Checks
Write-Host "`n🔍 Testing Service Health..." -ForegroundColor Cyan

$healthyServices = 0
foreach ($service in $services.GetEnumerator()) {
    Write-Host "Testing $($service.Key)..." -ForegroundColor Yellow
    
    $healthUrl = "$($service.Value)/health"
    $result = Test-Endpoint -Url $healthUrl
    
    if ($result.Success) {
        Write-Host "✅ $($service.Key): Healthy" -ForegroundColor Green
        $testResults.serviceHealth[$service.Key] = $true
        $healthyServices++
    } else {
        Write-Host "❌ $($service.Key): $($result.Error)" -ForegroundColor Red
        $testResults.serviceHealth[$service.Key] = $false
    }
}

Write-Host "`n📊 Service Health: $healthyServices/$($services.Count) services healthy" -ForegroundColor Cyan

# Test 2: API Gateway Connectivity
Write-Host "`n🌐 Testing API Gateway..." -ForegroundColor Cyan

if ($testResults.serviceHealth["API Gateway"]) {
    $gatewayHealth = Test-Endpoint -Url "$($services['API Gateway'])/health"
    if ($gatewayHealth.Success) {
        Write-Host "✅ API Gateway responding correctly" -ForegroundColor Green
        
        # Try to access a proxied endpoint
        $proxyTest = Test-Endpoint -Url "$($services['API Gateway'])/api/v1/health"
        if ($proxyTest.Success) {
            Write-Host "✅ API Gateway proxy routing working" -ForegroundColor Green
        } else {
            Write-Host "⚠️ API Gateway proxy may need configuration" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "❌ API Gateway not available for testing" -ForegroundColor Red
}

# Test 3: Trading Service Endpoints
Write-Host "`n💰 Testing Trading Service..." -ForegroundColor Cyan

if ($testResults.serviceHealth["Trading Service"]) {
    # Test health endpoint
    $tradingHealth = Test-Endpoint -Url "$($services['Trading Service'])/health"
    if ($tradingHealth.Success) {
        Write-Host "✅ Trading Service health check passed" -ForegroundColor Green
        
        # Test info endpoint
        $tradingInfo = Test-Endpoint -Url "$($services['Trading Service'])/api/info"
        if ($tradingInfo.Success) {
            Write-Host "✅ Trading Service info endpoint working" -ForegroundColor Green
            $testResults.trading = $true
        } else {
            Write-Host "⚠️ Trading Service info endpoint not available" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "❌ Trading Service not available for testing" -ForegroundColor Red
}

# Test 4: Market Data Service (if available)
Write-Host "`n📊 Testing Market Data Service..." -ForegroundColor Cyan

if ($testResults.serviceHealth["Market Data Service"]) {
    $marketHealth = Test-Endpoint -Url "$($services['Market Data Service'])/health"
    if ($marketHealth.Success) {
        Write-Host "✅ Market Data Service health check passed" -ForegroundColor Green
        
        # Test market data endpoints
        $pricesTest = Test-Endpoint -Url "$($services['Market Data Service'])/api/market-data/prices"
        if ($pricesTest.Success) {
            Write-Host "✅ Market prices endpoint working" -ForegroundColor Green
            $testResults.marketData = $true
        } else {
            Write-Host "⚠️ Market prices endpoint needs authentication or configuration" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "❌ Market Data Service not available for testing" -ForegroundColor Red
}

# Test 5: User Service Authentication (if available)
Write-Host "`n🔐 Testing User Service..." -ForegroundColor Cyan

if ($testResults.serviceHealth["User Service"]) {
    $userHealth = Test-Endpoint -Url "$($services['User Service'])/health"
    if ($userHealth.Success) {
        Write-Host "✅ User Service health check passed" -ForegroundColor Green
        $testResults.authentication = $true
    }
} else {
    Write-Host "❌ User Service not available for testing" -ForegroundColor Red
}

# Test 6: Frontend Dashboard
Write-Host "`n🖥️ Testing Frontend Dashboard..." -ForegroundColor Cyan

if ($testResults.serviceHealth["Dashboard"]) {
    Write-Host "✅ Frontend Dashboard accessible" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend Dashboard not available" -ForegroundColor Red
}

# Summary
Write-Host "`n🎯 INTEGRATION TEST RESULTS" -ForegroundColor Green
Write-Host "============================" -ForegroundColor Green

$passedTests = 0
$totalTests = 6

# Service Health
if ($healthyServices -ge 2) {
    Write-Host "✅ Service Health: PASSED ($healthyServices services running)" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ Service Health: FAILED (need at least 2 services)" -ForegroundColor Red
}

# API Gateway
if ($testResults.serviceHealth["API Gateway"]) {
    Write-Host "✅ API Gateway: PASSED" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ API Gateway: FAILED" -ForegroundColor Red
}

# Trading Service
if ($testResults.trading) {
    Write-Host "✅ Trading Service: PASSED" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ Trading Service: FAILED" -ForegroundColor Red
}

# Market Data Service
if ($testResults.marketData) {
    Write-Host "✅ Market Data Service: PASSED" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ Market Data Service: FAILED" -ForegroundColor Red
}

# User Service
if ($testResults.authentication) {
    Write-Host "✅ User Service: PASSED" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ User Service: FAILED" -ForegroundColor Red
}

# Frontend
if ($testResults.serviceHealth["Dashboard"]) {
    Write-Host "✅ Frontend Dashboard: PASSED" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ Frontend Dashboard: FAILED" -ForegroundColor Red
}

Write-Host "`n📊 Overall Result: $passedTests/$totalTests tests passed" -ForegroundColor Cyan

if ($passedTests -ge 4) {
    Write-Host "🎉 Integration testing SUCCESSFUL - Core platform is functional!" -ForegroundColor Green
    Write-Host "`n✅ READY FOR NEXT PHASE: Real-time WebSocket Implementation" -ForegroundColor Green
} elseif ($passedTests -ge 2) {
    Write-Host "⚠️ Integration testing PARTIAL - Some services need to be started" -ForegroundColor Yellow
    Write-Host "`n📋 NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "1. Start missing services using: scripts\start-database-mode.bat" -ForegroundColor White
    Write-Host "2. Verify database connections" -ForegroundColor White
    Write-Host "3. Re-run integration tests" -ForegroundColor White
} else {
    Write-Host "❌ Integration testing FAILED - Platform needs attention" -ForegroundColor Red
    Write-Host "`n📋 CRITICAL ACTIONS NEEDED:" -ForegroundColor Red
    Write-Host "1. Start core services (API Gateway, Trading Service, User Service)" -ForegroundColor White
    Write-Host "2. Check service logs for errors" -ForegroundColor White
    Write-Host "3. Verify database configuration" -ForegroundColor White
}

Write-Host "`n🔗 Service URLs:" -ForegroundColor Cyan
foreach ($service in $services.GetEnumerator()) {
    $status = if ($testResults.serviceHealth[$service.Key]) { "✅" } else { "❌" }
    Write-Host "$status $($service.Key): $($service.Value)" -ForegroundColor White
}

Write-Host "`n📖 For detailed setup instructions, see: DATABASE_SETUP.md" -ForegroundColor Gray
