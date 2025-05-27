# Platform3 Comprehensive Validation Runner
# Executes the complete validation suite for the 67-indicator system

Write-Host "🚀 Platform3 67-Indicator System - Comprehensive Validation" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""

# Set location to analytics service directory
$analyticsServiceDir = "d:\MD\Platform3\Platform3\services\analytics-service"
Set-Location $analyticsServiceDir

Write-Host "📂 Working Directory: $analyticsServiceDir" -ForegroundColor Yellow
Write-Host ""

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    Write-Host ""
}

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    Write-Host "📁 Created logs directory" -ForegroundColor Yellow
}

# Compile TypeScript if needed
Write-Host "🔨 Compiling TypeScript..." -ForegroundColor Yellow
npx tsc --noEmit
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ TypeScript compilation had issues, but continuing..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Starting Comprehensive Validation Suite..." -ForegroundColor Green
Write-Host "   - 🔬 Calculation Accuracy (4 tests)" -ForegroundColor Cyan
Write-Host "   - 📊 Real Data Processing (4 tests)" -ForegroundColor Cyan  
Write-Host "   - ⚡ Performance (5 tests)" -ForegroundColor Cyan
Write-Host "   - 🛡️ Error Handling (4 tests)" -ForegroundColor Cyan
Write-Host "   - 🔄 Integration (3 tests)" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Total: 20 comprehensive validation tests" -ForegroundColor Cyan
Write-Host ""

# Run the validation suite
npx ts-node src/test/run-comprehensive-validation.ts

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "🎉 VALIDATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "✅ System is ready for production deployment" -ForegroundColor Green
} else {
    Write-Host "⚠️ VALIDATION COMPLETED WITH ISSUES" -ForegroundColor Yellow
    Write-Host "🔍 Check the detailed reports for specific failures" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📊 Reports Generated:" -ForegroundColor Yellow
Write-Host "   📄 JSON: logs/comprehensive-validation-results.json" -ForegroundColor White
Write-Host "   📋 HTML: logs/validation-report.html" -ForegroundColor White
Write-Host "   📝 Log: logs/validation-test-results.log" -ForegroundColor White
Write-Host ""

# Open HTML report in browser if successful
if ($exitCode -eq 0) {
    $htmlReport = Join-Path $analyticsServiceDir "logs\validation-report.html"
    if (Test-Path $htmlReport) {
        Write-Host "🌐 Opening validation report in browser..." -ForegroundColor Green
        Start-Process $htmlReport
    }
}

Write-Host "🏁 Validation process completed." -ForegroundColor Green
Write-Host ""

exit $exitCode
