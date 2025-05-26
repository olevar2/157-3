#!/usr/bin/env pwsh
# PostgreSQL Database Setup Script for Windows
# Run this script to initialize the forex trading platform database

Write-Host "🗃️  Setting up PostgreSQL Database for Forex Trading Platform..." -ForegroundColor Green

# Configuration
$DB_NAME = "forex_platform"
$DB_USER = "postgres"
$DB_PASSWORD = "postgres"
$DB_HOST = "localhost"
$DB_PORT = "5432"

# Check if PostgreSQL is installed
Write-Host "📍 Checking PostgreSQL installation..." -ForegroundColor Yellow

try {
    $pgVersion = psql --version
    Write-Host "✅ PostgreSQL found: $pgVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ PostgreSQL not found. Please install PostgreSQL first." -ForegroundColor Red
    Write-Host "💡 Download from: https://www.postgresql.org/download/windows/" -ForegroundColor Cyan
    exit 1
}

# Test connection to PostgreSQL
Write-Host "🔗 Testing database connection..." -ForegroundColor Yellow

$env:PGPASSWORD = $DB_PASSWORD
$connectionTest = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "SELECT version();" postgres 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to connect to PostgreSQL. Please check:" -ForegroundColor Red
    Write-Host "   - PostgreSQL service is running" -ForegroundColor Yellow
    Write-Host "   - Username/password are correct" -ForegroundColor Yellow
    Write-Host "   - Connection parameters are valid" -ForegroundColor Yellow
    Write-Host "Error: $connectionTest" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Database connection successful" -ForegroundColor Green
}

# Create database if it doesn't exist
Write-Host "🆕 Creating database '$DB_NAME'..." -ForegroundColor Yellow

$createDbResult = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME;" postgres 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Database '$DB_NAME' created successfully" -ForegroundColor Green
} elseif ($createDbResult -like "*already exists*") {
    Write-Host "ℹ️  Database '$DB_NAME' already exists" -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to create database: $createDbResult" -ForegroundColor Red
    exit 1
}

# Initialize database schema
Write-Host "🏗️  Initializing database schema..." -ForegroundColor Yellow

$schemaPath = "d:\MD\Platform3\Platform3\database\init\001_create_database_structure.sql"
if (Test-Path $schemaPath) {
    $schemaResult = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $schemaPath 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database schema created successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create schema: $schemaResult" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Schema file not found: $schemaPath" -ForegroundColor Red
    exit 1
}

# Seed initial data
Write-Host "🌱 Seeding initial data..." -ForegroundColor Yellow

$seedPath = "d:\MD\Platform3\Platform3\database\init\002_seed_initial_data.sql"
if (Test-Path $seedPath) {
    $seedResult = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $seedPath 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Initial data seeded successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to seed data: $seedResult" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Seed file not found: $seedPath" -ForegroundColor Red
    exit 1
}

# Create functions and procedures
Write-Host "⚙️  Creating database functions..." -ForegroundColor Yellow

$functionsPath = "d:\MD\Platform3\Platform3\database\init\003_functions_and_procedures.sql"
if (Test-Path $functionsPath) {
    $functionsResult = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $functionsPath 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database functions created successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create functions: $functionsResult" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Functions file not found: $functionsPath" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "🔍 Verifying database setup..." -ForegroundColor Yellow

$verifyResult = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
SELECT 
    'Users' as table_name, 
    COUNT(*) as record_count 
FROM users 
UNION ALL 
SELECT 
    'Currency Pairs' as table_name, 
    COUNT(*) as record_count 
FROM currency_pairs 
UNION ALL 
SELECT 
    'Trading Accounts' as table_name, 
    COUNT(*) as record_count 
FROM trading_accounts
UNION ALL
SELECT 
    'System Config' as table_name, 
    COUNT(*) as record_count 
FROM system_config
ORDER BY table_name;" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Database verification successful:" -ForegroundColor Green
    Write-Host $verifyResult -ForegroundColor Cyan
} else {
    Write-Host "❌ Database verification failed: $verifyResult" -ForegroundColor Red
    exit 1
}

# Create database connection info file
$connectionInfo = @"
# Database Connection Information
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME

# Test connection:
# psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME

# Demo user credentials:
# Email: demo@forexplatform.com
# Password: DemoPassword123!

# Admin user credentials:
# Email: admin@forexplatform.com  
# Password: AdminSecure2024!
"@

$connectionInfo | Out-File -FilePath "d:\MD\Platform3\Platform3\database\connection_info.txt" -Encoding UTF8

Write-Host "" -ForegroundColor White
Write-Host "🎉 Database setup completed successfully!" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "📋 Summary:" -ForegroundColor Cyan
Write-Host "   • Database: $DB_NAME" -ForegroundColor White
Write-Host "   • Host: $DB_HOST:$DB_PORT" -ForegroundColor White
Write-Host "   • Connection URL: postgresql://$DB_USER:****@$DB_HOST:$DB_PORT/$DB_NAME" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "👤 Demo Users Created:" -ForegroundColor Cyan
Write-Host "   • Email: demo@forexplatform.com | Password: DemoPassword123!" -ForegroundColor White
Write-Host "   • Email: admin@forexplatform.com | Password: AdminSecure2024!" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "🔧 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Start User Service: cd services\user-service && npm install && npm run dev" -ForegroundColor White
Write-Host "   2. Start Trading Service: cd services\trading-service && npm install && npm run dev" -ForegroundColor White
Write-Host "   3. Update frontend to use real authentication" -ForegroundColor White
Write-Host "" -ForegroundColor White

# Clean up environment variable
$env:PGPASSWORD = $null
