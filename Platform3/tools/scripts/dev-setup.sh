#!/bin/bash

# Personal Forex Trading Platform - Development Setup Script
# This script sets up the complete development environment for the server-based platform

set -e

echo "🚀 Personal Forex Trading Platform - Development Setup"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker Compose is available"
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    # Logs directories
    mkdir -p logs/gateway
    mkdir -p logs/user-service
    mkdir -p logs/market-data
    mkdir -p logs/trading-service
    mkdir -p logs/risk-service
    mkdir -p logs/analytics
    mkdir -p logs/payment
    mkdir -p logs/notifications
    
    # Data directories
    mkdir -p data/postgres
    mkdir -p data/redis
    mkdir -p data/influxdb
    mkdir -p data/kafka
    mkdir -p data/prometheus
    mkdir -p data/grafana
    
    # Config directories
    mkdir -p infrastructure/monitoring/grafana/dashboards
    mkdir -p infrastructure/monitoring/grafana/datasources
    mkdir -p infrastructure/database/schemas
    
    print_success "Directories created successfully"
}

# Generate secure passwords if .env doesn't exist
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Please create one based on .env.example"
        return 1
    fi
    
    # Check if .env has default passwords and warn user
    if grep -q "ForexSecure2025!" .env; then
        print_warning "⚠️  Default passwords detected in .env file!"
        print_warning "⚠️  Please change all passwords before deploying to production!"
        echo ""
        echo "To generate secure passwords, you can use:"
        echo "openssl rand -base64 32"
        echo ""
    fi
    
    print_success "Environment configuration checked"
}

# Create Prometheus configuration
setup_prometheus() {
    print_status "Setting up Prometheus configuration..."
    
    cat > infrastructure/monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

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
EOF

    print_success "Prometheus configuration created"
}

# Create Grafana datasource configuration
setup_grafana() {
    print_status "Setting up Grafana configuration..."
    
    # Datasources
    cat > infrastructure/monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Dashboard provisioning
    cat > infrastructure/monitoring/grafana/dashboards/dashboard.yml << 'EOF'
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
EOF

    print_success "Grafana configuration created"
}

# Start the platform
start_platform() {
    print_status "Starting the Personal Forex Trading Platform..."
    
    # Pull latest images
    print_status "Pulling Docker images..."
    docker-compose pull
    
    # Start infrastructure services first
    print_status "Starting infrastructure services..."
    docker-compose up -d postgres redis influxdb zookeeper kafka prometheus grafana
    
    # Wait for databases to be ready
    print_status "Waiting for databases to be ready..."
    sleep 30
    
    # Check if services are healthy
    print_status "Checking service health..."
    
    # Wait for PostgreSQL
    until docker-compose exec -T postgres pg_isready -U forex_admin -d forex_trading; do
        print_status "Waiting for PostgreSQL to be ready..."
        sleep 5
    done
    
    print_success "Infrastructure services are running!"
    
    # Note about application services
    print_warning "Application services will be started after implementation in Phase 1F"
    
    echo ""
    echo "🎉 Development environment setup complete!"
    echo ""
    echo "📊 Access Points:"
    echo "  - Grafana Dashboard: http://localhost:3010 (admin/GrafanaSecure2025!)"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - InfluxDB: http://localhost:8086"
    echo "  - PostgreSQL: localhost:5432 (forex_admin/ForexSecure2025!)"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Services will be implemented in upcoming phases"
    echo "  2. Web dashboard will be available at http://localhost:8080"
    echo "  3. API Gateway will be at http://localhost:3000"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  - View logs: docker-compose logs -f [service-name]"
    echo "  - Stop platform: docker-compose down"
    echo "  - Restart service: docker-compose restart [service-name]"
    echo "  - View status: docker-compose ps"
}

# Main execution
main() {
    echo "Starting setup process..."
    echo ""
    
    check_docker
    check_docker_compose
    setup_directories
    setup_environment
    setup_prometheus
    setup_grafana
    start_platform
    
    print_success "Personal Forex Trading Platform development environment is ready!"
}

# Execute main function
main "$@"
