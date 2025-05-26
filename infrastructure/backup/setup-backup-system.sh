#!/bin/bash

# =============================================================================
# Platform3 Backup System Setup Script
# Automated setup and configuration for backup and recovery system
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/opt/platform3/backups}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VENV_DIR="${BACKUP_BASE_DIR}/venv"

# Logging
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}" >&2
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*${NC}"
}

# Check if running as root for system setup
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This is recommended for initial system setup."
    else
        info "Running as non-root user. Some operations may require sudo."
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check for required commands
    local required_commands=("docker" "docker-compose" "python3" "pip3" "pg_dump" "redis-cli")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        error "Missing required commands: ${missing_commands[*]}"
        error "Please install the missing dependencies and run this script again."
        exit 1
    fi
    
    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2)
    local required_version="3.8"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        error "Python 3.8 or higher is required. Found: $python_version"
        exit 1
    fi
    
    # Check available disk space
    local available_space=$(df "$BACKUP_BASE_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
    local required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        warn "Low disk space. Available: $(($available_space / 1024 / 1024))GB, Recommended: 10GB+"
    fi
    
    log "System requirements check completed successfully"
}

# Create directory structure
create_directories() {
    log "Creating backup directory structure..."
    
    local directories=(
        "$BACKUP_BASE_DIR"
        "$BACKUP_BASE_DIR/logs"
        "$BACKUP_BASE_DIR/config"
        "$BACKUP_BASE_DIR/scripts"
        "$BACKUP_BASE_DIR/temp"
        "$BACKUP_BASE_DIR/archive"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        else
            info "Directory already exists: $dir"
        fi
    done
    
    # Set appropriate permissions
    chmod 755 "$BACKUP_BASE_DIR"
    chmod 750 "$BACKUP_BASE_DIR/config"
    chmod 755 "$BACKUP_BASE_DIR/logs"
    
    log "Directory structure created successfully"
}

# Setup Python virtual environment
setup_python_environment() {
    log "Setting up Python virtual environment..."
    
    if [ ! -d "$PYTHON_VENV_DIR" ]; then
        python3 -m venv "$PYTHON_VENV_DIR"
        log "Created Python virtual environment: $PYTHON_VENV_DIR"
    else
        info "Python virtual environment already exists"
    fi
    
    # Activate virtual environment and install dependencies
    source "$PYTHON_VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        pip install -r "$SCRIPT_DIR/requirements.txt"
        log "Installed Python dependencies"
    else
        warn "requirements.txt not found, skipping Python dependencies"
    fi
    
    deactivate
    log "Python environment setup completed"
}

# Copy configuration files
setup_configuration() {
    log "Setting up configuration files..."
    
    # Copy backup configuration
    if [ -f "$SCRIPT_DIR/config/backup-config.yaml" ]; then
        cp "$SCRIPT_DIR/config/backup-config.yaml" "$BACKUP_BASE_DIR/config/"
        log "Copied backup configuration"
    fi
    
    # Create monitoring configuration
    cat > "$BACKUP_BASE_DIR/config/monitor.json" << EOF
{
    "backup_base_dir": "$BACKUP_BASE_DIR",
    "retention_days": 30,
    "check_interval_seconds": 60,
    "metrics_interval_seconds": 300,
    "alert_thresholds": {
        "backup_failure_count": 3,
        "storage_usage_percent": 85,
        "backup_duration_hours": 2,
        "data_integrity_failures": 1
    },
    "notifications": {
        "email": {
            "enabled": true,
            "smtp_server": "localhost",
            "smtp_port": 587,
            "recipients": ["admin@platform3.com"]
        },
        "slack": {
            "enabled": false,
            "webhook_url": ""
        }
    },
    "database": {
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "forex_trading",
            "user": "forex_admin",
            "password": "ForexSecure2025!"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": "RedisSecure2025!"
        }
    }
}
EOF
    
    log "Configuration files setup completed"
}

# Copy and setup scripts
setup_scripts() {
    log "Setting up backup scripts..."
    
    # Copy main backup script
    if [ -f "$SCRIPT_DIR/backup-strategy.sh" ]; then
        cp "$SCRIPT_DIR/backup-strategy.sh" "$BACKUP_BASE_DIR/scripts/"
        chmod +x "$BACKUP_BASE_DIR/scripts/backup-strategy.sh"
        log "Copied backup strategy script"
    fi
    
    # Copy monitoring script
    if [ -f "$SCRIPT_DIR/backup-monitoring.py" ]; then
        cp "$SCRIPT_DIR/backup-monitoring.py" "$BACKUP_BASE_DIR/scripts/"
        chmod +x "$BACKUP_BASE_DIR/scripts/backup-monitoring.py"
        log "Copied backup monitoring script"
    fi
    
    # Create wrapper scripts
    cat > "$BACKUP_BASE_DIR/scripts/run-backup.sh" << 'EOF'
#!/bin/bash
# Wrapper script to run backup with virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$BACKUP_BASE_DIR/venv"

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Run backup script
exec "$SCRIPT_DIR/backup-strategy.sh" "$@"
EOF
    
    cat > "$BACKUP_BASE_DIR/scripts/run-monitoring.sh" << 'EOF'
#!/bin/bash
# Wrapper script to run monitoring with virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$BACKUP_BASE_DIR/venv"

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Run monitoring script
exec python "$SCRIPT_DIR/backup-monitoring.py" "$@"
EOF
    
    chmod +x "$BACKUP_BASE_DIR/scripts/run-backup.sh"
    chmod +x "$BACKUP_BASE_DIR/scripts/run-monitoring.sh"
    
    log "Backup scripts setup completed"
}

# Setup systemd services (if running as root)
setup_systemd_services() {
    if [[ $EUID -ne 0 ]]; then
        warn "Not running as root, skipping systemd service setup"
        return 0
    fi
    
    log "Setting up systemd services..."
    
    # Backup monitoring service
    cat > /etc/systemd/system/platform3-backup-monitor.service << EOF
[Unit]
Description=Platform3 Backup Monitoring Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=platform3
Group=platform3
WorkingDirectory=$BACKUP_BASE_DIR
ExecStart=$BACKUP_BASE_DIR/scripts/run-monitoring.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Backup timer service
    cat > /etc/systemd/system/platform3-backup.service << EOF
[Unit]
Description=Platform3 Backup Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
User=platform3
Group=platform3
WorkingDirectory=$BACKUP_BASE_DIR
ExecStart=$BACKUP_BASE_DIR/scripts/run-backup.sh
StandardOutput=journal
StandardError=journal
EOF
    
    cat > /etc/systemd/system/platform3-backup.timer << EOF
[Unit]
Description=Platform3 Backup Timer
Requires=platform3-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    # Reload systemd and enable services
    systemctl daemon-reload
    systemctl enable platform3-backup-monitor.service
    systemctl enable platform3-backup.timer
    
    log "Systemd services setup completed"
}

# Setup cron jobs (if not using systemd)
setup_cron_jobs() {
    if [[ $EUID -eq 0 ]]; then
        info "Running as root, systemd services are preferred over cron jobs"
        return 0
    fi
    
    log "Setting up cron jobs..."
    
    # Create cron job for daily backup
    (crontab -l 2>/dev/null || true; echo "0 2 * * * $BACKUP_BASE_DIR/scripts/run-backup.sh") | crontab -
    
    log "Cron jobs setup completed"
}

# Test backup system
test_backup_system() {
    log "Testing backup system..."
    
    # Check if Docker containers are running
    if ! docker ps | grep -q "forex-"; then
        warn "Platform3 containers are not running. Backup test will be limited."
    else
        info "Platform3 containers are running, performing full test"
        
        # Run a test backup
        if [ -x "$BACKUP_BASE_DIR/scripts/run-backup.sh" ]; then
            log "Running test backup..."
            "$BACKUP_BASE_DIR/scripts/run-backup.sh" || warn "Test backup failed"
        fi
    fi
    
    # Test monitoring script
    if [ -x "$BACKUP_BASE_DIR/scripts/run-monitoring.sh" ]; then
        log "Testing monitoring script..."
        timeout 10 "$BACKUP_BASE_DIR/scripts/run-monitoring.sh" || info "Monitoring test completed"
    fi
    
    log "Backup system test completed"
}

# Display setup summary
display_summary() {
    echo
    echo "=============================================="
    echo "Platform3 Backup System Setup Complete"
    echo "=============================================="
    echo
    echo "Backup Directory: $BACKUP_BASE_DIR"
    echo "Configuration: $BACKUP_BASE_DIR/config/"
    echo "Scripts: $BACKUP_BASE_DIR/scripts/"
    echo "Logs: $BACKUP_BASE_DIR/logs/"
    echo
    echo "Manual Commands:"
    echo "  Run Backup: $BACKUP_BASE_DIR/scripts/run-backup.sh"
    echo "  Start Monitor: $BACKUP_BASE_DIR/scripts/run-monitoring.sh"
    echo
    if [[ $EUID -eq 0 ]]; then
        echo "Systemd Services:"
        echo "  Start Monitor: systemctl start platform3-backup-monitor"
        echo "  Enable Timer: systemctl start platform3-backup.timer"
        echo "  Check Status: systemctl status platform3-backup-monitor"
    else
        echo "Cron Jobs:"
        echo "  Daily backup scheduled at 2:00 AM"
        echo "  View cron jobs: crontab -l"
    fi
    echo
    echo "Next Steps:"
    echo "1. Review configuration in $BACKUP_BASE_DIR/config/"
    echo "2. Test backup: $BACKUP_BASE_DIR/scripts/run-backup.sh"
    echo "3. Start monitoring: $BACKUP_BASE_DIR/scripts/run-monitoring.sh"
    echo "4. Check logs: tail -f $BACKUP_BASE_DIR/logs/backup-monitor.log"
    echo
}

# Main setup function
main() {
    log "Starting Platform3 Backup System Setup..."
    
    check_permissions
    check_requirements
    create_directories
    setup_python_environment
    setup_configuration
    setup_scripts
    
    if [[ $EUID -eq 0 ]]; then
        setup_systemd_services
    else
        setup_cron_jobs
    fi
    
    test_backup_system
    display_summary
    
    log "Setup completed successfully!"
}

# Handle script interruption
trap 'error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"
