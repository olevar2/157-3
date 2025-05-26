#!/bin/bash

# =============================================================================
# Platform3 Forex Trading Platform - Comprehensive Backup Strategy
# Optimized for short-term trading data integrity and disaster recovery
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/opt/platform3/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Security enhancements
BACKUP_ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"
ENABLE_ENCRYPTION="${ENABLE_ENCRYPTION:-false}"
BACKUP_CHECKSUM_ALGORITHM="${BACKUP_CHECKSUM_ALGORITHM:-sha256}"

# Enhanced error handling
set -euo pipefail
trap 'handle_error $? $LINENO' ERR

handle_error() {
    local exit_code=$1
    local line_number=$2
    error "Script failed with exit code $exit_code at line $line_number"
    cleanup_on_error
    exit $exit_code
}

cleanup_on_error() {
    log "Performing cleanup after error..."
    # Remove incomplete backup directory
    if [[ -n "${BACKUP_DIR:-}" && -d "$BACKUP_DIR" ]]; then
        log "Removing incomplete backup directory: $BACKUP_DIR"
        rm -rf "$BACKUP_DIR"
    fi
}

# Logging
LOG_FILE="${BACKUP_BASE_DIR}/logs/backup-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Database connection parameters
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-forex_trading}"
POSTGRES_USER="${POSTGRES_USER:-forex_admin}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-ForexSecure2025!}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-RedisSecure2025!}"

INFLUXDB_HOST="${INFLUXDB_HOST:-localhost}"
INFLUXDB_PORT="${INFLUXDB_PORT:-8086}"
INFLUXDB_ORG="${INFLUXDB_ORG:-forex-trading}"
INFLUXDB_BUCKET="${INFLUXDB_BUCKET:-market-data}"
INFLUXDB_TOKEN="${INFLUXDB_TOKEN:-}"

# Backup timestamp
BACKUP_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="${BACKUP_BASE_DIR}/${BACKUP_TIMESTAMP}"

# Create backup directories
create_backup_structure() {
    log "Creating backup directory structure..."
    mkdir -p "${BACKUP_DIR}"/{postgresql,redis,influxdb,kafka,application,logs,monitoring}
    mkdir -p "${BACKUP_BASE_DIR}/logs"
}

# PostgreSQL backup - Critical trading data
backup_postgresql() {
    log "Starting PostgreSQL backup..."
    
    local pg_backup_file="${BACKUP_DIR}/postgresql/forex_trading_${BACKUP_TIMESTAMP}.sql"
    local pg_custom_file="${BACKUP_DIR}/postgresql/forex_trading_${BACKUP_TIMESTAMP}.custom"
    
    # Full SQL dump for maximum compatibility
    log "Creating PostgreSQL SQL dump..."
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --verbose \
        --no-password \
        --create \
        --clean \
        --if-exists \
        > "$pg_backup_file"
    
    # Custom format for faster restore
    log "Creating PostgreSQL custom format backup..."
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --verbose \
        --no-password \
        --format=custom \
        --compress="$COMPRESSION_LEVEL" \
        --file="$pg_custom_file"
    
    # Table-specific backups for critical trading data
    log "Creating table-specific backups for critical data..."
    local critical_tables=("orders" "positions" "transactions" "accounts" "market_data_cache")
    
    for table in "${critical_tables[@]}"; do
        local table_file="${BACKUP_DIR}/postgresql/${table}_${BACKUP_TIMESTAMP}.sql"
        PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --verbose \
            --no-password \
            --table="$table" \
            > "$table_file" 2>/dev/null || log "Warning: Table $table not found or empty"
    done
    
    # Compress SQL files
    log "Compressing PostgreSQL backups..."
    gzip -"$COMPRESSION_LEVEL" "${BACKUP_DIR}/postgresql"/*.sql
    
    # Encrypt backups if enabled
    encrypt_file "$pg_backup_file"
    encrypt_file "$pg_custom_file"
    for table in "${critical_tables[@]}"; do
        local table_file="${BACKUP_DIR}/postgresql/${table}_${BACKUP_TIMESTAMP}.sql.gz"
        encrypt_file "$table_file"
    done
    
    log "PostgreSQL backup completed successfully"
}

# Redis backup - Session and cache data
backup_redis() {
    log "Starting Redis backup..."
    
    local redis_backup_file="${BACKUP_DIR}/redis/redis_${BACKUP_TIMESTAMP}.rdb"
    local redis_config_file="${BACKUP_DIR}/redis/redis_config_${BACKUP_TIMESTAMP}.conf"
    
    # Create Redis backup using BGSAVE
    log "Triggering Redis background save..."
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning BGSAVE
    
    # Wait for background save to complete
    while [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning LASTSAVE)" = "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning LASTSAVE)" ]; do
        sleep 1
    done
    
    # Copy RDB file
    log "Copying Redis RDB file..."
    docker cp forex-redis:/data/dump.rdb "$redis_backup_file" || {
        error "Failed to copy Redis RDB file"
        return 1
    }
    
    # Backup Redis configuration
    log "Backing up Redis configuration..."
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning CONFIG GET '*' > "$redis_config_file"
    
    # Export key-value pairs for critical data
    log "Exporting critical Redis keys..."
    local redis_keys_file="${BACKUP_DIR}/redis/redis_keys_${BACKUP_TIMESTAMP}.txt"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning --scan --pattern "trading:*" | \
        xargs -I {} redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DUMP {} > "$redis_keys_file"
    
    # Encrypt backups if enabled
    encrypt_file "$redis_backup_file"
    encrypt_file "$redis_config_file"
    encrypt_file "$redis_keys_file"
    
    log "Redis backup completed successfully"
}

# InfluxDB backup - Time-series market data
backup_influxdb() {
    log "Starting InfluxDB backup..."
    
    local influx_backup_dir="${BACKUP_DIR}/influxdb"
    
    if [ -z "$INFLUXDB_TOKEN" ]; then
        log "Warning: INFLUXDB_TOKEN not set, skipping InfluxDB backup"
        return 0
    fi
    
    # Backup InfluxDB using influx CLI
    log "Creating InfluxDB backup..."
    docker exec forex-influxdb influx backup \
        --host "http://localhost:8086" \
        --token "$INFLUXDB_TOKEN" \
        --org "$INFLUXDB_ORG" \
        --bucket "$INFLUXDB_BUCKET" \
        "/tmp/influx_backup_${BACKUP_TIMESTAMP}" || {
        error "Failed to create InfluxDB backup"
        return 1
    }
    
    # Copy backup from container
    log "Copying InfluxDB backup from container..."
    docker cp "forex-influxdb:/tmp/influx_backup_${BACKUP_TIMESTAMP}" "$influx_backup_dir/"
    
    # Export recent market data as CSV for additional safety
    log "Exporting recent market data as CSV..."
    local csv_file="${influx_backup_dir}/market_data_${BACKUP_TIMESTAMP}.csv"
    docker exec forex-influxdb influx query \
        --host "http://localhost:8086" \
        --token "$INFLUXDB_TOKEN" \
        --org "$INFLUXDB_ORG" \
        'from(bucket: "market-data") |> range(start: -7d) |> yield()' \
        --raw > "$csv_file" || log "Warning: Failed to export CSV data"
    
    # Encrypt backups if enabled
    for file in "$influx_backup_dir"/*; do
        encrypt_file "$file"
    done
    
    log "InfluxDB backup completed successfully"
}

# Kafka backup - Event streams and topics
backup_kafka() {
    log "Starting Kafka backup..."
    
    local kafka_backup_dir="${BACKUP_DIR}/kafka"
    
    # Backup Kafka topics metadata
    log "Backing up Kafka topics metadata..."
    docker exec forex-kafka kafka-topics.sh \
        --bootstrap-server localhost:9092 \
        --list > "${kafka_backup_dir}/topics_${BACKUP_TIMESTAMP}.txt"
    
    # Backup topic configurations
    log "Backing up Kafka topic configurations..."
    while IFS= read -r topic; do
        docker exec forex-kafka kafka-configs.sh \
            --bootstrap-server localhost:9092 \
            --entity-type topics \
            --entity-name "$topic" \
            --describe > "${kafka_backup_dir}/config_${topic}_${BACKUP_TIMESTAMP}.txt"
    done < "${kafka_backup_dir}/topics_${BACKUP_TIMESTAMP}.txt"
    
    # Copy Kafka data directory
    log "Copying Kafka data directory..."
    docker cp forex-kafka:/var/lib/kafka/data "${kafka_backup_dir}/data" || {
        log "Warning: Failed to copy Kafka data directory"
    }
    
    # Encrypt backups if enabled
    for file in "${kafka_backup_dir}"/*; do
        encrypt_file "$file"
    done
    
    log "Kafka backup completed successfully"
}

# Application files backup
backup_application() {
    log "Starting application files backup..."
    
    local app_backup_dir="${BACKUP_DIR}/application"
    
    # Backup configuration files
    log "Backing up configuration files..."
    cp -r "$SCRIPT_DIR/../../" "$app_backup_dir/platform3_source" || {
        error "Failed to backup application source"
        return 1
    }
    
    # Remove node_modules and other large directories
    find "$app_backup_dir/platform3_source" -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$app_backup_dir/platform3_source" -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$app_backup_dir/platform3_source" -name "logs" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Backup Docker volumes
    log "Backing up Docker volumes..."
    local volumes=("postgres_data" "redis_data" "influxdb_data" "kafka_data" "prometheus_data" "grafana_data")
    
    for volume in "${volumes[@]}"; do
        log "Backing up volume: $volume"
        docker run --rm \
            -v "platform3_${volume}:/source:ro" \
            -v "${app_backup_dir}:/backup" \
            alpine:latest \
            tar czf "/backup/${volume}_${BACKUP_TIMESTAMP}.tar.gz" -C /source . || {
            log "Warning: Failed to backup volume $volume"
        }
    done
    
    # Encrypt backups if enabled
    for file in "$app_backup_dir"/*; do
        encrypt_file "$file"
    done
    
    log "Application files backup completed successfully"
}

# Monitoring and logs backup
backup_monitoring() {
    log "Starting monitoring data backup..."
    
    local monitoring_backup_dir="${BACKUP_DIR}/monitoring"
    
    # Backup Grafana dashboards and settings
    log "Backing up Grafana data..."
    docker cp forex-grafana:/var/lib/grafana "${monitoring_backup_dir}/grafana" || {
        log "Warning: Failed to backup Grafana data"
    }
    
    # Backup Prometheus data (last 7 days)
    log "Backing up Prometheus data..."
    docker cp forex-prometheus:/prometheus "${monitoring_backup_dir}/prometheus" || {
        log "Warning: Failed to backup Prometheus data"
    }
    
    # Backup application logs
    log "Backing up application logs..."
    mkdir -p "${monitoring_backup_dir}/logs"
    find "$SCRIPT_DIR/../../" -name "*.log" -mtime -7 -exec cp {} "${monitoring_backup_dir}/logs/" \; 2>/dev/null || true
    
    # Encrypt backups if enabled
    for file in "${monitoring_backup_dir}"/*; do
        encrypt_file "$file"
    done
    
    log "Monitoring data backup completed successfully"
}

# Create backup manifest
create_backup_manifest() {
    log "Creating backup manifest..."
    
    local manifest_file="${BACKUP_DIR}/backup_manifest.json"
    
    cat > "$manifest_file" << EOF
{
    "backup_timestamp": "$BACKUP_TIMESTAMP",
    "backup_type": "full",
    "platform_version": "3.0",
    "created_at": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "components": {
        "postgresql": {
            "version": "$(docker exec forex-postgres psql --version | head -1)",
            "database": "$POSTGRES_DB",
            "backup_files": [
                "postgresql/forex_trading_${BACKUP_TIMESTAMP}.sql.gz",
                "postgresql/forex_trading_${BACKUP_TIMESTAMP}.custom"
            ]
        },
        "redis": {
            "version": "$(docker exec forex-redis redis-server --version)",
            "backup_files": [
                "redis/redis_${BACKUP_TIMESTAMP}.rdb",
                "redis/redis_config_${BACKUP_TIMESTAMP}.conf"
            ]
        },
        "influxdb": {
            "version": "$(docker exec forex-influxdb influx version | head -1)",
            "org": "$INFLUXDB_ORG",
            "bucket": "$INFLUXDB_BUCKET",
            "backup_files": [
                "influxdb/influx_backup_${BACKUP_TIMESTAMP}/",
                "influxdb/market_data_${BACKUP_TIMESTAMP}.csv"
            ]
        },
        "kafka": {
            "version": "$(docker exec forex-kafka kafka-topics.sh --version | head -1)",
            "backup_files": [
                "kafka/topics_${BACKUP_TIMESTAMP}.txt",
                "kafka/data/"
            ]
        }
    },
    "backup_size_mb": $(du -sm "$BACKUP_DIR" | cut -f1),
    "retention_policy": {
        "retention_days": $RETENTION_DAYS,
        "cleanup_date": "$(date -d "+$RETENTION_DAYS days" -Iseconds)"
    }
}
EOF
    
    log "Backup manifest created: $manifest_file"
}

# Enhanced checksum generation
generate_checksums() {
    local backup_dir="$1"
    local checksum_file="${backup_dir}/checksums.${BACKUP_CHECKSUM_ALGORITHM}"
    
    log "Generating ${BACKUP_CHECKSUM_ALGORITHM} checksums..."
    find "$backup_dir" -type f ! -name "checksums.*" -exec ${BACKUP_CHECKSUM_ALGORITHM}sum {} \; > "$checksum_file"
    log "Checksums generated: $checksum_file"
}

# Encryption functions
encrypt_file() {
    local file="$1"
    
    if [[ "$ENABLE_ENCRYPTION" == "true" && -n "$BACKUP_ENCRYPTION_KEY" && -f "$file" ]]; then
        log "Encrypting file: $(basename "$file")"
        
        # Use AES-256-CBC with PBKDF2 for key derivation
        openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
            -in "$file" \
            -out "${file}.enc" \
            -k "$BACKUP_ENCRYPTION_KEY"
        
        if [[ $? -eq 0 ]]; then
            # Remove original file after successful encryption
            rm "$file"
            log "File encrypted successfully: $(basename "${file}.enc")"
        else
            error "Failed to encrypt file: $file"
            return 1
        fi
    fi
}

decrypt_file() {
    local encrypted_file="$1"
    local output_file="${encrypted_file%.enc}"
    
    if [[ -f "$encrypted_file" && -n "$BACKUP_ENCRYPTION_KEY" ]]; then
        log "Decrypting file: $(basename "$encrypted_file")"
        
        openssl enc -aes-256-cbc -d -salt -pbkdf2 -iter 100000 \
            -in "$encrypted_file" \
            -out "$output_file" \
            -k "$BACKUP_ENCRYPTION_KEY"
        
        if [[ $? -eq 0 ]]; then
            log "File decrypted successfully: $(basename "$output_file")"
            return 0
        else
            error "Failed to decrypt file: $encrypted_file"
            return 1
        fi
    else
        error "Encryption key not provided or file not found: $encrypted_file"
        return 1
    fi
}

# Remote transfer with retry mechanism
transfer_to_remote() {
    local source="$1"
    local destination="$2"
    local max_retries=3
    local retry_delay=5
    
    for ((i=1; i<=max_retries; i++)); do
        log "Attempting remote transfer (attempt $i/$max_retries)..."
        
        if rsync -avz --progress --timeout=300 "$source" "$destination"; then
            log "Remote transfer completed successfully"
            return 0
        else
            if [[ $i -lt $max_retries ]]; then
                log "Transfer failed, retrying in $retry_delay seconds..."
                sleep $retry_delay
                retry_delay=$((retry_delay * 2))  # Exponential backoff
            else
                error "Remote transfer failed after $max_retries attempts"
                return 1
            fi
        fi
    done
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "20*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; || {
        log "Warning: Failed to cleanup some old backups"
    }
    
    # Cleanup old log files
    find "$BACKUP_BASE_DIR/logs" -name "backup-*.log" -mtime +$RETENTION_DAYS -delete || true
    
    log "Cleanup completed"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    local verification_file="${BACKUP_DIR}/verification_${BACKUP_TIMESTAMP}.txt"
    
    {
        echo "Backup Verification Report"
        echo "========================="
        echo "Timestamp: $(date)"
        echo "Backup Directory: $BACKUP_DIR"
        echo ""
        
        echo "File Checksums:"
        find "$BACKUP_DIR" -type f -exec sha256sum {} \;
        
        echo ""
        echo "Directory Structure:"
        tree "$BACKUP_DIR" 2>/dev/null || find "$BACKUP_DIR" -type d
        
        echo ""
        echo "Backup Sizes:"
        du -sh "$BACKUP_DIR"/*
        
    } > "$verification_file"
    
    log "Backup verification completed: $verification_file"
}

# Main backup function
main() {
    log "Starting Platform3 Forex Trading Platform backup..."
    log "Backup timestamp: $BACKUP_TIMESTAMP"
    log "Backup directory: $BACKUP_DIR"
    
    # Check if Docker containers are running
    if ! docker ps | grep -q "forex-"; then
        error "Platform3 containers are not running. Please start the platform first."
        exit 1
    fi
    
    # Create backup structure
    create_backup_structure
    
    # Perform backups in parallel where possible
    {
        backup_postgresql &
        backup_redis &
        backup_influxdb &
        backup_kafka &
        wait
    }
    
    # Sequential backups for file operations
    backup_application
    backup_monitoring
    
    # Create manifest and verify
    create_backup_manifest
    verify_backup
    
    # Cleanup old backups
    cleanup_old_backups
    
    local backup_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    log "Backup completed successfully!"
    log "Backup location: $BACKUP_DIR"
    log "Backup size: $backup_size"
    log "Log file: $LOG_FILE"
    
    # Send notification (if configured)
    if command -v mail >/dev/null 2>&1 && [ -n "${BACKUP_EMAIL:-}" ]; then
        echo "Platform3 backup completed successfully. Size: $backup_size" | \
            mail -s "Platform3 Backup Success - $BACKUP_TIMESTAMP" "$BACKUP_EMAIL"
    fi
}

# Handle script interruption
trap 'error "Backup interrupted"; exit 1' INT TERM

# Run main function
main "$@"
