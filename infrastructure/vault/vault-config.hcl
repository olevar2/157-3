# HashiCorp Vault Configuration for Platform3
# Production-ready configuration with high availability

# Storage backend - using file storage for development, Consul for production
storage "file" {
  path = "/vault/data"
}

# Alternative storage for production (uncomment for production deployment)
# storage "consul" {
#   address = "consul-server-1:8500"
#   path    = "vault/"
# }

# Listener configuration
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = false
  tls_cert_file = "/vault/certs/vault-cert.pem"
  tls_key_file  = "/vault/certs/vault-key.pem"
  tls_min_version = "tls12"
}

# API address
api_addr = "https://vault:8200"

# Cluster configuration
cluster_addr = "https://vault:8201"

# UI configuration
ui = true

# Logging
log_level = "INFO"
log_format = "json"

# Disable mlock for development (enable for production)
disable_mlock = true

# Enable raw endpoint (disable for production)
raw_storage_endpoint = false

# Telemetry
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}

# Seal configuration (auto-unseal for production)
# seal "awskms" {
#   region     = "us-east-1"
#   kms_key_id = "your-kms-key-id"
# }

# Plugin directory
plugin_directory = "/vault/plugins"

# Maximum lease TTL
max_lease_ttl = "768h"
default_lease_ttl = "168h"

# Enable audit logging
# audit {
#   file {
#     file_path = "/vault/logs/audit.log"
#   }
# }
