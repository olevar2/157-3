# Platform3 Vault Policy
# Defines access permissions for Platform3 services

# Database secrets
path "secret/data/database/*" {
  capabilities = ["read", "list"]
}

# JWT secrets
path "secret/data/auth/*" {
  capabilities = ["read", "list"]
}

# Broker API keys
path "secret/data/brokers/*" {
  capabilities = ["read", "list"]
}

# Service-specific secrets
path "secret/data/services/+/config" {
  capabilities = ["read", "list"]
}

# Trading configuration
path "secret/data/trading/*" {
  capabilities = ["read", "list"]
}

# Risk management configuration
path "secret/data/risk/*" {
  capabilities = ["read", "list"]
}

# Analytics configuration
path "secret/data/analytics/*" {
  capabilities = ["read", "list"]
}

# Notification secrets
path "secret/data/notifications/*" {
  capabilities = ["read", "list"]
}

# Allow token renewal
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Allow token lookup
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

# System health check
path "sys/health" {
  capabilities = ["read"]
}
