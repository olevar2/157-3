# Consul Configuration for Platform3
datacenter = "dc1"
data_dir = "/consul/data"
log_level = "INFO"
server = true

# Performance and Limits
performance {
  raft_multiplier = 1
}

limits {
  http_max_conns_per_client = 200
  https_handshake_timeout = "5s"
  rpc_handshake_timeout = "5s"
  rpc_max_conns_per_client = 100
}

# UI Configuration
ui_config {
  enabled = true
}

# Connect (Service Mesh) Configuration
connect {
  enabled = true
}

# Ports Configuration
ports {
  grpc = 8502
  http = 8500
  https = 8501
  dns = 8600
}

# ACL Configuration (for production security)
acl = {
  enabled = true
  default_policy = "deny"
  enable_token_persistence = true
}

# Encryption Configuration
encrypt = "CONSUL_ENCRYPT_KEY_PLACEHOLDER"

# TLS Configuration for secure communication
tls {
  defaults {
    verify_incoming = true
    verify_outgoing = true
    ca_file = "/consul/config/certs/consul-ca.pem"
    cert_file = "/consul/config/certs/consul-server.pem"
    key_file = "/consul/config/certs/consul-server-key.pem"
  }
  internal_rpc {
    verify_server_hostname = true
  }
}

# Auto-encrypt for client certificates
auto_encrypt {
  allow_tls = true
}
