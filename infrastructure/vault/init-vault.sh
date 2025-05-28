#!/bin/bash
# Platform3 Vault Initialization Script
# Initializes Vault with Platform3 secrets and policies

set -e

echo "🔐 Initializing HashiCorp Vault for Platform3..."

# Wait for Vault to be ready
echo "⏳ Waiting for Vault to be ready..."
until curl -s http://vault:8200/v1/sys/health | grep -q '"initialized":true'; do
  echo "Waiting for Vault to initialize..."
  sleep 5
done

# Check if Vault is already initialized
if curl -s http://vault:8200/v1/sys/init | grep -q '"initialized":true'; then
  echo "✅ Vault is already initialized"
  exit 0
fi

echo "🚀 Initializing Vault..."

# Initialize Vault
INIT_RESPONSE=$(curl -s -X POST \
  -d '{"secret_shares": 5, "secret_threshold": 3}' \
  http://vault:8200/v1/sys/init)

# Extract keys and root token
UNSEAL_KEY_1=$(echo $INIT_RESPONSE | jq -r '.keys[0]')
UNSEAL_KEY_2=$(echo $INIT_RESPONSE | jq -r '.keys[1]')
UNSEAL_KEY_3=$(echo $INIT_RESPONSE | jq -r '.keys[2]')
ROOT_TOKEN=$(echo $INIT_RESPONSE | jq -r '.root_token')

echo "🔑 Unsealing Vault..."

# Unseal Vault
curl -s -X POST -d "{\"key\": \"$UNSEAL_KEY_1\"}" http://vault:8200/v1/sys/unseal
curl -s -X POST -d "{\"key\": \"$UNSEAL_KEY_2\"}" http://vault:8200/v1/sys/unseal
curl -s -X POST -d "{\"key\": \"$UNSEAL_KEY_3\"}" http://vault:8200/v1/sys/unseal

echo "📝 Setting up Platform3 policies..."

# Create Platform3 policy
curl -s -X PUT \
  -H "X-Vault-Token: $ROOT_TOKEN" \
  -d @/vault/policies/platform3-policy.hcl \
  http://vault:8200/v1/sys/policies/acl/platform3

echo "🔐 Creating Platform3 secrets..."

# Enable KV secrets engine
curl -s -X POST \
  -H "X-Vault-Token: $ROOT_TOKEN" \
  -d '{"type": "kv", "options": {"version": "2"}}' \
  http://vault:8200/v1/sys/mounts/secret

# Create database secrets
curl -s -X POST \
  -H "X-Vault-Token: $ROOT_TOKEN" \
  -d '{"data": {"password": "ForexSecure2025!", "url": "postgresql://forex_admin:ForexSecure2025!@postgres:5432/forex_trading"}}' \
  http://vault:8200/v1/secret/data/database/postgres

# Create JWT secrets
curl -s -X POST \
  -H "X-Vault-Token: $ROOT_TOKEN" \
  -d '{"data": {"secret": "forex-jwt-secret-super-secure-2025", "expires_in": "24h"}}' \
  http://vault:8200/v1/secret/data/auth/jwt

# Create Redis secrets
curl -s -X POST \
  -H "X-Vault-Token: $ROOT_TOKEN" \
  -d '{"data": {"password": "RedisSecure2025!", "url": "redis://redis:6379"}}' \
  http://vault:8200/v1/secret/data/database/redis

echo "🎯 Creating service token for Platform3..."

# Create service token
SERVICE_TOKEN=$(curl -s -X POST \
  -H "X-Vault-Token: $ROOT_TOKEN" \
  -d '{"policies": ["platform3"], "ttl": "768h", "renewable": true}' \
  http://vault:8200/v1/auth/token/create | jq -r '.auth.client_token')

echo "✅ Vault initialization complete!"
echo "🔑 Service Token: $SERVICE_TOKEN"
echo "💾 Save this token securely - it will be used by Platform3 services"

# Save tokens to file (for development only)
cat > /vault/tokens.txt << EOF
ROOT_TOKEN=$ROOT_TOKEN
SERVICE_TOKEN=$SERVICE_TOKEN
UNSEAL_KEY_1=$UNSEAL_KEY_1
UNSEAL_KEY_2=$UNSEAL_KEY_2
UNSEAL_KEY_3=$UNSEAL_KEY_3
EOF

chmod 600 /vault/tokens.txt
echo "📁 Tokens saved to /vault/tokens.txt"
