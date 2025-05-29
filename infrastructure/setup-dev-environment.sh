#!/bin/bash

echo "🚀 Setting up Platform3 Configuration Management Development Environment"

# Start infrastructure services
echo "📦 Starting Docker services..."
cd "$(dirname "$0")/.."
docker-compose -f infrastructure/docker-compose.yml up -d vault redis

# Wait for Vault to be ready
echo "⏳ Waiting for Vault to be ready..."
timeout=60
while ! docker-compose -f infrastructure/docker-compose.yml exec vault vault status > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ Vault failed to start within 60 seconds"
        exit 1
    fi
done

# Initialize Vault
echo "🔧 Initializing Vault..."
docker-compose -f infrastructure/docker-compose.yml exec vault sh -c '
    # Initialize vault if not already done
    if ! vault status | grep -q "Initialized.*true"; then
        vault operator init -key-shares=5 -key-threshold=3 > /tmp/vault-init.txt
        echo "Vault initialized. Keys saved to /tmp/vault-init.txt"
    fi
    
    # Unseal vault
    UNSEAL_KEYS=$(grep "Unseal Key" /tmp/vault-init.txt | head -3 | cut -d: -f2 | tr -d " ")
    for key in $UNSEAL_KEYS; do
        vault operator unseal $key
    done
    
    # Login with root token
    ROOT_TOKEN=$(grep "Initial Root Token" /tmp/vault-init.txt | cut -d: -f2 | tr -d " ")
    vault auth $ROOT_TOKEN
    
    # Enable KV engine
    vault secrets enable -path=platform3 kv-v2
    
    # Create initial configuration
    vault kv put platform3/config/database \
        host=localhost \
        port=5432 \
        username=platform3 \
        password=dev-secret
        
    vault kv put platform3/config/redis \
        host=localhost \
        port=6379 \
        password=redis-secret
        
    vault kv put platform3/feature-flags \
        new-ui=true \
        api-v2=false \
        debug-mode=true
'

echo "✅ Infrastructure setup complete!"
echo "🔧 Vault UI available at: http://localhost:8200"
echo "📊 Redis available at: localhost:6379"
