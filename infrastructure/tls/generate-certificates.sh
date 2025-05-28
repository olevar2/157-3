#!/bin/bash

# Platform3 Certificate Generation Script
# Generates CA and service certificates for mTLS authentication

set -e

# Configuration
CA_DIR="./ca"
CERTS_DIR="./certs"
SERVICES_DIR="./services"
VALIDITY_DAYS=365
KEY_SIZE=4096

# Service list for Platform3
SERVICES=(
    "api-gateway"
    "user-service"
    "trading-service"
    "market-data-service"
    "analytics-service"
    "notification-service"
    "compliance-service"
    "risk-service"
    "ml-service"
    "backtest-service"
    "data-quality-service"
    "order-execution-service"
    "qa-service"
    "service-discovery"
    "auth-service"
)

echo "🔐 Platform3 Certificate Generation Starting..."
echo "================================================"

# Create directory structure
mkdir -p $CA_DIR $CERTS_DIR $SERVICES_DIR

# Generate CA private key
echo "📋 Generating Certificate Authority (CA)..."
if [ ! -f "$CA_DIR/ca-key.pem" ]; then
    openssl genrsa -out $CA_DIR/ca-key.pem $KEY_SIZE
    echo "✅ CA private key generated"
else
    echo "⚠️  CA private key already exists, skipping..."
fi

# Generate CA certificate
if [ ! -f "$CA_DIR/ca-cert.pem" ]; then
    openssl req -new -x509 -days $VALIDITY_DAYS -key $CA_DIR/ca-key.pem \
        -out $CA_DIR/ca-cert.pem \
        -subj "/C=US/ST=CA/L=San Francisco/O=Platform3/OU=Security/CN=Platform3-CA"
    echo "✅ CA certificate generated"
else
    echo "⚠️  CA certificate already exists, skipping..."
fi

# Generate server certificates for each service
echo ""
echo "🔑 Generating service certificates..."
for service in "${SERVICES[@]}"; do
    echo "Processing: $service"
    
    SERVICE_DIR="$SERVICES_DIR/$service"
    mkdir -p $SERVICE_DIR
    
    # Generate service private key
    if [ ! -f "$SERVICE_DIR/$service-key.pem" ]; then
        openssl genrsa -out $SERVICE_DIR/$service-key.pem $KEY_SIZE
        echo "  ✅ Private key generated for $service"
    else
        echo "  ⚠️  Private key already exists for $service"
    fi
    
    # Generate certificate signing request (CSR)
    if [ ! -f "$SERVICE_DIR/$service-csr.pem" ]; then
        openssl req -new -key $SERVICE_DIR/$service-key.pem \
            -out $SERVICE_DIR/$service-csr.pem \
            -subj "/C=US/ST=CA/L=San Francisco/O=Platform3/OU=Services/CN=$service"
        echo "  ✅ CSR generated for $service"
    else
        echo "  ⚠️  CSR already exists for $service"
    fi
    
    # Create certificate extensions file
    cat > $SERVICE_DIR/$service-ext.cnf << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $service
DNS.2 = $service.forex-network
DNS.3 = localhost
DNS.4 = *.forex-network
IP.1 = 127.0.0.1
EOF
    
    # Generate service certificate signed by CA
    if [ ! -f "$SERVICE_DIR/$service-cert.pem" ]; then
        openssl x509 -req -in $SERVICE_DIR/$service-csr.pem \
            -CA $CA_DIR/ca-cert.pem -CAkey $CA_DIR/ca-key.pem \
            -CAcreateserial -out $SERVICE_DIR/$service-cert.pem \
            -days $VALIDITY_DAYS -extensions v3_req \
            -extfile $SERVICE_DIR/$service-ext.cnf
        echo "  ✅ Certificate generated for $service"
    else
        echo "  ⚠️  Certificate already exists for $service"
    fi
    
    # Verify certificate
    openssl verify -CAfile $CA_DIR/ca-cert.pem $SERVICE_DIR/$service-cert.pem > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✅ Certificate verified for $service"
    else
        echo "  ❌ Certificate verification failed for $service"
        exit 1
    fi
    
    # Clean up CSR and extension files
    rm -f $SERVICE_DIR/$service-csr.pem $SERVICE_DIR/$service-ext.cnf
done

# Generate client certificates for external access
echo ""
echo "👤 Generating client certificates..."
CLIENT_DIR="$CERTS_DIR/clients"
mkdir -p $CLIENT_DIR

# Admin client certificate
if [ ! -f "$CLIENT_DIR/admin-key.pem" ]; then
    openssl genrsa -out $CLIENT_DIR/admin-key.pem $KEY_SIZE
    openssl req -new -key $CLIENT_DIR/admin-key.pem \
        -out $CLIENT_DIR/admin-csr.pem \
        -subj "/C=US/ST=CA/L=San Francisco/O=Platform3/OU=Admin/CN=admin"
    
    cat > $CLIENT_DIR/admin-ext.cnf << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
EOF
    
    openssl x509 -req -in $CLIENT_DIR/admin-csr.pem \
        -CA $CA_DIR/ca-cert.pem -CAkey $CA_DIR/ca-key.pem \
        -CAcreateserial -out $CLIENT_DIR/admin-cert.pem \
        -days $VALIDITY_DAYS -extensions v3_req \
        -extfile $CLIENT_DIR/admin-ext.cnf
    
    rm -f $CLIENT_DIR/admin-csr.pem $CLIENT_DIR/admin-ext.cnf
    echo "✅ Admin client certificate generated"
fi

# Copy CA certificate to common locations
cp $CA_DIR/ca-cert.pem $CERTS_DIR/
echo "✅ CA certificate copied to certs directory"

# Set appropriate permissions
chmod 600 $CA_DIR/ca-key.pem
chmod 644 $CA_DIR/ca-cert.pem
find $SERVICES_DIR -name "*-key.pem" -exec chmod 600 {} \;
find $SERVICES_DIR -name "*-cert.pem" -exec chmod 644 {} \;
find $CLIENT_DIR -name "*-key.pem" -exec chmod 600 {} \;
find $CLIENT_DIR -name "*-cert.pem" -exec chmod 644 {} \;

echo ""
echo "🎉 Certificate generation completed successfully!"
echo "================================================"
echo "📁 Directory structure:"
echo "  $CA_DIR/          - Certificate Authority files"
echo "  $SERVICES_DIR/    - Service certificates"
echo "  $CLIENT_DIR/      - Client certificates"
echo ""
echo "🔒 Security notes:"
echo "  - Keep CA private key secure and backed up"
echo "  - Service private keys have restricted permissions (600)"
echo "  - Certificates are valid for $VALIDITY_DAYS days"
echo ""
echo "📋 Next steps:"
echo "  1. Distribute service certificates to respective containers"
echo "  2. Configure services to use mTLS"
echo "  3. Set up certificate rotation before expiry"
echo ""
echo "✨ Ready for mTLS deployment!"
