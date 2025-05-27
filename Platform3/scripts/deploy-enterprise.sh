#!/bin/bash

# Platform3 Enterprise Deployment Script
# Deploys the complete enterprise framework with shadow mode, rollback, and monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="platform3-enterprise"
REGISTRY="ghcr.io/platform3"
VERSION=${1:-"latest"}

echo -e "${BLUE}🚀 Platform3 Enterprise Deployment${NC}"
echo -e "${BLUE}====================================${NC}"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Create namespace
echo -e "${YELLOW}📦 Creating namespace...${NC}"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Build and push Docker images
echo -e "${YELLOW}🔨 Building Docker images...${NC}"

services=("shadow-mode-service" "deployment-service" "monitoring-service")

for service in "${services[@]}"; do
    echo -e "${BLUE}Building $service...${NC}"
    
    cd services/$service
    docker build -t $REGISTRY/$service:$VERSION .
    docker push $REGISTRY/$service:$VERSION
    cd ../..
    
    echo -e "${GREEN}✅ $service built and pushed${NC}"
done

# Deploy infrastructure components
echo -e "${YELLOW}🏗️ Deploying infrastructure...${NC}"

# Redis
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF

# Kafka
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-service
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kafka
  template:
    metadata:
      labels:
        app: kafka
    spec:
      containers:
      - name: kafka
        image: confluentinc/cp-kafka:latest
        ports:
        - containerPort: 9092
        env:
        - name: KAFKA_ZOOKEEPER_CONNECT
          value: "zookeeper-service:2181"
        - name: KAFKA_ADVERTISED_LISTENERS
          value: "PLAINTEXT://kafka-service:9092"
        - name: KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR
          value: "1"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: kafka-service
  namespace: $NAMESPACE
spec:
  selector:
    app: kafka
  ports:
  - port: 9092
    targetPort: 9092
EOF

echo -e "${GREEN}✅ Infrastructure deployed${NC}"

# Deploy enterprise services
echo -e "${YELLOW}🎯 Deploying enterprise services...${NC}"
kubectl apply -f k8s/enterprise-deployment.yaml

# Wait for deployments to be ready
echo -e "${YELLOW}⏳ Waiting for deployments to be ready...${NC}"

deployments=("shadow-mode-service" "deployment-service" "monitoring-service")

for deployment in "${deployments[@]}"; do
    echo -e "${BLUE}Waiting for $deployment...${NC}"
    kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=300s
    echo -e "${GREEN}✅ $deployment is ready${NC}"
done

# Verify deployment
echo -e "${YELLOW}🔍 Verifying deployment...${NC}"

for service in "${services[@]}"; do
    echo -e "${BLUE}Checking $service health...${NC}"
    
    # Port forward and check health
    kubectl port-forward -n $NAMESPACE service/$service 8080:$(kubectl get service $service -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}') &
    PID=$!
    
    sleep 5
    
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $service is healthy${NC}"
    else
        echo -e "${RED}❌ $service health check failed${NC}"
    fi
    
    kill $PID 2>/dev/null || true
done

# Display deployment information
echo -e "${GREEN}🎉 Platform3 Enterprise Deployment Complete!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "${YELLOW}📊 Deployment Summary:${NC}"
echo -e "Namespace: $NAMESPACE"
echo -e "Version: $VERSION"
echo -e "Services deployed: ${#services[@]}"
echo ""
echo -e "${YELLOW}🔗 Service Endpoints:${NC}"
echo -e "Shadow Mode Service: http://shadow-mode-service.$NAMESPACE:3010"
echo -e "Deployment Service: http://deployment-service.$NAMESPACE:3011"
echo -e "Monitoring Service: http://monitoring-service.$NAMESPACE:3012"
echo ""
echo -e "${YELLOW}📈 Monitoring:${NC}"
echo -e "Metrics: http://monitoring-service.$NAMESPACE:3012/metrics"
echo -e "Dashboard: http://monitoring-service.$NAMESPACE:3012/dashboard"
echo ""
echo -e "${YELLOW}🛠️ Management Commands:${NC}"
echo -e "View pods: kubectl get pods -n $NAMESPACE"
echo -e "View services: kubectl get services -n $NAMESPACE"
echo -e "View logs: kubectl logs -f deployment/<service-name> -n $NAMESPACE"
echo ""
echo -e "${GREEN}✅ Enterprise deployment framework is now active!${NC}"
