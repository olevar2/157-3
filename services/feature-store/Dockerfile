# Multi-stage Docker build for Feature Store microservice
# Optimized for production deployment with minimal footprint

# Stage 1: Python Feature Pipeline
FROM python:3.11-slim as python-base

WORKDIR /app/python

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY src/feature-pipeline.py .
COPY feature-definitions.yaml .

# Stage 2: Node.js API Server
FROM node:18-alpine as node-base

WORKDIR /app/node

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci --only=production

# Copy TypeScript source and compile
COPY tsconfig.json .
COPY src/feature-serving-api.ts src/
RUN npm run build

# Stage 3: Production runtime
FROM ubuntu:22.04

# Install Python and Node.js runtime
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    nodejs \
    npm \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python application
COPY --from=python-base /app/python /app/python
COPY --from=python-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy Node.js application
COPY --from=node-base /app/node /app/node

# Create logs directory
RUN mkdir -p /app/logs

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Environment variables
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
ENV NODE_ENV=production
ENV REDIS_HOST=redis
ENV KAFKA_BROKERS=kafka1:9092,kafka2:9092,kafka3:9092
ENV INFLUXDB_URL=http://influxdb:8086

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8081

# Start supervisor to manage both processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
