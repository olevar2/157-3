# Multi-stage Dockerfile for Data Quality Service
# Optimized for production deployment with minimal image size

# Stage 1: Python dependencies and application
FROM python:3.11-slim as python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Node.js dependencies (if needed)
FROM node:18-alpine as node-base

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production && npm cache clean --force

# Stage 3: Final application image
FROM python:3.11-slim as final

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for potential Node.js components)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Copy Python dependencies from python-base stage
COPY --from=python-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-base /usr/local/bin /usr/local/bin

# Copy Node.js dependencies from node-base stage
COPY --from=node-base /app/node_modules ./node_modules

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data config

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "quality-monitor.py"]
