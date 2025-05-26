# 🧠 AI Feature Store - Real-time Feature Engineering for Forex Trading

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-API-blue?logo=typescript)](https://typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-Pipeline-green?logo=python)](https://python.org)
[![Redis](https://img.shields.io/badge/Redis-Cache-red?logo=redis)](https://redis.io)
[![Kafka](https://img.shields.io/badge/Kafka-Streaming-orange?logo=apache-kafka)](https://kafka.apache.org)

## 🎯 Overview

The AI Feature Store is a high-performance microservice designed specifically for forex trading platforms. It provides real-time feature engineering, computation, and serving capabilities optimized for M1-H4 timeframes with sub-millisecond latency requirements.

### 🚀 Key Features

- **Real-time Feature Computation**: Processes tick data and generates 40+ trading features
- **Sub-millisecond Serving**: Redis-based caching for ultra-fast feature access
- **Microservices Architecture**: Scalable, independent deployment with Docker
- **Multiple Categories**: Microstructure, price action, technical indicators, session-based features
- **WebSocket Streaming**: Real-time feature updates for trading applications
- **High Availability**: Cluster-ready with automatic failover and load balancing

## 📊 Feature Categories

### 🔬 Microstructure Features
- Bid-ask spread analysis
- Order book imbalance detection
- Tick volume indicators
- Market depth analysis

### 📈 Price Action Features
- Price velocity and acceleration
- Local extrema detection
- Support/resistance levels
- Price momentum indicators

### 🔧 Technical Indicators
- RSI, MACD, Bollinger Bands
- Moving averages (SMA, EMA)
- Stochastic oscillators
- Volume-based indicators

### 🕐 Session-based Features
- Asian/London/NY session detection
- Session high/low tracking
- Cross-session momentum
- Time-based volatility patterns

### 📰 Sentiment Features
- News sentiment analysis
- Market sentiment indicators
- Fear & greed index
- Volatility sentiment

### 🔗 Correlation Features
- Inter-currency correlations
- Cross-asset correlations
- Correlation momentum
- Divergence detection

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Tick Data     │───▶│  Feature Store  │───▶│  Trading Apps   │
│   (Kafka)       │    │   Microservice  │    │   (WebSocket)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   (Cache)       │
                       └─────────────────┘
```

### Components

1. **Feature Pipeline (Python)**: Real-time feature computation from tick data
2. **Feature API (TypeScript)**: High-performance REST/WebSocket API
3. **Redis Cache**: Sub-millisecond feature serving
4. **Kafka Integration**: Real-time data streaming
5. **InfluxDB**: Time-series data storage

## 🚀 Quick Start

### Prerequisites

- Docker Desktop 4.0+
- Docker Compose 2.0+
- 8GB+ RAM available for containers
- Windows 10/11 or Linux

### 1. Clone and Setup

```powershell
# Clone the repository
git clone <repository-url>
cd Platform3/services/feature-store

# Run automated setup (Windows)
.\setup.ps1 -Action deploy
```

### 2. Verify Installation

```bash
# Check service health
curl http://localhost:8080/health

# Get feature definitions
curl http://localhost:8080/api/definitions

# Test feature retrieval
curl http://localhost:8080/api/features/EURUSD/all
```

### 3. WebSocket Connection

```javascript
// Connect to real-time feature stream
const ws = new WebSocket('ws://localhost:8081');

ws.onmessage = (event) => {
    const featureUpdate = JSON.parse(event.data);
    console.log('Feature update:', featureUpdate);
};
```

## 📚 API Documentation

### REST Endpoints

#### Get Single Feature
```http
GET /api/features/{symbol}/{feature}
```

#### Get Multiple Features
```http
POST /api/features/batch
Content-Type: application/json

{
  "symbol": "EURUSD",
  "features": ["rsi_14", "sma_20", "bb_position"],
  "format": "json"
}
```

#### Get Feature Vector (for ML)
```http
GET /api/features/{symbol}/vector?features=rsi_14,sma_20,macd
```

#### Subscribe to Real-time Updates
```http
POST /api/subscribe
Content-Type: application/json

{
  "symbols": ["EURUSD", "GBPUSD"],
  "features": ["rsi_14", "sma_20"],
  "update_frequency": "tick"
}
```

### WebSocket API

Connect to `ws://localhost:8081/ws/{subscription_id}` after creating a subscription.

```json
{
  "symbol": "EURUSD",
  "timestamp": "2025-05-25T10:30:00.000Z",
  "features": {
    "rsi_14": 65.3,
    "sma_20": 1.0845,
    "bb_position": 0.78
  },
  "event_type": "feature_update"
}
```

## ⚙️ Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka Configuration  
KAFKA_BROKERS=localhost:9092,localhost:9093,localhost:9094

# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=forex_trading_token
INFLUXDB_ORG=forex_trading_org

# API Configuration
PORT=8080
WS_PORT=8081
LOG_LEVEL=info
```

### Feature Definitions

Features are defined in `feature-definitions.yaml`:

```yaml
feature_categories:
  microstructure:
    bid_ask_spread:
      timeframes: [M1, M5]
      update_frequency: tick
      computation_type: real_time
      storage_ttl: 3600
```

## 🔧 Development

### Local Development Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Run feature pipeline
python src/feature-pipeline.py

# Run API server (in another terminal)
npm run dev
```

### Testing

```bash
# Run Python tests
python -m pytest tests/

# Run TypeScript tests
npm test

# Integration tests
npm run test:integration
```

## 📈 Performance Metrics

### Latency Targets
- Feature serving: < 1ms (95th percentile)
- Feature computation: < 10ms per tick
- WebSocket updates: < 5ms end-to-end

### Throughput Targets
- Tick processing: 100,000+ ticks/second
- Feature requests: 10,000+ requests/second
- Concurrent WebSocket connections: 1,000+

### Resource Usage
- Memory: ~2GB under normal load
- CPU: ~1 core under normal load
- Storage: ~10GB for 1 month of data

## 🐳 Docker Deployment

### Single Service
```bash
docker-compose up -d feature-store
```

### Full Stack
```bash
docker-compose up -d
```

### Scaling
```bash
# Scale feature store instances
docker-compose up -d --scale feature-store=3

# Scale Kafka brokers
docker-compose up -d --scale kafka1=1 --scale kafka2=1 --scale kafka3=1
```

## 📊 Monitoring

### Health Checks
- `/health` - Service health status
- `/ready` - Kubernetes readiness probe
- `/metrics` - Prometheus-compatible metrics

### Logging
- Application logs: `logs/feature-*.log`
- Error logs: `logs/error.log`
- Performance logs: `logs/performance.log`

### Metrics Dashboard
Access Grafana dashboard at `http://localhost:3000` (if enabled)

## 🔒 Security

### Authentication
- API key authentication for production
- JWT tokens for WebSocket connections
- Rate limiting on all endpoints

### Network Security
- TLS encryption for all external connections
- Internal service mesh with mTLS
- Network segmentation with Docker networks

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check Docker logs
docker-compose logs feature-store

# Check resource usage
docker stats
```

#### High Memory Usage
```bash
# Check Redis memory
docker exec forex-feature-redis redis-cli info memory

# Adjust cache TTL in configuration
```

#### Slow Feature Serving
```bash
# Check Redis latency
docker exec forex-feature-redis redis-cli --latency

# Monitor API metrics
curl http://localhost:8080/metrics
```

## 🔄 Backup and Recovery

### Data Backup
```bash
# Backup Redis data
docker exec forex-feature-redis redis-cli bgsave

# Backup InfluxDB data
docker exec forex-influxdb influx backup /backup
```

### Disaster Recovery
```bash
# Restore from backup
docker-compose down
docker volume rm feature-store_redis-data
docker-compose up -d
# Restore data files
```

## 📝 API Examples

### Python Client
```python
import requests
import json

# Get all features for EURUSD
response = requests.get('http://localhost:8080/api/features/EURUSD/all')
features = response.json()

print(f"RSI: {features['rsi_14']}")
print(f"SMA: {features['sma_20']}")
```

### JavaScript Client
```javascript
// Feature subscription
const response = await fetch('http://localhost:8080/api/subscribe', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbols: ['EURUSD'],
    features: ['rsi_14', 'sma_20'],
    update_frequency: 'tick'
  })
});

const subscription = await response.json();
console.log('Subscription ID:', subscription.subscription_id);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

- Documentation: [Wiki](wiki-url)
- Issues: [GitHub Issues](issues-url)
- Discord: [Trading Platform Discord](discord-url)
- Email: support@trading-platform.com

---

*Built with ❤️ for high-frequency forex trading*
