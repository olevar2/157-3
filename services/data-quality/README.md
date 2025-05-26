# Data Quality Framework

Comprehensive data validation and quality monitoring system for the AI Forex Trading Platform, optimized for short-term trading data integrity (M1-H4 timeframes).

## Overview

The Data Quality Framework provides real-time monitoring, validation, and anomaly detection for all trading data flowing through the platform. It ensures data integrity critical for accurate trading decisions and risk management.

## Features

### 🔍 **Real-time Data Validation**
- OHLC price consistency checks
- Bid-Ask spread validation for scalping
- Volume data validation
- Timestamp integrity verification
- Trading data validation (orders, positions, accounts)

### 📊 **Advanced Anomaly Detection**
- Statistical anomaly detection (Z-score, IQR, Isolation Forest)
- Pattern-based anomaly detection (gaps, spikes, volume anomalies)
- Temporal anomaly detection for data timing
- Machine learning-based anomaly scoring

### 🚨 **Intelligent Alerting**
- Multi-channel notifications (Email, Slack, Database)
- Severity-based escalation rules
- Auto-remediation for critical issues
- Real-time alert dashboard

### 📈 **Quality Metrics & Monitoring**
- Data quality score calculation
- Data freshness monitoring
- Anomaly rate tracking
- Comprehensive reporting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Quality Framework                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Validation     │  │   Anomaly       │  │   Quality    │ │
│  │   Engine        │  │  Detection      │  │  Metrics     │ │
│  │                 │  │                 │  │              │ │
│  │ • OHLC Rules    │  │ • Statistical   │  │ • Scores     │ │
│  │ • Spread Check  │  │ • Pattern-based │  │ • Freshness  │ │
│  │ • Volume Valid  │  │ • ML-based      │  │ • Reporting  │ │
│  │ • Timestamp     │  │ • Temporal      │  │ • Trends     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Alerting      │  │  Auto-Remediation│  │   Storage    │ │
│  │   System        │  │     Engine       │  │   Layer      │ │
│  │                 │  │                 │  │              │ │
│  │ • Email/Slack   │  │ • Order Reject  │  │ • PostgreSQL │ │
│  │ • Escalation    │  │ • Data Quarant. │  │ • Redis      │ │
│  │ • Dashboard     │  │ • Alert Trigger │  │ • InfluxDB   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL 15+
- Redis 6+
- InfluxDB 2.x

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Node.js Dependencies
```bash
npm install
```

## Configuration

### 1. Data Validation Rules
Edit `data-validation-rules.yaml` to configure validation rules:

```yaml
market_data:
  price_validation:
    ohlc_consistency:
      rule_id: "MD001"
      severity: "CRITICAL"
      checks:
        - name: "high_low_range"
          condition: "high >= low"
```

### 2. Environment Variables
Create `.env` file:

```env
# Database connections
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=platform3_trading
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Redis connection
REDIS_URL=redis://localhost:6379

# InfluxDB connection
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=platform3

# Alert configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@platform3.com
SMTP_PASSWORD=your-password

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## Usage

### Start Quality Monitor
```bash
# Start the real-time quality monitor
python quality-monitor.py
```

### Run Anomaly Detection
```bash
# Test anomaly detection
python anomaly-detection.py
```

### Start Node.js Service (if applicable)
```bash
# Start the service
npm start

# Development mode
npm run dev
```

## API Endpoints

### Quality Metrics
```http
GET /api/v1/quality/metrics
```
Returns current data quality metrics.

### Validation Results
```http
GET /api/v1/quality/validations?symbol=EURUSD&timeframe=M1
```
Get validation results for specific symbol and timeframe.

### Anomaly Detection
```http
POST /api/v1/quality/detect-anomalies
Content-Type: application/json

{
  "symbol": "EURUSD",
  "timestamp": "2024-12-19T10:00:00Z",
  "open": 1.0500,
  "high": 1.0510,
  "low": 1.0495,
  "close": 1.0505,
  "volume": 1000
}
```

### Alert Management
```http
GET /api/v1/quality/alerts?severity=CRITICAL&limit=50
```

## Validation Rules

### Market Data Rules

#### MD001 - OHLC Consistency
- **Purpose**: Ensure price data integrity
- **Checks**: High ≥ Low, Open/Close within High/Low range
- **Severity**: CRITICAL

#### MD002 - Price Movement Limits
- **Purpose**: Detect abnormal price movements
- **Thresholds**: M1: 0.5%, M5: 1.0%, M15: 2.0%, H1: 5.0%, H4: 10.0%
- **Severity**: HIGH

#### MD003 - Spread Validation
- **Purpose**: Validate bid-ask spreads for scalping
- **Thresholds**: EURUSD: 3 pips, GBPUSD: 4 pips, etc.
- **Severity**: HIGH

### Trading Data Rules

#### TD001 - Order Validation
- **Purpose**: Validate order data integrity
- **Checks**: Size limits (0.01-100 lots), positive prices
- **Severity**: CRITICAL

#### TD002 - Position Validation
- **Purpose**: Ensure position data consistency
- **Checks**: P&L calculations, margin requirements
- **Severity**: CRITICAL

## Anomaly Detection

### Statistical Methods
- **Z-Score**: Detects values beyond 3 standard deviations
- **IQR**: Uses interquartile range for outlier detection
- **Isolation Forest**: ML-based multivariate anomaly detection

### Pattern-Based Detection
- **Price Gaps**: Detects gaps > 0.5% between periods
- **Price Spikes**: Identifies sudden price movements
- **Volume Anomalies**: Detects unusual volume patterns

### Temporal Detection
- **Timing Anomalies**: Validates data arrival timing
- **Sequence Validation**: Ensures chronological order

## Alerting

### Severity Levels
- **CRITICAL**: Immediate notification, auto-stop trading
- **HIGH**: Immediate notification, manual review
- **MEDIUM**: Batch notification, scheduled review
- **LOW**: Daily report inclusion

### Notification Channels
- **Email**: Critical and high severity alerts
- **Slack**: Real-time notifications with context
- **Database**: All alerts stored for analysis

## Monitoring Dashboard

Access the monitoring dashboard at: `http://localhost:3000/quality-dashboard`

### Key Metrics
- Data Quality Score (target: 99%)
- Data Freshness Score (target: 98%)
- Anomaly Rate (target: <1%)
- Alert Counts by Severity

## Testing

### Run Tests
```bash
# Python tests
pytest tests/

# Node.js tests
npm test

# Coverage report
npm run test:coverage
```

### Test Data Quality
```bash
# Test with sample data
python test_quality_framework.py
```

## Performance

### Benchmarks
- **Validation Speed**: >10,000 records/second
- **Anomaly Detection**: <100ms per data point
- **Alert Latency**: <1 second for critical alerts
- **Memory Usage**: <500MB for 1M records

### Optimization Tips
1. Adjust window sizes based on data volume
2. Configure appropriate thresholds for your market
3. Use Redis for high-frequency data caching
4. Implement data retention policies

## Troubleshooting

### Common Issues

#### High False Positive Rate
- Adjust Z-score thresholds in configuration
- Review market-specific parameters
- Consider market volatility periods

#### Performance Issues
- Check database connection pooling
- Monitor Redis memory usage
- Review data retention settings

#### Missing Alerts
- Verify notification channel configuration
- Check alert severity thresholds
- Review escalation rules

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-validation-rule`)
3. Commit changes (`git commit -am 'Add new validation rule'`)
4. Push to branch (`git push origin feature/new-validation-rule`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Email: support@platform3.com
- Slack: #data-quality-support
- Documentation: https://docs.platform3.com/data-quality
