# Real-Time Market Data Ingestion & Processing Service

High-throughput data ingestion and processing pipeline for the Platform3 forex trading platform. Optimized for scalping, day trading, and swing trading strategies with minimal latency and maximum reliability.

## 🚀 Features

### High-Performance Data Processing
- **Ultra-fast ingestion**: 100,000+ ticks per second capacity
- **Sub-millisecond validation**: Real-time data quality assurance
- **Batch processing**: Configurable batch sizes for optimal throughput
- **Parallel processing**: Multi-threaded architecture for maximum performance

### Comprehensive Data Validation
- **Real-time validation**: Price, spread, volume, and timestamp checks
- **Statistical analysis**: Outlier detection using historical data
- **Quality scoring**: Comprehensive data quality metrics
- **Error handling**: Graceful degradation and error recovery

### Multi-Database Storage
- **InfluxDB**: Time-series tick data for scalping strategies
- **Redis**: Sub-millisecond caching for real-time access
- **PostgreSQL**: Aggregated OHLC data for analysis
- **Kafka**: Real-time streaming for downstream consumers

### Session-Aware Processing
- **Trading sessions**: Asian, London, NY, and Overlap detection
- **Session-based routing**: Optimized data flow per trading session
- **Performance monitoring**: Session-specific metrics and analytics

## 📋 Requirements

### System Requirements
- Python 3.9+
- 8GB+ RAM (16GB recommended for high-frequency trading)
- SSD storage for optimal I/O performance
- Multi-core CPU (8+ cores recommended)

### Database Dependencies
- PostgreSQL 15+
- Redis 7+
- InfluxDB 2.7+
- Kafka 3.0+

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Database Setup
```bash
# Ensure all databases are running
docker-compose up -d postgres redis influxdb kafka

# Verify connections
python -c "from RealTimeDataProcessor import RealTimeDataProcessor; print('Connections OK')"
```

## ⚙️ Configuration

### DataProcessorConfig
```python
config = DataProcessorConfig(
    # Database connections
    postgres_host="localhost",
    postgres_port=5432,
    redis_host="localhost",
    redis_port=6379,
    influx_url="http://localhost:8086",
    
    # Performance settings
    batch_size=1000,           # Ticks per batch
    flush_interval=0.1,        # 100ms flush interval
    max_workers=8,             # Thread pool size
    buffer_size=10000          # Queue buffer size
)
```

### ValidationConfig
```python
config = ValidationConfig(
    # Price validation
    min_price=0.0001,
    max_price=100.0,
    max_price_change_pct=5.0,
    
    # Spread validation
    min_spread=0.0001,
    max_spread=0.01,
    max_spread_pct=1.0,
    
    # Statistical validation
    price_outlier_threshold=3.0,
    volume_outlier_threshold=3.0
)
```

## 🚀 Usage

### Basic Usage
```python
import asyncio
from RealTimeDataProcessor import RealTimeDataProcessor, DataProcessorConfig, TickData
from DataValidator import DataValidator, ValidationConfig
from datetime import datetime, timezone

# Initialize processor
config = DataProcessorConfig()
processor = RealTimeDataProcessor(config)

# Initialize validator
validator = DataValidator(ValidationConfig())

# Start processing
async def main():
    # Start the processor
    processor_task = asyncio.create_task(processor.start())
    
    # Ingest tick data
    tick = TickData(
        symbol="EURUSD",
        timestamp=datetime.now(timezone.utc),
        bid=1.1000,
        ask=1.1005,
        volume=100,
        spread=0.0005,
        session="London"
    )
    
    # Validate and ingest
    if validator.is_tick_valid(tick):
        processor.ingest_tick(tick)
    
    await processor_task

asyncio.run(main())
```

### Advanced Usage with Custom Validation
```python
# Custom validation configuration
validation_config = ValidationConfig(
    max_price_change_pct=2.0,  # Stricter price change limits
    max_spread=0.005,          # Tighter spread limits
    price_outlier_threshold=2.5 # More sensitive outlier detection
)

validator = DataValidator(validation_config)

# Validate with detailed results
results = validator.validate_tick(tick)
for result in results:
    if result.level == ValidationLevel.ERROR:
        print(f"Validation error: {result.message}")
```

## 📊 Performance Metrics

### Target Performance
- **Throughput**: 100,000+ ticks/second
- **Latency**: <1ms validation time
- **Reliability**: 99.99% uptime
- **Data Quality**: >99.9% validation success rate

### Monitoring
```python
# Get real-time statistics
stats = processor.get_stats()
validation_stats = validator.get_validation_stats()

print(f"Processing rate: {stats['ticks_per_second']:.2f} ticks/sec")
print(f"Validation accuracy: {validation_stats['valid_percentage']:.2f}%")
```

## 🔧 Architecture

### Data Flow
1. **Ingestion**: Tick data received via `ingest_tick()`
2. **Validation**: Real-time quality checks and statistical analysis
3. **Batching**: Accumulate ticks for efficient processing
4. **Storage**: Parallel writes to InfluxDB, Redis, PostgreSQL
5. **Streaming**: Real-time distribution via Kafka

### Components
- **RealTimeDataProcessor**: Main processing engine
- **DataValidator**: Comprehensive validation system
- **TickData**: Standardized data structure
- **ValidationResult**: Detailed validation feedback

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/test_data_processor.py -v
python -m pytest tests/test_validator.py -v
```

### Performance Tests
```bash
python tests/performance_test.py
```

### Integration Tests
```bash
python tests/integration_test.py
```

## 📈 Optimization Tips

### High-Frequency Trading
- Increase `max_workers` for parallel processing
- Reduce `flush_interval` for lower latency
- Use SSD storage for database files
- Optimize network configuration for minimal latency

### Memory Optimization
- Adjust `buffer_size` based on available RAM
- Configure database connection pools
- Monitor memory usage during peak trading hours

### Database Tuning
- Optimize PostgreSQL for write-heavy workloads
- Configure Redis memory policies
- Set appropriate InfluxDB retention policies

## 🔍 Troubleshooting

### Common Issues
1. **High latency**: Check database connections and network
2. **Memory usage**: Adjust buffer sizes and batch processing
3. **Validation errors**: Review data quality and validation rules
4. **Connection failures**: Verify database availability and credentials

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
processor = RealTimeDataProcessor(config)
processor.logger.setLevel(logging.DEBUG)
```

## 📝 API Reference

### RealTimeDataProcessor
- `ingest_tick(tick: TickData) -> bool`: Ingest single tick
- `start() -> None`: Start processing pipeline
- `stop() -> None`: Stop processing gracefully
- `get_stats() -> Dict`: Get performance statistics

### DataValidator
- `validate_tick(tick: TickData) -> List[ValidationResult]`: Full validation
- `is_tick_valid(tick: TickData) -> bool`: Quick validation check
- `get_validation_stats() -> Dict`: Get validation statistics

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility

## 📄 License

Copyright (c) 2025 Platform3 Development Team. All rights reserved.
