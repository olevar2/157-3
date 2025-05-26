#!/bin/bash

# Redis Scalping Setup Script
# Configures Redis cluster optimized for high-frequency forex trading
# Enables sub-millisecond response times for scalping operations

set -e

echo "🚀 Setting up Redis Cluster for Forex Scalping..."

# Configuration variables
REDIS_PASSWORD="ScalpingRedis2025!"
CLUSTER_NODES=6
BASE_PORT=7000
DATA_DIR="./data/redis-cluster"
LOG_DIR="./logs/redis"

# Create necessary directories
echo "📁 Creating Redis directories..."
mkdir -p $DATA_DIR
mkdir -p $LOG_DIR
mkdir -p ./conf/redis

# Function to create Redis node configuration
create_node_config() {
    local port=$1
    local node_dir="$DATA_DIR/node-$port"
    local conf_file="./conf/redis/redis-$port.conf"
    
    mkdir -p $node_dir
    
    cat > $conf_file << EOF
# Redis Node $port Configuration for Scalping
include ./infrastructure/database/redis/redis-cluster-trading.conf

# Node-specific settings
port $port
cluster-announce-port $port
cluster-announce-bus-port $((port + 10000))

# Data directory
dir $node_dir

# Logging
logfile $LOG_DIR/redis-$port.log

# PID file
pidfile /var/run/redis/redis-$port.pid

# Node-specific memory settings
maxmemory 512mb

# Cluster node configuration file
cluster-config-file $node_dir/nodes-$port.conf

# Authentication
requirepass $REDIS_PASSWORD
masterauth $REDIS_PASSWORD
EOF

    echo "✅ Created configuration for Redis node on port $port"
}

echo "⚙️ Creating Redis cluster node configurations..."

# Create configurations for all cluster nodes
for i in $(seq 0 $((CLUSTER_NODES - 1))); do
    port=$((BASE_PORT + i))
    create_node_config $port
done

echo "🔧 Creating Redis cluster startup script..."

# Create cluster startup script
cat > ./start-redis-cluster.sh << 'EOF'
#!/bin/bash

# Start Redis Cluster for Scalping Operations
set -e

BASE_PORT=7000
NODES=6

echo "🚀 Starting Redis cluster nodes..."

# Start all Redis nodes
for i in $(seq 0 $((NODES - 1))); do
    port=$((BASE_PORT + i))
    echo "Starting Redis node on port $port..."
    
    redis-server ./conf/redis/redis-$port.conf --daemonize yes
    
    # Wait for node to start
    sleep 1
    
    # Check if node is running
    if redis-cli -p $port ping > /dev/null 2>&1; then
        echo "✅ Redis node $port started successfully"
    else
        echo "❌ Failed to start Redis node $port"
        exit 1
    fi
done

echo "⏳ Waiting for all nodes to initialize..."
sleep 3

# Create the cluster
echo "🔗 Creating Redis cluster..."
redis-cli --cluster create \
    127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
    127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
    --cluster-replicas 1 \
    --cluster-yes

echo "✅ Redis cluster created successfully!"

# Test cluster connectivity
echo "🧪 Testing cluster connectivity..."
for i in $(seq 0 $((NODES - 1))); do
    port=$((BASE_PORT + i))
    if redis-cli -p $port cluster nodes > /dev/null 2>&1; then
        echo "✅ Node $port: Cluster connectivity OK"
    else
        echo "❌ Node $port: Cluster connectivity failed"
    fi
done

echo "📊 Cluster status:"
redis-cli -p 7000 cluster info

echo "🎯 Redis cluster ready for scalping operations!"
EOF

chmod +x ./start-redis-cluster.sh

echo "🔧 Creating Redis cluster shutdown script..."

# Create cluster shutdown script
cat > ./stop-redis-cluster.sh << 'EOF'
#!/bin/bash

# Stop Redis Cluster
set -e

BASE_PORT=7000
NODES=6

echo "🛑 Stopping Redis cluster nodes..."

for i in $(seq 0 $((NODES - 1))); do
    port=$((BASE_PORT + i))
    echo "Stopping Redis node on port $port..."
    
    redis-cli -p $port shutdown nosave > /dev/null 2>&1 || echo "Node $port already stopped"
done

echo "✅ Redis cluster stopped successfully!"
EOF

chmod +x ./stop-redis-cluster.sh

echo "📝 Creating Redis monitoring script..."

# Create monitoring script
cat > ./monitor-redis-cluster.sh << 'EOF'
#!/bin/bash

# Monitor Redis Cluster Performance
BASE_PORT=7000
NODES=6

echo "📊 Redis Cluster Performance Monitor"
echo "====================================="

# Function to get node info
get_node_info() {
    local port=$1
    local info=$(redis-cli -p $port info memory 2>/dev/null)
    local latency=$(redis-cli -p $port --latency-history -i 1 | head -1 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        local memory=$(echo "$info" | grep used_memory_human | cut -d: -f2 | tr -d '\r')
        local keys=$(redis-cli -p $port dbsize 2>/dev/null)
        echo "Node $port: Memory: $memory, Keys: $keys"
    else
        echo "Node $port: OFFLINE"
    fi
}

# Monitor all nodes
while true; do
    clear
    echo "📊 Redis Cluster Status - $(date)"
    echo "=================================="
    
    for i in $(seq 0 $((NODES - 1))); do
        port=$((BASE_PORT + i))
        get_node_info $port
    done
    
    echo ""
    echo "🔄 Cluster Info:"
    redis-cli -p 7000 cluster info 2>/dev/null | grep -E "(cluster_state|cluster_slots|cluster_known_nodes)"
    
    echo ""
    echo "⚡ Recent Operations:"
    redis-cli -p 7000 info stats 2>/dev/null | grep -E "(total_commands_processed|instantaneous_ops_per_sec)"
    
    sleep 5
done
EOF

chmod +x ./monitor-redis-cluster.sh

echo "🧪 Creating Redis testing script..."

# Create testing script for scalping performance
cat > ./test-redis-scalping.sh << 'EOF'
#!/bin/bash

# Test Redis Cluster for Scalping Performance
set -e

PORT=7000
PASSWORD="ScalpingRedis2025!"

echo "🧪 Testing Redis Cluster for Scalping Performance..."

# Test basic connectivity
echo "🔌 Testing connectivity..."
redis-cli -p $PORT -a $PASSWORD ping

# Test set/get performance
echo "⚡ Testing SET/GET performance..."
redis-cli -p $PORT -a $PASSWORD eval "
for i=1,10000 do
    redis.call('SET', 'test:tick:' .. i, '1.2345|100|0.0001|' .. i)
end
return 'OK'
" 0

echo "📊 Testing GET performance..."
time redis-cli -p $PORT -a $PASSWORD eval "
for i=1,10000 do
    redis.call('GET', 'test:tick:' .. i)
end
return 'OK'
" 0

# Test hash operations for trading positions
echo "📈 Testing HASH operations for trading positions..."
redis-cli -p $PORT -a $PASSWORD eval "
for i=1,1000 do
    redis.call('HSET', 'position:' .. i, 
        'symbol', 'EURUSD',
        'side', 'BUY', 
        'quantity', '10000',
        'entry_price', '1.1234',
        'current_price', '1.1240',
        'pnl', '6.0'
    )
end
return 'OK'
" 0

# Test sorted sets for price levels
echo "📉 Testing ZADD for price levels..."
redis-cli -p $PORT -a $PASSWORD eval "
for i=1,1000 do
    local price = 1.1000 + (i * 0.0001)
    redis.call('ZADD', 'orderbook:EURUSD:bids', price, 'level:' .. i .. ':1000')
end
return 'OK'
" 0

# Test pub/sub for real-time signals
echo "📡 Testing PUB/SUB for real-time signals..."
redis-cli -p $PORT -a $PASSWORD publish "signals:EURUSD" "BUY|1.1234|0.95|1.1244|1.1224" &
redis-cli -p $PORT -a $PASSWORD publish "ticks:EURUSD" "1.1235|150|0.0001|$(date +%s)" &

# Test pipeline performance
echo "🚀 Testing pipeline performance..."
{
    for i in {1..1000}; do
        echo "SET scalping:test:$i 1.234$i"
    done
} | redis-cli -p $PORT -a $PASSWORD --pipe

# Cleanup test data
echo "🧹 Cleaning up test data..."
redis-cli -p $PORT -a $PASSWORD eval "
local keys = redis.call('KEYS', 'test:*')
for i=1,#keys do
    redis.call('DEL', keys[i])
end

keys = redis.call('KEYS', 'position:*')
for i=1,#keys do
    redis.call('DEL', keys[i])
end

keys = redis.call('KEYS', 'scalping:*')
for i=1,#keys do
    redis.call('DEL', keys[i])
end

redis.call('DEL', 'orderbook:EURUSD:bids')
return 'Cleaned up'
" 0

echo "✅ Redis scalping performance tests completed!"
echo "📊 Cluster is ready for high-frequency trading operations"
EOF

chmod +x ./test-redis-scalping.sh

echo "📖 Creating Redis usage documentation..."

# Create documentation
cat > ./README-Redis-Scalping.md << 'EOF'
# Redis Cluster for Forex Scalping

## Overview
High-performance Redis cluster optimized for forex scalping operations with sub-millisecond response times.

## Features
- 6-node cluster with replication
- Optimized for trading workloads
- Sub-millisecond latency for critical operations
- Automatic failover and recovery
- Session state management
- Real-time feature serving

## Quick Start

### Start Cluster
```bash
./start-redis-cluster.sh
```

### Stop Cluster
```bash
./stop-redis-cluster.sh
```

### Monitor Performance
```bash
./monitor-redis-cluster.sh
```

### Test Performance
```bash
./test-redis-scalping.sh
```

## Data Structures for Trading

### Real-time Tick Data
```bash
SET tick:EURUSD:latest "1.1234|150|0.0001|1640995200"
EXPIRE tick:EURUSD:latest 60
```

### Trading Positions
```bash
HSET position:12345 symbol EURUSD side BUY quantity 10000 entry_price 1.1234
```

### Order Book Levels
```bash
ZADD orderbook:EURUSD:bids 1.1234 "level1:1000"
ZADD orderbook:EURUSD:asks 1.1236 "level1:800"
```

### Real-time Signals
```bash
PUBLISH signals:EURUSD "BUY|1.1234|0.95|1.1244|1.1224"
```

### Session State
```bash
HSET session:trader123 active_trades 5 total_pnl 125.50 risk_level low
EXPIRE session:trader123 3600
```

## Performance Optimization

### Connection Pooling
Use connection pooling for multiple trading services to maintain consistent performance.

### Pipeline Operations
Use Redis pipelines for batch operations to reduce network round trips.

### Lua Scripts
Use Lua scripts for atomic operations involving multiple keys.

## Monitoring

### Key Metrics
- Memory usage per node
- Operations per second
- Average latency
- Cluster health status

### Alerts
- Node failures
- High memory usage (>80%)
- Slow operations (>1ms)
- Network partitions

## Security
- Password authentication enabled
- Dangerous commands disabled
- Access restricted to trading services
- Encrypted connections recommended
EOF

echo "✅ Redis Scalping Setup Complete!"
echo ""
echo "📊 Created Components:"
echo "  - Cluster configuration files (6 nodes)"
echo "  - Startup/shutdown scripts"
echo "  - Performance monitoring tools"
echo "  - Testing and validation scripts"
echo "  - Documentation and usage examples"
echo ""
echo "⚡ Performance Features:"
echo "  - Sub-millisecond response times"
echo "  - High availability with replication"
echo "  - Optimized for trading workloads"
echo "  - Session state management"
echo "  - Real-time feature serving"
echo ""
echo "🚀 To start the cluster:"
echo "  ./start-redis-cluster.sh"
echo ""
echo "📊 To monitor performance:"
echo "  ./monitor-redis-cluster.sh"
echo ""
echo "🧪 To test scalping performance:"
echo "  ./test-redis-scalping.sh"
