#!/bin/bash

# Kafka Scalping Topics Configuration
# Creates optimized topics for M1-M5 short-term trading strategies
# Supports sub-second latency for forex scalping operations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
KAFKA_BOOTSTRAP_SERVERS="localhost:9092,localhost:9093,localhost:9094"
REPLICATION_FACTOR=3
MIN_IN_SYNC_REPLICAS=2

echo -e "${BLUE}🚀 Creating Kafka Topics for Forex Scalping Platform${NC}"
echo -e "${BLUE}Bootstrap Servers: $KAFKA_BOOTSTRAP_SERVERS${NC}"
echo -e "${BLUE}Replication Factor: $REPLICATION_FACTOR${NC}"
echo ""

# Function to create topic with error handling
create_topic() {
    local topic_name=$1
    local partitions=$2
    local retention_ms=$3
    local segment_ms=$4
    local description=$5
    
    echo -e "${YELLOW}Creating topic: $topic_name${NC}"
    echo -e "  └─ $description"
    
    kafka-topics --create \
        --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
        --topic $topic_name \
        --partitions $partitions \
        --replication-factor $REPLICATION_FACTOR \
        --config min.insync.replicas=$MIN_IN_SYNC_REPLICAS \
        --config retention.ms=$retention_ms \
        --config segment.ms=$segment_ms \
        --config compression.type=lz4 \
        --config cleanup.policy=delete \
        --config max.message.bytes=1000000 \
        --config flush.ms=1000 \
        --config unclean.leader.election.enable=false \
        --config preallocate=true \
        --if-not-exists || {
        echo -e "${RED}Failed to create topic: $topic_name${NC}"
        return 1
    }
    
    echo -e "${GREEN}  ✅ Topic created successfully${NC}"
    echo ""
}

# Wait for Kafka to be ready
echo -e "${YELLOW}Waiting for Kafka cluster to be ready...${NC}"
timeout=60
while ! kafka-topics --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --list > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo -e "${RED}❌ Timeout waiting for Kafka cluster${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ Kafka cluster is ready${NC}"
echo ""

# === TICK DATA TOPICS (Ultra High Frequency) ===
echo -e "${BLUE}=== CREATING TICK DATA TOPICS ===${NC}"

# M1 Tick Data (1-second granularity)
create_topic "forex.tick.m1" 16 86400000 60000 "M1 tick data for scalping strategies (1-day retention)"

# Order Book Updates (Real-time)
create_topic "forex.orderbook.updates" 32 3600000 30000 "Real-time order book updates (1-hour retention)"

# Order Flow Data
create_topic "forex.orderflow" 16 21600000 60000 "Order flow analysis data (6-hour retention)"

# Microstructure Data
create_topic "forex.microstructure" 8 43200000 120000 "Market microstructure data (12-hour retention)"

# === AGGREGATED DATA TOPICS ===
echo -e "${BLUE}=== CREATING AGGREGATED DATA TOPICS ===${NC}"

# M5 Aggregated Data
create_topic "forex.bars.m5" 8 604800000 300000 "M5 OHLCV bars for day trading (7-day retention)"

# M15 Aggregated Data
create_topic "forex.bars.m15" 4 1209600000 900000 "M15 OHLCV bars for swing trading (14-day retention)"

# H1 Aggregated Data
create_topic "forex.bars.h1" 2 2592000000 3600000 "H1 OHLCV bars for position tracking (30-day retention)"

# === TRADING SIGNAL TOPICS ===
echo -e "${BLUE}=== CREATING TRADING SIGNAL TOPICS ===${NC}"

# Scalping Signals (M1-M5)
create_topic "forex.signals.scalping" 16 3600000 30000 "Scalping signals for M1-M5 strategies (1-hour retention)"

# Day Trading Signals (M15-H1)
create_topic "forex.signals.daytrading" 8 21600000 120000 "Day trading signals for M15-H1 strategies (6-hour retention)"

# Swing Trading Signals (H4+)
create_topic "forex.signals.swing" 4 86400000 300000 "Swing trading signals for H4+ strategies (24-hour retention)"

# Signal Confirmations
create_topic "forex.signals.confirmations" 8 7200000 60000 "Signal confirmation events (2-hour retention)"

# === ORDER EXECUTION TOPICS ===
echo -e "${BLUE}=== CREATING ORDER EXECUTION TOPICS ===${NC}"

# Order Requests
create_topic "forex.orders.requests" 16 86400000 60000 "Order execution requests (1-day retention)"

# Order Executions
create_topic "forex.orders.executions" 16 259200000 120000 "Order execution results (3-day retention)"

# Order Status Updates
create_topic "forex.orders.status" 8 86400000 60000 "Order status change events (1-day retention)"

# Fill Reports
create_topic "forex.orders.fills" 16 604800000 300000 "Order fill reports (7-day retention)"

# === RISK MANAGEMENT TOPICS ===
echo -e "${BLUE}=== CREATING RISK MANAGEMENT TOPICS ===${NC}"

# Position Updates
create_topic "forex.positions.updates" 8 259200000 120000 "Real-time position updates (3-day retention)"

# Risk Alerts
create_topic "forex.risk.alerts" 4 604800000 300000 "Risk management alerts (7-day retention)"

# Stop Loss Triggers
create_topic "forex.risk.stoploss" 8 86400000 60000 "Stop loss trigger events (1-day retention)"

# Margin Calls
create_topic "forex.risk.margin" 4 604800000 300000 "Margin call notifications (7-day retention)"

# === SESSION MANAGEMENT TOPICS ===
echo -e "${BLUE}=== CREATING SESSION MANAGEMENT TOPICS ===${NC}"

# Session Events (Market Open/Close)
create_topic "forex.sessions.events" 4 604800000 3600000 "Trading session events (7-day retention)"

# Session State
create_topic "forex.sessions.state" 2 86400000 1800000 "Current session state tracking (1-day retention)"

# Session Statistics
create_topic "forex.sessions.stats" 4 2592000000 3600000 "Session performance statistics (30-day retention)"

# === ANALYTICS TOPICS ===
echo -e "${BLUE}=== CREATING ANALYTICS TOPICS ===${NC}"

# Performance Metrics
create_topic "forex.analytics.performance" 4 2592000000 3600000 "Trading performance metrics (30-day retention)"

# P&L Updates
create_topic "forex.analytics.pnl" 8 2592000000 900000 "Real-time P&L calculations (30-day retention)"

# Correlation Analysis
create_topic "forex.analytics.correlations" 2 86400000 1800000 "Currency pair correlation data (1-day retention)"

# Volatility Measurements
create_topic "forex.analytics.volatility" 4 604800000 900000 "Real-time volatility calculations (7-day retention)"

# === EXTERNAL DATA TOPICS ===
echo -e "${BLUE}=== CREATING EXTERNAL DATA TOPICS ===${NC}"

# News Events
create_topic "forex.news.events" 4 604800000 300000 "Economic news and events (7-day retention)"

# Economic Calendar
create_topic "forex.economic.calendar" 2 2592000000 3600000 "Economic calendar events (30-day retention)"

# Central Bank Updates
create_topic "forex.centralbank.updates" 2 2592000000 3600000 "Central bank policy updates (30-day retention)"

# === DEAD LETTER TOPICS ===
echo -e "${BLUE}=== CREATING ERROR HANDLING TOPICS ===${NC}"

# Dead Letter Queue
create_topic "forex.deadletter.queue" 4 604800000 300000 "Failed message processing queue (7-day retention)"

# Processing Errors
create_topic "forex.errors.processing" 4 259200000 300000 "Processing error events (3-day retention)"

# Data Quality Issues
create_topic "forex.errors.dataquality" 2 86400000 600000 "Data quality issue reports (1-day retention)"

echo ""
echo -e "${GREEN}🎉 All Kafka topics created successfully!${NC}"
echo ""

# List all created topics
echo -e "${BLUE}=== TOPIC SUMMARY ===${NC}"
kafka-topics --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --list | grep "forex\." | sort

echo ""
echo -e "${GREEN}✅ Kafka scalping infrastructure is ready for high-frequency trading!${NC}"
echo -e "${YELLOW}📊 Monitor topics at: http://localhost:9021${NC}"
echo -e "${YELLOW}🔧 Schema Registry at: http://localhost:8081${NC}"
