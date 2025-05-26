// Tick Data Schema for High-Frequency Forex Scalping
// Optimized for M1-M5 timeframe strategies with ultra-fast queries
// Supports 1M+ tick data points per second with sub-millisecond response

// =============================================================================
// CORE TICK DATA MEASUREMENT SCHEMA
// =============================================================================

// Primary forex tick data structure
// Used for M1-M5 scalping strategies
schema_forex_tick = {
    measurement: "forex_tick",
    tags: {
        symbol: "string",           // Currency pair (e.g., "EURUSD", "GBPJPY")
        broker: "string",           // Data provider/broker
        session: "string",          // Trading session (asian/london/new_york)
        timeframe: "string",        // M1, M5, M15, H1, etc.
        data_type: "string"         // bid, ask, mid, last
    },
    fields: {
        price: "float",             // Tick price
        volume: "int",              // Tick volume
        spread: "float",            // Bid-ask spread
        bid: "float",               // Bid price
        ask: "float",               // Ask price
        high: "float",              // Period high
        low: "float",               // Period low
        open: "float",              // Period open
        close: "float"              // Period close
    },
    timestamp: "time"               // Nanosecond precision timestamp
}

// =============================================================================
// ORDER FLOW DATA SCHEMA
// =============================================================================

// Level 2 order book data for scalping
schema_order_flow = {
    measurement: "order_flow",
    tags: {
        symbol: "string",           // Currency pair
        side: "string",             // bid/ask
        level: "int",               // Order book level (1-20)
        source: "string"            // Data source
    },
    fields: {
        price: "float",             // Price level
        size: "float",              // Order size at level
        orders: "int",              // Number of orders
        cumulative_size: "float",   // Cumulative size from best price
        imbalance: "float"          // Bid-ask imbalance ratio
    },
    timestamp: "time"
}

// =============================================================================
// SCALPING SIGNALS SCHEMA
// =============================================================================

// Real-time scalping signals
schema_scalping_signals = {
    measurement: "scalping_signal",
    tags: {
        symbol: "string",           // Currency pair
        signal_type: "string",      // BUY, SELL, CLOSE
        strategy: "string",         // Strategy name
        timeframe: "string",        // Signal timeframe
        session: "string"           // Trading session
    },
    fields: {
        confidence: "float",        // Signal confidence (0-100)
        price: "float",             // Signal price
        target_price: "float",      // Take profit target
        stop_price: "float",        // Stop loss price
        risk_reward: "float",       // Risk-reward ratio
        expected_duration: "int",   // Expected trade duration (seconds)
        signal_strength: "float"    // Technical strength indicator
    },
    timestamp: "time"
}

// =============================================================================
// TECHNICAL INDICATORS SCHEMA
// =============================================================================

// Fast technical indicators for scalping
schema_technical_indicators = {
    measurement: "technical_indicator",
    tags: {
        symbol: "string",           // Currency pair
        indicator: "string",        // Indicator name (RSI, MACD, etc.)
        timeframe: "string",        // Calculation timeframe
        period: "int"               // Indicator period
    },
    fields: {
        value: "float",             // Indicator value
        signal: "string",           // BUY/SELL/NEUTRAL
        overbought: "bool",         // Overbought condition
        oversold: "bool",           // Oversold condition
        divergence: "bool"          // Price-indicator divergence
    },
    timestamp: "time"
}

// =============================================================================
// SESSION PERFORMANCE SCHEMA
// =============================================================================

// Trading session performance tracking
schema_session_performance = {
    measurement: "session_performance",
    tags: {
        session: "string",          // asian/london/new_york
        strategy: "string",         // Strategy name
        symbol: "string"            // Currency pair
    },
    fields: {
        trades_count: "int",        // Number of trades
        winning_trades: "int",      // Winning trades count
        losing_trades: "int",       // Losing trades count
        total_pips: "float",        // Total pips gained/lost
        profit_loss: "float",       // P&L in base currency
        win_rate: "float",          // Win rate percentage
        avg_trade_duration: "int",  // Average trade duration (seconds)
        max_drawdown: "float",      // Maximum drawdown
        sharpe_ratio: "float"       // Risk-adjusted return
    },
    timestamp: "time"
}

// =============================================================================
// MARKET MICROSTRUCTURE SCHEMA
// =============================================================================

// Market microstructure data for advanced scalping
schema_microstructure = {
    measurement: "market_microstructure",
    tags: {
        symbol: "string",           // Currency pair
        event_type: "string",       // trade, quote, order
        aggressor: "string"         // buyer/seller initiated
    },
    fields: {
        price: "float",             // Event price
        size: "float",              // Event size
        milliseconds: "int",        // Intra-second timing
        price_impact: "float",      // Price impact of trade
        effective_spread: "float",  // Effective spread
        realized_spread: "float"    // Realized spread
    },
    timestamp: "time"
}

// =============================================================================
// VOLATILITY METRICS SCHEMA
// =============================================================================

// Real-time volatility for scalping risk management
schema_volatility = {
    measurement: "volatility_metrics",
    tags: {
        symbol: "string",           // Currency pair
        timeframe: "string",        // Measurement timeframe
        volatility_type: "string"   // historical, implied, realized
    },
    fields: {
        volatility: "float",        // Volatility percentage
        atr: "float",               // Average True Range
        volatility_rank: "float",   // Volatility percentile rank
        vol_surface: "float",       // Volatility surface value
        regime: "string"            // Low/Medium/High volatility regime
    },
    timestamp: "time"
}

// =============================================================================
// TRADING EXECUTION SCHEMA
// =============================================================================

// Trade execution data for performance analysis
schema_trade_execution = {
    measurement: "trade_execution",
    tags: {
        trade_id: "string",         // Unique trade identifier
        symbol: "string",           // Currency pair
        side: "string",             // BUY/SELL
        strategy: "string",         // Strategy name
        session: "string"           // Trading session
    },
    fields: {
        entry_price: "float",       // Entry price
        exit_price: "float",        // Exit price
        quantity: "float",          // Trade size
        pips: "float",              // Pips gained/lost
        profit_loss: "float",       // P&L in base currency
        duration: "int",            // Trade duration (seconds)
        slippage: "float",          // Execution slippage
        commission: "float",        // Trading commission
        swap: "float"               // Overnight swap
    },
    timestamp: "time"
}

// =============================================================================
// INDEX OPTIMIZATION FOR HIGH-FREQUENCY QUERIES
// =============================================================================

// Indexes optimized for scalping queries
scalping_indexes = [
    // Primary tick data index
    {
        measurement: "forex_tick",
        tagKey: "symbol",
        fields: ["price", "volume", "spread"],
        retention: "7d"
    },
    
    // Signal detection index
    {
        measurement: "scalping_signal", 
        tagKey: "symbol",
        fields: ["confidence", "signal_type"],
        retention: "3d"
    },
    
    // Session performance index
    {
        measurement: "session_performance",
        tagKey: "session",
        fields: ["win_rate", "total_pips"],
        retention: "30d"
    },
    
    // Order flow index
    {
        measurement: "order_flow",
        tagKey: "symbol",
        fields: ["price", "size", "imbalance"],
        retention: "1d"
    }
]

// =============================================================================
// CONTINUOUS QUERIES FOR REAL-TIME AGGREGATION
// =============================================================================

// M1 to M5 aggregation for scalping context
m1_to_m5_aggregation = '
    SELECT 
        first(open) as open,
        max(high) as high,
        min(low) as low,
        last(close) as close,
        sum(volume) as volume,
        mean(spread) as avg_spread
    INTO "m5_scalping_data"."autogen".:MEASUREMENT
    FROM "m1_tick_data"."autogen"./^forex_tick$/
    GROUP BY time(5m), symbol, session
    fill(none)
'

// Real-time signal strength calculation
signal_strength_calc = '
    SELECT 
        mean(confidence) as avg_confidence,
        count(signal_type) as signal_count,
        last(price) as current_price
    INTO "scalping_signals"."autogen".signal_summary
    FROM "scalping_signals"."autogen".scalping_signal
    GROUP BY time(30s), symbol, strategy
    fill(previous)
'

// Session-based performance aggregation
session_performance_agg = '
    SELECT 
        sum(trades_count) as total_trades,
        sum(total_pips) as session_pips,
        mean(win_rate) as avg_win_rate
    INTO "trading_performance"."autogen".session_summary
    FROM "session_performance"."autogen".session_performance
    GROUP BY time(1h), session, strategy
    fill(0)
'

// =============================================================================
// SCALPING-OPTIMIZED QUERY TEMPLATES
// =============================================================================

// Template for ultra-fast tick data retrieval
fast_tick_query = '
    from(bucket: "m1-tick-data")
    |> range(start: -5m)
    |> filter(fn: (r) => r["_measurement"] == "forex_tick")
    |> filter(fn: (r) => r["symbol"] == "${symbol}")
    |> filter(fn: (r) => r["_field"] == "price")
    |> last()
'

// Template for scalping signal detection
scalping_signal_query = '
    from(bucket: "scalping-signals")
    |> range(start: -1m)
    |> filter(fn: (r) => r["_measurement"] == "scalping_signal")
    |> filter(fn: (r) => r["symbol"] == "${symbol}")
    |> filter(fn: (r) => r["confidence"] > 70.0)
    |> sort(columns: ["_time"], desc: true)
    |> limit(n: 1)
'

// Template for order flow analysis
order_flow_query = '
    from(bucket: "order-flow")
    |> range(start: -30s)
    |> filter(fn: (r) => r["_measurement"] == "order_flow")
    |> filter(fn: (r) => r["symbol"] == "${symbol}")
    |> pivot(rowKey:["_time"], columnKey: ["side"], valueColumn: "_value")
    |> map(fn: (r) => ({
        r with
        imbalance: (r.bid - r.ask) / (r.bid + r.ask)
    }))
'
