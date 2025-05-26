// Session-Based Data Organization for Forex Trading
// Optimizes data storage and retrieval for Asian, London, and New York sessions
// Enables session-specific scalping and day trading strategies

// =============================================================================
// TRADING SESSION DEFINITIONS
// =============================================================================

// Trading session time definitions (UTC)
session_definitions = {
    asian: {
        start_hour: 23,           // 23:00 UTC (previous day)
        end_hour: 8,              // 08:00 UTC
        timezone: "Asia/Tokyo",
        description: "Asian trading session (Tokyo, Singapore, Hong Kong)",
        volatility: "low-medium",
        major_pairs: ["USDJPY", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY"],
        peak_hours: [0, 1, 2, 3, 4, 5, 6, 7]  // UTC hours of highest activity
    },
    
    london: {
        start_hour: 8,            // 08:00 UTC
        end_hour: 16,             // 16:00 UTC  
        timezone: "Europe/London",
        description: "London trading session (European markets)",
        volatility: "high",
        major_pairs: ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY"],
        peak_hours: [8, 9, 10, 11, 12, 13, 14, 15]  // UTC hours of highest activity
    },
    
    new_york: {
        start_hour: 13,           // 13:00 UTC
        end_hour: 22,             // 22:00 UTC
        timezone: "America/New_York", 
        description: "New York trading session (North American markets)",
        volatility: "high",
        major_pairs: ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
        peak_hours: [13, 14, 15, 16, 17, 18, 19, 20, 21]  // UTC hours of highest activity
    }
}

// Session overlap periods (highest volatility)
session_overlaps = {
    asian_london: {
        start_hour: 8,
        end_hour: 9,
        description: "Asian-London overlap (1 hour)",
        volatility: "very_high",
        best_pairs: ["EURJPY", "GBPJPY", "AUDJPY"]
    },
    
    london_new_york: {
        start_hour: 13,
        end_hour: 16,
        description: "London-New York overlap (3 hours)",
        volatility: "highest",
        best_pairs: ["EURUSD", "GBPUSD", "USDCHF", "USDCAD"]
    }
}

// =============================================================================
// SESSION BUCKET CREATION QUERIES
// =============================================================================

// Create bucket for Asian trading session
create_asian_session_bucket = '
import "influxdata/influxdb/schema"

// Create Asian session bucket with optimized retention
schema.createBucket(
    bucket: "asian-session-data",
    org: "forex-scalping", 
    retention: 720h,  // 30 days
    description: "Asian trading session tick data and signals"
)

// Create measurement for Asian session ticks
schema.createMeasurement(
    bucket: "asian-session-data",
    measurement: "asian_tick",
    tags: [
        {key: "symbol", type: "string"},
        {key: "pair_type", type: "string"},  // major, minor, exotic
        {key: "volatility_regime", type: "string"},  // low, medium, high
        {key: "hour_utc", type: "string"}
    ],
    fields: [
        {key: "price", type: "float"},
        {key: "volume", type: "int"},
        {key: "spread", type: "float"},
        {key: "session_momentum", type: "float"},
        {key: "asian_bias", type: "string"}  // bullish, bearish, neutral
    ]
)
'

// Create bucket for London trading session  
create_london_session_bucket = '
import "influxdata/influxdb/schema"

// Create London session bucket with optimized retention
schema.createBucket(
    bucket: "london-session-data",
    org: "forex-scalping",
    retention: 720h,  // 30 days
    description: "London trading session tick data and signals"
)

// Create measurement for London session ticks
schema.createMeasurement(
    bucket: "london-session-data", 
    measurement: "london_tick",
    tags: [
        {key: "symbol", type: "string"},
        {key: "pair_type", type: "string"},
        {key: "volatility_regime", type: "string"},
        {key: "hour_utc", type: "string"},
        {key: "ecb_event", type: "string"}  // none, rate_decision, speech
    ],
    fields: [
        {key: "price", type: "float"},
        {key: "volume", type: "int"}, 
        {key: "spread", type: "float"},
        {key: "session_momentum", type: "float"},
        {key: "london_bias", type: "string"},
        {key: "breakout_potential", type: "float"}
    ]
)
'

// Create bucket for New York trading session
create_new_york_session_bucket = '
import "influxdata/influxdb/schema"

// Create New York session bucket with optimized retention
schema.createBucket(
    bucket: "new-york-session-data",
    org: "forex-scalping",
    retention: 720h,  // 30 days  
    description: "New York trading session tick data and signals"
)

// Create measurement for New York session ticks
schema.createMeasurement(
    bucket: "new-york-session-data",
    measurement: "new_york_tick", 
    tags: [
        {key: "symbol", type: "string"},
        {key: "pair_type", type: "string"},
        {key: "volatility_regime", type: "string"}, 
        {key: "hour_utc", type: "string"},
        {key: "fed_event", type: "string"},  // none, rate_decision, fomc, speech
        {key: "nfp_week", type: "bool"}  // Non-farm payrolls week
    ],
    fields: [
        {key: "price", type: "float"},
        {key: "volume", type: "int"},
        {key: "spread", type: "float"},
        {key: "session_momentum", type: "float"},
        {key: "new_york_bias", type: "string"},
        {key: "dollar_strength", type: "float"}
    ]
)
'

// =============================================================================
// SESSION-SPECIFIC DATA ROUTING TASKS
// =============================================================================

// Route tick data to Asian session bucket
asian_session_routing = '
option task = {name: "route-asian-session", every: 1m}

import "date"
import "timezone"

// Get incoming tick data
data = from(bucket: "m1-tick-data")
    |> range(start: -1m)
    |> filter(fn: (r) => r["_measurement"] == "forex_tick")

// Filter for Asian session hours (23:00-08:00 UTC)
asian_data = data
    |> filter(fn: (r) => {
        hour = date.hour(t: r._time)
        return hour >= 23 or hour < 8
    })
    |> map(fn: (r) => ({
        r with
        session: "asian",
        volatility_regime: if r.spread < 0.0002 then "low" 
                          else if r.spread < 0.0005 then "medium"
                          else "high",
        pair_type: if contains(value: r.symbol, set: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]) then "major"
                  else if contains(value: r.symbol, set: ["EURJPY", "GBPJPY", "EURGBP"]) then "minor"  
                  else "exotic",
        hour_utc: string(v: date.hour(t: r._time)),
        asian_bias: if r.volume > 1000 then "active" else "quiet"
    }))
    |> set(key: "_measurement", value: "asian_tick")
    |> to(bucket: "asian-session-data", org: "forex-scalping")
'

// Route tick data to London session bucket
london_session_routing = '
option task = {name: "route-london-session", every: 1m}

import "date"

// Get incoming tick data
data = from(bucket: "m1-tick-data")
    |> range(start: -1m)
    |> filter(fn: (r) => r["_measurement"] == "forex_tick")

// Filter for London session hours (08:00-16:00 UTC)
london_data = data
    |> filter(fn: (r) => {
        hour = date.hour(t: r._time)
        return hour >= 8 and hour < 16
    })
    |> map(fn: (r) => ({
        r with
        session: "london",
        volatility_regime: if r.spread < 0.0001 then "low"
                          else if r.spread < 0.0003 then "medium" 
                          else "high",
        pair_type: if contains(value: r.symbol, set: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]) then "major"
                  else if contains(value: r.symbol, set: ["EURJPY", "GBPJPY", "EURGBP"]) then "minor"
                  else "exotic", 
        hour_utc: string(v: date.hour(t: r._time)),
        london_bias: if r.volume > 2000 then "strong" else "weak",
        breakout_potential: r.volume / r.spread  // Volume to spread ratio
    }))
    |> set(key: "_measurement", value: "london_tick")
    |> to(bucket: "london-session-data", org: "forex-scalping")
'

// Route tick data to New York session bucket
new_york_session_routing = '
option task = {name: "route-new-york-session", every: 1m}

import "date"

// Get incoming tick data  
data = from(bucket: "m1-tick-data")
    |> range(start: -1m)
    |> filter(fn: (r) => r["_measurement"] == "forex_tick")

// Filter for New York session hours (13:00-22:00 UTC)
new_york_data = data
    |> filter(fn: (r) => {
        hour = date.hour(t: r._time)
        return hour >= 13 and hour < 22
    })
    |> map(fn: (r) => ({
        r with
        session: "new_york",
        volatility_regime: if r.spread < 0.0001 then "low"
                          else if r.spread < 0.0004 then "medium"
                          else "high",
        pair_type: if contains(value: r.symbol, set: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]) then "major" 
                  else if contains(value: r.symbol, set: ["USDCAD", "AUDUSD", "NZDUSD"]) then "minor"
                  else "exotic",
        hour_utc: string(v: date.hour(t: r._time)),
        new_york_bias: if r.volume > 1500 then "active" else "quiet",
        dollar_strength: if contains(value: r.symbol, set: ["USDJPY", "USDCAD", "USDCHF"]) then r.price else 1.0 / r.price
    }))
    |> set(key: "_measurement", value: "new_york_tick") 
    |> to(bucket: "new-york-session-data", org: "forex-scalping")
'

// =============================================================================
// SESSION OVERLAP DETECTION
// =============================================================================

// Detect and route session overlap data
session_overlap_detection = '
option task = {name: "detect-session-overlaps", every: 1m}

import "date"

// Get all tick data
data = from(bucket: "m1-tick-data")
    |> range(start: -1m)
    |> filter(fn: (r) => r["_measurement"] == "forex_tick")

// Detect Asian-London overlap (08:00-09:00 UTC)
asian_london_overlap = data
    |> filter(fn: (r) => {
        hour = date.hour(t: r._time)
        return hour == 8
    })
    |> map(fn: (r) => ({
        r with
        overlap_type: "asian_london",
        volatility_boost: r.volume * 1.5,  // Expected volatility increase
        opportunity_score: r.volume / r.spread * 2.0
    }))
    |> set(key: "_measurement", value: "session_overlap")
    |> to(bucket: "scalping-signals", org: "forex-scalping")

// Detect London-New York overlap (13:00-16:00 UTC) 
london_ny_overlap = data
    |> filter(fn: (r) => {
        hour = date.hour(t: r._time)
        return hour >= 13 and hour < 16
    })
    |> map(fn: (r) => ({
        r with
        overlap_type: "london_new_york",
        volatility_boost: r.volume * 2.0,  // Highest volatility period
        opportunity_score: r.volume / r.spread * 3.0
    }))
    |> set(key: "_measurement", value: "session_overlap")
    |> to(bucket: "scalping-signals", org: "forex-scalping")
'

// =============================================================================
// SESSION-SPECIFIC PERFORMANCE ANALYTICS
// =============================================================================

// Calculate session-specific trading performance
session_performance_calc = '
option task = {name: "session-performance-analytics", every: 1h}

import "date"

// Asian session performance
asian_performance = from(bucket: "asian-session-data")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_measurement"] == "asian_tick")
    |> group(columns: ["symbol"])
    |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    |> map(fn: (r) => ({
        r with
        session: "asian",
        avg_spread: r._value,
        session_hour: date.hour(t: r._time),
        liquidity_score: if r._value < 0.0002 then "high" else "low"
    }))
    |> set(key: "_measurement", value: "session_analytics")
    |> to(bucket: "trading-performance", org: "forex-scalping")

// London session performance
london_performance = from(bucket: "london-session-data")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_measurement"] == "london_tick")
    |> group(columns: ["symbol"])
    |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    |> map(fn: (r) => ({
        r with
        session: "london",
        avg_spread: r._value,
        session_hour: date.hour(t: r._time),
        liquidity_score: if r._value < 0.0001 then "high" else "medium"
    }))
    |> set(key: "_measurement", value: "session_analytics")
    |> to(bucket: "trading-performance", org: "forex-scalping")

// New York session performance
ny_performance = from(bucket: "new-york-session-data")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_measurement"] == "new_york_tick")
    |> group(columns: ["symbol"])
    |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    |> map(fn: (r) => ({
        r with
        session: "new_york",
        avg_spread: r._value,
        session_hour: date.hour(t: r._time),
        liquidity_score: if r._value < 0.0002 then "high" else "medium"
    }))
    |> set(key: "_measurement", value: "session_analytics")
    |> to(bucket: "trading-performance", org: "forex-scalping")
'

// =============================================================================
// SESSION-OPTIMIZED QUERY TEMPLATES
// =============================================================================

// Query template for session-specific data retrieval
session_data_template = '
// Get session data for specific timeframe
session_query = (session_name, symbol, timeframe) => {
    bucket_name = session_name + "-session-data"
    measurement_name = session_name + "_tick"
    
    return from(bucket: bucket_name)
        |> range(start: timeframe)
        |> filter(fn: (r) => r["_measurement"] == measurement_name)
        |> filter(fn: (r) => r["symbol"] == symbol)
        |> sort(columns: ["_time"], desc: false)
}

// Get current session activity
current_session_activity = (symbol) => {
    hour = date.hour(t: now())
    
    session = if hour >= 23 or hour < 8 then "asian"
             else if hour >= 8 and hour < 16 then "london"  
             else if hour >= 13 and hour < 22 then "new_york"
             else "closed"
    
    return session_query(session: session, symbol: symbol, timeframe: "-1h")
}

// Get session overlap opportunities
overlap_opportunities = (symbol) => {
    return from(bucket: "scalping-signals")
        |> range(start: -1h)
        |> filter(fn: (r) => r["_measurement"] == "session_overlap")
        |> filter(fn: (r) => r["symbol"] == symbol)
        |> filter(fn: (r) => r["opportunity_score"] > 1000.0)
        |> sort(columns: ["opportunity_score"], desc: true)
        |> limit(n: 10)
}
'
