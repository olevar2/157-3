-- Real-time Features Lua Scripts for Redis
-- Optimized for high-frequency forex trading operations
-- Provides atomic operations for scalping and day trading

-- ============================================================================
-- TICK DATA OPERATIONS
-- ============================================================================

-- Store tick data with automatic expiration and indexing
-- KEYS[1]: tick key (e.g., "tick:EURUSD:latest")
-- KEYS[2]: tick history key (e.g., "tick:EURUSD:history")
-- ARGV[1]: price, ARGV[2]: volume, ARGV[3]: spread, ARGV[4]: timestamp
local function store_tick_data()
    local tick_key = KEYS[1]
    local history_key = KEYS[2]
    local price = tonumber(ARGV[1])
    local volume = tonumber(ARGV[2])
    local spread = tonumber(ARGV[3])
    local timestamp = tonumber(ARGV[4])
    
    -- Store latest tick data
    local tick_data = price .. "|" .. volume .. "|" .. spread .. "|" .. timestamp
    redis.call('SET', tick_key, tick_data)
    redis.call('EXPIRE', tick_key, 60)  -- 1 minute expiration
    
    -- Add to history with score as timestamp
    redis.call('ZADD', history_key, timestamp, tick_data)
    
    -- Keep only last 1000 ticks in history
    local count = redis.call('ZCARD', history_key)
    if count > 1000 then
        redis.call('ZREMRANGEBYRANK', history_key, 0, count - 1001)
    end
    
    -- Update price statistics
    local stats_key = string.gsub(tick_key, ":latest", ":stats")
    redis.call('HSET', stats_key, 
        'last_price', price,
        'last_volume', volume,
        'last_spread', spread,
        'last_update', timestamp
    )
    redis.call('EXPIRE', stats_key, 300)  -- 5 minutes expiration
    
    return "OK"
end

-- ============================================================================
-- TRADING POSITION MANAGEMENT
-- ============================================================================

-- Update trading position atomically
-- KEYS[1]: position key (e.g., "position:12345")
-- ARGV[1]: current_price, ARGV[2]: unrealized_pnl, ARGV[3]: timestamp
local function update_position()
    local position_key = KEYS[1]
    local current_price = tonumber(ARGV[1])
    local unrealized_pnl = tonumber(ARGV[2])
    local timestamp = tonumber(ARGV[3])
    
    -- Check if position exists
    if redis.call('EXISTS', position_key) == 0 then
        return {err = "Position not found"}
    end
    
    -- Get current position data
    local position = redis.call('HGETALL', position_key)
    local pos_data = {}
    for i = 1, #position, 2 do
        pos_data[position[i]] = position[i + 1]
    end
    
    -- Update position fields
    redis.call('HSET', position_key,
        'current_price', current_price,
        'unrealized_pnl', unrealized_pnl,
        'last_update', timestamp
    )
    
    -- Update position in active positions index
    local symbol = pos_data.symbol or "UNKNOWN"
    local active_key = "active_positions:" .. symbol
    redis.call('ZADD', active_key, timestamp, position_key)
    
    -- Check for stop loss or take profit
    local entry_price = tonumber(pos_data.entry_price or 0)
    local stop_loss = tonumber(pos_data.stop_loss or 0)
    local take_profit = tonumber(pos_data.take_profit or 0)
    local side = pos_data.side or "BUY"
    
    local trigger_exit = false
    local exit_reason = ""
    
    if side == "BUY" then
        if stop_loss > 0 and current_price <= stop_loss then
            trigger_exit = true
            exit_reason = "STOP_LOSS"
        elseif take_profit > 0 and current_price >= take_profit then
            trigger_exit = true
            exit_reason = "TAKE_PROFIT"
        end
    else -- SELL
        if stop_loss > 0 and current_price >= stop_loss then
            trigger_exit = true
            exit_reason = "STOP_LOSS"
        elseif take_profit > 0 and current_price <= take_profit then
            trigger_exit = true
            exit_reason = "TAKE_PROFIT"
        end
    end
    
    if trigger_exit then
        -- Add to exit queue
        local exit_data = position_key .. "|" .. exit_reason .. "|" .. current_price .. "|" .. timestamp
        redis.call('LPUSH', 'exit_queue', exit_data)
        redis.call('EXPIRE', 'exit_queue', 300)
        
        return {exit_triggered = true, reason = exit_reason, price = current_price}
    end
    
    return {updated = true, pnl = unrealized_pnl}
end

-- ============================================================================
-- ORDER BOOK OPERATIONS
-- ============================================================================

-- Update order book levels atomically
-- KEYS[1]: bids key, KEYS[2]: asks key
-- ARGV[1]: bid_prices (comma-separated), ARGV[2]: bid_sizes, ARGV[3]: ask_prices, ARGV[4]: ask_sizes
local function update_order_book()
    local bids_key = KEYS[1]
    local asks_key = KEYS[2]
    local bid_prices = ARGV[1]
    local bid_sizes = ARGV[2]
    local ask_prices = ARGV[3]
    local ask_sizes = ARGV[4]
    
    -- Clear existing order book
    redis.call('DEL', bids_key, asks_key)
    
    -- Parse and add bid levels
    local bid_price_list = {}
    local bid_size_list = {}
    for price in string.gmatch(bid_prices, "[^,]+") do
        table.insert(bid_price_list, tonumber(price))
    end
    for size in string.gmatch(bid_sizes, "[^,]+") do
        table.insert(bid_size_list, tonumber(size))
    end
    
    -- Add bids (higher prices have higher scores)
    for i = 1, math.min(#bid_price_list, #bid_size_list) do
        local level_data = "level" .. i .. ":" .. bid_size_list[i]
        redis.call('ZADD', bids_key, bid_price_list[i], level_data)
    end
    
    -- Parse and add ask levels  
    local ask_price_list = {}
    local ask_size_list = {}
    for price in string.gmatch(ask_prices, "[^,]+") do
        table.insert(ask_price_list, tonumber(price))
    end
    for size in string.gmatch(ask_sizes, "[^,]+") do
        table.insert(ask_size_list, tonumber(size))
    end
    
    -- Add asks (lower prices have lower scores)
    for i = 1, math.min(#ask_price_list, #ask_size_list) do
        local level_data = "level" .. i .. ":" .. ask_size_list[i]
        redis.call('ZADD', asks_key, ask_price_list[i], level_data)
    end
    
    -- Set expiration
    redis.call('EXPIRE', bids_key, 10)  -- 10 seconds
    redis.call('EXPIRE', asks_key, 10)
    
    -- Calculate spread and imbalance
    local best_bid = redis.call('ZREVRANGE', bids_key, 0, 0, 'WITHSCORES')
    local best_ask = redis.call('ZRANGE', asks_key, 0, 0, 'WITHSCORES')
    
    if #best_bid > 0 and #best_ask > 0 then
        local spread = tonumber(best_ask[2]) - tonumber(best_bid[2])
        local bid_size = tonumber(string.match(best_bid[1], ":(%d+)"))
        local ask_size = tonumber(string.match(best_ask[1], ":(%d+)"))
        local imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        
        return {
            spread = spread,
            imbalance = imbalance,
            best_bid = tonumber(best_bid[2]),
            best_ask = tonumber(best_ask[2])
        }
    end
    
    return "OK"
end

-- ============================================================================
-- SIGNAL PROCESSING
-- ============================================================================

-- Process and store trading signal with validation
-- KEYS[1]: signal key, KEYS[2]: signals index key
-- ARGV[1]: signal_type, ARGV[2]: confidence, ARGV[3]: price, ARGV[4]: symbol, ARGV[5]: timestamp
local function process_signal()
    local signal_key = KEYS[1]
    local index_key = KEYS[2]
    local signal_type = ARGV[1]
    local confidence = tonumber(ARGV[2])
    local price = tonumber(ARGV[3])
    local symbol = ARGV[4]
    local timestamp = tonumber(ARGV[5])
    
    -- Validate signal confidence
    if confidence < 50 then
        return {err = "Signal confidence too low"}
    end
    
    -- Check for conflicting signals in last 30 seconds
    local recent_signals = redis.call('ZRANGEBYSCORE', index_key, timestamp - 30, timestamp)
    for i = 1, #recent_signals do
        local signal_data = redis.call('HGETALL', recent_signals[i])
        if signal_data and #signal_data > 0 then
            local existing_type = signal_data[2]  -- Assuming field order
            if existing_type ~= signal_type then
                return {err = "Conflicting signal detected", conflict_with = existing_type}
            end
        end
    end
    
    -- Store signal data
    redis.call('HSET', signal_key,
        'signal_type', signal_type,
        'confidence', confidence,
        'price', price,
        'symbol', symbol,
        'timestamp', timestamp,
        'status', 'ACTIVE'
    )
    redis.call('EXPIRE', signal_key, 300)  -- 5 minutes
    
    -- Add to signals index
    redis.call('ZADD', index_key, timestamp, signal_key)
    
    -- Keep only last 100 signals
    local count = redis.call('ZCARD', index_key)
    if count > 100 then
        redis.call('ZREMRANGEBYRANK', index_key, 0, count - 101)
    end
    
    -- Publish signal to subscribers
    local signal_channel = "signals:" .. symbol
    local signal_message = signal_type .. "|" .. confidence .. "|" .. price .. "|" .. timestamp
    redis.call('PUBLISH', signal_channel, signal_message)
    
    return {stored = true, published = true, confidence = confidence}
end

-- ============================================================================
-- SESSION MANAGEMENT
-- ============================================================================

-- Update trading session state
-- KEYS[1]: session key (e.g., "session:trader123")
-- ARGV[1]: active_trades, ARGV[2]: total_pnl, ARGV[3]: risk_level, ARGV[4]: timestamp
local function update_session()
    local session_key = KEYS[1]
    local active_trades = tonumber(ARGV[1])
    local total_pnl = tonumber(ARGV[2])
    local risk_level = ARGV[3]
    local timestamp = tonumber(ARGV[4])
    
    -- Update session data
    redis.call('HSET', session_key,
        'active_trades', active_trades,
        'total_pnl', total_pnl,
        'risk_level', risk_level,
        'last_activity', timestamp
    )
    redis.call('EXPIRE', session_key, 3600)  -- 1 hour
    
    -- Update session statistics
    redis.call('HINCRBY', session_key, 'total_updates', 1)
    
    -- Check risk limits
    local max_trades = tonumber(redis.call('HGET', session_key, 'max_trades') or 10)
    local max_loss = tonumber(redis.call('HGET', session_key, 'max_loss') or -1000)
    
    local alerts = {}
    if active_trades > max_trades then
        table.insert(alerts, "MAX_TRADES_EXCEEDED")
    end
    if total_pnl < max_loss then
        table.insert(alerts, "MAX_LOSS_EXCEEDED")
    end
    
    if #alerts > 0 then
        -- Store alerts
        for i = 1, #alerts do
            redis.call('LPUSH', 'session_alerts:' .. session_key, alerts[i] .. "|" .. timestamp)
        end
        redis.call('EXPIRE', 'session_alerts:' .. session_key, 3600)
        
        return {updated = true, alerts = alerts}
    end
    
    return {updated = true, status = "OK"}
end

-- ============================================================================
-- PERFORMANCE ANALYTICS
-- ============================================================================

-- Calculate real-time performance metrics
-- KEYS[1]: performance key (e.g., "performance:EURUSD")
-- ARGV[1]: trade_pnl, ARGV[2]: trade_duration, ARGV[3]: timestamp
local function update_performance()
    local perf_key = KEYS[1]
    local trade_pnl = tonumber(ARGV[1])
    local trade_duration = tonumber(ARGV[2])
    local timestamp = tonumber(ARGV[3])
    
    -- Increment trade counters
    redis.call('HINCRBY', perf_key, 'total_trades', 1)
    
    if trade_pnl > 0 then
        redis.call('HINCRBY', perf_key, 'winning_trades', 1)
        redis.call('HINCRBYFLOAT', perf_key, 'total_profit', trade_pnl)
    else
        redis.call('HINCRBY', perf_key, 'losing_trades', 1)
        redis.call('HINCRBYFLOAT', perf_key, 'total_loss', math.abs(trade_pnl))
    end
    
    -- Update total P&L
    redis.call('HINCRBYFLOAT', perf_key, 'total_pnl', trade_pnl)
    
    -- Update average trade duration
    local current_avg = tonumber(redis.call('HGET', perf_key, 'avg_duration') or 0)
    local total_trades = tonumber(redis.call('HGET', perf_key, 'total_trades'))
    local new_avg = ((current_avg * (total_trades - 1)) + trade_duration) / total_trades
    redis.call('HSET', perf_key, 'avg_duration', new_avg)
    
    -- Calculate win rate
    local winning = tonumber(redis.call('HGET', perf_key, 'winning_trades') or 0)
    local win_rate = (winning / total_trades) * 100
    redis.call('HSET', perf_key, 'win_rate', win_rate)
    
    -- Set expiration
    redis.call('EXPIRE', perf_key, 86400)  -- 24 hours
    
    return {
        total_trades = total_trades,
        win_rate = win_rate,
        total_pnl = redis.call('HGET', perf_key, 'total_pnl'),
        avg_duration = new_avg
    }
end

-- ============================================================================
-- SCRIPT REGISTRATION
-- ============================================================================

-- Return script functions for registration
return {
    store_tick_data = store_tick_data,
    update_position = update_position,
    update_order_book = update_order_book,
    process_signal = process_signal,
    update_session = update_session,
    update_performance = update_performance
}
