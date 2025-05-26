-- Forex Trading Platform - Database Functions and Procedures
-- Phase 1B: Core Database Architecture Implementation
-- Created: December 24, 2024

-- Function to safely update account balance with audit trail
CREATE OR REPLACE FUNCTION update_account_balance(
    p_account_id UUID,
    p_amount DECIMAL(20,8),
    p_transaction_type VARCHAR(50),
    p_reference_id UUID DEFAULT NULL,
    p_description TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    current_balance DECIMAL(20,8);
    new_balance DECIMAL(20,8);
BEGIN
    -- Get current balance with row locking
    SELECT balance INTO current_balance
    FROM trading_accounts 
    WHERE id = p_account_id
    FOR UPDATE;
    
    -- Check if account exists
    IF current_balance IS NULL THEN
        RAISE EXCEPTION 'Account not found: %', p_account_id;
    END IF;
    
    -- Calculate new balance
    new_balance := current_balance + p_amount;
    
    -- Prevent negative balance for non-margin operations
    IF new_balance < 0 AND p_transaction_type NOT IN ('margin_call', 'forced_closure') THEN
        RAISE EXCEPTION 'Insufficient balance. Current: %, Required: %', current_balance, -p_amount;
    END IF;
    
    -- Update account balance
    UPDATE trading_accounts 
    SET balance = new_balance,
        equity = new_balance, -- Simplified for now
        updated_at = NOW(),
        last_activity = NOW()
    WHERE id = p_account_id;
    
    -- Insert balance history record
    INSERT INTO balance_history (
        account_id,
        balance_before,
        balance_after,
        transaction_type,
        reference_id,
        description
    ) VALUES (
        p_account_id,
        current_balance,
        new_balance,
        p_transaction_type,
        p_reference_id,
        p_description
    );
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate position profit/loss
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    p_position_id UUID,
    p_current_price DECIMAL(12,6)
)
RETURNS DECIMAL(12,2) AS $$
DECLARE
    pos_record RECORD;
    price_diff DECIMAL(12,6);
    pip_value DECIMAL(12,6);
    pnl DECIMAL(12,2);
    pair_info RECORD;
BEGIN
    -- Get position details
    SELECT * INTO pos_record
    FROM positions 
    WHERE id = p_position_id AND status = 'open';
    
    IF NOT FOUND THEN
        RETURN 0;
    END IF;
    
    -- Get currency pair info
    SELECT * INTO pair_info
    FROM currency_pairs 
    WHERE symbol = pos_record.symbol;
    
    -- Calculate price difference based on position type
    IF pos_record.position_type = 'buy' THEN
        price_diff := p_current_price - pos_record.open_price;
    ELSE
        price_diff := pos_record.open_price - p_current_price;
    END IF;
    
    -- Calculate pip value (simplified calculation)
    -- For USD base pairs: pip_value = volume * 10^(-pip_precision)
    -- For non-USD base pairs: pip_value = volume * 10^(-pip_precision) * exchange_rate
    IF pair_info.quote_currency = 'USD' THEN
        pip_value := pos_record.volume * POWER(10, -pair_info.pip_precision);
    ELSE
        -- Simplified: assume 1.0 exchange rate for non-USD pairs
        pip_value := pos_record.volume * POWER(10, -pair_info.pip_precision);
    END IF;
    
    -- Calculate P&L
    pnl := price_diff * pip_value * POWER(10, pair_info.pip_precision);
    
    RETURN ROUND(pnl, 2);
END;
$$ LANGUAGE plpgsql;

-- Function to update position profit/loss
CREATE OR REPLACE FUNCTION update_position_pnl(
    p_position_id UUID,
    p_current_price DECIMAL(12,6)
)
RETURNS BOOLEAN AS $$
DECLARE
    calculated_pnl DECIMAL(12,2);
BEGIN
    -- Calculate P&L
    calculated_pnl := calculate_position_pnl(p_position_id, p_current_price);
    
    -- Update position
    UPDATE positions 
    SET current_price = p_current_price,
        profit_loss = calculated_pnl
    WHERE id = p_position_id AND status = 'open';
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to close a position
CREATE OR REPLACE FUNCTION close_position(
    p_position_id UUID,
    p_close_price DECIMAL(12,6),
    p_comment TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    pos_record RECORD;
    final_pnl DECIMAL(12,2);
    trade_history_id UUID;
BEGIN
    -- Get position details with lock
    SELECT * INTO pos_record
    FROM positions 
    WHERE id = p_position_id AND status = 'open'
    FOR UPDATE;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Position not found or already closed: %', p_position_id;
    END IF;
    
    -- Calculate final P&L
    final_pnl := calculate_position_pnl(p_position_id, p_close_price);
    
    -- Update position status
    UPDATE positions 
    SET status = 'closed',
        current_price = p_close_price,
        profit_loss = final_pnl,
        closed_at = NOW(),
        comment = COALESCE(comment, '') || COALESCE(' ' || p_comment, '')
    WHERE id = p_position_id;
    
    -- Create trade history record
    INSERT INTO trade_history (
        account_id,
        position_id,
        symbol,
        trade_type,
        volume,
        open_price,
        close_price,
        commission,
        swap,
        profit_loss,
        opened_at,
        closed_at,
        comment
    ) VALUES (
        pos_record.account_id,
        p_position_id,
        pos_record.symbol,
        pos_record.position_type,
        pos_record.volume,
        pos_record.open_price,
        p_close_price,
        pos_record.commission,
        pos_record.swap,
        final_pnl,
        pos_record.opened_at,
        NOW(),
        p_comment
    ) RETURNING id INTO trade_history_id;
    
    -- Update account balance
    PERFORM update_account_balance(
        pos_record.account_id,
        final_pnl - pos_record.commission - pos_record.swap,
        'trade_' || CASE WHEN final_pnl >= 0 THEN 'profit' ELSE 'loss' END,
        trade_history_id,
        'Position closed: ' || pos_record.symbol || ' - ' || final_pnl::TEXT
    );
    
    RETURN trade_history_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check margin requirements
CREATE OR REPLACE FUNCTION check_margin_requirement(
    p_account_id UUID,
    p_symbol VARCHAR(10),
    p_volume DECIMAL(10,5)
)
RETURNS DECIMAL(12,2) AS $$
DECLARE
    account_record RECORD;
    pair_record RECORD;
    current_price DECIMAL(12,6);
    margin_required DECIMAL(12,2);
BEGIN
    -- Get account details
    SELECT * INTO account_record
    FROM trading_accounts 
    WHERE id = p_account_id AND is_active = true;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Account not found or inactive: %', p_account_id;
    END IF;
    
    -- Get currency pair details
    SELECT * INTO pair_record
    FROM currency_pairs 
    WHERE symbol = p_symbol AND is_tradeable = true;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Currency pair not found or not tradeable: %', p_symbol;
    END IF;
    
    -- Get current market price
    SELECT (bid + ask) / 2 INTO current_price
    FROM market_prices 
    WHERE symbol = p_symbol
    ORDER BY timestamp DESC 
    LIMIT 1;
    
    IF current_price IS NULL THEN
        RAISE EXCEPTION 'No current price available for: %', p_symbol;
    END IF;
    
    -- Calculate margin required
    margin_required := p_volume * current_price * pair_record.margin_requirement * (100.0 / account_record.leverage);
    
    RETURN margin_required;
END;
$$ LANGUAGE plpgsql;

-- Function to validate trade before execution
CREATE OR REPLACE FUNCTION validate_trade(
    p_account_id UUID,
    p_symbol VARCHAR(10),
    p_volume DECIMAL(10,5),
    p_trade_type trade_type_enum
)
RETURNS BOOLEAN AS $$
DECLARE
    account_record RECORD;
    pair_record RECORD;
    margin_required DECIMAL(12,2);
BEGIN
    -- Get account details
    SELECT * INTO account_record
    FROM trading_accounts 
    WHERE id = p_account_id AND is_active = true;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Account not found or inactive';
    END IF;
    
    -- Check if trading is enabled
    IF NOT (SELECT config_value::BOOLEAN FROM system_config WHERE config_key = 'trading_enabled') THEN
        RAISE EXCEPTION 'Trading is currently disabled';
    END IF;
    
    -- Get currency pair details
    SELECT * INTO pair_record
    FROM currency_pairs 
    WHERE symbol = p_symbol AND is_tradeable = true;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Currency pair not tradeable: %', p_symbol;
    END IF;
    
    -- Validate volume
    IF p_volume < pair_record.min_trade_size OR p_volume > pair_record.max_trade_size THEN
        RAISE EXCEPTION 'Invalid trade size. Min: %, Max: %', pair_record.min_trade_size, pair_record.max_trade_size;
    END IF;
    
    -- Check margin requirement
    margin_required := check_margin_requirement(p_account_id, p_symbol, p_volume);
    
    IF margin_required > account_record.margin_available THEN
        RAISE EXCEPTION 'Insufficient margin. Required: %, Available: %', margin_required, account_record.margin_available;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < NOW() OR is_active = false;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get account summary
CREATE OR REPLACE FUNCTION get_account_summary(p_account_id UUID)
RETURNS TABLE (
    account_id UUID,
    account_number VARCHAR(20),
    balance DECIMAL(20,8),
    equity DECIMAL(20,8),
    margin_used DECIMAL(20,8),
    margin_available DECIMAL(20,8),
    margin_level DECIMAL(10,2),
    open_positions_count BIGINT,
    total_profit_loss DECIMAL(12,2),
    daily_profit_loss DECIMAL(12,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ta.id,
        ta.account_number,
        ta.balance,
        ta.equity,
        ta.margin_used,
        ta.margin_available,
        ta.margin_level,
        COALESCE(pos_stats.position_count, 0) as open_positions_count,
        COALESCE(pos_stats.total_pnl, 0) as total_profit_loss,
        COALESCE(daily_stats.daily_pnl, 0) as daily_profit_loss
    FROM trading_accounts ta
    LEFT JOIN (
        SELECT 
            account_id,
            COUNT(*) as position_count,
            SUM(profit_loss) as total_pnl
        FROM positions 
        WHERE account_id = p_account_id AND status = 'open'
        GROUP BY account_id
    ) pos_stats ON ta.id = pos_stats.account_id
    LEFT JOIN (
        SELECT 
            account_id,
            SUM(profit_loss) as daily_pnl
        FROM trade_history 
        WHERE account_id = p_account_id 
        AND closed_at >= CURRENT_DATE
        GROUP BY account_id
    ) daily_stats ON ta.id = daily_stats.account_id
    WHERE ta.id = p_account_id;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for new functions
CREATE INDEX IF NOT EXISTS idx_positions_account_status ON positions(account_id, status);
CREATE INDEX IF NOT EXISTS idx_trade_history_account_closed ON trade_history(account_id, closed_at);
CREATE INDEX IF NOT EXISTS idx_balance_history_transaction_type ON balance_history(transaction_type);

-- Create views for common queries
CREATE OR REPLACE VIEW v_account_positions AS
SELECT 
    p.*,
    cp.base_currency,
    cp.quote_currency,
    cp.pip_precision,
    mp.bid as current_bid,
    mp.ask as current_ask,
    ((mp.bid + mp.ask) / 2) as current_mid_price
FROM positions p
JOIN currency_pairs cp ON p.symbol = cp.symbol
LEFT JOIN market_prices mp ON p.symbol = mp.symbol
WHERE p.status = 'open';

CREATE OR REPLACE VIEW v_daily_trade_summary AS
SELECT 
    th.account_id,
    DATE(th.closed_at) as trade_date,
    COUNT(*) as trades_count,
    SUM(CASE WHEN th.profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN th.profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(th.profit_loss) as total_pnl,
    AVG(th.profit_loss) as avg_pnl,
    MAX(th.profit_loss) as best_trade,
    MIN(th.profit_loss) as worst_trade,
    SUM(th.commission) as total_commission,
    SUM(th.swap) as total_swap
FROM trade_history th
GROUP BY th.account_id, DATE(th.closed_at);

-- Grant necessary permissions (adjust as needed for your user setup)
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO forex_user;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO forex_user;
