-- Personal Forex Trading Platform - Database Initialization
-- PostgreSQL Schema Setup for Owner-Only Trading Platform
-- Version: 1.0.0

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ========================================
-- ENUM TYPES
-- ========================================

-- User status for platform owner
CREATE TYPE user_status_type AS ENUM (
    'active',
    'inactive',
    'locked'
);

-- Account types
CREATE TYPE account_type AS ENUM (
    'demo',
    'live',
    'paper'
);

-- Order types
CREATE TYPE order_type AS ENUM (
    'market',
    'limit',
    'stop',
    'stop_limit',
    'trailing_stop'
);

-- Order status
CREATE TYPE order_status AS ENUM (
    'pending',
    'filled',
    'partial',
    'cancelled',
    'expired',
    'rejected'
);

-- Order side
CREATE TYPE order_side AS ENUM (
    'buy',
    'sell'
);

-- Position status
CREATE TYPE position_status AS ENUM (
    'open',
    'closed',
    'partial'
);

-- Transaction types
CREATE TYPE transaction_type AS ENUM (
    'deposit',
    'withdrawal',
    'trade',
    'fee',
    'interest',
    'dividend'
);

-- Risk level
CREATE TYPE risk_level AS ENUM (
    'low',
    'medium',
    'high',
    'critical'
);

-- ========================================
-- CORE TABLES
-- ========================================

-- Platform owner/user table (single user system)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    status user_status_type DEFAULT 'active',
    timezone VARCHAR(50) DEFAULT 'UTC',
    locale VARCHAR(10) DEFAULT 'en-US',
    last_login_at TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trading accounts
CREATE TABLE trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_type account_type NOT NULL,
    account_number VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    balance DECIMAL(20,8) DEFAULT 0,
    equity DECIMAL(20,8) DEFAULT 0,
    margin_used DECIMAL(20,8) DEFAULT 0,
    margin_available DECIMAL(20,8) DEFAULT 0,
    margin_level DECIMAL(10,4) DEFAULT 0,
    leverage INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Currency pairs and instruments
CREATE TABLE instruments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    base_currency VARCHAR(3) NOT NULL,
    quote_currency VARCHAR(3) NOT NULL,
    pip_size DECIMAL(10,8) NOT NULL,
    min_trade_size DECIMAL(15,6) NOT NULL,
    max_trade_size DECIMAL(15,6) NOT NULL,
    margin_requirement DECIMAL(8,4) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Orders table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    order_type order_type NOT NULL,
    side order_side NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(15,6),
    stop_price DECIMAL(15,6),
    limit_price DECIMAL(15,6),
    filled_quantity DECIMAL(15,6) DEFAULT 0,
    remaining_quantity DECIMAL(15,6) NOT NULL,
    avg_fill_price DECIMAL(15,6),
    status order_status DEFAULT 'pending',
    time_in_force VARCHAR(10) DEFAULT 'GTC',
    order_tag VARCHAR(50),
    broker_order_id VARCHAR(100),
    commission DECIMAL(10,4) DEFAULT 0,
    swap DECIMAL(10,4) DEFAULT 0,
    profit_loss DECIMAL(15,6) DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    filled_at TIMESTAMP,
    cancelled_at TIMESTAMP
);

-- Positions table
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    side order_side NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    entry_price DECIMAL(15,6) NOT NULL,
    current_price DECIMAL(15,6),
    stop_loss DECIMAL(15,6),
    take_profit DECIMAL(15,6),
    unrealized_pnl DECIMAL(15,6) DEFAULT 0,
    realized_pnl DECIMAL(15,6) DEFAULT 0,
    commission DECIMAL(10,4) DEFAULT 0,
    swap DECIMAL(10,4) DEFAULT 0,
    status position_status DEFAULT 'open',
    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trade executions/fills
CREATE TABLE trade_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id),
    position_id UUID REFERENCES positions(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    side order_side NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(15,6) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    execution_time TIMESTAMP DEFAULT NOW(),
    broker_execution_id VARCHAR(100),
    liquidity_flag VARCHAR(10)
);

-- Account transactions
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id),
    transaction_type transaction_type NOT NULL,
    amount DECIMAL(15,6) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    balance_before DECIMAL(20,8) NOT NULL,
    balance_after DECIMAL(20,8) NOT NULL,
    reference_id UUID, -- Can reference orders, positions, etc.
    reference_type VARCHAR(50),
    description TEXT,
    external_transaction_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Risk management rules
CREATE TABLE risk_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading_accounts(id),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- 'position_size', 'daily_loss', 'drawdown', etc.
    threshold_value DECIMAL(15,6) NOT NULL,
    threshold_currency VARCHAR(3),
    risk_level risk_level NOT NULL,
    is_active BOOLEAN DEFAULT true,
    action_on_breach VARCHAR(50) DEFAULT 'alert', -- 'alert', 'close_positions', 'disable_trading'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Risk alerts and violations
CREATE TABLE risk_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id),
    rule_id UUID REFERENCES risk_rules(id),
    alert_type VARCHAR(50) NOT NULL,
    risk_level risk_level NOT NULL,
    current_value DECIMAL(15,6),
    threshold_value DECIMAL(15,6),
    message TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Market data snapshots (for caching latest prices)
CREATE TABLE market_data_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    bid DECIMAL(15,6) NOT NULL,
    ask DECIMAL(15,6) NOT NULL,
    mid_price DECIMAL(15,6) NOT NULL,
    spread DECIMAL(10,6) NOT NULL,
    volume DECIMAL(20,2) DEFAULT 0,
    high_24h DECIMAL(15,6),
    low_24h DECIMAL(15,6),
    change_24h DECIMAL(8,4),
    change_percent_24h DECIMAL(8,4),
    timestamp TIMESTAMP DEFAULT NOW(),
    source VARCHAR(50)
);

-- Platform settings and configuration
CREATE TABLE platform_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    setting_type VARCHAR(20) DEFAULT 'string', -- 'string', 'number', 'boolean', 'json'
    description TEXT,
    is_system BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Audit logs for all platform activities
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Session management
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity_at TIMESTAMP DEFAULT NOW()
);

-- ========================================
-- INDEXES FOR PERFORMANCE
-- ========================================

-- User indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);

-- Trading account indexes
CREATE INDEX idx_trading_accounts_user_id ON trading_accounts(user_id);
CREATE INDEX idx_trading_accounts_account_type ON trading_accounts(account_type);
CREATE INDEX idx_trading_accounts_active ON trading_accounts(is_active);

-- Order indexes
CREATE INDEX idx_orders_account_id ON orders(account_id);
CREATE INDEX idx_orders_instrument_id ON orders(instrument_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_side_status ON orders(side, status);

-- Position indexes
CREATE INDEX idx_positions_account_id ON positions(account_id);
CREATE INDEX idx_positions_instrument_id ON positions(instrument_id);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_opened_at ON positions(opened_at);

-- Transaction indexes
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_created_at ON transactions(created_at);

-- Market data indexes
CREATE INDEX idx_market_data_instrument_id ON market_data_snapshots(instrument_id);
CREATE INDEX idx_market_data_timestamp ON market_data_snapshots(timestamp);
CREATE UNIQUE INDEX idx_market_data_unique ON market_data_snapshots(instrument_id, source);

-- Audit log indexes
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- Session indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_active ON user_sessions(is_active);

-- ========================================
-- INITIAL DATA SETUP
-- ========================================

-- Insert default platform owner (update with your details)
INSERT INTO users (id, email, password_hash, first_name, last_name, status) VALUES (
    uuid_generate_v4(),
    'owner@forexplatform.com',
    '$2b$10$example.hash.here', -- Replace with actual bcrypt hash
    'Platform',
    'Owner',
    'active'
);

-- Insert major forex pairs
INSERT INTO instruments (symbol, name, base_currency, quote_currency, pip_size, min_trade_size, max_trade_size, margin_requirement) VALUES
('EURUSD', 'Euro vs US Dollar', 'EUR', 'USD', 0.0001, 0.01, 100.0, 0.02),
('GBPUSD', 'British Pound vs US Dollar', 'GBP', 'USD', 0.0001, 0.01, 100.0, 0.02),
('USDJPY', 'US Dollar vs Japanese Yen', 'USD', 'JPY', 0.01, 0.01, 100.0, 0.02),
('USDCHF', 'US Dollar vs Swiss Franc', 'USD', 'CHF', 0.0001, 0.01, 100.0, 0.02),
('AUDUSD', 'Australian Dollar vs US Dollar', 'AUD', 'USD', 0.0001, 0.01, 100.0, 0.02),
('USDCAD', 'US Dollar vs Canadian Dollar', 'USD', 'CAD', 0.0001, 0.01, 100.0, 0.02),
('NZDUSD', 'New Zealand Dollar vs US Dollar', 'NZD', 'USD', 0.0001, 0.01, 100.0, 0.02),
('EURJPY', 'Euro vs Japanese Yen', 'EUR', 'JPY', 0.01, 0.01, 100.0, 0.02),
('GBPJPY', 'British Pound vs Japanese Yen', 'GBP', 'JPY', 0.01, 0.01, 100.0, 0.02),
('EURGBP', 'Euro vs British Pound', 'EUR', 'GBP', 0.0001, 0.01, 100.0, 0.02);

-- Insert default platform settings
INSERT INTO platform_settings (setting_key, setting_value, setting_type, description, is_system) VALUES
('platform_name', 'Personal Forex Trading Platform', 'string', 'Platform display name', true),
('default_leverage', '100', 'number', 'Default leverage for new accounts', false),
('max_positions', '50', 'number', 'Maximum open positions allowed', false),
('session_timeout', '3600', 'number', 'Session timeout in seconds', true),
('risk_monitoring_enabled', 'true', 'boolean', 'Enable real-time risk monitoring', false),
('auto_close_on_margin_call', 'true', 'boolean', 'Automatically close positions on margin call', false),
('market_data_refresh_rate', '100', 'number', 'Market data refresh rate in milliseconds', false);

-- Create a default demo account for the owner
INSERT INTO trading_accounts (user_id, account_type, account_number, name, currency, balance, equity, margin_available, leverage)
SELECT 
    id,
    'demo',
    'DEMO-' || EXTRACT(EPOCH FROM NOW())::TEXT,
    'Demo Account',
    'USD',
    100000.00,
    100000.00,
    100000.00,
    100
FROM users WHERE email = 'owner@forexplatform.com' LIMIT 1;

-- ========================================
-- TRIGGERS AND FUNCTIONS
-- ========================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update timestamp triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_platform_settings_updated_at BEFORE UPDATE ON platform_settings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to create audit log entries
CREATE OR REPLACE FUNCTION create_audit_log()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_logs (
        user_id,
        action,
        resource_type,
        resource_id,
        old_values,
        new_values,
        created_at
    ) VALUES (
        COALESCE(current_setting('app.current_user_id', true)::UUID, NULL),
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id::TEXT, OLD.id::TEXT),
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
        NOW()
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to important tables
CREATE TRIGGER audit_users AFTER INSERT OR UPDATE OR DELETE ON users FOR EACH ROW EXECUTE FUNCTION create_audit_log();
CREATE TRIGGER audit_orders AFTER INSERT OR UPDATE OR DELETE ON orders FOR EACH ROW EXECUTE FUNCTION create_audit_log();
CREATE TRIGGER audit_positions AFTER INSERT OR UPDATE OR DELETE ON positions FOR EACH ROW EXECUTE FUNCTION create_audit_log();
CREATE TRIGGER audit_transactions AFTER INSERT OR UPDATE OR DELETE ON transactions FOR EACH ROW EXECUTE FUNCTION create_audit_log();

-- ========================================
-- VIEWS FOR COMMON QUERIES
-- ========================================

-- Account summary view
CREATE VIEW account_summary AS
SELECT 
    ta.id as account_id,
    ta.account_number,
    ta.name as account_name,
    ta.account_type,
    ta.currency,
    ta.balance,
    ta.equity,
    ta.margin_used,
    ta.margin_available,
    ta.margin_level,
    ta.leverage,
    COUNT(DISTINCT p.id) as open_positions,
    COUNT(DISTINCT o.id) as pending_orders,
    COALESCE(SUM(p.unrealized_pnl), 0) as total_unrealized_pnl,
    ta.created_at
FROM trading_accounts ta
LEFT JOIN positions p ON ta.id = p.account_id AND p.status = 'open'
LEFT JOIN orders o ON ta.id = o.account_id AND o.status = 'pending'
WHERE ta.is_active = true
GROUP BY ta.id, ta.account_number, ta.name, ta.account_type, ta.currency, 
         ta.balance, ta.equity, ta.margin_used, ta.margin_available, 
         ta.margin_level, ta.leverage, ta.created_at;

-- Position summary view
CREATE VIEW position_summary AS
SELECT 
    p.id as position_id,
    p.account_id,
    i.symbol,
    i.name as instrument_name,
    p.side,
    p.quantity,
    p.entry_price,
    p.current_price,
    p.stop_loss,
    p.take_profit,
    p.unrealized_pnl,
    p.realized_pnl,
    p.commission,
    p.swap,
    p.status,
    p.opened_at,
    EXTRACT(EPOCH FROM (NOW() - p.opened_at))/3600 as hours_open
FROM positions p
JOIN instruments i ON p.instrument_id = i.id
WHERE p.status = 'open';

-- Market overview view
CREATE VIEW market_overview AS
SELECT 
    i.symbol,
    i.name,
    i.base_currency,
    i.quote_currency,
    md.bid,
    md.ask,
    md.mid_price,
    md.spread,
    md.high_24h,
    md.low_24h,
    md.change_24h,
    md.change_percent_24h,
    md.volume,
    md.timestamp as last_update
FROM instruments i
LEFT JOIN market_data_snapshots md ON i.id = md.instrument_id
WHERE i.is_active = true
ORDER BY i.symbol;

COMMIT;
