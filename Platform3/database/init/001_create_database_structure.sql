-- Forex Trading Platform - PostgreSQL Database Schema
-- Phase 1B: Core Database Architecture Implementation
-- Created: December 24, 2024

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create custom types
CREATE TYPE user_status_enum AS ENUM ('pending', 'active', 'suspended', 'closed');
CREATE TYPE account_type_enum AS ENUM ('demo', 'live', 'paper');
CREATE TYPE trade_status_enum AS ENUM ('pending', 'open', 'closed', 'cancelled');
CREATE TYPE trade_type_enum AS ENUM ('buy', 'sell');
CREATE TYPE order_type_enum AS ENUM ('market', 'limit', 'stop', 'stop_limit');
CREATE TYPE currency_enum AS ENUM ('USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD');

-- Users table with comprehensive profile management
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    country VARCHAR(2), -- ISO country code
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(5) DEFAULT 'en',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE,
    status user_status_enum DEFAULT 'pending',
    kyc_status VARCHAR(20) DEFAULT 'not_started',
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(32),
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    preferences JSONB DEFAULT '{}',
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Trading accounts for users
CREATE TABLE trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_number VARCHAR(20) UNIQUE NOT NULL,
    account_type account_type_enum NOT NULL,
    base_currency currency_enum NOT NULL DEFAULT 'USD',
    balance DECIMAL(20,8) DEFAULT 0.00000000,
    equity DECIMAL(20,8) DEFAULT 0.00000000,
    margin_used DECIMAL(20,8) DEFAULT 0.00000000,
    margin_available DECIMAL(20,8) DEFAULT 0.00000000,
    margin_level DECIMAL(10,2) DEFAULT 0.00,
    leverage INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_balance CHECK (balance >= 0),
    CONSTRAINT valid_leverage CHECK (leverage > 0 AND leverage <= 1000),
    CONSTRAINT valid_margin_level CHECK (margin_level >= 0)
);

-- Currency pairs and instruments
CREATE TABLE currency_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) UNIQUE NOT NULL, -- e.g., EURUSD, GBPJPY
    base_currency currency_enum NOT NULL,
    quote_currency currency_enum NOT NULL,
    pip_precision INTEGER DEFAULT 4,
    min_trade_size DECIMAL(10,5) DEFAULT 0.01000,
    max_trade_size DECIMAL(15,5) DEFAULT 100.00000,
    margin_requirement DECIMAL(5,4) DEFAULT 0.0100, -- 1% = 0.01
    swap_long DECIMAL(10,6) DEFAULT 0.000000,
    swap_short DECIMAL(10,6) DEFAULT 0.000000,
    is_tradeable BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Real-time market data prices
CREATE TABLE market_prices (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    bid DECIMAL(12,6) NOT NULL,
    ask DECIMAL(12,6) NOT NULL,
    spread DECIMAL(8,6) GENERATED ALWAYS AS (ask - bid) STORED,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    volume BIGINT DEFAULT 0,
    high_24h DECIMAL(12,6),
    low_24h DECIMAL(12,6),
    open_24h DECIMAL(12,6),
    CONSTRAINT valid_prices CHECK (bid > 0 AND ask > 0 AND ask >= bid)
);

-- Trading positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    position_type trade_type_enum NOT NULL,
    volume DECIMAL(10,5) NOT NULL,
    open_price DECIMAL(12,6) NOT NULL,
    current_price DECIMAL(12,6),
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    commission DECIMAL(10,2) DEFAULT 0.00,
    swap DECIMAL(10,2) DEFAULT 0.00,
    profit_loss DECIMAL(12,2) DEFAULT 0.00,
    status trade_status_enum DEFAULT 'open',
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    comment TEXT,
    magic_number INTEGER,
    CONSTRAINT valid_volume CHECK (volume > 0),
    CONSTRAINT valid_open_price CHECK (open_price > 0),
    CONSTRAINT valid_stop_loss CHECK (stop_loss IS NULL OR stop_loss > 0),
    CONSTRAINT valid_take_profit CHECK (take_profit IS NULL OR take_profit > 0)
);

-- Trade orders (pending orders)
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    order_type order_type_enum NOT NULL,
    trade_type trade_type_enum NOT NULL,
    volume DECIMAL(10,5) NOT NULL,
    price DECIMAL(12,6),
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    expiration TIMESTAMP WITH TIME ZONE,
    status trade_status_enum DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    comment TEXT,
    magic_number INTEGER,
    CONSTRAINT valid_order_volume CHECK (volume > 0),
    CONSTRAINT valid_order_price CHECK (price IS NULL OR price > 0)
);

-- Trade history for closed positions
CREATE TABLE trade_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    position_id UUID REFERENCES positions(id),
    symbol VARCHAR(10) NOT NULL,
    trade_type trade_type_enum NOT NULL,
    volume DECIMAL(10,5) NOT NULL,
    open_price DECIMAL(12,6) NOT NULL,
    close_price DECIMAL(12,6) NOT NULL,
    commission DECIMAL(10,2) DEFAULT 0.00,
    swap DECIMAL(10,2) DEFAULT 0.00,
    profit_loss DECIMAL(12,2) NOT NULL,
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    duration INTERVAL GENERATED ALWAYS AS (closed_at - opened_at) STORED,
    comment TEXT
);

-- Account balance history
CREATE TABLE balance_history (
    id BIGSERIAL PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    balance_before DECIMAL(20,8) NOT NULL,
    balance_after DECIMAL(20,8) NOT NULL,
    amount DECIMAL(20,8) GENERATED ALWAYS AS (balance_after - balance_before) STORED,
    transaction_type VARCHAR(50) NOT NULL, -- 'deposit', 'withdrawal', 'trade_profit', 'trade_loss', 'commission', 'swap'
    reference_id UUID, -- Can reference trade_history.id, positions.id, etc.
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions for authentication
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Comprehensive audit trail
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    account_id UUID REFERENCES trading_accounts(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    severity VARCHAR(20) DEFAULT 'info' -- 'info', 'warning', 'error', 'critical'
);

-- API rate limiting
CREATE TABLE rate_limits (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    ip_address INET,
    endpoint VARCHAR(255) NOT NULL,
    requests_count INTEGER DEFAULT 1,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- System configuration
CREATE TABLE system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance optimization
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE INDEX idx_trading_accounts_user_id ON trading_accounts(user_id);
CREATE INDEX idx_trading_accounts_account_number ON trading_accounts(account_number);
CREATE INDEX idx_trading_accounts_is_active ON trading_accounts(is_active);

CREATE INDEX idx_market_prices_symbol ON market_prices(symbol);
CREATE INDEX idx_market_prices_timestamp ON market_prices(timestamp DESC);
CREATE INDEX idx_market_prices_symbol_timestamp ON market_prices(symbol, timestamp DESC);

CREATE INDEX idx_positions_account_id ON positions(account_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_opened_at ON positions(opened_at DESC);

CREATE INDEX idx_orders_account_id ON orders(account_id);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);

CREATE INDEX idx_trade_history_account_id ON trade_history(account_id);
CREATE INDEX idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX idx_trade_history_closed_at ON trade_history(closed_at DESC);

CREATE INDEX idx_balance_history_account_id ON balance_history(account_id);
CREATE INDEX idx_balance_history_created_at ON balance_history(created_at DESC);

CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_session_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_resource_type ON audit_logs(resource_type);

CREATE INDEX idx_rate_limits_user_id ON rate_limits(user_id);
CREATE INDEX idx_rate_limits_ip_address ON rate_limits(ip_address);
CREATE INDEX idx_rate_limits_expires_at ON rate_limits(expires_at);

-- Create triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_currency_pairs_updated_at BEFORE UPDATE ON currency_pairs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for generating unique account numbers
CREATE OR REPLACE FUNCTION generate_account_number()
RETURNS TEXT AS $$
DECLARE
    new_number TEXT;
    prefix TEXT := 'FXT';
    counter INTEGER;
BEGIN
    -- Get current counter from system_config
    SELECT COALESCE(config_value::INTEGER, 100000) INTO counter
    FROM system_config 
    WHERE config_key = 'account_number_counter';
    
    -- If counter doesn't exist, create it
    IF counter IS NULL THEN
        counter := 100000;
        INSERT INTO system_config (config_key, config_value, description) 
        VALUES ('account_number_counter', counter::TEXT, 'Auto-incrementing counter for account numbers');
    END IF;
    
    -- Generate new account number
    new_number := prefix || LPAD(counter::TEXT, 7, '0');
    
    -- Increment counter
    UPDATE system_config 
    SET config_value = (counter + 1)::TEXT 
    WHERE config_key = 'account_number_counter';
    
    RETURN new_number;
END;
$$ LANGUAGE plpgsql;
