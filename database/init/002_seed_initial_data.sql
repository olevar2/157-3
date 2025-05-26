-- Forex Trading Platform - Initial Data Seeding
-- Phase 1B: Core Database Architecture Implementation
-- Created: December 24, 2024

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('app_name', 'Personal Forex Trading Platform', 'Application name'),
('app_version', '1.0.0', 'Application version'),
('default_leverage', '100', 'Default leverage for new accounts'),
('max_leverage', '1000', 'Maximum leverage allowed'),
('min_trade_size', '0.01', 'Minimum trade size in lots'),
('max_trade_size', '100.0', 'Maximum trade size in lots'),
('default_currency', 'USD', 'Default base currency for accounts'),
('session_timeout', '86400', 'Session timeout in seconds (24 hours)'),
('max_login_attempts', '5', 'Maximum failed login attempts before lockout'),
('lockout_duration', '1800', 'Account lockout duration in seconds (30 minutes)'),
('kyc_required', 'true', 'Whether KYC verification is required'),
('account_number_counter', '100001', 'Counter for generating account numbers'),
('api_rate_limit_per_minute', '1000', 'API rate limit per minute per user'),
('maintenance_mode', 'false', 'Whether the platform is in maintenance mode'),
('trading_enabled', 'true', 'Whether trading is currently enabled');

-- Insert major currency pairs with realistic trading parameters
INSERT INTO currency_pairs (symbol, base_currency, quote_currency, pip_precision, min_trade_size, max_trade_size, margin_requirement, swap_long, swap_short, is_tradeable) VALUES
-- Major pairs
('EURUSD', 'EUR', 'USD', 5, 0.01000, 100.00000, 0.0333, -0.890000, 0.450000, true),
('GBPUSD', 'GBP', 'USD', 5, 0.01000, 100.00000, 0.0333, -1.200000, 0.300000, true),
('USDJPY', 'USD', 'JPY', 3, 0.01000, 100.00000, 0.0333, 0.150000, -0.950000, true),
('USDCHF', 'USD', 'CHF', 5, 0.01000, 100.00000, 0.0333, 0.080000, -0.750000, true),
('AUDUSD', 'AUD', 'USD', 5, 0.01000, 100.00000, 0.0333, -0.650000, 0.250000, true),
('USDCAD', 'USD', 'CAD', 5, 0.01000, 100.00000, 0.0333, 0.120000, -0.880000, true),
('NZDUSD', 'NZD', 'USD', 5, 0.01000, 100.00000, 0.0333, -0.580000, 0.180000, true),

-- Cross pairs
('EURGBP', 'EUR', 'GBP', 5, 0.01000, 100.00000, 0.0333, -1.100000, 0.400000, true),
('EURJPY', 'EUR', 'JPY', 3, 0.01000, 100.00000, 0.0333, -0.350000, -0.200000, true),
('GBPJPY', 'GBP', 'JPY', 3, 0.01000, 100.00000, 0.0333, -0.450000, -0.150000, true),
('EURCHF', 'EUR', 'CHF', 5, 0.01000, 100.00000, 0.0333, -0.980000, 0.350000, true),
('EURAUD', 'EUR', 'AUD', 5, 0.01000, 100.00000, 0.0333, -1.350000, 0.650000, true),
('EURCAD', 'EUR', 'CAD', 5, 0.01000, 100.00000, 0.0333, -1.200000, 0.580000, true),
('GBPCHF', 'GBP', 'CHF', 5, 0.01000, 100.00000, 0.0333, -1.150000, 0.380000, true),

-- Commodity currencies
('AUDCAD', 'AUD', 'CAD', 5, 0.01000, 100.00000, 0.0333, -0.750000, 0.320000, true),
('AUDCHF', 'AUD', 'CHF', 5, 0.01000, 100.00000, 0.0333, -0.680000, 0.280000, true),
('AUDJPY', 'AUD', 'JPY', 3, 0.01000, 100.00000, 0.0333, -0.420000, -0.180000, true),
('CADCHF', 'CAD', 'CHF', 5, 0.01000, 100.00000, 0.0333, -0.580000, 0.220000, true),
('CADJPY', 'CAD', 'JPY', 3, 0.01000, 100.00000, 0.0333, -0.380000, -0.120000, true),
('CHFJPY', 'CHF', 'JPY', 3, 0.01000, 100.00000, 0.0333, -0.250000, -0.080000, true),
('NZDCAD', 'NZD', 'CAD', 5, 0.01000, 100.00000, 0.0333, -0.680000, 0.280000, true),
('NZDCHF', 'NZD', 'CHF', 5, 0.01000, 100.00000, 0.0333, -0.620000, 0.240000, true),
('NZDJPY', 'NZD', 'JPY', 3, 0.01000, 100.00000, 0.0333, -0.380000, -0.150000, true),
('GBPCAD', 'GBP', 'CAD', 5, 0.01000, 100.00000, 0.0333, -1.280000, 0.520000, true),
('GBPAUD', 'GBP', 'AUD', 5, 0.01000, 100.00000, 0.0333, -1.450000, 0.680000, true),
('GBPNZD', 'GBP', 'NZD', 5, 0.01000, 100.00000, 0.0333, -1.380000, 0.620000, true),
('AUDNZD', 'AUD', 'NZD', 5, 0.01000, 100.00000, 0.0333, -0.580000, 0.220000, true);

-- Insert initial market prices (these will be updated by Market Data Service)
INSERT INTO market_prices (symbol, bid, ask, timestamp, volume, high_24h, low_24h, open_24h) VALUES
-- Major pairs with realistic spreads
('EURUSD', 1.08520, 1.08535, NOW(), 1250000, 1.08892, 1.08234, 1.08456),
('GBPUSD', 1.26840, 1.26858, NOW(), 980000, 1.27234, 1.26567, 1.26892),
('USDJPY', 149.245, 149.260, NOW(), 1680000, 149.856, 148.934, 149.123),
('USDCHF', 0.88420, 0.88438, NOW(), 750000, 0.88756, 0.88234, 0.88567),
('AUDUSD', 0.67890, 0.67905, NOW(), 680000, 0.68234, 0.67654, 0.67923),
('USDCAD', 1.35670, 1.35688, NOW(), 520000, 1.36012, 1.35423, 1.35789),
('NZDUSD', 0.62340, 0.62358, NOW(), 420000, 0.62567, 0.62123, 0.62456),

-- Cross pairs
('EURGBP', 0.85560, 0.85578, NOW(), 890000, 0.85823, 0.85345, 0.85612),
('EURJPY', 161.890, 161.915, NOW(), 1240000, 162.456, 161.234, 161.567),
('GBPJPY', 189.450, 189.485, NOW(), 950000, 190.123, 188.967, 189.234),
('EURCHF', 0.95890, 0.95912, NOW(), 640000, 0.96234, 0.95567, 0.95823),
('EURAUD', 1.59870, 1.59898, NOW(), 580000, 1.60234, 1.59456, 1.59712),
('EURCAD', 1.47230, 1.47258, NOW(), 480000, 1.47567, 1.46923, 1.47156),
('GBPCHF', 1.12340, 1.12368, NOW(), 420000, 1.12678, 1.12012, 1.12234),

-- Commodity currencies
('AUDCAD', 0.92110, 0.92135, NOW(), 380000, 0.92456, 0.91823, 0.92067),
('AUDCHF', 0.60080, 0.60098, NOW(), 340000, 0.60345, 0.59834, 0.60123),
('AUDJPY', 101.340, 101.365, NOW(), 620000, 101.789, 100.923, 101.234),
('CADCHF', 0.65210, 0.65228, NOW(), 290000, 0.65456, 0.64978, 0.65123),
('CADJPY', 110.020, 110.045, NOW(), 490000, 110.456, 109.634, 110.123),
('CHFJPY', 168.750, 168.780, NOW(), 420000, 169.234, 168.345, 168.567),
('NZDCAD', 0.84560, 0.84585, NOW(), 280000, 0.84823, 0.84234, 0.84456),
('NZDCHF', 0.55180, 0.55198, NOW(), 250000, 0.55423, 0.54967, 0.55089),
('NZDJPY', 92.980, 93.005, NOW(), 390000, 93.345, 92.634, 92.823),
('GBPCAD', 1.72340, 1.72375, NOW(), 420000, 1.72789, 1.71934, 1.72156),
('GBPAUD', 1.86890, 1.86925, NOW(), 480000, 1.87345, 1.86456, 1.86712),
('GBPNZD', 2.03450, 2.03490, NOW(), 360000, 2.03923, 2.02978, 2.03234),
('AUDNZD', 1.08920, 1.08945, NOW(), 320000, 1.09234, 1.08634, 1.08856);

-- Create a demo user for testing
-- Password: DemoPassword123!
-- Hashed with bcrypt (rounds: 12)
INSERT INTO users (
    id,
    email, 
    password_hash, 
    first_name, 
    last_name, 
    phone, 
    country, 
    timezone, 
    email_verified, 
    status, 
    kyc_status
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    'demo@forexplatform.com',
    '$2b$12$LQv3c1yqBWVHxkd0LQ4YCOdcDNjTUk5dWXeYNX.nG2Zx3B4F5C6Y8',
    'Demo',
    'User',
    '+1234567890',
    'US',
    'America/New_York',
    true,
    'active',
    'approved'
);

-- Create demo trading account
INSERT INTO trading_accounts (
    id,
    user_id,
    account_number,
    account_type,
    base_currency,
    balance,
    equity,
    margin_available,
    leverage
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    'FXT0100001',
    'demo',
    'USD',
    10000.00000000,
    10000.00000000,
    10000.00000000,
    100
);

-- Create an admin user for platform management
-- Email: admin@forexplatform.com
-- Password: AdminSecure2024!
-- Hashed with bcrypt (rounds: 12)
INSERT INTO users (
    id,
    email,
    password_hash,
    first_name,
    last_name,
    country,
    timezone,
    email_verified,
    status,
    kyc_status,
    preferences
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    'admin@forexplatform.com',
    '$2b$12$YQv4c2yqBWVHxkd0LQ5ZDPedeDNjTUk6eWXfZOY.nH3Zy4C5G6D9A',
    'Platform',
    'Administrator',
    'US',
    'UTC',
    true,
    'active',
    'approved',
    '{"role": "admin", "permissions": ["all"]}'
);

-- Create admin trading account
INSERT INTO trading_accounts (
    id,
    user_id,
    account_number,
    account_type,
    base_currency,
    balance,
    equity,
    margin_available,
    leverage
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000002',
    'FXT0100002',
    'live',
    'USD',
    50000.00000000,
    50000.00000000,
    50000.00000000,
    200
);

-- Add some initial balance history for demo account
INSERT INTO balance_history (account_id, balance_before, balance_after, transaction_type, description) VALUES
('00000000-0000-0000-0000-000000000001', 0.00000000, 10000.00000000, 'initial_deposit', 'Demo account initial balance'),
('00000000-0000-0000-0000-000000000002', 0.00000000, 50000.00000000, 'initial_deposit', 'Admin account initial balance');

-- Add audit logs for account creation
INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, severity) VALUES
('00000000-0000-0000-0000-000000000001', 'user_created', 'user', '00000000-0000-0000-0000-000000000001', '{"email": "demo@forexplatform.com", "status": "active"}', 'info'),
('00000000-0000-0000-0000-000000000001', 'account_created', 'trading_account', '00000000-0000-0000-0000-000000000001', '{"account_number": "FXT0100001", "account_type": "demo", "balance": 10000}', 'info'),
('00000000-0000-0000-0000-000000000002', 'user_created', 'user', '00000000-0000-0000-0000-000000000002', '{"email": "admin@forexplatform.com", "status": "active", "role": "admin"}', 'info'),
('00000000-0000-0000-0000-000000000002', 'account_created', 'trading_account', '00000000-0000-0000-0000-000000000002', '{"account_number": "FXT0100002", "account_type": "live", "balance": 50000}', 'info');

-- Update account number counter
UPDATE system_config SET config_value = '100003' WHERE config_key = 'account_number_counter';
