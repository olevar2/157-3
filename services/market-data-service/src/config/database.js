const { Pool } = require('pg');

// Database configuration
const dbConfig = {
  user: process.env.DB_USER || 'postgres',
  host: process.env.DB_HOST || 'localhost',
  database: process.env.DB_NAME || 'platform3_market_data',
  password: process.env.DB_PASSWORD || 'password',
  port: process.env.DB_PORT || 5432,
  
  // Connection pool settings
  max: 20, // Maximum number of clients in pool
  idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
  connectionTimeoutMillis: 2000, // Return error after 2 seconds if connection could not be established
  maxUses: 7500, // Close (and replace) a connection after it has been used 7500 times
};

// Create connection pool
const pool = new Pool(dbConfig);

// Pool event handlers
pool.on('connect', (client) => {
  console.log('New client connected to market data database');
});

pool.on('error', (err, client) => {
  console.error('Unexpected error on idle client', err);
  process.exit(-1);
});

pool.on('acquire', (client) => {
  console.log('Client acquired from pool');
});

pool.on('remove', (client) => {
  console.log('Client removed from pool');
});

/**
 * Execute a query
 */
const query = async (text, params) => {
  const start = Date.now();
  try {
    const res = await pool.query(text, params);
    const duration = Date.now() - start;
    console.log('Executed query', { text, duration, rows: res.rowCount });
    return res;
  } catch (error) {
    console.error('Database query error:', error);
    throw error;
  }
};

/**
 * Get a client from the pool
 */
const getClient = async () => {
  try {
    const client = await pool.connect();
    return client;
  } catch (error) {
    console.error('Error getting client from pool:', error);
    throw error;
  }
};

/**
 * Initialize database tables
 */
const initializeTables = async () => {
  try {
    console.log('Initializing market data database tables...');
    
    // Create price_data table
    await query(`
      CREATE TABLE IF NOT EXISTS price_data (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        bid DECIMAL(10, 6) NOT NULL,
        ask DECIMAL(10, 6) NOT NULL,
        spread DECIMAL(8, 6) NOT NULL,
        change_24h DECIMAL(8, 4) DEFAULT 0,
        volume_24h BIGINT DEFAULT 0,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        INDEX idx_symbol_timestamp (symbol, timestamp),
        INDEX idx_timestamp (timestamp)
      );
    `);

    // Create market_stats table
    await query(`
      CREATE TABLE IF NOT EXISTS market_stats (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL UNIQUE,
        high_24h DECIMAL(10, 6),
        low_24h DECIMAL(10, 6),
        open_24h DECIMAL(10, 6),
        close_24h DECIMAL(10, 6),
        volume_24h BIGINT DEFAULT 0,
        trades_count_24h INTEGER DEFAULT 0,
        volatility DECIMAL(8, 6) DEFAULT 0,
        last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        INDEX idx_symbol (symbol),
        INDEX idx_last_updated (last_updated)
      );
    `);

    // Create instrument_config table
    await query(`
      CREATE TABLE IF NOT EXISTS instrument_config (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL UNIQUE,
        name VARCHAR(100) NOT NULL,
        base_currency VARCHAR(3) NOT NULL,
        quote_currency VARCHAR(3) NOT NULL,
        pip_size DECIMAL(10, 8) NOT NULL DEFAULT 0.0001,
        min_spread DECIMAL(8, 6) NOT NULL DEFAULT 0.0001,
        max_spread DECIMAL(8, 6) NOT NULL DEFAULT 0.005,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        INDEX idx_symbol (symbol),
        INDEX idx_active (is_active)
      );
    `);

    console.log('Market data database tables initialized successfully');
    
    // Insert default instrument configurations
    await insertDefaultInstruments();
    
  } catch (error) {
    console.error('Error initializing database tables:', error);
    throw error;
  }
};

/**
 * Insert default trading instruments
 */
const insertDefaultInstruments = async () => {
  try {
    const instruments = [
      { symbol: 'EURUSD', name: 'Euro / US Dollar', base: 'EUR', quote: 'USD', pipSize: 0.0001 },
      { symbol: 'GBPUSD', name: 'British Pound / US Dollar', base: 'GBP', quote: 'USD', pipSize: 0.0001 },
      { symbol: 'USDJPY', name: 'US Dollar / Japanese Yen', base: 'USD', quote: 'JPY', pipSize: 0.01 },
      { symbol: 'USDCHF', name: 'US Dollar / Swiss Franc', base: 'USD', quote: 'CHF', pipSize: 0.0001 },
      { symbol: 'AUDUSD', name: 'Australian Dollar / US Dollar', base: 'AUD', quote: 'USD', pipSize: 0.0001 },
      { symbol: 'USDCAD', name: 'US Dollar / Canadian Dollar', base: 'USD', quote: 'CAD', pipSize: 0.0001 },
      { symbol: 'NZDUSD', name: 'New Zealand Dollar / US Dollar', base: 'NZD', quote: 'USD', pipSize: 0.0001 },
      { symbol: 'EURGBP', name: 'Euro / British Pound', base: 'EUR', quote: 'GBP', pipSize: 0.0001 },
      { symbol: 'EURJPY', name: 'Euro / Japanese Yen', base: 'EUR', quote: 'JPY', pipSize: 0.01 },
      { symbol: 'GBPJPY', name: 'British Pound / Japanese Yen', base: 'GBP', quote: 'JPY', pipSize: 0.01 },
      { symbol: 'CHFJPY', name: 'Swiss Franc / Japanese Yen', base: 'CHF', quote: 'JPY', pipSize: 0.01 },
      { symbol: 'EURCHF', name: 'Euro / Swiss Franc', base: 'EUR', quote: 'CHF', pipSize: 0.0001 },
      { symbol: 'AUDCAD', name: 'Australian Dollar / Canadian Dollar', base: 'AUD', quote: 'CAD', pipSize: 0.0001 },
      { symbol: 'GBPCHF', name: 'British Pound / Swiss Franc', base: 'GBP', quote: 'CHF', pipSize: 0.0001 }
    ];

    for (const instrument of instruments) {
      await query(`
        INSERT INTO instrument_config (symbol, name, base_currency, quote_currency, pip_size)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (symbol) DO NOTHING;
      `, [instrument.symbol, instrument.name, instrument.base, instrument.quote, instrument.pipSize]);
    }

    console.log('Default instruments inserted successfully');
  } catch (error) {
    console.error('Error inserting default instruments:', error);
    throw error;
  }
};

/**
 * Test database connection
 */
const testConnection = async () => {
  try {
    const result = await query('SELECT NOW() as current_time');
    console.log('Database connection successful:', result.rows[0].current_time);
    return true;
  } catch (error) {
    console.error('Database connection failed:', error);
    return false;
  }
};

/**
 * Graceful shutdown
 */
const closePool = async () => {
  try {
    await pool.end();
    console.log('Database pool has ended');
  } catch (error) {
    console.error('Error closing database pool:', error);
  }
};

// Handle process exit
process.on('SIGINT', closePool);
process.on('SIGTERM', closePool);
process.on('exit', closePool);

module.exports = {
  query,
  getClient,
  pool,
  initializeTables,
  testConnection,
  closePool
};
