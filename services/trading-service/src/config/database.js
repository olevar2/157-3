const { Pool } = require('pg');

// Database configuration
const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'platform3',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password',
  max: 20, // Maximum number of clients in the pool
  idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
  connectionTimeoutMillis: 2000, // Return an error after 2 seconds if connection could not be established
};

// Create connection pool
const pool = new Pool(dbConfig);

// Test database connection
const testConnection = async () => {
  try {
    const client = await pool.connect();
    console.log('✅ Trading Service: Database connected successfully');
    
    // Test query
    const result = await client.query('SELECT NOW()');
    console.log('📊 Trading Service: Database query test passed:', result.rows[0].now);
    
    client.release();
    return true;
  } catch (error) {
    console.error('❌ Trading Service: Database connection failed:', error.message);
    return false;
  }
};

// Graceful shutdown
const closePool = async () => {
  try {
    await pool.end();
    console.log('🔌 Trading Service: Database pool closed');
  } catch (error) {
    console.error('❌ Trading Service: Error closing database pool:', error.message);
  }
};

// Handle connection errors
pool.on('error', (err) => {
  console.error('❌ Trading Service: Unexpected database error:', err.message);
});

pool.on('connect', () => {
  console.log('🔗 Trading Service: New database connection established');
});

pool.on('remove', () => {
  console.log('🔌 Trading Service: Database connection removed from pool');
});

module.exports = {
  pool,
  testConnection,
  closePool
};
