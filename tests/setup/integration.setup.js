/**
 * Integration Test Setup for Platform3 Phase 2
 * Configures environment for testing service interactions and database operations
 */

const axios = require('axios');
const { spawn } = require('child_process');

// Wait for service to be ready
const waitForService = async (url, timeout = 30000) => {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    try {
      await axios.get(url, { timeout: 5000 });
      return true;
    } catch (error) {
      await global.testUtils.sleep(1000);
    }
  }
  
  throw new Error(`Service at ${url} not ready after ${timeout}ms`);
};

// Setup test database
const setupTestDatabase = async () => {
  const { database } = global.testConfig;
  
  try {
    // Create test database if it doesn't exist
    const setupCommands = [
      `createdb -h ${database.host} -p ${database.port} -U ${database.user} ${database.database}`,
      `psql -h ${database.host} -p ${database.port} -U ${database.user} -d ${database.database} -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"`
    ];
    
    for (const command of setupCommands) {
      try {
        await execCommand(command);
      } catch (error) {
        // Database might already exist, continue
        console.log(`Database setup command failed (might be expected): ${error.message}`);
      }
    }
    
    console.log('✅ Test database setup completed');
  } catch (error) {
    console.error('❌ Test database setup failed:', error.message);
    throw error;
  }
};

// Execute shell command
const execCommand = (command) => {
  return new Promise((resolve, reject) => {
    const child = spawn('sh', ['-c', command], { stdio: 'pipe' });
    
    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    child.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`Command failed with code ${code}: ${stderr}`));
      }
    });
  });
};

// Setup Redis for testing
const setupTestRedis = async () => {
  const { redis } = global.testConfig;
  
  try {
    const Redis = require('redis');
    const client = Redis.createClient({
      host: redis.host,
      port: redis.port,
      db: redis.db
    });
    
    await client.connect();
    await client.flushDb(); // Clear test database
    await client.disconnect();
    
    console.log('✅ Test Redis setup completed');
  } catch (error) {
    console.error('❌ Test Redis setup failed:', error.message);
    // Don't throw - Redis might not be required for all tests
  }
};

// Global setup for integration tests
beforeAll(async () => {
  console.log('🔧 Setting up integration test environment...');
  
  try {
    // Setup test database
    await setupTestDatabase();
    
    // Setup test Redis
    await setupTestRedis();
    
    // Wait for core services to be ready
    const coreServices = [
      global.testConfig.services.configService + '/health',
      global.testConfig.services.authService + '/health'
    ];
    
    for (const serviceUrl of coreServices) {
      try {
        await waitForService(serviceUrl, 10000);
        console.log(`✅ Service ready: ${serviceUrl}`);
      } catch (error) {
        console.warn(`⚠️  Service not ready: ${serviceUrl} - ${error.message}`);
      }
    }
    
    console.log('🚀 Integration test environment ready');
    
  } catch (error) {
    console.error('❌ Integration test setup failed:', error.message);
    throw error;
  }
}, 60000); // 60 second timeout for setup

// Cleanup after all integration tests
afterAll(async () => {
  console.log('🧹 Cleaning up integration test environment...');
  
  try {
    // Clear test data
    await global.testUtils.database.clearTestData();
    
    console.log('✅ Integration test cleanup completed');
  } catch (error) {
    console.error('❌ Integration test cleanup failed:', error.message);
  }
});

// Enhanced utilities for integration tests
global.integrationUtils = {
  // Service health checks
  checkServiceHealth: async (serviceName) => {
    const serviceUrl = global.testConfig.services[serviceName];
    if (!serviceUrl) {
      throw new Error(`Unknown service: ${serviceName}`);
    }
    
    try {
      const response = await axios.get(`${serviceUrl}/health`, { timeout: 5000 });
      return {
        healthy: response.status === 200,
        response: response.data,
        responseTime: response.headers['x-response-time'] || 'unknown'
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message,
        responseTime: 'timeout'
      };
    }
  },
  
  // Database utilities
  database: {
    query: async (sql, params = []) => {
      const { Client } = require('pg');
      const client = new Client(global.testConfig.database);
      
      try {
        await client.connect();
        const result = await client.query(sql, params);
        await client.end();
        return result;
      } catch (error) {
        await client.end();
        throw error;
      }
    },
    
    seedData: async (tableName, data) => {
      const columns = Object.keys(data);
      const values = Object.values(data);
      const placeholders = values.map((_, i) => `$${i + 1}`).join(', ');
      
      const sql = `INSERT INTO ${tableName} (${columns.join(', ')}) VALUES (${placeholders}) RETURNING *`;
      
      const result = await global.integrationUtils.database.query(sql, values);
      return result.rows[0];
    },
    
    clearTable: async (tableName) => {
      await global.integrationUtils.database.query(`TRUNCATE TABLE ${tableName} CASCADE`);
    }
  },
  
  // API utilities
  api: {
    makeRequest: async (method, url, data = null, headers = {}) => {
      const config = {
        method,
        url,
        headers: {
          'Content-Type': 'application/json',
          'x-test-mode': 'true',
          ...headers
        },
        timeout: 10000
      };
      
      if (data) {
        config.data = data;
      }
      
      try {
        const response = await axios(config);
        return {
          success: true,
          status: response.status,
          data: response.data,
          headers: response.headers
        };
      } catch (error) {
        return {
          success: false,
          status: error.response?.status || 0,
          error: error.message,
          data: error.response?.data || null
        };
      }
    },
    
    authenticateUser: async (credentials = { username: 'testuser', password: 'testpass' }) => {
      const response = await global.integrationUtils.api.makeRequest(
        'POST',
        global.testConfig.services.authService + '/auth/login',
        credentials
      );
      
      if (response.success) {
        return response.data.token;
      }
      
      throw new Error(`Authentication failed: ${response.error}`);
    }
  }
};

console.log('🔗 Integration test utilities loaded');
console.log('📊 Available services:', Object.keys(global.testConfig.services));
console.log('🗄️  Database:', global.testConfig.database.database);
console.log('🔴 Redis DB:', global.testConfig.redis.db);
