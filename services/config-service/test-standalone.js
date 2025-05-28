#!/usr/bin/env node

/**
 * Standalone Configuration Service Test
 * Tests the configuration service without external dependencies
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

// Mock Configuration Manager
class MockConfigurationManager {
  constructor(config) {
    this.config = config;
    this.mockData = {
      'test-service:development': {
        database: {
          host: 'localhost',
          port: 5432,
          name: 'test_db'
        },
        jwt: {
          secret: 'test-jwt-secret',
          expiresIn: '24h'
        },
        redis: {
          host: 'localhost',
          port: 6379
        },
        'test-key': 'test-value-' + Date.now()
      }
    };
    this.history = [];
    console.log('✅ Mock Configuration Manager initialized');
  }

  async getConfiguration(request) {
    const key = `${request.service}:${request.environment}`;
    const config = this.mockData[key] || {};
    
    // Filter by requested keys if specified
    if (request.keys && request.keys.length > 0) {
      const filteredConfig = {};
      for (const k of request.keys) {
        if (config[k] !== undefined) {
          filteredConfig[k] = config[k];
        }
      }
      return {
        service: request.service,
        environment: request.environment,
        configuration: filteredConfig,
        version: 1,
        timestamp: new Date()
      };
    }

    return {
      service: request.service,
      environment: request.environment,
      configuration: config,
      version: 1,
      timestamp: new Date()
    };
  }

  async updateConfiguration(service, environment, key, value, userId) {
    const configKey = `${service}:${environment}`;
    if (!this.mockData[configKey]) {
      this.mockData[configKey] = {};
    }
    
    const oldValue = this.mockData[configKey][key];
    this.mockData[configKey][key] = value;
    
    // Record history
    this.history.push({
      id: `${Date.now()}-${Math.random()}`,
      key,
      oldValue,
      newValue: value,
      environment,
      service,
      changedBy: userId || 'system',
      changedAt: new Date()
    });
    
    console.log(`✅ Configuration updated: ${service}:${environment}:${key} = ${value}`);
  }

  async getFeatureFlag(name, service, environment) {
    // Mock feature flags
    const flags = {
      'test-feature': {
        name: 'test-feature',
        enabled: true,
        environment: environment || 'development',
        service: service,
        rolloutPercentage: 100,
        createdAt: new Date(),
        updatedAt: new Date()
      },
      'advanced-analytics': {
        name: 'advanced-analytics',
        enabled: true,
        environment: environment || 'development',
        rolloutPercentage: 100,
        createdAt: new Date(),
        updatedAt: new Date()
      }
    };
    
    return flags[name] || null;
  }

  async registerService(registration) {
    console.log(`✅ Service registered: ${registration.serviceName}:${registration.environment}`);
  }

  getConfigurationHistory(service, environment, key) {
    let history = this.history;
    
    if (service) {
      history = history.filter(h => h.service === service);
    }
    if (environment) {
      history = history.filter(h => h.environment === environment);
    }
    if (key) {
      history = history.filter(h => h.key === key);
    }
    
    return history.sort((a, b) => b.changedAt.getTime() - a.changedAt.getTime());
  }

  async healthCheck() {
    return {
      vault: true,  // Mock as healthy
      redis: true,  // Mock as healthy
      cache: true
    };
  }
}

// Create Express app
const app = express();
const PORT = process.env.PORT || 3007;

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize Mock Configuration Manager
const configManager = new MockConfigurationManager({
  endpoint: 'http://mock-vault:8200',
  token: 'mock-token'
});

// API Key authentication middleware
const authenticateApiKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  const validApiKeys = ['config-service-key-1', 'config-service-key-2', 'test-api-key'];

  if (!apiKey || !validApiKeys.includes(apiKey)) {
    return res.status(401).json({ error: 'Invalid or missing API key' });
  }

  next();
};

// Routes
app.get('/health', async (req, res) => {
  try {
    const health = await configManager.healthCheck();
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'config-service',
      version: '1.0.0',
      checks: health
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'config-service',
      version: '1.0.0',
      error: 'Health check failed'
    });
  }
});

app.post('/api/v1/config', authenticateApiKey, async (req, res) => {
  try {
    const request = req.body;
    
    if (!request.service || !request.environment) {
      return res.status(400).json({
        error: 'Missing required fields: service and environment'
      });
    }

    const configuration = await configManager.getConfiguration(request);
    
    res.json({
      success: true,
      data: configuration
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to retrieve configuration',
      message: error.message
    });
  }
});

app.put('/api/v1/config/:service/:environment/:key', authenticateApiKey, async (req, res) => {
  try {
    const { service, environment, key } = req.params;
    const { value } = req.body;
    const userId = req.headers['x-user-id'];

    if (value === undefined) {
      return res.status(400).json({
        error: 'Missing required field: value'
      });
    }

    await configManager.updateConfiguration(service, environment, key, value, userId);
    
    res.json({
      success: true,
      message: 'Configuration updated successfully'
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to update configuration',
      message: error.message
    });
  }
});

app.get('/api/v1/feature-flags/:name', authenticateApiKey, async (req, res) => {
  try {
    const { name } = req.params;
    const { service, environment } = req.query;

    const featureFlag = await configManager.getFeatureFlag(name, service, environment);

    if (!featureFlag) {
      return res.status(404).json({
        error: 'Feature flag not found'
      });
    }

    res.json({
      success: true,
      data: featureFlag
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to retrieve feature flag',
      message: error.message
    });
  }
});

app.post('/api/v1/register', authenticateApiKey, async (req, res) => {
  try {
    const registration = req.body;
    
    if (!registration.serviceName || !registration.environment || !registration.configKeys) {
      return res.status(400).json({
        error: 'Missing required fields: serviceName, environment, configKeys'
      });
    }

    registration.lastHeartbeat = new Date();
    await configManager.registerService(registration);
    
    res.json({
      success: true,
      message: 'Service registered successfully'
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to register service',
      message: error.message
    });
  }
});

app.get('/api/v1/history', authenticateApiKey, async (req, res) => {
  try {
    const { service, environment, key } = req.query;
    
    const history = configManager.getConfigurationHistory(service, environment, key);
    
    res.json({
      success: true,
      data: history
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to retrieve configuration history',
      message: error.message
    });
  }
});

app.get('/metrics', (req, res) => {
  res.set('Content-Type', 'text/plain');
  res.send(`
# HELP config_service_requests_total Total number of requests
# TYPE config_service_requests_total counter
config_service_requests_total 100

# HELP config_service_vault_connection Vault connection status
# TYPE config_service_vault_connection gauge
config_service_vault_connection 1

# HELP config_service_redis_connection Redis connection status
# TYPE config_service_redis_connection gauge
config_service_redis_connection 1
  `);
});

// Error handling
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error.message);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.originalUrl} not found`
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Mock Configuration Service started successfully on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🔧 API endpoint: http://localhost:${PORT}/api/v1/config`);
  console.log(`🎯 Ready for testing!`);
});

module.exports = app;
