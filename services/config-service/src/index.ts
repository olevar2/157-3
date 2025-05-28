import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import dotenv from 'dotenv';
import { ConfigurationManager } from './ConfigurationManager';
import logger from './utils/logger';
import { VaultConfig, ConfigurationRequest, FeatureFlag, ServiceRegistration } from './types';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3007;

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Request logging middleware
app.use((req, res, next) => {
  logger.info('HTTP Request', {
    method: req.method,
    url: req.url,
    userAgent: req.get('User-Agent'),
    ip: req.ip
  });
  next();
});

// Initialize Configuration Manager
const vaultConfig: VaultConfig = {
  endpoint: process.env.VAULT_URL || 'http://vault:8200',
  token: process.env.VAULT_TOKEN || '',
  namespace: process.env.VAULT_NAMESPACE,
  timeout: parseInt(process.env.VAULT_TIMEOUT || '5000'),
  retries: parseInt(process.env.VAULT_RETRIES || '3')
};

const configManager = new ConfigurationManager(vaultConfig);

// API Key authentication middleware
const authenticateApiKey = (req: express.Request, res: express.Response, next: express.NextFunction) => {
  const apiKey = req.headers['x-api-key'] as string;
  const validApiKeys = process.env.VALID_API_KEYS?.split(',') || [];

  if (!apiKey || !validApiKeys.includes(apiKey)) {
    return res.status(401).json({ error: 'Invalid or missing API key' });
  }

  next();
};

// Routes

/**
 * Health check endpoint
 */
app.get('/health', async (req, res) => {
  try {
    const health = await configManager.healthCheck();
    const status = health.vault && health.cache ? 200 : 503;
    
    res.status(status).json({
      status: status === 200 ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'config-service',
      version: '1.0.0',
      checks: health
    });
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'config-service',
      version: '1.0.0',
      error: 'Health check failed'
    });
  }
});

/**
 * Get configuration for a service
 */
app.post('/api/v1/config', authenticateApiKey, async (req, res) => {
  try {
    const request: ConfigurationRequest = req.body;
    
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
    logger.error('Failed to get configuration', { error, body: req.body });
    res.status(500).json({
      error: 'Failed to retrieve configuration',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Update configuration value
 */
app.put('/api/v1/config/:service/:environment/:key', authenticateApiKey, async (req, res) => {
  try {
    const { service, environment, key } = req.params;
    const { value } = req.body;
    const userId = req.headers['x-user-id'] as string;

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
    logger.error('Failed to update configuration', { error, params: req.params, body: req.body });
    res.status(500).json({
      error: 'Failed to update configuration',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Get feature flag
 */
app.get('/api/v1/feature-flags/:name', authenticateApiKey, async (req, res) => {
  try {
    const { name } = req.params;
    const { service, environment } = req.query;

    const featureFlag = await configManager.getFeatureFlag(
      name,
      service as string,
      environment as string
    );

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
    logger.error('Failed to get feature flag', { error, params: req.params, query: req.query });
    res.status(500).json({
      error: 'Failed to retrieve feature flag',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Register service for configuration updates
 */
app.post('/api/v1/register', authenticateApiKey, async (req, res) => {
  try {
    const registration: ServiceRegistration = req.body;
    
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
    logger.error('Failed to register service', { error, body: req.body });
    res.status(500).json({
      error: 'Failed to register service',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Get configuration history
 */
app.get('/api/v1/history', authenticateApiKey, async (req, res) => {
  try {
    const { service, environment, key } = req.query;
    
    const history = configManager.getConfigurationHistory(
      service as string,
      environment as string,
      key as string
    );
    
    res.json({
      success: true,
      data: history
    });
  } catch (error) {
    logger.error('Failed to get configuration history', { error, query: req.query });
    res.status(500).json({
      error: 'Failed to retrieve configuration history',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Get service metrics
 */
app.get('/metrics', (req, res) => {
  // Basic metrics endpoint for monitoring
  res.set('Content-Type', 'text/plain');
  res.send(`
# HELP config_service_requests_total Total number of requests
# TYPE config_service_requests_total counter
config_service_requests_total 0

# HELP config_service_vault_connection Vault connection status
# TYPE config_service_vault_connection gauge
config_service_vault_connection 1

# HELP config_service_redis_connection Redis connection status
# TYPE config_service_redis_connection gauge
config_service_redis_connection 1
  `);
});

// Error handling middleware
app.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error', { error, url: req.url, method: req.method });
  
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.originalUrl} not found`
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`🚀 Configuration Service started successfully`, {
    port: PORT,
    environment: process.env.NODE_ENV || 'development',
    vaultEndpoint: vaultConfig.endpoint
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

export default app;
