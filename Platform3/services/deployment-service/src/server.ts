/**
 * Deployment Service Server
 * Enterprise deployment and rollback management for Platform3
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { createLogger, format, transports } from 'winston';
import { createClient } from 'redis';
import { RollbackManager } from './RollbackManager';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3011;

// Logger setup
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.errors({ stack: true }),
    format.json()
  ),
  defaultMeta: { service: 'deployment-service' },
  transports: [
    new transports.File({ filename: 'logs/error.log', level: 'error' }),
    new transports.File({ filename: 'logs/combined.log' }),
    new transports.Console({
      format: format.combine(
        format.colorize(),
        format.simple()
      )
    })
  ]
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Redis client
const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Rollback Manager
let rollbackManager: RollbackManager;

// Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'deployment-service',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

app.get('/rollback/statistics', (req, res) => {
  try {
    const stats = rollbackManager?.getRollbackStatistics() || {};
    res.json({
      rollbackStats: stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error getting rollback statistics:', error);
    res.status(500).json({ error: 'Failed to get statistics' });
  }
});

app.post('/rollback/trigger', async (req, res) => {
  try {
    const { serviceId, reason } = req.body;
    
    if (!serviceId) {
      return res.status(400).json({ error: 'serviceId is required' });
    }

    await rollbackManager.manualRollback(serviceId, reason || 'manual-trigger');
    
    res.json({ 
      message: `Rollback triggered for ${serviceId}`,
      serviceId,
      reason: reason || 'manual-trigger',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error triggering rollback:', error);
    res.status(500).json({ error: 'Failed to trigger rollback' });
  }
});

app.get('/deployment/status', (req, res) => {
  try {
    const stats = rollbackManager?.getRollbackStatistics() || {};
    res.json({
      deploymentStatus: 'active',
      rollbackCapability: 'enabled',
      statistics: stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error getting deployment status:', error);
    res.status(500).json({ error: 'Failed to get status' });
  }
});

// Error handling middleware
app.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  
  if (rollbackManager) {
    await rollbackManager.stop();
  }
  
  await redis.quit();
  process.exit(0);
});

// Initialize and start server
async function startServer() {
  try {
    // Connect to Redis
    await redis.connect();
    logger.info('Connected to Redis');

    // Initialize Rollback Manager
    const config = {
      enabled: process.env.ROLLBACK_ENABLED !== 'false',
      healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL || '30000'),
      rollbackThresholds: {
        latencyMs: parseInt(process.env.LATENCY_THRESHOLD || '100'),
        errorRate: parseFloat(process.env.ERROR_RATE_THRESHOLD || '0.01'),
        accuracyThreshold: parseFloat(process.env.ACCURACY_THRESHOLD || '0.75'),
        resourceUsage: parseFloat(process.env.RESOURCE_USAGE_THRESHOLD || '0.9')
      },
      services: [
        {
          name: 'analytics-service',
          version: '1.0.0',
          previousVersion: '0.9.0',
          healthEndpoint: 'http://analytics-service:3001/health',
          rollbackCommand: 'kubectl rollout undo deployment/analytics-service',
          dependencies: []
        },
        {
          name: 'trading-service',
          version: '1.0.0',
          previousVersion: '0.9.0',
          healthEndpoint: 'http://trading-service:3002/health',
          rollbackCommand: 'kubectl rollout undo deployment/trading-service',
          dependencies: ['analytics-service']
        },
        {
          name: 'user-service',
          version: '1.0.0',
          previousVersion: '0.9.0',
          healthEndpoint: 'http://user-service:3003/health',
          rollbackCommand: 'kubectl rollout undo deployment/user-service',
          dependencies: []
        },
        {
          name: 'api-gateway',
          version: '1.0.0',
          previousVersion: '0.9.0',
          healthEndpoint: 'http://api-gateway:3000/health',
          rollbackCommand: 'kubectl rollout undo deployment/api-gateway',
          dependencies: ['analytics-service', 'trading-service', 'user-service']
        }
      ]
    };

    rollbackManager = new RollbackManager(config, logger, redis);

    // Start the server
    app.listen(PORT, () => {
      logger.info(`Deployment Service running on port ${PORT}`);
    });

    // Start rollback monitoring
    await rollbackManager.start();
    logger.info('Rollback Manager started');

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();
