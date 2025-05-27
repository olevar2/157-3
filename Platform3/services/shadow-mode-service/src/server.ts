/**
 * Shadow Mode Service Server
 * Enterprise deployment shadow mode orchestration for Platform3
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { createLogger, format, transports } from 'winston';
import { createClient } from 'redis';
import { Kafka } from 'kafkajs';
import { ShadowModeOrchestrator } from './ShadowModeOrchestrator';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3010;

// Logger setup
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.errors({ stack: true }),
    format.json()
  ),
  defaultMeta: { service: 'shadow-mode-service' },
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

// Kafka setup
const kafka = new Kafka({
  clientId: 'shadow-mode-service',
  brokers: [process.env.KAFKA_BROKER || 'localhost:9092']
});

const kafkaConsumer = kafka.consumer({ groupId: 'shadow-mode-group' });
const kafkaProducer = kafka.producer();

// Shadow Mode Orchestrator
let shadowOrchestrator: ShadowModeOrchestrator;

// Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'shadow-mode-service',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

app.get('/metrics', async (req, res) => {
  try {
    const stats = shadowOrchestrator?.getStatistics() || {};
    res.json({
      shadowMode: stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error getting metrics:', error);
    res.status(500).json({ error: 'Failed to get metrics' });
  }
});

app.post('/shadow-mode/start', async (req, res) => {
  try {
    if (shadowOrchestrator) {
      await shadowOrchestrator.start();
      res.json({ message: 'Shadow mode started successfully' });
    } else {
      res.status(500).json({ error: 'Shadow orchestrator not initialized' });
    }
  } catch (error) {
    logger.error('Error starting shadow mode:', error);
    res.status(500).json({ error: 'Failed to start shadow mode' });
  }
});

app.post('/shadow-mode/stop', async (req, res) => {
  try {
    if (shadowOrchestrator) {
      await shadowOrchestrator.stop();
      res.json({ message: 'Shadow mode stopped successfully' });
    } else {
      res.status(500).json({ error: 'Shadow orchestrator not initialized' });
    }
  } catch (error) {
    logger.error('Error stopping shadow mode:', error);
    res.status(500).json({ error: 'Failed to stop shadow mode' });
  }
});

app.get('/shadow-mode/status', (req, res) => {
  try {
    const stats = shadowOrchestrator?.getStatistics() || {};
    res.json({
      enabled: true,
      statistics: stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error getting shadow mode status:', error);
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
  
  if (shadowOrchestrator) {
    await shadowOrchestrator.stop();
  }
  
  await redis.quit();
  await kafkaConsumer.disconnect();
  await kafkaProducer.disconnect();
  
  process.exit(0);
});

// Initialize and start server
async function startServer() {
  try {
    // Connect to Redis
    await redis.connect();
    logger.info('Connected to Redis');

    // Connect to Kafka
    await kafkaConsumer.connect();
    await kafkaProducer.connect();
    logger.info('Connected to Kafka');

    // Initialize Shadow Mode Orchestrator
    const config = {
      enabled: process.env.SHADOW_MODE_ENABLED === 'true',
      trafficMirrorPercentage: parseInt(process.env.TRAFFIC_MIRROR_PERCENTAGE || '10'),
      comparisonThreshold: parseFloat(process.env.COMPARISON_THRESHOLD || '0.95'),
      maxLatencyMs: parseInt(process.env.MAX_LATENCY_MS || '100'),
      enabledServices: (process.env.ENABLED_SERVICES || 'analytics-service,trading-service,ml-service').split(',')
    };

    shadowOrchestrator = new ShadowModeOrchestrator(
      config,
      logger,
      redis,
      kafkaConsumer,
      kafkaProducer
    );

    // Start the server
    app.listen(PORT, () => {
      logger.info(`Shadow Mode Service running on port ${PORT}`);
    });

    // Start shadow mode if enabled
    if (config.enabled) {
      await shadowOrchestrator.start();
      logger.info('Shadow Mode Orchestrator started');
    }

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();
