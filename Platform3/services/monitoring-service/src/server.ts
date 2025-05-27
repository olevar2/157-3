/**
 * Monitoring Service Server
 * Enterprise performance monitoring for Platform3
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { createLogger, format, transports } from 'winston';
import { createClient } from 'redis';
import { PerformanceMonitor } from './PerformanceMonitor';
import { register } from 'prom-client';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3012;

// Logger setup
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.errors({ stack: true }),
    format.json()
  ),
  defaultMeta: { service: 'monitoring-service' },
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

// Performance Monitor
let performanceMonitor: PerformanceMonitor;

// Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'monitoring-service',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

app.get('/metrics', async (req, res) => {
  try {
    // Return Prometheus metrics
    const metrics = await performanceMonitor?.getPrometheusMetrics() || await register.metrics();
    res.set('Content-Type', register.contentType);
    res.end(metrics);
  } catch (error) {
    logger.error('Error getting Prometheus metrics:', error);
    res.status(500).json({ error: 'Failed to get metrics' });
  }
});

app.get('/dashboard', (req, res) => {
  try {
    const dashboard = performanceMonitor?.getPerformanceDashboard() || {};
    res.json(dashboard);
  } catch (error) {
    logger.error('Error getting performance dashboard:', error);
    res.status(500).json({ error: 'Failed to get dashboard' });
  }
});

app.get('/alerts', (req, res) => {
  try {
    // This would return active alerts
    res.json({
      alerts: [],
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error getting alerts:', error);
    res.status(500).json({ error: 'Failed to get alerts' });
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
  
  if (performanceMonitor) {
    await performanceMonitor.stop();
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

    // Initialize Performance Monitor
    const config = {
      enabled: process.env.MONITORING_ENABLED !== 'false',
      metricsInterval: parseInt(process.env.METRICS_INTERVAL || '30000'),
      alertThresholds: {
        latencyP99: parseInt(process.env.LATENCY_P99_THRESHOLD || '100'),
        errorRate: parseFloat(process.env.ERROR_RATE_THRESHOLD || '0.01'),
        throughput: parseInt(process.env.THROUGHPUT_THRESHOLD || '100'),
        accuracy: parseFloat(process.env.ACCURACY_THRESHOLD || '0.75'),
        availability: parseFloat(process.env.AVAILABILITY_THRESHOLD || '0.999')
      },
      services: [
        'analytics-service',
        'trading-service',
        'user-service',
        'api-gateway',
        'ml-service'
      ],
      businessMetrics: process.env.BUSINESS_METRICS_ENABLED === 'true'
    };

    performanceMonitor = new PerformanceMonitor(config, logger, redis);

    // Start the server
    app.listen(PORT, () => {
      logger.info(`Monitoring Service running on port ${PORT}`);
    });

    // Start performance monitoring
    await performanceMonitor.start();
    logger.info('Performance Monitor started');

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();
