require('dotenv').config();

const ServiceDiscoveryMiddleware = require('../../../shared/communication/service_discovery_middleware');
const Platform3MessageQueue = require('../../../shared/communication/redis_message_queue');
const HealthCheckEndpoint = require('../../../shared/communication/health_check_endpoint');
const logger = require('../../../shared/logging/platform3_logger');

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');


// Platform3 Service Mesh Integration
const correlationMiddleware = require('../../shared/middleware/correlation_middleware');
const { circuitBreakerMiddleware } = require('../../shared/middleware/circuit_breaker_middleware');
// Import event system components
const EventBus = require('./core/EventBus');
const MessageQueue = require('./core/MessageQueue');
const KafkaManager = require('./core/KafkaManager');
const EventStore = require('./core/EventStore');
const logger = require('./utils/logger');

const app = express();
// Apply service mesh middleware
app.use(correlationMiddleware);
app.use(circuitBreakerMiddleware('event-system'));


// Platform3 Microservices Integration
const serviceDiscovery = new ServiceDiscoveryMiddleware('services', PORT || 3000);
const messageQueue = new Platform3MessageQueue();
const healthCheck = new HealthCheckEndpoint('services', [
    {
        name: 'redis',
        check: async () => {
            return { healthy: true, responseTime: 0 };
        }
    }
]);

// Apply service discovery middleware
app.use(serviceDiscovery.middleware());

// Add health check endpoints
app.use('/api', healthCheck.getRouter());

// Register service with Consul on startup
serviceDiscovery.registerService().catch(err => {
    logger.error('Failed to register service', { error: err.message });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
});

process.on('SIGINT', async () => {
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
});

const PORT = process.env.PORT || 3005;

// Security and middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));
app.use(compression());
app.use(morgan('combined', { stream: { write: (msg) => logger.info(msg.trim()) } }));
app.use(express.json({ limit: '10mb' }));

// Initialize event system components
let eventBus, messageQueue, kafkaManager, eventStore;

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const redisStatus = await eventBus.isHealthy();
    const queueStatus = await messageQueue.isHealthy();
    const kafkaStatus = kafkaManager ? await kafkaManager.isHealthy() : false;
    const storeStatus = await eventStore.isHealthy();

    const overallStatus = redisStatus && queueStatus && storeStatus ? 'healthy' : 'degraded';

    res.status(overallStatus === 'healthy' ? 200 : 503).json({
      status: overallStatus,
      service: 'event-system',
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      components: {
        eventBus: redisStatus ? 'healthy' : 'unhealthy',
        messageQueue: queueStatus ? 'healthy' : 'unhealthy',
        kafka: kafkaStatus ? 'healthy' : 'disabled',
        eventStore: storeStatus ? 'healthy' : 'unhealthy'
      }
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      service: 'event-system',
      error: error.message
    });
  }
});

// Service info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    service: 'event-system',
    version: '1.0.0',
    description: 'Message Queue & Event System for Forex Trading Platform',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    features: [
      'Real-time event streaming',
      'Message queuing with Redis/Bull',
      'Kafka integration (optional)',
      'Event sourcing and storage',
      'Dead letter queues',
      'Event replay capabilities',
      'Pub/Sub messaging',
      'Event monitoring'
    ],
    topics: [
      'market-data-feed',
      'trade-executions', 
      'user-activities',
      'risk-alerts',
      'compliance-events'
    ],
    queues: [
      'email-notifications',
      'sms-alerts',
      'document-processing',
      'payment-processing'
    ]
  });
});

// Event publishing endpoint
app.post('/api/events/publish', async (req, res) => {
  try {
    const { topic, event, metadata } = req.body;
    
    if (!topic || !event) {
      return res.status(400).json({
        error: 'Topic and event are required'
      });
    }

    // Publish to event bus
    const eventId = await eventBus.publish(topic, event, metadata);
    
    // Store event for sourcing
    await eventStore.store(eventId, topic, event, metadata);

    logger.info(`Event published: ${topic}`, { eventId, event });

    res.json({
      success: true,
      eventId,
      topic,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error publishing event: ${error.message}`);
    res.status(500).json({
      error: 'Failed to publish event',
      details: error.message
    });
  }
});

// Queue job endpoint
app.post('/api/queue/add', async (req, res) => {
  try {
    const { queue, job, data, options } = req.body;
    
    if (!queue || !job) {
      return res.status(400).json({
        error: 'Queue and job are required'
      });
    }

    const jobId = await messageQueue.addJob(queue, job, data, options);

    logger.info(`Job queued: ${queue}/${job}`, { jobId, data });

    res.json({
      success: true,
      jobId,
      queue,
      job,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error queuing job: ${error.message}`);
    res.status(500).json({
      error: 'Failed to queue job',
      details: error.message
    });
  }
});

// Get event history
app.get('/api/events/history', async (req, res) => {
  try {
    const { topic, limit = 100, offset = 0 } = req.query;
    
    const events = await eventStore.getEvents(topic, { limit, offset });

    res.json({
      success: true,
      events,
      total: events.length,
      topic: topic || 'all'
    });
  } catch (error) {
    logger.error(`Error fetching event history: ${error.message}`);
    res.status(500).json({
      error: 'Failed to fetch event history',
      details: error.message
    });
  }
});

// Get queue status
app.get('/api/queue/status', async (req, res) => {
  try {
    const status = await messageQueue.getStatus();

    res.json({
      success: true,
      queues: status,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error fetching queue status: ${error.message}`);
    res.status(500).json({
      error: 'Failed to fetch queue status',
      details: error.message
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`, { 
    stack: err.stack,
    url: req.url,
    method: req.method
  });
  
  res.status(err.status || 500).json({
    error: 'Internal server error',
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    timestamp: new Date().toISOString()
  });
});

// Initialize and start server
const startServer = async () => {
  try {
    // Initialize event system components
    eventBus = new EventBus();
    messageQueue = new MessageQueue();
    eventStore = new EventStore();
    
    // Initialize Kafka if enabled
    if (process.env.ENABLE_KAFKA === 'true') {
      kafkaManager = new KafkaManager();
      await kafkaManager.initialize();
    }

    // Initialize all components
    await eventBus.initialize();
    await messageQueue.initialize();
    await eventStore.initialize();

    // Start HTTP server
    const server = app.listen(PORT, () => {
      logger.info(`🚀 Event System running on port ${PORT}`);
      logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`🔄 Event Bus: ${eventBus.isConnected() ? 'Connected' : 'Disconnected'}`);
      logger.info(`📬 Message Queue: ${messageQueue.isConnected() ? 'Connected' : 'Disconnected'}`);
      logger.info(`📚 Event Store: ${eventStore.isConnected() ? 'Connected' : 'Disconnected'}`);
      if (kafkaManager) {
        logger.info(`🎯 Kafka: ${kafkaManager.isConnected() ? 'Connected' : 'Disconnected'}`);
      }
    });

    // Graceful shutdown
    const gracefulShutdown = async (signal) => {
      logger.info(`Received ${signal}, shutting down gracefully...`);
      
      // Close all components
      if (kafkaManager) await kafkaManager.disconnect();
      await messageQueue.close();
      await eventBus.disconnect();
      await eventStore.close();
      
      server.close(() => {
        logger.info('Event System HTTP server closed');
        process.exit(0);
      });
    };

    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));

    return server;
  } catch (error) {
    logger.error(`Failed to start Event System: ${error.message}`);
    process.exit(1);
  }
};

// Start the server
startServer();

module.exports = app;
