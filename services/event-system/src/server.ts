import dotenv from 'dotenv';
dotenv.config();

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import { EventBus } from './core/EventBus';
import { MessageQueue } from './core/MessageQueue';
import { KafkaManager } from './core/KafkaManager';
import { EventStore } from './core/EventStore';
import { Logger } from './utils/Logger';
import { 
  PublishEventRequest, 
  SubscribeRequest, 
  QueueJobRequest,
  EventData,
  HealthStatus 
} from './types';

const app = express();
const PORT = process.env.PORT || 3005;
const logger = new Logger('EventSystemServer');

// Security and middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));
app.use(compression());
app.use(morgan('combined', { stream: { write: (msg: string) => logger.info(msg.trim()) } }));
app.use(express.json({ limit: '10mb' }));

// Initialize event system components
let eventBus: EventBus;
let messageQueue: MessageQueue;
let kafkaManager: KafkaManager | null = null;
let eventStore: EventStore;

// Error handling middleware
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// Health check endpoint
app.get('/health', async (req: Request, res: Response) => {
  try {
    let redisStatus;
    try {
      redisStatus = eventBus ? await eventBus.getHealth() : { 
        service: 'EventBus', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: 'Not initialized' } 
      };
    } catch (e: any) {
      redisStatus = { 
        service: 'EventBus', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: e.message || 'Connection failed' } 
      };
    }
    
    let queueStatus;
    try {
      queueStatus = messageQueue ? await messageQueue.getHealth() : { 
        service: 'MessageQueue', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: 'Not initialized' } 
      };
    } catch (e: any) {
      queueStatus = { 
        service: 'MessageQueue', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: e.message || 'Connection failed' } 
      };
    }
    
    let kafkaStatus;
    try {
      kafkaStatus = kafkaManager ? await kafkaManager.getHealth() : { 
        service: 'KafkaManager', 
        status: 'healthy' as const, 
        timestamp: new Date(), 
        details: { message: 'Kafka disabled' } 
      };
    } catch (e: any) {
      kafkaStatus = { 
        service: 'KafkaManager', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: e.message || 'Connection failed' } 
      };
    }
    
    let storeStatus;
    try {
      storeStatus = eventStore ? await eventStore.getHealth() : { 
        service: 'EventStore', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: 'Not initialized' } 
      };
    } catch (e: any) {
      storeStatus = { 
        service: 'EventStore', 
        status: 'unhealthy' as const, 
        timestamp: new Date(), 
        details: { error: e.message || 'Connection failed' } 
      };
    }

    const healthyComponents = [redisStatus, queueStatus, storeStatus].filter(s => s.status === 'healthy').length;
    const totalComponents = 3; // EventBus, MessageQueue, EventStore (Kafka is optional)
    const overallHealthy = healthyComponents === totalComponents;
    const degraded = healthyComponents > 0 && healthyComponents < totalComponents;

    const response = {
      status: overallHealthy ? 'healthy' : (degraded ? 'degraded' : 'unhealthy'),
      service: 'event-system',
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      uptime: Math.floor(process.uptime()),
      components: {
        eventBus: redisStatus.status === 'healthy' ? 'healthy' : 'unhealthy',
        messageQueue: queueStatus.status === 'healthy' ? 'healthy' : 'unhealthy',
        kafka: kafkaStatus.status === 'healthy' ? 'healthy' : 'disabled',
        eventStore: storeStatus.status === 'healthy' ? 'healthy' : 'unhealthy'
      },
      details: {
        eventBus: redisStatus.details,
        messageQueue: queueStatus.details,
        kafka: kafkaStatus.details,
        eventStore: storeStatus.details,
        redis: {
          url: process.env.REDIS_URL || 'redis://localhost:6379',
          status: healthyComponents > 0 ? 'connected' : 'disconnected'
        }
      }
    };

    res.status(overallHealthy ? 200 : (degraded ? 503 : 503)).json(response);
  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(503).json({
      status: 'unhealthy',
      service: 'event-system',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Service info endpoint
app.get('/api/info', (req: Request, res: Response) => {
  res.json({
    service: 'event-system',
    version: '1.0.0',
    description: 'Message Queue & Event System for Forex Trading Platform',
    uptime: Math.floor(process.uptime()),
    timestamp: new Date().toISOString(),
    features: [
      'Real-time event streaming with Redis Pub/Sub',
      'Message queuing with Bull/BullMQ',
      'Apache Kafka integration (optional)',
      'Event sourcing and storage',
      'Dead letter queues for failed messages',
      'Event replay capabilities',
      'WebSocket real-time notifications',
      'Event monitoring and metrics'
    ],
    eventTypes: [
      'market-data-update',
      'trade-execution',
      'order-placed',
      'order-cancelled',
      'user-login',
      'user-logout',
      'risk-alert',
      'compliance-violation',
      'payment-processed',
      'kyc-verification-complete'
    ],
    queues: [
      'email-notifications',
      'sms-alerts',
      'push-notifications',
      'document-processing',
      'payment-processing',
      'risk-calculations',
      'compliance-checks'
    ]
  });
});

// Event publishing endpoint
app.post('/api/events/publish', async (req: Request, res: Response): Promise<void> => {
  try {
    const eventRequest: PublishEventRequest = req.body;
    
    // Validate required fields
    if (!eventRequest.type || !eventRequest.source || eventRequest.data === undefined) {
      res.status(400).json({
        error: 'Missing required fields',
        required: ['type', 'source', 'data']
      });
      return;
    }

    const eventData: EventData = {
      eventId: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      eventType: eventRequest.type,
      timestamp: Date.now(),
      data: eventRequest.data,
      metadata: eventRequest.metadata,
      correlationId: eventRequest.correlationId,
      source: eventRequest.source
    };

    // Publish to event bus
    await eventBus.publishEvent(eventData.eventType, eventData);
    
    // Store in event store (using aggregateId from source or generate one)
    const aggregateId = eventRequest.metadata?.aggregateId || `${eventRequest.source}_${Date.now()}`;
    await eventStore.storeEvent(aggregateId, eventData);

    // Optional: Publish to Kafka if enabled
    if (kafkaManager) {
      await kafkaManager.publishEvent(eventData.eventType, eventData);
    }

    logger.info(`Event published: ${eventData.eventType}`, { eventId: eventData.eventId });

    res.status(201).json({
      success: true,
      eventId: eventData.eventId,
      timestamp: eventData.timestamp
    });
  } catch (error) {
    logger.error('Failed to publish event:', error);
    res.status(500).json({
      error: 'Failed to publish event',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Event subscription endpoint
app.post('/api/events/subscribe', async (req: Request, res: Response): Promise<void> => {
  try {
    const subscribeRequest: SubscribeRequest = req.body;
    
    if (!subscribeRequest.eventType) {
      res.status(400).json({
        error: 'Missing required field: eventType'
      });
      return;
    }

    const subscriptionId = await eventBus.subscribe(
      subscribeRequest.eventType,
      async (event: EventData) => {
        // Handle webhook notification if provided
        if (subscribeRequest.webhookUrl) {
          // This would typically be handled by a separate notification service
          logger.info(`Webhook notification for ${event.eventType}`, { 
            webhookUrl: subscribeRequest.webhookUrl,
            eventId: event.eventId 
          });
        }
      }
    );

    res.status(201).json({
      success: true,
      subscriptionId,
      eventType: subscribeRequest.eventType
    });
  } catch (error) {
    logger.error('Failed to create subscription:', error);
    res.status(500).json({
      error: 'Failed to create subscription',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Queue job endpoint
app.post('/api/queue/job', async (req: Request, res: Response): Promise<void> => {
  try {
    const jobRequest: QueueJobRequest = req.body;
    
    if (!jobRequest.type || jobRequest.data === undefined) {
      res.status(400).json({
        error: 'Missing required fields',
        required: ['type', 'data']
      });
      return;
    }    const jobOptions: any = {
      priority: jobRequest.priority,
      delay: jobRequest.delay
    };

    // Handle backoff properly - convert string to object if needed
    if (jobRequest.options?.backoff) {
      if (typeof jobRequest.options.backoff === 'string') {
        jobOptions.backoff = { type: jobRequest.options.backoff };
      } else {
        jobOptions.backoff = jobRequest.options.backoff;
      }
    }

    // Add other options
    if (jobRequest.options?.attempts) jobOptions.attempts = jobRequest.options.attempts;
    if (jobRequest.options?.removeOnComplete !== undefined) jobOptions.removeOnComplete = jobRequest.options.removeOnComplete;
    if (jobRequest.options?.removeOnFail !== undefined) jobOptions.removeOnFail = jobRequest.options.removeOnFail;

    const job = await messageQueue.addJob(jobRequest.type, jobRequest.data, jobOptions);

    logger.info(`Job queued: ${jobRequest.type}`, { jobId: job.id });

    res.status(201).json({
      success: true,
      jobId: job.id,
      type: jobRequest.type,      priority: jobRequest.priority || 0
    });
  } catch (error) {
    logger.error('Failed to queue job:', error);
    res.status(500).json({
      error: 'Failed to queue job',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Event replay endpoint
app.post('/api/events/replay', async (req: Request, res: Response): Promise<void> => {
  try {
    const { eventType, fromTimestamp, toTimestamp, aggregateId, limit = 100 } = req.body;
    
    if (!aggregateId) {
      res.status(400).json({
        error: 'Missing required field: aggregateId'
      });
      return;
    }

    const eventStream = await eventStore.getEventStream(aggregateId, fromTimestamp, toTimestamp);
    
    // Re-publish events
    for (const event of eventStream.events) {
      await eventBus.publishEvent(event.eventType, event);
    }

    logger.info(`Event replay completed: ${eventStream.events.length} events`, { eventType, aggregateId });

    res.json({
      success: true,
      eventsReplayed: eventStream.events.length,
      eventType,
      aggregateId
    });
  } catch (error) {
    logger.error('Failed to replay events:', error);
    res.status(500).json({
      error: 'Failed to replay events',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Metrics endpoint
app.get('/api/metrics', async (req: Request, res: Response): Promise<void> => {
  try {
    const metrics = {
      eventBus: eventBus.getHealth(),
      messageQueue: messageQueue.getHealth(),
      eventStore: eventStore.getHealth(),
      kafkaManager: kafkaManager?.getHealth() || { service: 'KafkaManager', status: 'disabled' },
      uptime: Math.floor(process.uptime()),
      memoryUsage: process.memoryUsage(),
      timestamp: new Date().toISOString()
    };

    res.json(metrics);
  } catch (error) {
    logger.error('Failed to get metrics:', error);
    res.status(500).json({
      error: 'Failed to get metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Initialize services
async function initializeServices(): Promise<void> {
  logger.info('Initializing Event System services...');

  // Initialize Redis-based components with correct config structure
  const redisUrl = `redis://${process.env.REDIS_PASSWORD ? `:${process.env.REDIS_PASSWORD}@` : ''}${process.env.REDIS_HOST || 'localhost'}:${process.env.REDIS_PORT || '6379'}/${process.env.REDIS_DB || '0'}`;
  
  logger.info(`Connecting to Redis at: ${redisUrl.replace(/:.*@/, ':***@')}`);
  
  const eventBusConfig = {
    redisUrl,
    maxRetries: 3,
    retryDelayMs: 1000
  };

  const messageQueueConfig = {
    redisUrl,
    concurrency: parseInt(process.env.QUEUE_CONCURRENCY || '5')
  };

  const eventStoreConfig = {
    redisUrl,
    database: parseInt(process.env.REDIS_DB || '0'),
    snapshotTtl: 86400 // 24 hours
  };

  // Initialize components
  eventBus = new EventBus(eventBusConfig);
  messageQueue = new MessageQueue(messageQueueConfig);
  eventStore = new EventStore(eventStoreConfig);

  // Try to connect to each service with individual error handling
  let redisAvailable = false;
  
  try {
    logger.info('Connecting to EventBus...');
    await eventBus.connect();
    logger.info('EventBus connected successfully');
    redisAvailable = true;
  } catch (error) {
    logger.error('Failed to connect EventBus to Redis:', error);
    logger.warn('EventBus will operate in degraded mode');
  }

  try {
    logger.info('Connecting to MessageQueue...');
    await messageQueue.connect();
    logger.info('MessageQueue connected successfully');
  } catch (error) {
    logger.error('Failed to connect MessageQueue to Redis:', error);
    logger.warn('MessageQueue will operate in degraded mode');
  }
  try {
    logger.info('Connecting to EventStore...');
    await Promise.race([
      eventStore.connect(),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('EventStore connection timeout')), 10000)
      )
    ]);
    logger.info('EventStore connected successfully');
  } catch (error) {
    logger.error('Failed to connect EventStore to Redis:', error);
    logger.warn('EventStore will operate in degraded mode');
  }

  // Initialize Kafka if configured
  if (process.env.KAFKA_ENABLED === 'true' && process.env.KAFKA_BROKERS) {
    try {
      const kafkaConfig = {
        brokers: process.env.KAFKA_BROKERS.split(','),
        clientId: process.env.KAFKA_CLIENT_ID || 'event-system'
      };
      
      kafkaManager = new KafkaManager(kafkaConfig);
      await kafkaManager.connect();
      logger.info('Kafka integration enabled and connected');
    } catch (error) {
      logger.error('Failed to connect to Kafka:', error);
      logger.warn('Kafka integration disabled');
      kafkaManager = null;
    }
  } else {
    logger.info('Kafka integration disabled (KAFKA_ENABLED=false or no brokers configured)');
  }

  if (redisAvailable) {
    logger.info('Event System services initialized successfully with Redis');
  } else {
    logger.warn('Event System services initialized in DEGRADED MODE - Redis unavailable');
    logger.info('To start Redis using Docker: docker run -d --name redis-event-system -p 6379:6379 redis:7-alpine');
  }
}

// Graceful shutdown
async function gracefulShutdown(): Promise<void> {
  logger.info('Shutting down Event System...');
  
  try {
    if (eventBus) {
      await eventBus.disconnect().catch((e: any) => logger.warn('EventBus disconnect error:', e.message));
    }
    if (messageQueue) {
      await messageQueue.disconnect().catch((e: any) => logger.warn('MessageQueue disconnect error:', e.message));
    }
    if (eventStore) {
      await eventStore.disconnect().catch((e: any) => logger.warn('EventStore disconnect error:', e.message));
    }
    if (kafkaManager) {
      await kafkaManager.disconnect().catch((e: any) => logger.warn('KafkaManager disconnect error:', e.message));
    }
    
    logger.info('Event System shutdown complete');
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown:', error);
    process.exit(1);
  }
}

// Handle shutdown signals
process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);

// Start server
async function startServer(): Promise<void> {
  await initializeServices();
  
  app.listen(PORT, () => {
    logger.info(`Event System running on port ${PORT}`);
    logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
    logger.info(`Health check: http://localhost:${PORT}/health`);
    logger.info(`API info: http://localhost:${PORT}/api/info`);
  });
}

startServer().catch((error) => {
  logger.error('Failed to start server:', error);
  process.exit(1);
});
