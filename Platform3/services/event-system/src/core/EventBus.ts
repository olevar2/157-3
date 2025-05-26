import Redis from 'ioredis';
import { EventEmitter } from 'events';
import { Logger } from '../utils/Logger';
import { EventData, EventBusConfig, HealthStatus } from '../types';

export class EventBus extends EventEmitter {
  private redis: Redis;
  private subscriber: Redis;
  private logger: Logger;
  private config: EventBusConfig;
  private isConnected: boolean = false;
  private subscriptions: Map<string, Set<(data: EventData) => Promise<void>>> = new Map();

  constructor(config: EventBusConfig) {    super();
    this.config = config;
    this.logger = new Logger('EventBus');
    
    this.redis = new Redis(config.redisUrl, {
      enableReadyCheck: true,
      maxRetriesPerRequest: config.maxRetries || 3,
      lazyConnect: true
    });

    this.subscriber = new Redis(config.redisUrl, {
      enableReadyCheck: true,
      maxRetriesPerRequest: config.maxRetries || 3,
      lazyConnect: true
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redis.on('connect', () => {
      this.logger.info('EventBus Redis publisher connected');
    });

    this.redis.on('ready', () => {
      this.isConnected = true;
      this.logger.info('EventBus Redis publisher ready');
    });

    this.redis.on('error', (error) => {
      this.logger.error('EventBus Redis publisher error:', error);
      this.isConnected = false;
    });

    this.subscriber.on('message', (channel: string, message: string) => {
      this.handleMessage(channel, message);
    });

    this.subscriber.on('error', (error) => {
      this.logger.error('EventBus Redis subscriber error:', error);
    });
  }

  async connect(): Promise<void> {
    try {
      await this.redis.connect();
      await this.subscriber.connect();
      this.logger.info('EventBus connected to Redis successfully');
    } catch (error) {
      this.logger.error('Failed to connect EventBus to Redis:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      await this.redis.quit();
      await this.subscriber.quit();
      this.isConnected = false;
      this.logger.info('EventBus disconnected from Redis');
    } catch (error) {
      this.logger.error('Error disconnecting EventBus from Redis:', error);
      throw error;
    }
  }

  async publishEvent(eventType: string, eventData: EventData): Promise<void> {
    try {
      const channel = `events:${eventType}`;
      const message = JSON.stringify(eventData);
      
      await this.redis.publish(channel, message);
      
      this.logger.info(`Event published to channel ${channel}`, {
        eventId: eventData.eventId,
        eventType: eventData.eventType
      });
    } catch (error) {
      this.logger.error(`Failed to publish event ${eventType}:`, error);
      throw error;
    }
  }

  async subscribe(eventType: string, handler: (event: EventData) => Promise<void>): Promise<string> {
    try {
      const channel = `events:${eventType}`;
      
      // Add handler to subscriptions
      if (!this.subscriptions.has(channel)) {
        this.subscriptions.set(channel, new Set());
        await this.subscriber.subscribe(channel);
      }
      
      this.subscriptions.get(channel)!.add(handler);
      
      const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      this.logger.info(`Subscribed to channel ${channel}`, { subscriptionId });
      
      return subscriptionId;
    } catch (error) {
      this.logger.error(`Failed to subscribe to ${eventType}:`, error);
      throw error;
    }
  }

  async unsubscribe(eventType: string, handler: (event: EventData) => Promise<void>): Promise<void> {
    try {
      const channel = `events:${eventType}`;
      const handlers = this.subscriptions.get(channel);
      
      if (handlers) {
        handlers.delete(handler);
        
        if (handlers.size === 0) {
          this.subscriptions.delete(channel);
          await this.subscriber.unsubscribe(channel);
        }
      }
      
      this.logger.info(`Unsubscribed from channel ${channel}`);
    } catch (error) {
      this.logger.error(`Failed to unsubscribe from ${eventType}:`, error);
      throw error;
    }
  }

  private async handleMessage(channel: string, message: string): Promise<void> {
    try {
      const eventData: EventData = JSON.parse(message);
      const handlers = this.subscriptions.get(channel);
      
      if (!handlers || handlers.size === 0) {
        return;
      }

      // Execute all handlers for this channel
      const promises = Array.from(handlers).map(async (handler) => {
        try {
          await handler(eventData);
        } catch (error) {
          this.logger.error(`Error in event handler for channel ${channel}:`, error);
          
          // Send to dead letter queue if configured
          await this.sendToDeadLetterQueue(channel, eventData, error as Error);
        }
      });

      await Promise.allSettled(promises);
      
      this.logger.debug(`Processed event on channel ${channel}`, {
        eventId: eventData.eventId,
        handlersCount: handlers.size
      });
      
    } catch (error) {
      this.logger.error(`Failed to handle message on channel ${channel}:`, error);
    }
  }

  private async sendToDeadLetterQueue(channel: string, eventData: EventData, error: Error): Promise<void> {
    try {
      const dlqChannel = `${channel}.dlq`;
      const dlqEvent: EventData = {
        eventId: `dlq_${Date.now()}`,
        eventType: 'dead_letter_event',
        timestamp: Date.now(),
        data: {
          originalChannel: channel,
          originalEvent: eventData,
          error: error.message,
          timestamp: Date.now()
        }
      };

      await this.redis.publish(dlqChannel, JSON.stringify(dlqEvent));
      this.logger.warn(`Event sent to dead letter queue: ${dlqChannel}`);
      
    } catch (dlqError) {
      this.logger.error('Failed to send event to dead letter queue:', dlqError);
    }
  }

  getHealth(): HealthStatus {
    return {
      service: 'EventBus',
      status: this.isConnected ? 'healthy' : 'unhealthy',
      timestamp: new Date(),
      details: {
        connected: this.isConnected,
        subscriptions: this.subscriptions.size,
        redisUrl: this.config.redisUrl.replace(/\/\/[^@]*@/, '//***@') // Hide credentials
      }
    };
  }
}
