const Redis = require('ioredis');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

class EventBus {
  constructor() {
    this.redis = null;
    this.subscriber = null;
    this.publisher = null;
    this.listeners = new Map();
  }

  async initialize() {
    try {
      const redisConfig = {
        host: process.env.REDIS_HOST || 'localhost',
        port: process.env.REDIS_PORT || 6379,
        password: process.env.REDIS_PASSWORD,
        db: parseInt(process.env.REDIS_DB) || 0,
        retryDelayOnFailover: 100,
        enableReadyCheck: false,
        maxRetriesPerRequest: null,
        lazyConnect: true
      };

      // Create Redis connections for pub/sub
      this.publisher = new Redis(redisConfig);
      this.subscriber = new Redis(redisConfig);
      this.redis = new Redis(redisConfig);

      // Connect to Redis
      await this.publisher.connect();
      await this.subscriber.connect();
      await this.redis.connect();

      // Set up event handlers
      this.subscriber.on('message', this.handleMessage.bind(this));
      this.subscriber.on('error', (error) => {
        logger.error(`Redis subscriber error: ${error.message}`);
      });

      logger.info('Event Bus initialized successfully');
    } catch (error) {
      logger.error(`Failed to initialize Event Bus: ${error.message}`);
      throw error;
    }
  }

  async publish(topic, event, metadata = {}) {
    try {
      const eventId = uuidv4();
      const eventData = {
        id: eventId,
        topic,
        event,
        metadata: {
          ...metadata,
          timestamp: new Date().toISOString(),
          source: 'event-system'
        }
      };

      // Publish to Redis pub/sub
      await this.publisher.publish(`event:${topic}`, JSON.stringify(eventData));

      // Also store in a sorted set for replay capability
      await this.redis.zadd(
        `events:${topic}`, 
        Date.now(), 
        JSON.stringify(eventData)
      );

      // Expire old events after 7 days
      await this.redis.expire(`events:${topic}`, 7 * 24 * 60 * 60);

      logger.debug(`Event published to topic: ${topic}`, { eventId, event });
      return eventId;
    } catch (error) {
      logger.error(`Failed to publish event: ${error.message}`);
      throw error;
    }
  }

  async subscribe(topic, callback) {
    try {
      const channel = `event:${topic}`;
      
      if (!this.listeners.has(channel)) {
        this.listeners.set(channel, new Set());
        await this.subscriber.subscribe(channel);
      }
      
      this.listeners.get(channel).add(callback);
      logger.info(`Subscribed to topic: ${topic}`);
    } catch (error) {
      logger.error(`Failed to subscribe to topic ${topic}: ${error.message}`);
      throw error;
    }
  }

  async unsubscribe(topic, callback) {
    try {
      const channel = `event:${topic}`;
      
      if (this.listeners.has(channel)) {
        this.listeners.get(channel).delete(callback);
        
        if (this.listeners.get(channel).size === 0) {
          this.listeners.delete(channel);
          await this.subscriber.unsubscribe(channel);
        }
      }
      
      logger.info(`Unsubscribed from topic: ${topic}`);
    } catch (error) {
      logger.error(`Failed to unsubscribe from topic ${topic}: ${error.message}`);
      throw error;
    }
  }

  async handleMessage(channel, message) {
    try {
      const eventData = JSON.parse(message);
      const callbacks = this.listeners.get(channel);
      
      if (callbacks) {
        for (const callback of callbacks) {
          try {
            await callback(eventData);
          } catch (error) {
            logger.error(`Error in event callback: ${error.message}`, { channel, eventData });
          }
        }
      }
    } catch (error) {
      logger.error(`Error handling message: ${error.message}`, { channel, message });
    }
  }

  async getRecentEvents(topic, limit = 100) {
    try {
      const events = await this.redis.zrevrange(
        `events:${topic}`, 
        0, 
        limit - 1, 
        'WITHSCORES'
      );

      const result = [];
      for (let i = 0; i < events.length; i += 2) {
        const eventData = JSON.parse(events[i]);
        const timestamp = events[i + 1];
        result.push({ ...eventData, score: timestamp });
      }

      return result;
    } catch (error) {
      logger.error(`Failed to get recent events: ${error.message}`);
      throw error;
    }
  }

  async isHealthy() {
    try {
      await this.redis.ping();
      return true;
    } catch (error) {
      return false;
    }
  }

  isConnected() {
    return this.redis && this.redis.status === 'ready';
  }

  async disconnect() {
    try {
      if (this.subscriber) await this.subscriber.disconnect();
      if (this.publisher) await this.publisher.disconnect();
      if (this.redis) await this.redis.disconnect();
      logger.info('Event Bus disconnected');
    } catch (error) {
      logger.error(`Error disconnecting Event Bus: ${error.message}`);
    }
  }
}

module.exports = EventBus;
