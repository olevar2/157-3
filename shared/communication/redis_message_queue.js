
const redis = require('redis');
const logger = require('../../../shared/logging/platform3_logger');

class Platform3MessageQueue {
    constructor(redisConfig = {}) {
        this.client = redis.createClient({
            host: redisConfig.host || process.env.REDIS_HOST || 'localhost',
            port: redisConfig.port || process.env.REDIS_PORT || 6379,
            password: redisConfig.password || process.env.REDIS_PASSWORD,
            retry_strategy: (options) => {
                if (options.error && options.error.code === 'ECONNREFUSED') {
                    logger.error('Redis server connection refused');
                    return new Error('Redis server connection refused');
                }
                if (options.total_retry_time > 1000 * 60 * 60) {
                    logger.error('Redis retry time exhausted');
                    return new Error('Retry time exhausted');
                }
                if (options.attempt > 10) {
                    return undefined;
                }
                return Math.min(options.attempt * 100, 3000);
            }
        });

        this.subscriber = redis.createClient({
            host: redisConfig.host || process.env.REDIS_HOST || 'localhost',
            port: redisConfig.port || process.env.REDIS_PORT || 6379,
            password: redisConfig.password || process.env.REDIS_PASSWORD
        });

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.client.on('connect', () => {
            logger.info('Redis client connected');
        });

        this.client.on('error', (err) => {
            logger.error('Redis client error', { error: err.message });
        });

        this.subscriber.on('error', (err) => {
            logger.error('Redis subscriber error', { error: err.message });
        });
    }

    async publishMessage(channel, message, correlationId = null) {
        try {
            const messageData = {
                payload: message,
                timestamp: new Date().toISOString(),
                correlationId: correlationId || this.generateMessageId(),
                service: process.env.SERVICE_NAME || 'unknown'
            };

            const result = await this.client.publish(channel, JSON.stringify(messageData));
            logger.info('Message published to Redis', {
                channel,
                correlationId: messageData.correlationId,
                subscribers: result
            });

            return messageData.correlationId;
        } catch (error) {
            logger.error('Failed to publish message to Redis', {
                channel,
                error: error.message
            });
            throw error;
        }
    }

    async subscribeToChannel(channel, callback) {
        try {
            await this.subscriber.subscribe(channel);
            this.subscriber.on('message', (receivedChannel, message) => {
                if (receivedChannel === channel) {
                    try {
                        const messageData = JSON.parse(message);
                        logger.info('Message received from Redis', {
                            channel: receivedChannel,
                            correlationId: messageData.correlationId
                        });
                        callback(messageData);
                    } catch (parseError) {
                        logger.error('Failed to parse Redis message', {
                            channel: receivedChannel,
                            error: parseError.message
                        });
                    }
                }
            });

            logger.info('Subscribed to Redis channel', { channel });
        } catch (error) {
            logger.error('Failed to subscribe to Redis channel', {
                channel,
                error: error.message
            });
            throw error;
        }
    }

    generateMessageId() {
        return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    async disconnect() {
        try {
            await this.client.quit();
            await this.subscriber.quit();
            logger.info('Redis connections closed');
        } catch (error) {
            logger.error('Error closing Redis connections', { error: error.message });
        }
    }
}

module.exports = Platform3MessageQueue;
