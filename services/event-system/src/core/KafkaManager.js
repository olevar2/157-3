const { Kafka } = require('kafkajs');
const logger = require('../utils/logger');

class KafkaManager {
    constructor(config = {}) {
        this.config = {
            clientId: config.clientId || process.env.KAFKA_CLIENT_ID || 'platform3-event-system',
            brokers: config.brokers || (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
            groupId: config.groupId || process.env.KAFKA_GROUP_ID || 'platform3-events',
            ...config
        };

        this.kafka = Kafka({
            clientId: this.config.clientId,
            brokers: this.config.brokers,
            retry: {
                initialRetryTime: 100,
                retries: 8
            }
        });

        this.producer = null;
        this.consumer = null;
        this.admin = null;
        this.isConnected = false;
        this.topics = new Set();
        this.messageHandlers = new Map();
    }

    /**
     * Initialize Kafka connection
     */
    async connect() {
        try {
            this.producer = this.kafka.producer({
                maxInFlightRequests: 1,
                idempotent: true,
                transactionTimeout: 30000
            });

            this.consumer = this.kafka.consumer({
                groupId: this.config.groupId,
                sessionTimeout: 30000,
                rebalanceTimeout: 60000,
                heartbeatInterval: 3000
            });

            this.admin = this.kafka.admin();

            await this.producer.connect();
            await this.consumer.connect();
            await this.admin.connect();

            this.isConnected = true;
            logger.info('KafkaManager connected successfully');

            // Setup error handlers
            this.setupErrorHandlers();

        } catch (error) {
            logger.error('Error connecting to Kafka:', error);
            throw error;
        }
    }

    /**
     * Setup error handlers for Kafka components
     */
    setupErrorHandlers() {
        this.producer.on('producer.disconnect', () => {
            logger.warn('Kafka producer disconnected');
            this.isConnected = false;
        });

        this.consumer.on('consumer.disconnect', () => {
            logger.warn('Kafka consumer disconnected');
            this.isConnected = false;
        });

        this.producer.on('producer.network.request_timeout', (payload) => {
            logger.error('Kafka producer request timeout:', payload);
        });

        this.consumer.on('consumer.network.request_timeout', (payload) => {
            logger.error('Kafka consumer request timeout:', payload);
        });
    }

    /**
     * Create topics if they don't exist
     * @param {Array} topicConfigs - Array of topic configurations
     */
    async createTopics(topicConfigs) {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            const existingTopics = await this.admin.listTopics();
            const newTopics = topicConfigs.filter(config => !existingTopics.includes(config.topic));

            if (newTopics.length > 0) {
                await this.admin.createTopics({
                    topics: newTopics.map(config => ({
                        topic: config.topic,
                        numPartitions: config.numPartitions || 3,
                        replicationFactor: config.replicationFactor || 1,
                        configEntries: config.configEntries || []
                    }))
                });

                logger.info(`Created topics: ${newTopics.map(t => t.topic).join(', ')}`);
            }

            // Track topics
            topicConfigs.forEach(config => this.topics.add(config.topic));

        } catch (error) {
            logger.error('Error creating topics:', error);
            throw error;
        }
    }

    /**
     * Publish message to Kafka topic
     * @param {string} topic - Topic name
     * @param {Object} message - Message data
     * @param {Object} options - Publishing options
     */
    async publishMessage(topic, message, options = {}) {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            const kafkaMessage = {
                key: options.key || message.id || Date.now().toString(),
                value: JSON.stringify(message),
                partition: options.partition,
                timestamp: options.timestamp || Date.now().toString(),
                headers: {
                    source: 'platform3-event-system',
                    contentType: 'application/json',
                    ...options.headers
                }
            };

            const result = await this.producer.send({
                topic,
                messages: [kafkaMessage]
            });

            logger.info(`Message published to topic ${topic}:`, {
                partition: result[0].partition,
                offset: result[0].offset
            });

            return result[0];

        } catch (error) {
            logger.error(`Error publishing message to topic ${topic}:`, error);
            throw error;
        }
    }

    /**
     * Publish batch of messages
     * @param {string} topic - Topic name
     * @param {Array} messages - Array of messages
     * @param {Object} options - Publishing options
     */
    async publishBatch(topic, messages, options = {}) {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            const kafkaMessages = messages.map((message, index) => ({
                key: options.keyGenerator ? options.keyGenerator(message, index) : 
                     message.id || `${Date.now()}_${index}`,
                value: JSON.stringify(message),
                partition: options.partition,
                timestamp: Date.now().toString(),
                headers: {
                    source: 'platform3-event-system',
                    contentType: 'application/json',
                    batchIndex: index.toString(),
                    ...options.headers
                }
            }));

            const result = await this.producer.send({
                topic,
                messages: kafkaMessages
            });

            logger.info(`Batch of ${messages.length} messages published to topic ${topic}`);
            return result;

        } catch (error) {
            logger.error(`Error publishing batch to topic ${topic}:`, error);
            throw error;
        }
    }

    /**
     * Subscribe to topic and register message handler
     * @param {string} topic - Topic name
     * @param {Function} handler - Message handler function
     * @param {Object} options - Subscription options
     */
    async subscribe(topic, handler, options = {}) {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            // Register handler
            this.messageHandlers.set(topic, {
                handler,
                options
            });

            // Subscribe to topic
            await this.consumer.subscribe({
                topic,
                fromBeginning: options.fromBeginning || false
            });

            logger.info(`Subscribed to topic: ${topic}`);

        } catch (error) {
            logger.error(`Error subscribing to topic ${topic}:`, error);
            throw error;
        }
    }

    /**
     * Start consuming messages
     */
    async startConsuming() {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            await this.consumer.run({
                eachMessage: async ({ topic, partition, message }) => {
                    try {
                        const handler = this.messageHandlers.get(topic);
                        if (!handler) {
                            logger.warn(`No handler registered for topic: ${topic}`);
                            return;
                        }

                        // Parse message
                        const parsedMessage = {
                            topic,
                            partition,
                            offset: message.offset,
                            key: message.key ? message.key.toString() : null,
                            value: JSON.parse(message.value.toString()),
                            timestamp: message.timestamp,
                            headers: message.headers || {}
                        };

                        // Call handler
                        await handler.handler(parsedMessage);

                        logger.debug(`Message processed from topic ${topic}:`, {
                            partition,
                            offset: message.offset
                        });

                    } catch (error) {
                        logger.error(`Error processing message from topic ${topic}:`, error);
                        
                        // Handle DLQ or retry logic here if needed
                        if (error.retry !== false) {
                            throw error; // Let Kafka handle retry
                        }
                    }
                }
            });

            logger.info('Kafka consumer started');

        } catch (error) {
            logger.error('Error starting Kafka consumer:', error);
            throw error;
        }
    }

    /**
     * Get topic metadata
     * @param {Array} topics - Array of topic names
     */
    async getTopicMetadata(topics = []) {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            const metadata = await this.admin.fetchTopicMetadata({
                topics: topics.length > 0 ? topics : Array.from(this.topics)
            });

            return metadata;

        } catch (error) {
            logger.error('Error fetching topic metadata:', error);
            throw error;
        }
    }

    /**
     * Get consumer group information
     */
    async getConsumerGroupInfo() {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            const groups = await this.admin.listGroups();
            const groupInfo = await this.admin.describeGroups([this.config.groupId]);

            return {
                groups,
                currentGroup: groupInfo.groups[0]
            };

        } catch (error) {
            logger.error('Error getting consumer group info:', error);
            throw error;
        }
    }

    /**
     * Reset consumer group offset
     * @param {string} topic - Topic name
     * @param {number} partition - Partition number
     * @param {string} offset - Offset ('earliest', 'latest', or specific offset)
     */
    async resetOffset(topic, partition, offset = 'earliest') {
        try {
            if (!this.isConnected) {
                throw new Error('KafkaManager not connected');
            }

            await this.consumer.stop();

            await this.admin.resetOffsets({
                groupId: this.config.groupId,
                topic,
                earliest: offset === 'earliest',
                latest: offset === 'latest',
                ...(typeof offset === 'number' && { partitions: [{ partition, offset }] })
            });

            logger.info(`Reset offset for topic ${topic}, partition ${partition} to ${offset}`);

        } catch (error) {
            logger.error('Error resetting offset:', error);
            throw error;
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const metadata = await this.admin.fetchTopicMetadata({ topics: [] });
            return {
                status: 'healthy',
                connected: this.isConnected,
                topics: Array.from(this.topics),
                brokers: metadata.brokers.length,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            return {
                status: 'unhealthy',
                connected: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Disconnect from Kafka
     */
    async disconnect() {
        try {
            if (this.producer) {
                await this.producer.disconnect();
            }
            if (this.consumer) {
                await this.consumer.disconnect();
            }
            if (this.admin) {
                await this.admin.disconnect();
            }

            this.isConnected = false;
            logger.info('KafkaManager disconnected');

        } catch (error) {
            logger.error('Error disconnecting from Kafka:', error);
            throw error;
        }
    }
}

module.exports = KafkaManager;
