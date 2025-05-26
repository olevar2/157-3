import { Kafka, Producer, Consumer, EachMessagePayload, LogEntry } from 'kafkajs';
import { EventEmitter } from 'events';
import { EventData, KafkaConfig, HealthStatus } from '../types';
import { Logger } from '../utils/Logger';

export class KafkaManager extends EventEmitter {
  private kafka: Kafka;
  private producer: Producer | null = null;
  private consumers: Map<string, Consumer> = new Map();
  private isConnected: boolean = false;
  private logger: Logger;
  private config: KafkaConfig;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectInterval: number = 5000;

  constructor(config: KafkaConfig) {
    super();
    this.config = config;
    this.logger = new Logger('KafkaManager');
    
    this.kafka = new Kafka({
      clientId: config.clientId,
      brokers: config.brokers,
      connectionTimeout: config.connectionTimeout || 3000,
      requestTimeout: config.requestTimeout || 30000,
      retry: {
        initialRetryTime: 100,
        retries: 8
      },
      logLevel: 2, // WARN level
      logCreator: () => (entry: LogEntry) => {
        const { level, log } = entry;
        this.logger.info(`Kafka ${level}: ${log.message}`, { 
          ...log, 
          timestamp: log.timestamp 
        });
      }
    });
  }

  async connect(): Promise<void> {
    try {
      this.logger.info('Connecting to Kafka cluster...');
      
      // Initialize producer
      this.producer = this.kafka.producer({
        maxInFlightRequests: 1,
        idempotent: true,
        transactionTimeout: 30000,
        retry: {
          initialRetryTime: 100,
          retries: 8
        }
      });

      await this.producer.connect();
      this.isConnected = true;
      this.reconnectAttempts = 0;
      
      this.logger.info('Successfully connected to Kafka cluster');
      this.emit('connected');
      
    } catch (error) {
      this.logger.error('Failed to connect to Kafka:', error);
      await this.handleReconnection();
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      this.logger.info('Disconnecting from Kafka cluster...');
      
      // Disconnect all consumers
      for (const [topic, consumer] of this.consumers.entries()) {
        await consumer.disconnect();
        this.logger.info(`Consumer for topic ${topic} disconnected`);
      }
      this.consumers.clear();

      // Disconnect producer
      if (this.producer) {
        await this.producer.disconnect();
        this.producer = null;
      }

      this.isConnected = false;
      this.logger.info('Successfully disconnected from Kafka cluster');
      this.emit('disconnected');
      
    } catch (error) {
      this.logger.error('Error disconnecting from Kafka:', error);
      throw error;
    }
  }

  async publishEvent(topic: string, eventData: EventData): Promise<void> {
    if (!this.producer || !this.isConnected) {
      throw new Error('Kafka producer not connected');
    }

    try {
      const message = {
        key: eventData.correlationId || eventData.eventId,
        value: JSON.stringify(eventData),
        timestamp: eventData.timestamp.toString(),
        headers: {
          eventType: eventData.eventType,
          version: '1.0',
          source: eventData.source || 'event-system'
        }
      };

      await this.producer.send({
        topic,
        messages: [message]
      });

      this.logger.info(`Event published to Kafka topic ${topic}`, {
        eventId: eventData.eventId,
        eventType: eventData.eventType,
        correlationId: eventData.correlationId
      });

    } catch (error) {
      this.logger.error(`Failed to publish event to Kafka topic ${topic}:`, error);
      throw error;
    }
  }

  async subscribeToTopic(
    topic: string, 
    groupId: string,
    handler: (eventData: EventData) => Promise<void>
  ): Promise<void> {
    try {
      const consumer = this.kafka.consumer({
        groupId,
        sessionTimeout: 30000,
        heartbeatInterval: 3000,
        maxWaitTimeInMs: 5000,
        retry: {
          initialRetryTime: 100,
          retries: 8
        }
      });

      await consumer.connect();
      await consumer.subscribe({ topic, fromBeginning: false });

      await consumer.run({
        eachMessage: async ({ topic, partition, message }: EachMessagePayload) => {
          try {
            if (!message.value) {
              this.logger.warn('Received message with null value', { topic, partition });
              return;
            }

            const eventData: EventData = JSON.parse(message.value.toString());
            
            this.logger.info(`Processing message from Kafka topic ${topic}`, {
              eventId: eventData.eventId,
              eventType: eventData.eventType,
              partition,
              offset: message.offset
            });

            await handler(eventData);

          } catch (error) {
            this.logger.error(`Error processing message from topic ${topic}:`, error);
            
            // Send to dead letter queue topic
            await this.sendToDeadLetterQueue(topic, message, error as Error);
          }
        }
      });

      this.consumers.set(topic, consumer);
      this.logger.info(`Successfully subscribed to Kafka topic ${topic} with group ${groupId}`);

    } catch (error) {
      this.logger.error(`Failed to subscribe to Kafka topic ${topic}:`, error);
      throw error;
    }
  }

  async unsubscribeFromTopic(topic: string): Promise<void> {
    const consumer = this.consumers.get(topic);
    if (!consumer) {
      this.logger.warn(`No consumer found for topic ${topic}`);
      return;
    }

    try {
      await consumer.disconnect();
      this.consumers.delete(topic);
      this.logger.info(`Successfully unsubscribed from Kafka topic ${topic}`);
    } catch (error) {
      this.logger.error(`Error unsubscribing from topic ${topic}:`, error);
      throw error;
    }
  }

  async createTopics(topics: Array<{ topic: string; numPartitions?: number; replicationFactor?: number }>): Promise<void> {
    try {
      const admin = this.kafka.admin();
      await admin.connect();

      const topicConfigs = topics.map(({ topic, numPartitions = 3, replicationFactor = 1 }) => ({
        topic,
        numPartitions,
        replicationFactor,
        configEntries: [
          { name: 'cleanup.policy', value: 'delete' },
          { name: 'retention.ms', value: '604800000' }, // 7 days
          { name: 'segment.ms', value: '86400000' } // 1 day
        ]
      }));

      await admin.createTopics({
        topics: topicConfigs,
        waitForLeaders: true
      });

      await admin.disconnect();
      this.logger.info('Kafka topics created successfully', { topics: topics.map(t => t.topic) });

    } catch (error) {
      this.logger.error('Failed to create Kafka topics:', error);
      throw error;
    }
  }

  async getTopics(): Promise<string[]> {
    try {
      const admin = this.kafka.admin();
      await admin.connect();
      
      const metadata = await admin.fetchTopicMetadata();
      const topics = metadata.topics.map(topic => topic.name);
      
      await admin.disconnect();
      return topics;
      
    } catch (error) {
      this.logger.error('Failed to fetch Kafka topics:', error);
      throw error;
    }
  }

  getHealth(): HealthStatus {
    return {
      service: 'KafkaManager',
      status: this.isConnected ? 'healthy' : 'unhealthy',
      timestamp: new Date(),
      details: {
        connected: this.isConnected,
        producerConnected: !!this.producer,
        activeConsumers: this.consumers.size,
        reconnectAttempts: this.reconnectAttempts
      }
    };
  }

  private async sendToDeadLetterQueue(topic: string, message: any, error: Error): Promise<void> {
    try {
      const deadLetterTopic = `${topic}.dlq`;
      const deadLetterEvent: EventData = {
        eventId: `dlq-${Date.now()}`,
        eventType: 'dead_letter_message',
        timestamp: Date.now(),
        data: {
          originalTopic: topic,
          originalMessage: message.value?.toString(),
          error: error.message,
          offset: message.offset,
          partition: message.partition
        }
      };

      await this.publishEvent(deadLetterTopic, deadLetterEvent);
      
    } catch (dlqError) {
      this.logger.error('Failed to send message to dead letter queue:', dlqError);
    }
  }

  private async handleReconnection(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.logger.error('Max reconnection attempts reached. Giving up.');
      this.emit('maxReconnectAttemptsReached');
      return;
    }

    this.reconnectAttempts++;
    this.logger.info(`Attempting to reconnect to Kafka (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        this.logger.error('Reconnection attempt failed:', error);
        await this.handleReconnection();
      }
    }, this.reconnectInterval);
  }
}
