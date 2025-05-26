const Queue = require('bull');
const Redis = require('ioredis');
const logger = require('../utils/logger');

class MessageQueue {
  constructor() {
    this.redis = null;
    this.queues = new Map();
    this.processors = new Map();
  }

  async initialize() {
    try {
      const redisConfig = {
        host: process.env.REDIS_HOST || 'localhost',
        port: process.env.REDIS_PORT || 6379,
        password: process.env.REDIS_PASSWORD,
        db: parseInt(process.env.REDIS_QUEUE_DB) || 1
      };

      this.redis = new Redis(redisConfig);

      // Initialize predefined queues
      const queueNames = [
        'email-notifications',
        'sms-alerts',
        'document-processing',
        'payment-processing',
        'trade-executions',
        'risk-alerts'
      ];

      for (const queueName of queueNames) {
        await this.createQueue(queueName);
      }

      // Set up default processors
      this.setupDefaultProcessors();

      logger.info('Message Queue system initialized successfully');
    } catch (error) {
      logger.error(`Failed to initialize Message Queue: ${error.message}`);
      throw error;
    }
  }

  async createQueue(queueName, options = {}) {
    try {
      const defaultOptions = {
        redis: {
          host: process.env.REDIS_HOST || 'localhost',
          port: process.env.REDIS_PORT || 6379,
          password: process.env.REDIS_PASSWORD,
          db: parseInt(process.env.REDIS_QUEUE_DB) || 1
        },
        defaultJobOptions: {
          removeOnComplete: 10,
          removeOnFail: 5,
          attempts: 3,
          backoff: {
            type: 'exponential',
            delay: 2000
          }
        },
        ...options
      };

      const queue = new Queue(queueName, defaultOptions);

      // Set up event listeners
      queue.on('completed', (job, result) => {
        logger.info(`Job completed: ${job.queue.name}/${job.id}`, { result });
      });

      queue.on('failed', (job, error) => {
        logger.error(`Job failed: ${job.queue.name}/${job.id}`, { error: error.message });
      });

      queue.on('stalled', (job) => {
        logger.warn(`Job stalled: ${job.queue.name}/${job.id}`);
      });

      this.queues.set(queueName, queue);
      logger.info(`Queue created: ${queueName}`);
      
      return queue;
    } catch (error) {
      logger.error(`Failed to create queue ${queueName}: ${error.message}`);
      throw error;
    }
  }

  async addJob(queueName, jobType, data, options = {}) {
    try {
      let queue = this.queues.get(queueName);
      
      if (!queue) {
        queue = await this.createQueue(queueName);
      }

      const job = await queue.add(jobType, data, {
        priority: options.priority || 0,
        delay: options.delay || 0,
        attempts: options.attempts || 3,
        ...options
      });

      logger.debug(`Job added to queue: ${queueName}/${jobType}`, { jobId: job.id, data });
      return job.id;
    } catch (error) {
      logger.error(`Failed to add job to queue ${queueName}: ${error.message}`);
      throw error;
    }
  }

  async addProcessor(queueName, jobType, processor) {
    try {
      let queue = this.queues.get(queueName);
      
      if (!queue) {
        queue = await this.createQueue(queueName);
      }

      // Store processor for reference
      const processorKey = `${queueName}:${jobType}`;
      this.processors.set(processorKey, processor);

      // Add processor to queue
      queue.process(jobType, processor);
      
      logger.info(`Processor added: ${queueName}/${jobType}`);
    } catch (error) {
      logger.error(`Failed to add processor for ${queueName}/${jobType}: ${error.message}`);
      throw error;
    }
  }

  setupDefaultProcessors() {
    // Email notification processor
    this.addProcessor('email-notifications', 'send-email', async (job) => {
      const { to, subject, body, template } = job.data;
      logger.info(`Processing email: ${subject} to ${to}`);
      
      // Simulate email sending
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return { success: true, messageId: `email_${Date.now()}` };
    });

    // SMS alert processor
    this.addProcessor('sms-alerts', 'send-sms', async (job) => {
      const { to, message } = job.data;
      logger.info(`Processing SMS: ${message} to ${to}`);
      
      // Simulate SMS sending
      await new Promise(resolve => setTimeout(resolve, 500));
      
      return { success: true, messageId: `sms_${Date.now()}` };
    });

    // Document processing processor
    this.addProcessor('document-processing', 'process-kyc', async (job) => {
      const { userId, documentType, documentUrl } = job.data;
      logger.info(`Processing KYC document: ${documentType} for user ${userId}`);
      
      // Simulate document processing
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      return { 
        success: true, 
        status: 'approved',
        confidence: 0.95,
        processedAt: new Date().toISOString()
      };
    });

    // Payment processing processor
    this.addProcessor('payment-processing', 'process-payment', async (job) => {
      const { userId, amount, currency, method } = job.data;
      logger.info(`Processing payment: ${amount} ${currency} for user ${userId}`);
      
      // Simulate payment processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      return { 
        success: true, 
        transactionId: `txn_${Date.now()}`,
        status: 'completed',
        processedAt: new Date().toISOString()
      };
    });

    // Trade execution processor
    this.addProcessor('trade-executions', 'execute-trade', async (job) => {
      const { tradeId, userId, symbol, side, quantity, price } = job.data;
      logger.info(`Executing trade: ${symbol} ${side} ${quantity} @ ${price}`);
      
      // Simulate trade execution
      await new Promise(resolve => setTimeout(resolve, 100));
      
      return { 
        success: true, 
        executionId: `exec_${Date.now()}`,
        executedPrice: price * (1 + (Math.random() - 0.5) * 0.001), // Small slippage
        executedAt: new Date().toISOString()
      };
    });

    // Risk alert processor
    this.addProcessor('risk-alerts', 'process-risk-alert', async (job) => {
      const { userId, alertType, severity, message } = job.data;
      logger.warn(`Risk alert: ${alertType} (${severity}) for user ${userId}`);
      
      // Simulate risk processing
      await new Promise(resolve => setTimeout(resolve, 300));
      
      return { 
        success: true, 
        alertId: `alert_${Date.now()}`,
        action: severity === 'high' ? 'block_trading' : 'monitor',
        processedAt: new Date().toISOString()
      };
    });
  }

  async getStatus() {
    try {
      const status = {};
      
      for (const [queueName, queue] of this.queues) {
        const waiting = await queue.getWaiting();
        const active = await queue.getActive();
        const completed = await queue.getCompleted();
        const failed = await queue.getFailed();
        
        status[queueName] = {
          waiting: waiting.length,
          active: active.length,
          completed: completed.length,
          failed: failed.length,
          total: waiting.length + active.length + completed.length + failed.length
        };
      }
      
      return status;
    } catch (error) {
      logger.error(`Failed to get queue status: ${error.message}`);
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

  async close() {
    try {
      // Close all queues
      for (const [queueName, queue] of this.queues) {
        await queue.close();
      }
      
      // Disconnect Redis
      if (this.redis) {
        await this.redis.disconnect();
      }
      
      logger.info('Message Queue system closed');
    } catch (error) {
      logger.error(`Error closing Message Queue: ${error.message}`);
    }
  }
}

module.exports = MessageQueue;
