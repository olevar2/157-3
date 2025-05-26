import Bull, { Queue, Job, JobOptions } from 'bull';
import Redis from 'ioredis';
import { Logger } from '../utils/Logger';
import { QueueJob, MessageQueueConfig, HealthStatus } from '../types';

export class MessageQueue {
  private redis: Redis;
  private queues: Map<string, Queue> = new Map();
  private config: MessageQueueConfig;
  private logger: Logger;
  private isConnected: boolean = false;
  constructor(config: MessageQueueConfig) {
    this.config = config;
    this.logger = new Logger('MessageQueue');
    
    this.redis = new Redis(config.redisUrl, {
      enableReadyCheck: true,
      maxRetriesPerRequest: 3,
      lazyConnect: true
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redis.on('connect', () => {
      this.logger.info('MessageQueue Redis client connecting...');
    });

    this.redis.on('ready', () => {
      this.isConnected = true;
      this.logger.info('MessageQueue Redis client connected and ready');
    });

    this.redis.on('error', (error) => {
      this.logger.error('MessageQueue Redis client error:', error);
      this.isConnected = false;
    });

    this.redis.on('end', () => {
      this.isConnected = false;
      this.logger.info('MessageQueue Redis client connection ended');
    });
  }

  async connect(): Promise<void> {
    try {
      await this.redis.connect();
      this.logger.info('MessageQueue connected to Redis successfully');
    } catch (error) {
      this.logger.error('Failed to connect MessageQueue to Redis:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      // Close all queues
      for (const [name, queue] of this.queues.entries()) {
        await queue.close();
        this.logger.info(`Queue ${name} closed`);
      }
      this.queues.clear();

      // Close Redis connection
      if (this.isConnected) {
        await this.redis.quit();
        this.logger.info('MessageQueue disconnected from Redis');
      }
    } catch (error) {
      this.logger.error('Error disconnecting MessageQueue from Redis:', error);
      throw error;
    }
  }

  async addJob(queueName: string, jobData: any, options?: JobOptions): Promise<Job> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
      
      const jobOptions: JobOptions = {
        removeOnComplete: this.config.removeOnComplete || 100,
        removeOnFail: this.config.removeOnFail || 50,
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 2000
        },
        ...options
      };

      const job = await queue.add(jobData, jobOptions);
      
      this.logger.info(`Job added to queue ${queueName}`, {
        jobId: job.id,
        priority: job.opts.priority,
        delay: job.opts.delay
      });

      return job;
    } catch (error) {
      this.logger.error(`Failed to add job to queue ${queueName}:`, error);
      throw error;
    }
  }

  async processQueue(queueName: string, processor: (job: Job) => Promise<any>): Promise<void> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
      
      queue.process(this.config.concurrency || 1, async (job: Job) => {
        this.logger.info(`Processing job ${job.id} from queue ${queueName}`, {
          jobId: job.id,
          data: job.data
        });

        try {
          const result = await processor(job);
          
          this.logger.info(`Job ${job.id} completed successfully`, {
            jobId: job.id,
            result
          });

          return result;
        } catch (error) {
          this.logger.error(`Job ${job.id} failed:`, error);
          throw error;
        }
      });

      this.logger.info(`Queue ${queueName} processor registered`);
    } catch (error) {
      this.logger.error(`Failed to setup processor for queue ${queueName}:`, error);
      throw error;
    }
  }

  async getJob(queueName: string, jobId: string): Promise<Job | null> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
      return await queue.getJob(jobId);
    } catch (error) {
      this.logger.error(`Failed to get job ${jobId} from queue ${queueName}:`, error);
      return null;
    }
  }

  async removeJob(queueName: string, jobId: string): Promise<void> {
    try {
      const job = await this.getJob(queueName, jobId);
      if (job) {
        await job.remove();
        this.logger.info(`Job ${jobId} removed from queue ${queueName}`);
      }
    } catch (error) {
      this.logger.error(`Failed to remove job ${jobId} from queue ${queueName}:`, error);
      throw error;
    }
  }

  async getQueueStats(queueName: string): Promise<any> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
        const [waiting, active, completed, failed, delayed] = await Promise.all([
        queue.getWaiting(),
        queue.getActive(),
        queue.getCompleted(),
        queue.getFailed(),
        queue.getDelayed()
      ]);

      return {
        name: queueName,
        waiting: waiting.length,
        active: active.length,
        completed: completed.length,
        failed: failed.length,
        delayed: delayed.length,
        paused: await queue.isPaused()
      };
    } catch (error) {
      this.logger.error(`Failed to get stats for queue ${queueName}:`, error);
      throw error;
    }
  }

  async getAllQueueStats(): Promise<any[]> {
    const stats: any[] = [];
    
    for (const queueName of this.queues.keys()) {
      try {
        const queueStats = await this.getQueueStats(queueName);
        stats.push(queueStats);
      } catch (error) {
        this.logger.error(`Failed to get stats for queue ${queueName}:`, error);
      }
    }

    return stats;
  }

  async pauseQueue(queueName: string): Promise<void> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
      await queue.pause();
      this.logger.info(`Queue ${queueName} paused`);
    } catch (error) {
      this.logger.error(`Failed to pause queue ${queueName}:`, error);
      throw error;
    }
  }

  async resumeQueue(queueName: string): Promise<void> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
      await queue.resume();
      this.logger.info(`Queue ${queueName} resumed`);
    } catch (error) {
      this.logger.error(`Failed to resume queue ${queueName}:`, error);
      throw error;
    }
  }

  async cleanQueue(queueName: string, grace: number = 0, status: string = 'completed'): Promise<Job[]> {
    try {
      const queue = await this.getOrCreateQueue(queueName);
      const jobs = await queue.clean(grace, status as any);
      
      this.logger.info(`Cleaned ${jobs.length} ${status} jobs from queue ${queueName}`);
      return jobs;
    } catch (error) {
      this.logger.error(`Failed to clean queue ${queueName}:`, error);
      throw error;
    }
  }

  private async getOrCreateQueue(queueName: string): Promise<Queue> {
    if (!this.queues.has(queueName)) {
      const redisConfig = {
        host: this.redis.options.host,
        port: this.redis.options.port,
        password: this.redis.options.password,
        db: this.redis.options.db
      };

      const queue = new Bull(queueName, {
        redis: redisConfig,
        defaultJobOptions: {
          removeOnComplete: this.config.removeOnComplete || 100,
          removeOnFail: this.config.removeOnFail || 50
        }
      });

      // Setup queue event handlers
      queue.on('error', (error) => {
        this.logger.error(`Queue ${queueName} error:`, error);
      });

      queue.on('waiting', (jobId) => {
        this.logger.debug(`Job ${jobId} waiting in queue ${queueName}`);
      });

      queue.on('active', (job) => {
        this.logger.debug(`Job ${job.id} active in queue ${queueName}`);
      });

      queue.on('completed', (job, result) => {
        this.logger.debug(`Job ${job.id} completed in queue ${queueName}`, { result });
      });

      queue.on('failed', (job, error) => {
        this.logger.error(`Job ${job.id} failed in queue ${queueName}:`, error);
      });

      queue.on('paused', () => {
        this.logger.info(`Queue ${queueName} paused`);
      });

      queue.on('resumed', () => {
        this.logger.info(`Queue ${queueName} resumed`);
      });

      this.queues.set(queueName, queue);
      this.logger.info(`Queue ${queueName} created`);
    }

    return this.queues.get(queueName)!;
  }

  // Setup default processors for common job types
  async setupDefaultProcessors(): Promise<void> {
    // Email processor
    await this.processQueue('email', async (job: Job) => {
      const { to, subject, body, template } = job.data;
      
      this.logger.info('Processing email job', { to, subject });
      
      // Simulate email sending
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return { success: true, messageId: `email_${Date.now()}` };
    });

    // SMS processor
    await this.processQueue('sms', async (job: Job) => {
      const { to, message } = job.data;
      
      this.logger.info('Processing SMS job', { to });
      
      // Simulate SMS sending
      await new Promise(resolve => setTimeout(resolve, 500));
      
      return { success: true, messageId: `sms_${Date.now()}` };
    });

    // Payment processor
    await this.processQueue('payment', async (job: Job) => {
      const { amount, currency, paymentMethod, userId } = job.data;
      
      this.logger.info('Processing payment job', { amount, currency, userId });
      
      // Simulate payment processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      return { 
        success: true, 
        transactionId: `txn_${Date.now()}`,
        amount,
        currency
      };
    });

    // Notification processor
    await this.processQueue('notification', async (job: Job) => {
      const { userId, title, message, type } = job.data;
      
      this.logger.info('Processing notification job', { userId, type });
      
      // Simulate notification sending
      await new Promise(resolve => setTimeout(resolve, 300));
      
      return { success: true, notificationId: `notif_${Date.now()}` };
    });

    this.logger.info('Default queue processors setup completed');
  }

  getHealth(): HealthStatus {
    return {
      service: 'MessageQueue',
      status: this.isConnected ? 'healthy' : 'unhealthy',
      timestamp: new Date(),
      details: {
        connected: this.isConnected,
        activeQueues: this.queues.size,
        redisUrl: this.config.redisUrl.replace(/\/\/[^@]*@/, '//***@') // Hide credentials
      }
    };
  }
}
