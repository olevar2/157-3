export interface EventData {
  eventId: string;
  eventType: string;
  timestamp: number;
  data: any;
  metadata?: Record<string, any>;
  correlationId?: string;
  causationId?: string;
  source?: string;
  userId?: string;
  sessionId?: string;
}

export interface QueueJob {
  id: string;
  type: string;
  data: any;
  priority?: number;
  delay?: number;
  attempts?: number;
  backoff?: string | number;
  removeOnComplete?: boolean;
  removeOnFail?: boolean;
}

export interface EventSubscription {
  id: string;
  eventType: string;
  callback: (event: EventData) => Promise<void>;
  filter?: (event: EventData) => boolean;
  deadLetterQueue?: boolean;
}

export interface HealthStatus {
  service: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: Date;
  details?: Record<string, any>;
}

export interface EventBusConfig {
  redisUrl: string;
  maxRetries?: number;
  retryDelayMs?: number;
}

export interface MessageQueueConfig {
  redisUrl: string;
  concurrency?: number;
  removeOnComplete?: number;
  removeOnFail?: number;
}

export interface KafkaConfig {
  brokers: string[];
  clientId: string;
  connectionTimeout?: number;
  requestTimeout?: number;
}

export interface EventStoreConfig {
  redisUrl: string;
  password?: string;
  database?: number;
  connectTimeout?: number;
  snapshotTtl?: number;
}

// New interfaces for EventStore functionality
export interface EventQuery {
  aggregateIds?: string[];
  eventTypes?: string[];
  fromTimestamp?: number;
  toTimestamp?: number;
  correlationId?: string;
  limit?: number;
}

export interface EventStream {
  aggregateId: string;
  events: EventData[];
  version: number;
  timestamp: number;
}

export interface PublishEventRequest {
  type: string;
  source: string;
  data: any;
  metadata?: Record<string, any>;
  correlationId?: string;
  userId?: string;
}

export interface SubscribeRequest {
  eventType: string;
  webhookUrl?: string;
  filter?: Record<string, any>;
  deadLetterQueue?: boolean;
}

export interface QueueJobRequest {
  type: string;
  data: any;
  priority?: number;
  delay?: number;
  options?: {
    attempts?: number;
    backoff?: string | number;
    removeOnComplete?: boolean;
    removeOnFail?: boolean;
  };
}

export interface EventMetrics {
  eventsPublished: number;
  eventsConsumed: number;
  queueJobs: number;
  failedJobs: number;
  avgProcessingTime: number;
  errorRate: number;
}
