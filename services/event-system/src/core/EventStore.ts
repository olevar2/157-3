import { createClient, RedisClientType } from 'redis';
import { EventData, EventStoreConfig, HealthStatus, EventQuery, EventStream } from '../types';
import { Logger } from '../utils/Logger';

export class EventStore {
  private redisClient: RedisClientType;
  private logger: Logger;
  private config: EventStoreConfig;
  private isConnected: boolean = false;
  private readonly STREAM_KEY_PREFIX = 'events:stream:';
  private readonly AGGREGATE_KEY_PREFIX = 'events:aggregate:';
  private readonly SNAPSHOT_KEY_PREFIX = 'events:snapshot:';

  constructor(config: EventStoreConfig) {
    this.config = config;
    this.logger = new Logger('EventStore');
      this.redisClient = createClient({
      url: config.redisUrl,
      password: config.password,
      database: config.database || 0,
      socket: {
        connectTimeout: config.connectTimeout || 5000,
        reconnectStrategy: (retries) => Math.min(retries * 50, 500)
      }
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redisClient.on('connect', () => {
      this.logger.info('EventStore Redis client connecting...');
    });

    this.redisClient.on('ready', () => {
      this.isConnected = true;
      this.logger.info('EventStore Redis client connected and ready');
    });

    this.redisClient.on('error', (error) => {
      this.logger.error('EventStore Redis client error:', error);
      this.isConnected = false;
    });

    this.redisClient.on('end', () => {
      this.isConnected = false;
      this.logger.info('EventStore Redis client connection ended');
    });
  }

  async connect(): Promise<void> {
    try {
      await this.redisClient.connect();
      this.logger.info('EventStore connected to Redis successfully');
    } catch (error) {
      this.logger.error('Failed to connect EventStore to Redis:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      if (this.isConnected) {
        await this.redisClient.quit();
        this.logger.info('EventStore disconnected from Redis');
      }
    } catch (error) {
      this.logger.error('Error disconnecting EventStore from Redis:', error);
      throw error;
    }
  }

  async storeEvent(aggregateId: string, eventData: EventData, expectedVersion?: number): Promise<string> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const streamKey = `${this.STREAM_KEY_PREFIX}${aggregateId}`;
      const aggregateKey = `${this.AGGREGATE_KEY_PREFIX}${aggregateId}`;
      
      // Check expected version if provided (optimistic concurrency control)
      if (expectedVersion !== undefined) {
        const currentVersion = await this.getCurrentVersion(aggregateId);
        if (currentVersion !== expectedVersion) {
          throw new Error(`Concurrency conflict: expected version ${expectedVersion}, but current version is ${currentVersion}`);
        }
      }      // Prepare event data for storage - filter out undefined values
      const eventPayload: Record<string, string> = {
        eventId: eventData.eventId,
        eventType: eventData.eventType,
        aggregateId,
        timestamp: eventData.timestamp.toString(),
        data: JSON.stringify(eventData.data),
        metadata: JSON.stringify(eventData.metadata || {}),
        source: eventData.source || 'event-store',
        version: '*' // Redis Streams auto-generates version
      };

      // Add optional fields only if they exist
      if (eventData.correlationId) {
        eventPayload.correlationId = eventData.correlationId;
      }
      if (eventData.causationId) {
        eventPayload.causationId = eventData.causationId;
      }

      // Store event in Redis Stream
      const eventId = await this.redisClient.xAdd(streamKey, '*', eventPayload);
      
      // Update aggregate version and last event timestamp
      await this.redisClient.hSet(aggregateKey, {
        lastEventId: eventId,
        lastEventTimestamp: eventData.timestamp.toString(),
        version: await this.getStreamLength(streamKey)
      });

      this.logger.info(`Event stored successfully`, {
        aggregateId,
        eventId: eventData.eventId,
        eventType: eventData.eventType,
        streamEventId: eventId
      });

      return eventId;

    } catch (error) {
      this.logger.error(`Failed to store event for aggregate ${aggregateId}:`, error);
      throw error;
    }
  }

  async getEventStream(aggregateId: string, fromVersion?: number, toVersion?: number): Promise<EventStream> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const streamKey = `${this.STREAM_KEY_PREFIX}${aggregateId}`;
      
      // Determine start and end positions
      let start = fromVersion ? `${fromVersion}` : '-';
      let end = toVersion ? `${toVersion}` : '+';
      
      // Read events from stream
      const events = await this.redisClient.xRange(streamKey, start, end);
      
      const eventStream: EventStream = {
        aggregateId,
        events: events.map(event => this.parseStoredEvent(event)),
        version: await this.getCurrentVersion(aggregateId),
        timestamp: Date.now()
      };

      this.logger.info(`Retrieved event stream for aggregate ${aggregateId}`, {
        eventCount: events.length,
        fromVersion,
        toVersion
      });

      return eventStream;

    } catch (error) {
      this.logger.error(`Failed to get event stream for aggregate ${aggregateId}:`, error);
      throw error;
    }
  }

  async getEvent(aggregateId: string, eventId: string): Promise<EventData | null> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const streamKey = `${this.STREAM_KEY_PREFIX}${aggregateId}`;
      
      // Get specific event from stream
      const events = await this.redisClient.xRange(streamKey, eventId, eventId);
      
      if (events.length === 0) {
        return null;
      }

      return this.parseStoredEvent(events[0]);

    } catch (error) {
      this.logger.error(`Failed to get event ${eventId} for aggregate ${aggregateId}:`, error);
      throw error;
    }
  }

  async queryEvents(query: EventQuery): Promise<EventData[]> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const results: EventData[] = [];
      
      if (query.aggregateIds && query.aggregateIds.length > 0) {
        // Query specific aggregates
        for (const aggregateId of query.aggregateIds) {
          const stream = await this.getEventStream(
            aggregateId, 
            query.fromTimestamp, 
            query.toTimestamp
          );
          results.push(...stream.events);
        }
      } else {
        // Query all aggregates (expensive operation - use with caution)
        const aggregatePattern = `${this.STREAM_KEY_PREFIX}*`;
        const streamKeys = await this.redisClient.keys(aggregatePattern);
        
        for (const streamKey of streamKeys) {
          const aggregateId = streamKey.replace(this.STREAM_KEY_PREFIX, '');
          const stream = await this.getEventStream(
            aggregateId,
            query.fromTimestamp,
            query.toTimestamp
          );
          results.push(...stream.events);
        }
      }

      // Apply filters
      let filteredResults = results;
      
      if (query.eventTypes && query.eventTypes.length > 0) {
        filteredResults = filteredResults.filter(event => 
          query.eventTypes!.includes(event.eventType)
        );
      }

      if (query.correlationId) {
        filteredResults = filteredResults.filter(event => 
          event.correlationId === query.correlationId
        );
      }

      // Sort by timestamp
      filteredResults.sort((a, b) => a.timestamp - b.timestamp);

      // Apply limit
      if (query.limit && query.limit > 0) {
        filteredResults = filteredResults.slice(0, query.limit);
      }

      this.logger.info(`Query executed successfully`, {
        totalResults: results.length,
        filteredResults: filteredResults.length,
        query
      });

      return filteredResults;

    } catch (error) {
      this.logger.error('Failed to execute event query:', error);
      throw error;
    }
  }

  async saveSnapshot(aggregateId: string, version: number, snapshotData: any): Promise<void> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const snapshotKey = `${this.SNAPSHOT_KEY_PREFIX}${aggregateId}`;
      
      const snapshot = {
        aggregateId,
        version: version.toString(),
        timestamp: Date.now().toString(),
        data: JSON.stringify(snapshotData)
      };

      await this.redisClient.hSet(snapshotKey, snapshot);
      
      // Set expiration for snapshot (optional, based on config)
      if (this.config.snapshotTtl) {
        await this.redisClient.expire(snapshotKey, this.config.snapshotTtl);
      }

      this.logger.info(`Snapshot saved for aggregate ${aggregateId}`, { version });

    } catch (error) {
      this.logger.error(`Failed to save snapshot for aggregate ${aggregateId}:`, error);
      throw error;
    }
  }

  async getSnapshot(aggregateId: string): Promise<{ version: number; data: any; timestamp: number } | null> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const snapshotKey = `${this.SNAPSHOT_KEY_PREFIX}${aggregateId}`;
      const snapshot = await this.redisClient.hGetAll(snapshotKey);

      if (!snapshot || Object.keys(snapshot).length === 0) {
        return null;
      }

      return {
        version: parseInt(snapshot.version),
        data: JSON.parse(snapshot.data),
        timestamp: parseInt(snapshot.timestamp)
      };

    } catch (error) {
      this.logger.error(`Failed to get snapshot for aggregate ${aggregateId}:`, error);
      throw error;
    }
  }

  async getCurrentVersion(aggregateId: string): Promise<number> {
    try {
      const streamKey = `${this.STREAM_KEY_PREFIX}${aggregateId}`;
      return await this.getStreamLength(streamKey);
    } catch (error) {
      this.logger.error(`Failed to get current version for aggregate ${aggregateId}:`, error);
      return 0;
    }
  }

  async getStreamLength(streamKey: string): Promise<number> {
    try {
      return await this.redisClient.xLen(streamKey);
    } catch (error) {
      return 0;
    }
  }

  async trimStream(aggregateId: string, maxLength: number): Promise<void> {
    if (!this.isConnected) {
      throw new Error('EventStore not connected to Redis');
    }

    try {
      const streamKey = `${this.STREAM_KEY_PREFIX}${aggregateId}`;
      await this.redisClient.xTrim(streamKey, 'MAXLEN', maxLength, { strategyModifier: '~' });
      
      this.logger.info(`Stream trimmed for aggregate ${aggregateId}`, { maxLength });

    } catch (error) {
      this.logger.error(`Failed to trim stream for aggregate ${aggregateId}:`, error);
      throw error;
    }
  }

  getHealth(): HealthStatus {
    return {
      service: 'EventStore',
      status: this.isConnected ? 'healthy' : 'unhealthy',
      timestamp: new Date(),
      details: {
        connected: this.isConnected,
        redisUrl: this.config.redisUrl.replace(/\/\/[^@]*@/, '//***@'), // Hide credentials
        database: this.config.database || 0
      }
    };
  }

  private parseStoredEvent(redisEvent: any): EventData {
    const [eventId, fields] = redisEvent;
    const fieldMap: { [key: string]: string } = {};
    
    // Convert Redis stream fields array to object
    for (let i = 0; i < fields.length; i += 2) {
      fieldMap[fields[i]] = fields[i + 1];
    }

    return {
      eventId: fieldMap.eventId,
      eventType: fieldMap.eventType,
      timestamp: parseInt(fieldMap.timestamp),
      data: JSON.parse(fieldMap.data || '{}'),
      metadata: JSON.parse(fieldMap.metadata || '{}'),
      correlationId: fieldMap.correlationId,
      causationId: fieldMap.causationId,
      source: fieldMap.source
    };
  }
}
