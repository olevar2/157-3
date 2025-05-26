const Redis = require('ioredis');
const logger = require('../utils/logger');

class EventStore {
    constructor(redisConfig = {}) {
        this.redis = new Redis({
            host: redisConfig.host || process.env.REDIS_HOST || 'localhost',
            port: redisConfig.port || process.env.REDIS_PORT || 6379,
            password: redisConfig.password || process.env.REDIS_PASSWORD,
            db: redisConfig.db || process.env.REDIS_DB || 1,
            retryDelayOnFailover: 100,
            maxRetriesPerRequest: 3,
            lazyConnect: true
        });

        this.redis.on('connect', () => {
            logger.info('EventStore connected to Redis');
        });

        this.redis.on('error', (error) => {
            logger.error('EventStore Redis connection error:', error);
        });
    }

    /**
     * Store an event in the event store
     * @param {string} streamId - Stream identifier
     * @param {Object} event - Event data
     * @param {Object} metadata - Event metadata
     * @returns {Promise<string>} Event ID
     */
    async appendEvent(streamId, event, metadata = {}) {
        try {
            const eventId = this.generateEventId();
            const timestamp = Date.now();
            
            const eventData = {
                id: eventId,
                streamId,
                eventType: event.type,
                data: JSON.stringify(event.data),
                metadata: JSON.stringify({
                    ...metadata,
                    timestamp,
                    version: await this.getStreamVersion(streamId) + 1
                }),
                createdAt: new Date().toISOString()
            };

            // Store in stream-specific sorted set
            await this.redis.zadd(
                `stream:${streamId}`,
                timestamp,
                JSON.stringify(eventData)
            );

            // Store in global event log
            await this.redis.zadd(
                'events:global',
                timestamp,
                JSON.stringify(eventData)
            );

            // Update stream metadata
            await this.redis.hset(`stream:${streamId}:meta`, {
                lastEventId: eventId,
                lastUpdated: timestamp,
                version: eventData.metadata.version || 1
            });

            logger.info(`Event stored: ${eventId} in stream ${streamId}`);
            return eventId;

        } catch (error) {
            logger.error('Error storing event:', error);
            throw error;
        }
    }

    /**
     * Get events from a stream
     * @param {string} streamId - Stream identifier
     * @param {number} fromVersion - Starting version (optional)
     * @param {number} maxCount - Maximum events to return
     * @returns {Promise<Array>} Array of events
     */
    async getEvents(streamId, fromVersion = 0, maxCount = 100) {
        try {
            const key = `stream:${streamId}`;
            const events = await this.redis.zrange(key, fromVersion, fromVersion + maxCount - 1);
            
            return events.map(eventStr => {
                const event = JSON.parse(eventStr);
                return {
                    ...event,
                    data: JSON.parse(event.data),
                    metadata: JSON.parse(event.metadata)
                };
            });

        } catch (error) {
            logger.error(`Error getting events from stream ${streamId}:`, error);
            throw error;
        }
    }

    /**
     * Get events by time range
     * @param {number} startTime - Start timestamp
     * @param {number} endTime - End timestamp
     * @param {number} maxCount - Maximum events to return
     * @returns {Promise<Array>} Array of events
     */
    async getEventsByTimeRange(startTime, endTime, maxCount = 1000) {
        try {
            const events = await this.redis.zrangebyscore(
                'events:global',
                startTime,
                endTime,
                'LIMIT', 0, maxCount
            );

            return events.map(eventStr => {
                const event = JSON.parse(eventStr);
                return {
                    ...event,
                    data: JSON.parse(event.data),
                    metadata: JSON.parse(event.metadata)
                };
            });

        } catch (error) {
            logger.error('Error getting events by time range:', error);
            throw error;
        }
    }

    /**
     * Get stream metadata
     * @param {string} streamId - Stream identifier
     * @returns {Promise<Object>} Stream metadata
     */
    async getStreamMetadata(streamId) {
        try {
            const metadata = await this.redis.hgetall(`stream:${streamId}:meta`);
            return {
                ...metadata,
                version: parseInt(metadata.version) || 0,
                lastUpdated: parseInt(metadata.lastUpdated) || 0
            };
        } catch (error) {
            logger.error(`Error getting stream metadata for ${streamId}:`, error);
            throw error;
        }
    }

    /**
     * Get current version of a stream
     * @param {string} streamId - Stream identifier
     * @returns {Promise<number>} Current version
     */
    async getStreamVersion(streamId) {
        try {
            const version = await this.redis.hget(`stream:${streamId}:meta`, 'version');
            return parseInt(version) || 0;
        } catch (error) {
            logger.error(`Error getting stream version for ${streamId}:`, error);
            return 0;
        }
    }

    /**
     * Create a snapshot of stream state
     * @param {string} streamId - Stream identifier
     * @param {Object} state - Current state
     * @param {number} version - Version at snapshot
     * @returns {Promise<void>}
     */
    async createSnapshot(streamId, state, version) {
        try {
            const snapshot = {
                streamId,
                state: JSON.stringify(state),
                version,
                createdAt: new Date().toISOString()
            };

            await this.redis.hset(
                `snapshot:${streamId}`,
                version,
                JSON.stringify(snapshot)
            );

            logger.info(`Snapshot created for stream ${streamId} at version ${version}`);

        } catch (error) {
            logger.error(`Error creating snapshot for ${streamId}:`, error);
            throw error;
        }
    }

    /**
     * Get latest snapshot for a stream
     * @param {string} streamId - Stream identifier
     * @returns {Promise<Object|null>} Latest snapshot or null
     */
    async getLatestSnapshot(streamId) {
        try {
            const snapshots = await this.redis.hgetall(`snapshot:${streamId}`);
            if (!snapshots || Object.keys(snapshots).length === 0) {
                return null;
            }

            // Get the highest version
            const versions = Object.keys(snapshots).map(v => parseInt(v)).sort((a, b) => b - a);
            const latestVersion = versions[0];
            
            const snapshot = JSON.parse(snapshots[latestVersion]);
            return {
                ...snapshot,
                state: JSON.parse(snapshot.state)
            };

        } catch (error) {
            logger.error(`Error getting latest snapshot for ${streamId}:`, error);
            return null;
        }
    }

    /**
     * List all streams
     * @returns {Promise<Array>} Array of stream IDs
     */
    async listStreams() {
        try {
            const keys = await this.redis.keys('stream:*:meta');
            return keys.map(key => key.replace('stream:', '').replace(':meta', ''));
        } catch (error) {
            logger.error('Error listing streams:', error);
            throw error;
        }
    }

    /**
     * Generate unique event ID
     * @returns {string} Event ID
     */
    generateEventId() {
        return `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Close the connection
     */
    async close() {
        if (this.redis) {
            await this.redis.quit();
            logger.info('EventStore connection closed');
        }
    }
}

module.exports = EventStore;
