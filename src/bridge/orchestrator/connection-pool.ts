import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { BridgeMessage, ChannelConfig } from '../types';
import { DualChannelWebSocket } from '../channels/dual-channel.websocket';
import { Logger } from '../../shared/logging/logger';

/**
 * Connection pool configuration
 */
export interface ConnectionPoolConfig {
    minSize: number;     // Minimum number of connections to maintain
    maxSize: number;     // Maximum number of connections allowed
    idleTimeout: number; // Time in milliseconds before idle connections are closed
}

/**
 * Connection with metadata
 */
interface PooledConnection {
    id: string;
    connection: DualChannelWebSocket;
    inUse: boolean;
    lastUsed: number;
}

/**
 * Connection Pool
 * 
 * Manages a pool of WebSocket connections for the TypeScript-Python bridge.
 * Handles connection creation, reuse, health checks, and cleanup of idle connections.
 */
export class ConnectionPool extends EventEmitter {
    private connections: Map<string, PooledConnection> = new Map();
    private waitQueue: Array<{ resolve: (connection: DualChannelWebSocket) => void, reject: (error: Error) => void }> = [];
    private maintenanceInterval: NodeJS.Timeout | null = null;
    private logger: Logger;

    constructor(
        private configs: ChannelConfig[],
        private poolConfig: ConnectionPoolConfig
    ) {
        super();
        this.logger = new Logger('ConnectionPool');
        this.initialize();
    }

    /**
     * Initialize the connection pool
     */
    private async initialize(): Promise<void> {
        // Create minimum connections
        for (let i = 0; i < this.poolConfig.minSize; i++) {
            await this.createConnection();
        }
        
        // Start maintenance routine
        this.startMaintenance();
        this.logger.info(`Connection pool initialized with ${this.poolConfig.minSize} connections`);
    }

    /**
     * Start maintenance routine to clean up idle connections
     */
    private startMaintenance(): void {
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
        }
        
        this.maintenanceInterval = setInterval(() => {
            this.performMaintenance();
        }, 30000); // Run every 30 seconds
    }
    
    /**
     * Perform pool maintenance
     */
    private performMaintenance(): void {
        const now = Date.now();
        let idleCount = 0;
        
        for (const [id, conn] of this.connections.entries()) {
            // Skip connections in use
            if (conn.inUse) continue;
            
            // Check if connection is idle
            const idleTime = now - conn.lastUsed;
            if (idleTime > this.poolConfig.idleTimeout) {
                // Only close if we're above min size
                if (this.connections.size > this.poolConfig.minSize) {
                    this.closeConnection(id);
                    this.logger.debug(`Closed idle connection: ${id} (idle for ${idleTime}ms)`);
                }
            } else {
                idleCount++;
            }
        }
        
        this.logger.debug(`Pool maintenance completed. Active: ${this.connections.size}, Idle: ${idleCount}`);
    }

    /**
     * Get a connection from the pool
     */
    async getConnection(): Promise<DualChannelWebSocket> {
        // First, try to find an idle connection
        for (const [id, conn] of this.connections.entries()) {
            if (!conn.inUse) {
                conn.inUse = true;
                conn.lastUsed = Date.now();
                this.logger.debug(`Reusing existing connection: ${id}`);
                return conn.connection;
            }
        }
        
        // If pool isn't at max size, create a new connection
        if (this.connections.size < this.poolConfig.maxSize) {
            const connection = await this.createConnection();
            const pooledConn = this.connections.get(connection.id)!;
            pooledConn.inUse = true;
            pooledConn.lastUsed = Date.now();
            return connection;
        }
        
        // Otherwise, wait for a connection to be released
        this.logger.debug('Connection pool at capacity, waiting for available connection');
        return new Promise<DualChannelWebSocket>((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                const index = this.waitQueue.findIndex(item => item.resolve === resolve);
                if (index !== -1) {
                    this.waitQueue.splice(index, 1);
                }
                reject(new Error('Timed out waiting for connection'));
            }, 5000); // 5 second timeout
            
            this.waitQueue.push({
                resolve: (connection) => {
                    clearTimeout(timeoutId);
                    resolve(connection);
                },
                reject: (error) => {
                    clearTimeout(timeoutId);
                    reject(error);
                }
            });
        });
    }

    /**
     * Release a connection back to the pool
     */
    releaseConnection(connection: DualChannelWebSocket): void {
        // Find the connection in the pool
        for (const [id, conn] of this.connections.entries()) {
            if (conn.connection === connection) {
                conn.inUse = false;
                conn.lastUsed = Date.now();
                
                // Check if someone is waiting for a connection
                if (this.waitQueue.length > 0) {
                    const waiting = this.waitQueue.shift()!;
                    conn.inUse = true;
                    waiting.resolve(conn.connection);
                    this.logger.debug(`Connection ${id} released and immediately provided to waiting request`);
                } else {
                    this.logger.debug(`Connection ${id} released back to pool`);
                }
                return;
            }
        }
        
        this.logger.warn('Released connection not found in pool');
    }
    
    /**
     * Create a new connection
     */
    private async createConnection(): Promise<DualChannelWebSocket> {
        // Choose a config randomly for load distribution
        const config = this.configs[Math.floor(Math.random() * this.configs.length)];
        
        // Create connection ID
        const connectionId = uuidv4();
        
        try {
            // Create the connection
            const connection = new DualChannelWebSocket(config);
            
            // Set up connection event handlers
            this.setupConnectionHandlers(connection, connectionId);
            
            // Store in pool
            this.connections.set(connectionId, {
                id: connectionId,
                connection,
                inUse: false,
                lastUsed: Date.now()
            });
            
            this.emit('connection:created', connectionId);
            return connection;
        } catch (error) {
            this.emit('connection:error', connectionId, error);
            throw new Error(`Failed to create connection: ${error}`);
        }
    }

    /**
     * Set up event handlers for a connection
     */
    private setupConnectionHandlers(connection: DualChannelWebSocket, id: string): void {
        // Forward message events
        connection.on('message', (message) => {
            this.emit('message', message);
        });
        
        // Handle errors
        connection.on('error', (error) => {
            this.emit('connection:error', id, error);
        });
        
        // Handle disconnect
        connection.on('channel:disconnected', (channelName) => {
            this.emit('channel:disconnected', { id, channelName });
        });
    }

    /**
     * Close a specific connection
     */
    private closeConnection(id: string): void {
        const conn = this.connections.get(id);
        if (conn) {
            this.connections.delete(id);
            // Remove listeners to prevent memory leaks
            conn.connection.removeAllListeners();
            this.emit('connection:closed', id);
        }
    }

    /**
     * Close all connections in the pool
     */
    async closeAll(): Promise<void> {
        // Stop maintenance
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
            this.maintenanceInterval = null;
        }
        
        // Reject all waiting requests
        for (const waiting of this.waitQueue) {
            waiting.reject(new Error('Connection pool shutting down'));
        }
        this.waitQueue = [];
        
        // Close all connections
        for (const id of this.connections.keys()) {
            this.closeConnection(id);
        }
        
        this.logger.info('All connections closed');
    }
    
    /**
     * Get the number of active connections
     */
    get size(): number {
        return this.connections.size;
    }
    
    /**
     * Get the number of connections in use
     */
    get activeConnectionCount(): number {
        let count = 0;
        for (const conn of this.connections.values()) {
            if (conn.inUse) count++;
        }
        return count;
    }
    
    /**
     * Get the number of idle connections
     */
    get idleConnectionCount(): number {
        return this.size - this.activeConnectionCount;
    }
}
