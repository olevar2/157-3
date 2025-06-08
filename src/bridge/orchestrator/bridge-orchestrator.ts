import { EventEmitter } from 'events';
import { BridgeMessage, ChannelConfig, BridgeMetrics } from '../types';
import { DualChannelWebSocket } from '../channels/dual-channel.websocket';
import { ConnectionPool } from './connection-pool';
import { AutoRecoveryManager } from './auto-recovery-manager';
import { HealthMonitor } from './health-monitor';
import { CircuitBreaker } from './circuit-breaker';
import { Logger } from '../../shared/logging';

/**
 * BridgeOrchestrator
 * 
 * Central management component for the TypeScript-Python bridge.
 * Manages connection pools, recovery, health monitoring, and orchestrates
 * all cross-language communication with production-grade reliability.
 */
export class BridgeOrchestrator extends EventEmitter {
    private connectionPool: ConnectionPool;
    private recoveryManager: AutoRecoveryManager;
    private healthMonitor: HealthMonitor;
    private circuitBreakers: Map<string, CircuitBreaker> = new Map();
    private logger: Logger;
    private metrics: BridgeMetrics;

    constructor(
        private configs: ChannelConfig[],
        private poolConfig = { minSize: 2, maxSize: 10, idleTimeout: 30000 }
    ) {
        super();
        this.logger = Logger.getLogger('BridgeOrchestrator');
        
        // Initialize metrics
        this.metrics = {
            latency: new Map(),
            errors: new Map(),
            throughput: { in: 0, out: 0 },
            uptime: 0
        };

        this.initialize();
    }

    private initialize(): void {
        // Initialize connection pool
        this.connectionPool = new ConnectionPool(this.configs, this.poolConfig);
        
        // Initialize recovery manager
        this.recoveryManager = new AutoRecoveryManager(this.connectionPool);
        
        // Initialize health monitor
        this.healthMonitor = new HealthMonitor(this, this.connectionPool);
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start health monitoring
        this.healthMonitor.startMonitoring();
        
        // Log initialization
        this.logger.info('Bridge orchestrator initialized with pool size: ' + 
            `min=${this.poolConfig.minSize}, max=${this.poolConfig.maxSize}`);
    }
    
    private setupEventListeners(): void {
        this.connectionPool.on('connection:created', (id) => {
            this.logger.info(`Connection created: ${id}`);
            this.emit('connection:created', id);
        });
        
        this.connectionPool.on('connection:closed', (id) => {
            this.logger.info(`Connection closed: ${id}`);
            this.emit('connection:closed', id);
        });
        
        this.connectionPool.on('connection:error', (id, error) => {
            this.logger.error(`Connection error: ${id}`, error);
            this.recordError('connection');
            this.emit('connection:error', { id, error });
        });
        
        this.recoveryManager.on('recovery:initiated', (details) => {
            this.logger.warn('Recovery initiated', details);
            this.emit('recovery:initiated', details);
        });
        
        this.recoveryManager.on('recovery:completed', (details) => {
            this.logger.info('Recovery completed', details);
            this.emit('recovery:completed', details);
        });
        
        this.healthMonitor.on('health:alert', (alert) => {
            this.logger.warn('Health alert', alert);
            this.emit('health:alert', alert);
        });
    }

    /**
     * Send a message through the bridge
     */
    async sendMessage(message: BridgeMessage): Promise<void> {
        const operationId = `send-${message.id}`;
        const startTime = Date.now();
        
        // Check circuit breaker for message type
        const circuitBreaker = this.getOrCreateCircuitBreaker(message.type);
        if (circuitBreaker.isOpen()) {
            this.recordError('circuit-open');
            throw new Error(`Circuit open for message type: ${message.type}`);
        }
        
        try {
            // Get connection from pool
            const connection = await this.connectionPool.getConnection();
            
            // Add timestamp if not present
            if (!message.timestamp) {
                message.timestamp = Date.now();
            }
            
            // Send message
            await connection.send(message);
            
            // Return connection to pool
            this.connectionPool.releaseConnection(connection);
            
            // Record metrics
            this.recordLatency('send', Date.now() - startTime);
            this.metrics.throughput.out++;
            
            // Record success for circuit breaker
            circuitBreaker.recordSuccess();
            
        } catch (error) {
            // Record error
            this.recordError('send');
            
            // Record failure for circuit breaker
            circuitBreaker.recordFailure();
            
            // Log error
            this.logger.error(`Failed to send message ${message.id}`, error);
            
            // Re-throw error
            throw error;
        }
    }
    
    /**
     * Register a message handler
     */
    onMessage(handler: (message: BridgeMessage) => void): void {
        this.connectionPool.on('message', (message) => {
            try {
                // Record metrics
                this.metrics.throughput.in++;
                
                // Execute handler
                handler(message);
            } catch (error) {
                this.logger.error('Error in message handler', error);
                this.recordError('handler');
            }
        });
    }
    
    /**
     * Get bridge health status
     */
    getHealth(): { status: 'healthy' | 'degraded' | 'unhealthy', details: any } {
        return this.healthMonitor.getHealth();
    }
    
    /**
     * Get bridge metrics
     */
    getMetrics(): BridgeMetrics {
        // Update uptime
        this.metrics.uptime = process.uptime();
        return this.metrics;
    }

    /**
     * Record latency for an operation
     */
    private recordLatency(operation: string, latency: number): void {
        if (!this.metrics.latency.has(operation)) {
            this.metrics.latency.set(operation, []);
        }
        
        const latencies = this.metrics.latency.get(operation)!;
        latencies.push(latency);
        
        // Keep only the last 100 measurements
        if (latencies.length > 100) {
            latencies.shift();
        }
    }
    
    /**
     * Record error for an operation
     */
    private recordError(operation: string): void {
        if (!this.metrics.errors.has(operation)) {
            this.metrics.errors.set(operation, 0);
        }
        
        const currentCount = this.metrics.errors.get(operation)!;
        this.metrics.errors.set(operation, currentCount + 1);
    }
    
    /**
     * Get or create circuit breaker for a specific message type
     */
    private getOrCreateCircuitBreaker(messageType: string): CircuitBreaker {
        if (!this.circuitBreakers.has(messageType)) {
            this.circuitBreakers.set(
                messageType, 
                new CircuitBreaker({
                    failureThreshold: 5,
                    resetTimeout: 30000,
                    rollingCountWindow: 10000,
                    name: `bridge-${messageType}`
                })
            );
        }
        
        return this.circuitBreakers.get(messageType)!;
    }
    
    /**
     * Shutdown the bridge orchestrator
     */
    async shutdown(): Promise<void> {
        this.logger.info('Shutting down bridge orchestrator');
        
        // Stop health monitoring
        this.healthMonitor.stopMonitoring();
        
        // Close all connections
        await this.connectionPool.closeAll();
        
        this.logger.info('Bridge orchestrator shutdown complete');
    }
}
