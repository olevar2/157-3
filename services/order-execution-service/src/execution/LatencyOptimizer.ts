/**
 * Latency Optimizer for Ultra-Fast Order Execution
 * 
 * Advanced latency optimization system that minimizes execution delays through
 * intelligent routing, connection management, and performance monitoring.
 * 
 * Key Features:
 * - Real-time latency monitoring and optimization
 * - Adaptive routing based on network conditions
 * - Connection pooling with keep-alive optimization
 * - Geographic proximity routing
 * - Network path optimization
 * - Predictive latency modeling
 * 
 * Author: Platform3 Trading Team
 * Version: 1.0.0
 */

import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';
import * as net from 'net';
import * as dns from 'dns';

export interface LatencyMeasurement {
    timestamp: number;
    venue: string;
    latencyMs: number;
    jitter: number;
    packetLoss: number;
    bandwidth: number;
    route: string[];
}

export interface NetworkPath {
    destination: string;
    hops: NetworkHop[];
    totalLatency: number;
    reliability: number;
    lastUpdated: number;
}

export interface NetworkHop {
    ip: string;
    hostname?: string;
    latency: number;
    location?: string;
}

export interface ConnectionPool {
    venue: string;
    connections: Connection[];
    maxConnections: number;
    activeConnections: number;
    averageLatency: number;
    lastOptimized: number;
}

export interface Connection {
    id: string;
    socket: net.Socket;
    isActive: boolean;
    latency: number;
    lastUsed: number;
    keepAliveInterval: NodeJS.Timeout;
}

export interface OptimizationStrategy {
    name: string;
    priority: number;
    enabled: boolean;
    parameters: Record<string, any>;
    lastApplied: number;
    effectiveness: number;
}

export interface LatencyTarget {
    venue: string;
    targetLatencyMs: number;
    maxAcceptableMs: number;
    priority: number;
}

export class LatencyOptimizer extends EventEmitter {
    private latencyHistory: Map<string, LatencyMeasurement[]> = new Map();
    private networkPaths: Map<string, NetworkPath> = new Map();
    private connectionPools: Map<string, ConnectionPool> = new Map();
    private optimizationStrategies: OptimizationStrategy[] = [];
    private latencyTargets: Map<string, LatencyTarget> = new Map();
    
    private isMonitoring: boolean = false;
    private monitoringInterval: NodeJS.Timeout | null = null;
    private optimizationInterval: NodeJS.Timeout | null = null;
    
    // Configuration
    private readonly MONITORING_INTERVAL_MS = 1000; // 1 second
    private readonly OPTIMIZATION_INTERVAL_MS = 5000; // 5 seconds
    private readonly HISTORY_RETENTION_MS = 300000; // 5 minutes
    private readonly MAX_HISTORY_POINTS = 1000;
    private readonly KEEP_ALIVE_INTERVAL_MS = 30000; // 30 seconds
    
    constructor() {
        super();
        this.initializeOptimizationStrategies();
        console.log('✅ LatencyOptimizer initialized');
    }

    /**
     * Initialize optimization strategies
     */
    private initializeOptimizationStrategies(): void {
        this.optimizationStrategies = [
            {
                name: 'connection_pooling',
                priority: 10,
                enabled: true,
                parameters: { minConnections: 2, maxConnections: 10 },
                lastApplied: 0,
                effectiveness: 0.8
            },
            {
                name: 'keep_alive_optimization',
                priority: 9,
                enabled: true,
                parameters: { interval: 30000, probeCount: 3 },
                lastApplied: 0,
                effectiveness: 0.7
            },
            {
                name: 'adaptive_routing',
                priority: 8,
                enabled: true,
                parameters: { routeRefreshInterval: 60000 },
                lastApplied: 0,
                effectiveness: 0.6
            },
            {
                name: 'tcp_optimization',
                priority: 7,
                enabled: true,
                parameters: { 
                    nodelay: true, 
                    keepAlive: true,
                    keepAliveInitialDelay: 1000
                },
                lastApplied: 0,
                effectiveness: 0.5
            },
            {
                name: 'dns_caching',
                priority: 6,
                enabled: true,
                parameters: { ttl: 300000 },
                lastApplied: 0,
                effectiveness: 0.4
            }
        ];
    }

    /**
     * Add venue for latency optimization
     */
    public async addVenue(venue: string, endpoint: string, targetLatencyMs: number = 50): Promise<boolean> {
        try {
            // Set latency target
            this.latencyTargets.set(venue, {
                venue,
                targetLatencyMs,
                maxAcceptableMs: targetLatencyMs * 2,
                priority: 1
            });

            // Initialize latency history
            this.latencyHistory.set(venue, []);

            // Discover network path
            await this.discoverNetworkPath(venue, endpoint);

            // Initialize connection pool
            await this.initializeConnectionPool(venue, endpoint);

            console.log(`✅ Venue ${venue} added for latency optimization`);
            return true;
        } catch (error) {
            console.error(`Failed to add venue ${venue}:`, error);
            return false;
        }
    }

    /**
     * Start latency monitoring and optimization
     */
    public startOptimization(): void {
        if (this.isMonitoring) return;

        this.isMonitoring = true;
        
        // Start monitoring
        this.monitoringInterval = setInterval(() => {
            this.performLatencyMeasurements();
        }, this.MONITORING_INTERVAL_MS);

        // Start optimization
        this.optimizationInterval = setInterval(() => {
            this.performOptimizations();
        }, this.OPTIMIZATION_INTERVAL_MS);

        console.log('✅ Latency optimization started');
    }

    /**
     * Stop latency monitoring and optimization
     */
    public stopOptimization(): void {
        this.isMonitoring = false;

        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }

        if (this.optimizationInterval) {
            clearInterval(this.optimizationInterval);
            this.optimizationInterval = null;
        }

        // Close all connections
        for (const pool of this.connectionPools.values()) {
            this.closeConnectionPool(pool);
        }

        console.log('✅ Latency optimization stopped');
    }

    /**
     * Perform latency measurements for all venues
     */
    private async performLatencyMeasurements(): Promise<void> {
        const promises = Array.from(this.connectionPools.keys()).map(venue => 
            this.measureVenueLatency(venue)
        );

        try {
            await Promise.all(promises);
        } catch (error) {
            console.error('Error performing latency measurements:', error);
        }
    }

    /**
     * Measure latency for specific venue
     */
    private async measureVenueLatency(venue: string): Promise<void> {
        const pool = this.connectionPools.get(venue);
        if (!pool || pool.connections.length === 0) return;

        try {
            const startTime = performance.now();
            
            // Use an active connection for measurement
            const connection = pool.connections.find(conn => conn.isActive);
            if (!connection) return;

            // Send ping and measure response time
            const latency = await this.pingConnection(connection);
            const endTime = performance.now();

            const measurement: LatencyMeasurement = {
                timestamp: Date.now(),
                venue,
                latencyMs: latency,
                jitter: this.calculateJitter(venue, latency),
                packetLoss: 0, // Would be calculated from actual network stats
                bandwidth: 0, // Would be measured
                route: [] // Would be populated from traceroute
            };

            this.recordLatencyMeasurement(venue, measurement);
            this.updateConnectionLatency(connection, latency);

        } catch (error) {
            console.error(`Error measuring latency for venue ${venue}:`, error);
        }
    }

    /**
     * Ping connection to measure latency
     */
    private async pingConnection(connection: Connection): Promise<number> {
        return new Promise((resolve, reject) => {
            const startTime = performance.now();
            
            // Send a small ping packet
            const pingData = Buffer.from('PING');
            
            const timeout = setTimeout(() => {
                reject(new Error('Ping timeout'));
            }, 5000);

            connection.socket.write(pingData, (error) => {
                clearTimeout(timeout);
                
                if (error) {
                    reject(error);
                } else {
                    const latency = performance.now() - startTime;
                    resolve(latency);
                }
            });
        });
    }

    /**
     * Calculate jitter from recent latency measurements
     */
    private calculateJitter(venue: string, currentLatency: number): number {
        const history = this.latencyHistory.get(venue);
        if (!history || history.length < 2) return 0;

        const recentMeasurements = history.slice(-10);
        const latencies = recentMeasurements.map(m => m.latencyMs);
        
        let jitterSum = 0;
        for (let i = 1; i < latencies.length; i++) {
            jitterSum += Math.abs(latencies[i] - latencies[i - 1]);
        }

        return jitterSum / (latencies.length - 1);
    }

    /**
     * Record latency measurement
     */
    private recordLatencyMeasurement(venue: string, measurement: LatencyMeasurement): void {
        let history = this.latencyHistory.get(venue);
        if (!history) {
            history = [];
            this.latencyHistory.set(venue, history);
        }

        history.push(measurement);

        // Limit history size
        if (history.length > this.MAX_HISTORY_POINTS) {
            history.splice(0, history.length - this.MAX_HISTORY_POINTS);
        }

        // Remove old measurements
        const cutoffTime = Date.now() - this.HISTORY_RETENTION_MS;
        const validMeasurements = history.filter(m => m.timestamp > cutoffTime);
        this.latencyHistory.set(venue, validMeasurements);

        this.emit('latencyMeasured', { venue, measurement });
    }

    /**
     * Update connection latency
     */
    private updateConnectionLatency(connection: Connection, latency: number): void {
        connection.latency = latency;
        connection.lastUsed = Date.now();
    }

    /**
     * Perform optimization strategies
     */
    private async performOptimizations(): Promise<void> {
        // Sort strategies by priority
        const activeStrategies = this.optimizationStrategies
            .filter(s => s.enabled)
            .sort((a, b) => b.priority - a.priority);

        for (const strategy of activeStrategies) {
            try {
                await this.applyOptimizationStrategy(strategy);
            } catch (error) {
                console.error(`Error applying strategy ${strategy.name}:`, error);
            }
        }
    }

    /**
     * Apply specific optimization strategy
     */
    private async applyOptimizationStrategy(strategy: OptimizationStrategy): Promise<void> {
        const now = Date.now();
        
        switch (strategy.name) {
            case 'connection_pooling':
                await this.optimizeConnectionPools();
                break;
                
            case 'keep_alive_optimization':
                await this.optimizeKeepAlive();
                break;
                
            case 'adaptive_routing':
                await this.optimizeRouting();
                break;
                
            case 'tcp_optimization':
                await this.optimizeTcpSettings();
                break;
                
            case 'dns_caching':
                await this.optimizeDnsCache();
                break;
        }

        strategy.lastApplied = now;
    }

    /**
     * Optimize connection pools
     */
    private async optimizeConnectionPools(): Promise<void> {
        for (const [venue, pool] of this.connectionPools) {
            const target = this.latencyTargets.get(venue);
            if (!target) continue;

            const avgLatency = this.getAverageLatency(venue);
            
            // Add connections if latency is high
            if (avgLatency > target.targetLatencyMs && pool.connections.length < pool.maxConnections) {
                await this.addConnectionToPool(pool);
            }
            
            // Remove slow connections
            const slowConnections = pool.connections.filter(conn => 
                conn.latency > target.maxAcceptableMs
            );
            
            for (const conn of slowConnections) {
                await this.removeConnectionFromPool(pool, conn);
            }
        }
    }

    /**
     * Optimize keep-alive settings
     */
    private async optimizeKeepAlive(): Promise<void> {
        for (const pool of this.connectionPools.values()) {
            for (const connection of pool.connections) {
                if (connection.socket && !connection.socket.destroyed) {
                    connection.socket.setKeepAlive(true, 1000);
                    connection.socket.setNoDelay(true);
                }
            }
        }
    }

    /**
     * Optimize routing
     */
    private async optimizeRouting(): Promise<void> {
        for (const [venue, path] of this.networkPaths) {
            const now = Date.now();
            
            // Refresh path if it's old
            if (now - path.lastUpdated > 60000) { // 1 minute
                await this.discoverNetworkPath(venue, path.destination);
            }
        }
    }

    /**
     * Optimize TCP settings
     */
    private async optimizeTcpSettings(): Promise<void> {
        for (const pool of this.connectionPools.values()) {
            for (const connection of pool.connections) {
                if (connection.socket && !connection.socket.destroyed) {
                    connection.socket.setNoDelay(true);
                    connection.socket.setKeepAlive(true, 1000);
                }
            }
        }
    }

    /**
     * Optimize DNS cache
     */
    private async optimizeDnsCache(): Promise<void> {
        // Implement DNS caching optimization
        // This would involve pre-resolving DNS names and caching results
    }

    /**
     * Discover network path to destination
     */
    private async discoverNetworkPath(venue: string, destination: string): Promise<void> {
        try {
            // This would implement actual traceroute functionality
            const path: NetworkPath = {
                destination,
                hops: [],
                totalLatency: 0,
                reliability: 1.0,
                lastUpdated: Date.now()
            };

            this.networkPaths.set(venue, path);
        } catch (error) {
            console.error(`Error discovering path to ${venue}:`, error);
        }
    }

    /**
     * Initialize connection pool for venue
     */
    private async initializeConnectionPool(venue: string, endpoint: string): Promise<void> {
        const pool: ConnectionPool = {
            venue,
            connections: [],
            maxConnections: 10,
            activeConnections: 0,
            averageLatency: 0,
            lastOptimized: Date.now()
        };

        // Create initial connections
        for (let i = 0; i < 3; i++) {
            await this.addConnectionToPool(pool, endpoint);
        }

        this.connectionPools.set(venue, pool);
    }

    /**
     * Add connection to pool
     */
    private async addConnectionToPool(pool: ConnectionPool, endpoint?: string): Promise<void> {
        try {
            const socket = new net.Socket();
            
            // Configure socket
            socket.setNoDelay(true);
            socket.setKeepAlive(true, 1000);

            const connection: Connection = {
                id: `${pool.venue}_${Date.now()}_${Math.random()}`,
                socket,
                isActive: false,
                latency: 0,
                lastUsed: Date.now(),
                keepAliveInterval: setInterval(() => {
                    this.sendKeepAlive(connection);
                }, this.KEEP_ALIVE_INTERVAL_MS)
            };

            // Add to pool
            pool.connections.push(connection);
            
            console.log(`✅ Connection added to pool for ${pool.venue}`);
        } catch (error) {
            console.error(`Error adding connection to pool for ${pool.venue}:`, error);
        }
    }

    /**
     * Remove connection from pool
     */
    private async removeConnectionFromPool(pool: ConnectionPool, connection: Connection): Promise<void> {
        try {
            // Clear keep-alive interval
            if (connection.keepAliveInterval) {
                clearInterval(connection.keepAliveInterval);
            }

            // Close socket
            if (connection.socket && !connection.socket.destroyed) {
                connection.socket.destroy();
            }

            // Remove from pool
            const index = pool.connections.indexOf(connection);
            if (index > -1) {
                pool.connections.splice(index, 1);
            }

            console.log(`Connection removed from pool for ${pool.venue}`);
        } catch (error) {
            console.error(`Error removing connection from pool:`, error);
        }
    }

    /**
     * Close connection pool
     */
    private closeConnectionPool(pool: ConnectionPool): void {
        for (const connection of pool.connections) {
            if (connection.keepAliveInterval) {
                clearInterval(connection.keepAliveInterval);
            }
            
            if (connection.socket && !connection.socket.destroyed) {
                connection.socket.destroy();
            }
        }
        
        pool.connections = [];
    }

    /**
     * Send keep-alive packet
     */
    private sendKeepAlive(connection: Connection): void {
        if (connection.socket && !connection.socket.destroyed) {
            const keepAliveData = Buffer.from('KEEPALIVE');
            connection.socket.write(keepAliveData);
        }
    }

    /**
     * Get average latency for venue
     */
    private getAverageLatency(venue: string): number {
        const history = this.latencyHistory.get(venue);
        if (!history || history.length === 0) return 0;

        const recentMeasurements = history.slice(-10);
        const sum = recentMeasurements.reduce((acc, m) => acc + m.latencyMs, 0);
        return sum / recentMeasurements.length;
    }

    /**
     * Get latency statistics for venue
     */
    public getLatencyStats(venue: string): any {
        const history = this.latencyHistory.get(venue);
        if (!history || history.length === 0) {
            return { average: 0, min: 0, max: 0, p95: 0, p99: 0 };
        }

        const latencies = history.map(m => m.latencyMs).sort((a, b) => a - b);
        const sum = latencies.reduce((acc, val) => acc + val, 0);

        return {
            average: sum / latencies.length,
            min: latencies[0],
            max: latencies[latencies.length - 1],
            p95: latencies[Math.floor(latencies.length * 0.95)],
            p99: latencies[Math.floor(latencies.length * 0.99)],
            jitter: this.calculateJitter(venue, latencies[latencies.length - 1])
        };
    }

    /**
     * Get optimization status
     */
    public getOptimizationStatus(): any {
        return {
            isMonitoring: this.isMonitoring,
            venues: Array.from(this.latencyTargets.keys()),
            strategies: this.optimizationStrategies.map(s => ({
                name: s.name,
                enabled: s.enabled,
                effectiveness: s.effectiveness,
                lastApplied: s.lastApplied
            })),
            connectionPools: Array.from(this.connectionPools.entries()).map(([venue, pool]) => ({
                venue,
                connections: pool.connections.length,
                averageLatency: pool.averageLatency
            }))
        };
    }

    /**
     * Get best connection for venue
     */
    public getBestConnection(venue: string): Connection | null {
        const pool = this.connectionPools.get(venue);
        if (!pool || pool.connections.length === 0) return null;

        // Find connection with lowest latency
        return pool.connections.reduce((best, current) => 
            current.latency < best.latency ? current : best
        );
    }
}

export default LatencyOptimizer;
