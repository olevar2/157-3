/**
 * AI Feature Store Serving API
 * High-performance real-time feature serving for forex trading decisions
 * 
 * Optimized for sub-millisecond response times to support scalping strategies
 * Supports both individual feature queries and batch feature retrieval
 */

import express from 'express';
import Redis from 'ioredis';
import WebSocket from 'ws';
import http from 'http';
import compression from 'compression';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { createHash } from 'crypto';
import { performance } from 'perf_hooks';
import winston from 'winston';

// Types for TypeScript
interface FeatureQuery {
    symbol: string;
    features: string[];
    includeHistory?: boolean;
    includeLags?: boolean;
    timeframe?: string;
}

interface FeatureResponse {
    symbol: string;
    timestamp: string;
    features: Record<string, any>;
    metadata: {
        computation_time_ms: number;
        cache_hit: boolean;
        session: string;
    };
}

interface FeatureSubscription {
    clientId: string;
    symbol: string;
    features: string[];
    updateFrequency: number; // ms
}

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'logs/feature-api-error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/feature-api-combined.log' }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

class FeatureServingAPI {
    private app: express.Application;
    private server: http.Server;
    private wss: WebSocket.Server;
    private redisClient: Redis;
    private redisSubscriber: Redis;
    private featureCache: Map<string, any>;
    private subscriptions: Map<string, FeatureSubscription[]>;
    private performanceMetrics: Map<string, number[]>;

    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.wss = new WebSocket.Server({ server: this.server });
        this.featureCache = new Map();
        this.subscriptions = new Map();
        this.performanceMetrics = new Map();
        
        this.setupRedis();
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
        this.startMetricsCollection();
    }

    private setupRedis(): void {
        // Main Redis client for feature queries
        this.redisClient = new Redis({
            host: 'localhost',
            port: 6379,
            retryDelayOnFailover: 100,
            enableReadyCheck: false,
            maxRetriesPerRequest: 1,
            lazyConnect: true,
            keepAlive: 30000,
            connectTimeout: 1000,
            commandTimeout: 500  // 500ms timeout for ultra-fast responses
        });

        // Separate client for pub/sub
        this.redisSubscriber = new Redis({
            host: 'localhost',
            port: 6379,
            retryDelayOnFailover: 100,
            enableReadyCheck: false,
            maxRetriesPerRequest: 1,
            lazyConnect: true
        });

        // Subscribe to feature updates
        this.redisSubscriber.subscribe('computed-features', (err, count) => {
            if (err) {
                logger.error('Redis subscription error:', err);
            } else {
                logger.info(`Subscribed to ${count} Redis channels`);
            }
        });

        this.redisSubscriber.on('message', (channel, message) => {
            this.handleFeatureUpdate(channel, JSON.parse(message));
        });

        logger.info('Redis connections established for feature serving');
    }

    private setupMiddleware(): void {
        // Security and performance middleware
        this.app.use(helmet());
        this.app.use(compression());
        this.app.use(cors({
            origin: process.env.NODE_ENV === 'production' ? 
                ['https://trading-dashboard.com'] : 
                ['http://localhost:3000', 'http://localhost:5173']
        }));

        // Rate limiting for API protection
        const limiter = rateLimit({
            windowMs: 1000, // 1 second
            max: 1000, // 1000 requests per second (high for trading)
            message: 'Too many requests, please try again later',
            standardHeaders: true,
            legacyHeaders: false
        });
        this.app.use('/api/', limiter);

        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));

        // Performance monitoring middleware
        this.app.use((req, res, next) => {
            req.startTime = performance.now();
            next();
        });

        logger.info('Middleware setup completed');
    }

    private setupRoutes(): void {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                redis_connected: this.redisClient.status === 'ready',
                uptime: process.uptime()
            });
        });

        // Get single feature for a symbol
        this.app.get('/api/features/:symbol/:feature', async (req, res) => {
            try {
                const startTime = performance.now();
                const { symbol, feature } = req.params;
                const includeHistory = req.query.includeHistory === 'true';
                const includeLags = req.query.includeLags === 'true';

                const result = await this.getFeature(symbol, feature, {
                    includeHistory,
                    includeLags
                });

                const responseTime = performance.now() - startTime;
                this.recordMetric('single_feature_query', responseTime);

                res.json({
                    ...result,
                    metadata: {
                        ...result.metadata,
                        response_time_ms: responseTime
                    }
                });

            } catch (error) {
                logger.error('Single feature query error:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        });

        // Get multiple features for a symbol
        this.app.post('/api/features/batch', async (req, res) => {
            try {
                const startTime = performance.now();
                const query: FeatureQuery = req.body;

                const result = await this.getBatchFeatures(query);

                const responseTime = performance.now() - startTime;
                this.recordMetric('batch_feature_query', responseTime);

                res.json({
                    ...result,
                    metadata: {
                        ...result.metadata,
                        response_time_ms: responseTime
                    }
                });

            } catch (error) {
                logger.error('Batch feature query error:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        });

        // Get all available features for a symbol
        this.app.get('/api/features/:symbol', async (req, res) => {
            try {
                const startTime = performance.now();
                const { symbol } = req.params;

                const result = await this.getAllFeatures(symbol);

                const responseTime = performance.now() - startTime;
                this.recordMetric('all_features_query', responseTime);

                res.json({
                    ...result,
                    metadata: {
                        ...result.metadata,
                        response_time_ms: responseTime
                    }
                });

            } catch (error) {
                logger.error('All features query error:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        });

        // Get feature statistics and metadata
        this.app.get('/api/features/:symbol/:feature/stats', async (req, res) => {
            try {
                const { symbol, feature } = req.params;
                const stats = await this.getFeatureStats(symbol, feature);
                res.json(stats);

            } catch (error) {
                logger.error('Feature stats query error:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        });

        // Get performance metrics
        this.app.get('/api/metrics', (req, res) => {
            const metrics = this.getPerformanceMetrics();
            res.json(metrics);
        });

        // Feature definition endpoint
        this.app.get('/api/definitions', (req, res) => {
            // Return feature definitions from YAML config
            res.json({
                categories: [
                    'microstructure',
                    'price_action', 
                    'technical_indicators',
                    'session_based',
                    'sentiment',
                    'correlation',
                    'ml_derived'
                ],
                supported_symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
                timeframes: ['tick', 'm1', 'm5', 'm15', 'h1', 'h4']
            });
        });

        logger.info('API routes setup completed');
    }

    private setupWebSocket(): void {
        this.wss.on('connection', (ws, req) => {
            const clientId = this.generateClientId();
            logger.info(`WebSocket client connected: ${clientId}`);

            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    await this.handleWebSocketMessage(ws, clientId, message);
                } catch (error) {
                    logger.error('WebSocket message error:', error);
                    ws.send(JSON.stringify({ error: 'Invalid message format' }));
                }
            });

            ws.on('close', () => {
                this.cleanup(clientId);
                logger.info(`WebSocket client disconnected: ${clientId}`);
            });

            ws.on('error', (error) => {
                logger.error(`WebSocket error for ${clientId}:`, error);
            });

            // Send welcome message
            ws.send(JSON.stringify({
                type: 'connected',
                clientId,
                timestamp: new Date().toISOString()
            }));
        });

        logger.info('WebSocket server setup completed');
    }

    private async handleWebSocketMessage(ws: WebSocket, clientId: string, message: any): Promise<void> {
        switch (message.type) {
            case 'subscribe':
                await this.handleSubscription(ws, clientId, message.payload);
                break;
            
            case 'unsubscribe':
                this.handleUnsubscription(clientId, message.payload);
                break;
            
            case 'query':
                const result = await this.getBatchFeatures(message.payload);
                ws.send(JSON.stringify({
                    type: 'query_response',
                    requestId: message.requestId,
                    data: result
                }));
                break;
            
            default:
                ws.send(JSON.stringify({ error: 'Unknown message type' }));
        }
    }

    private async handleSubscription(ws: WebSocket, clientId: string, subscription: FeatureSubscription): Promise<void> {
        if (!this.subscriptions.has(clientId)) {
            this.subscriptions.set(clientId, []);
        }

        this.subscriptions.get(clientId)!.push({
            ...subscription,
            clientId
        });

        // Send initial features
        const features = await this.getBatchFeatures({
            symbol: subscription.symbol,
            features: subscription.features
        });

        ws.send(JSON.stringify({
            type: 'feature_update',
            data: features
        }));

        logger.info(`Client ${clientId} subscribed to ${subscription.symbol} features`);
    }

    private handleUnsubscription(clientId: string, payload: any): void {
        const subscriptions = this.subscriptions.get(clientId);
        if (subscriptions) {
            const filtered = subscriptions.filter(sub => 
                !(sub.symbol === payload.symbol && 
                  JSON.stringify(sub.features) === JSON.stringify(payload.features))
            );
            this.subscriptions.set(clientId, filtered);
        }
    }

    private async getFeature(symbol: string, featureName: string, options: any = {}): Promise<any> {
        const cacheKey = `${symbol}:${featureName}`;
        const startTime = performance.now();
        
        try {
            // Check cache first
            if (this.featureCache.has(cacheKey)) {
                const cached = this.featureCache.get(cacheKey);
                if (Date.now() - cached.timestamp < 100) { // 100ms cache TTL
                    return {
                        symbol,
                        feature: featureName,
                        value: cached.value,
                        timestamp: cached.timestamp,
                        metadata: {
                            computation_time_ms: performance.now() - startTime,
                            cache_hit: true,
                            session: this.getCurrentSession()
                        }
                    };
                }
            }

            // Get from Redis
            const featureKey = `features:${symbol}:${featureName}`;
            const featureData = await this.redisClient.hgetall(featureKey);

            if (Object.keys(featureData).length === 0) {
                throw new Error(`Feature ${featureName} not found for ${symbol}`);
            }

            let result = {
                symbol,
                feature: featureName,
                value: parseFloat(featureData.current) || 0,
                timestamp: featureData.timestamp || new Date().toISOString(),
                metadata: {
                    computation_time_ms: performance.now() - startTime,
                    cache_hit: false,
                    session: this.getCurrentSession()
                }
            };

            // Include lag features if requested
            if (options.includeLags) {
                const lagKeys = Object.keys(featureData).filter(key => key.startsWith('lag_'));
                result['lags'] = {};
                lagKeys.forEach(key => {
                    result['lags'][key] = parseFloat(featureData[key]) || 0;
                });
            }

            // Include historical data if requested
            if (options.includeHistory) {
                const historyKey = `history:${symbol}:${featureName}`;
                const history = await this.redisClient.lrange(historyKey, 0, 19); // Last 20 values
                result['history'] = history.map(h => parseFloat(h));
            }

            // Cache the result
            this.featureCache.set(cacheKey, {
                value: result.value,
                timestamp: Date.now()
            });

            return result;

        } catch (error) {
            logger.error(`Error getting feature ${featureName} for ${symbol}:`, error);
            throw error;
        }
    }

    private async getBatchFeatures(query: FeatureQuery): Promise<FeatureResponse> {
        const startTime = performance.now();
        const features: Record<string, any> = {};

        try {
            // Use Redis pipeline for batch operations
            const pipeline = this.redisClient.pipeline();

            query.features.forEach(feature => {
                const featureKey = `features:${query.symbol}:${feature}`;
                pipeline.hgetall(featureKey);
            });

            const results = await pipeline.exec();

            // Process results
            query.features.forEach((feature, index) => {
                const [err, featureData] = results[index];
                if (!err && featureData && Object.keys(featureData).length > 0) {
                    features[feature] = {
                        value: parseFloat(featureData.current) || 0,
                        timestamp: featureData.timestamp
                    };

                    // Include window functions if available
                    Object.keys(featureData).forEach(key => {
                        if (key.startsWith('rolling_') || key.startsWith('exponential_')) {
                            features[feature][key] = parseFloat(featureData[key]) || 0;
                        }
                    });
                }
            });

            return {
                symbol: query.symbol,
                timestamp: new Date().toISOString(),
                features,
                metadata: {
                    computation_time_ms: performance.now() - startTime,
                    cache_hit: false,
                    session: this.getCurrentSession()
                }
            };

        } catch (error) {
            logger.error(`Error getting batch features for ${query.symbol}:`, error);
            throw error;
        }
    }

    private async getAllFeatures(symbol: string): Promise<any> {
        const startTime = performance.now();
        
        try {
            // Get all feature keys for the symbol
            const pattern = `features:${symbol}:*`;
            const keys = await this.redisClient.keys(pattern);

            const pipeline = this.redisClient.pipeline();
            keys.forEach(key => pipeline.hgetall(key));

            const results = await pipeline.exec();
            const features: Record<string, any> = {};

            keys.forEach((key, index) => {
                const [err, featureData] = results[index];
                if (!err && featureData && Object.keys(featureData).length > 0) {
                    const featureName = key.split(':').pop();
                    features[featureName] = {
                        value: parseFloat(featureData.current) || 0,
                        timestamp: featureData.timestamp
                    };
                }
            });

            return {
                symbol,
                timestamp: new Date().toISOString(),
                features,
                feature_count: Object.keys(features).length,
                metadata: {
                    computation_time_ms: performance.now() - startTime,
                    cache_hit: false,
                    session: this.getCurrentSession()
                }
            };

        } catch (error) {
            logger.error(`Error getting all features for ${symbol}:`, error);
            throw error;
        }
    }

    private async getFeatureStats(symbol: string, feature: string): Promise<any> {
        try {
            const historyKey = `history:${symbol}:${feature}`;
            const history = await this.redisClient.lrange(historyKey, 0, 99); // Last 100 values
            
            if (history.length === 0) {
                return { error: 'No historical data available' };
            }

            const values = history.map(h => parseFloat(h)).filter(v => !isNaN(v));
            
            return {
                symbol,
                feature,
                statistics: {
                    count: values.length,
                    mean: values.reduce((a, b) => a + b, 0) / values.length,
                    min: Math.min(...values),
                    max: Math.max(...values),
                    std: this.calculateStandardDeviation(values),
                    last_updated: new Date().toISOString()
                }
            };

        } catch (error) {
            logger.error(`Error getting feature stats for ${symbol}:${feature}:`, error);
            throw error;
        }
    }

    private handleFeatureUpdate(channel: string, message: any): void {
        // Broadcast to subscribed WebSocket clients
        const { symbol, features } = message;

        this.subscriptions.forEach((subscriptions, clientId) => {
            subscriptions.forEach(subscription => {
                if (subscription.symbol === symbol) {
                    // Check if any subscribed features were updated
                    const relevantFeatures = subscription.features.filter(f => features[f]);
                    
                    if (relevantFeatures.length > 0) {
                        this.broadcastToClient(clientId, {
                            type: 'feature_update',
                            data: {
                                symbol,
                                timestamp: message.timestamp,
                                features: Object.keys(features)
                                    .filter(f => relevantFeatures.includes(f))
                                    .reduce((obj, key) => {
                                        obj[key] = features[key];
                                        return obj;
                                    }, {})
                            }
                        });
                    }
                }
            });
        });
    }

    private broadcastToClient(clientId: string, message: any): void {
        this.wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                // Note: In production, you'd want to track client IDs properly
                client.send(JSON.stringify(message));
            }
        });
    }

    private generateClientId(): string {
        return createHash('md5').update(`${Date.now()}-${Math.random()}`).digest('hex').substring(0, 16);
    }

    private getCurrentSession(): string {
        const hour = new Date().getHours();
        if (hour >= 21 || hour < 6) return 'asian';
        if (hour >= 7 && hour < 16) return 'london';
        if (hour >= 12 && hour < 21) return 'newyork';
        return 'overlap';
    }

    private recordMetric(operation: string, duration: number): void {
        if (!this.performanceMetrics.has(operation)) {
            this.performanceMetrics.set(operation, []);
        }
        
        const metrics = this.performanceMetrics.get(operation)!;
        metrics.push(duration);
        
        // Keep only last 1000 measurements
        if (metrics.length > 1000) {
            metrics.shift();
        }
    }

    private getPerformanceMetrics(): any {
        const metrics: any = {};
        
        this.performanceMetrics.forEach((durations, operation) => {
            if (durations.length > 0) {
                metrics[operation] = {
                    count: durations.length,
                    avg_ms: durations.reduce((a, b) => a + b, 0) / durations.length,
                    min_ms: Math.min(...durations),
                    max_ms: Math.max(...durations),
                    p95_ms: this.calculatePercentile(durations, 0.95),
                    p99_ms: this.calculatePercentile(durations, 0.99)
                };
            }
        });

        return {
            timestamp: new Date().toISOString(),
            metrics,
            cache_size: this.featureCache.size,
            active_subscriptions: Array.from(this.subscriptions.values()).reduce((a, b) => a + b.length, 0),
            websocket_clients: this.wss.clients.size
        };
    }

    private calculatePercentile(values: number[], percentile: number): number {
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.ceil(sorted.length * percentile) - 1;
        return sorted[index];
    }

    private calculateStandardDeviation(values: number[]): number {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    private cleanup(clientId: string): void {
        this.subscriptions.delete(clientId);
    }

    private startMetricsCollection(): void {
        // Clear old cache entries every 30 seconds
        setInterval(() => {
            const now = Date.now();
            this.featureCache.forEach((value, key) => {
                if (now - value.timestamp > 30000) { // 30 second TTL
                    this.featureCache.delete(key);
                }
            });
        }, 30000);

        // Log performance metrics every minute
        setInterval(() => {
            const metrics = this.getPerformanceMetrics();
            logger.info('Performance metrics:', metrics);
        }, 60000);
    }

    public start(port: number = 3001): void {
        this.server.listen(port, () => {
            logger.info(`Feature Serving API started on port ${port}`);
            logger.info(`WebSocket endpoint: ws://localhost:${port}`);
            logger.info(`Health check: http://localhost:${port}/health`);
        });

        // Graceful shutdown
        process.on('SIGTERM', () => {
            logger.info('Shutting down Feature Serving API...');
            this.server.close(() => {
                this.redisClient.disconnect();
                this.redisSubscriber.disconnect();
                logger.info('Feature Serving API shutdown complete');
                process.exit(0);
            });
        });
    }
}

// Start the server
const api = new FeatureServingAPI();
api.start(3001);
