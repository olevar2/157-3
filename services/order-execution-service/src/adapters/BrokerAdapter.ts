/**
 * Base Broker Adapter Interface
 * 
 * Abstract base class for all broker integrations providing a unified
 * interface for order execution, market data, and account management
 * across multiple brokers and liquidity providers.
 * 
 * Key Features:
 * - Unified broker interface abstraction
 * - Standardized order management
 * - Real-time market data streaming
 * - Account and position management
 * - Error handling and reconnection logic
 * - Performance monitoring and metrics
 * 
 * Author: Platform3 Trading Team
 * Version: 1.0.0
 */

import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';

export enum BrokerType {
    FOREX = 'FOREX',
    CFD = 'CFD',
    CRYPTO = 'CRYPTO',
    FUTURES = 'FUTURES',
    STOCKS = 'STOCKS'
}

export enum ConnectionStatus {
    DISCONNECTED = 'DISCONNECTED',
    CONNECTING = 'CONNECTING',
    CONNECTED = 'CONNECTED',
    RECONNECTING = 'RECONNECTING',
    ERROR = 'ERROR'
}

export enum OrderType {
    MARKET = 'MARKET',
    LIMIT = 'LIMIT',
    STOP = 'STOP',
    STOP_LIMIT = 'STOP_LIMIT',
    IOC = 'IOC',
    FOK = 'FOK'
}

export enum OrderSide {
    BUY = 'BUY',
    SELL = 'SELL'
}

export enum OrderStatus {
    PENDING = 'PENDING',
    SUBMITTED = 'SUBMITTED',
    PARTIALLY_FILLED = 'PARTIALLY_FILLED',
    FILLED = 'FILLED',
    CANCELLED = 'CANCELLED',
    REJECTED = 'REJECTED',
    EXPIRED = 'EXPIRED'
}

export interface BrokerConfig {
    brokerId: string;
    brokerName: string;
    brokerType: BrokerType;
    apiEndpoint: string;
    apiKey: string;
    apiSecret: string;
    sandbox: boolean;
    maxRetries: number;
    retryDelay: number;
    heartbeatInterval: number;
    requestTimeout: number;
    rateLimits: {
        ordersPerSecond: number;
        requestsPerMinute: number;
    };
}

export interface OrderRequest {
    clientOrderId: string;
    symbol: string;
    side: OrderSide;
    type: OrderType;
    quantity: number;
    price?: number;
    stopPrice?: number;
    timeInForce?: string;
    metadata?: Record<string, any>;
}

export interface OrderResponse {
    brokerOrderId: string;
    clientOrderId: string;
    status: OrderStatus;
    symbol: string;
    side: OrderSide;
    type: OrderType;
    quantity: number;
    executedQuantity: number;
    remainingQuantity: number;
    price?: number;
    executedPrice?: number;
    timestamp: number;
    fees?: number;
    errorMessage?: string;
}

export interface MarketDataTick {
    symbol: string;
    bid: number;
    ask: number;
    bidSize: number;
    askSize: number;
    timestamp: number;
    spread: number;
}

export interface Position {
    symbol: string;
    side: 'LONG' | 'SHORT';
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
    realizedPnL: number;
    timestamp: number;
}

export interface AccountInfo {
    accountId: string;
    balance: number;
    equity: number;
    margin: number;
    freeMargin: number;
    marginLevel: number;
    currency: string;
    leverage: number;
    timestamp: number;
}

export interface BrokerMetrics {
    connectionUptime: number;
    totalOrders: number;
    successfulOrders: number;
    rejectedOrders: number;
    averageLatency: number;
    lastHeartbeat: number;
    errorCount: number;
    reconnectionCount: number;
}

export abstract class BrokerAdapter extends EventEmitter {
    protected config: BrokerConfig;
    protected connectionStatus: ConnectionStatus = ConnectionStatus.DISCONNECTED;
    protected metrics: BrokerMetrics;
    protected lastHeartbeat: number = 0;
    protected reconnectAttempts: number = 0;
    protected heartbeatInterval?: NodeJS.Timeout;
    protected rateLimitQueue: Array<{ timestamp: number; type: string }> = [];
    
    constructor(config: BrokerConfig) {
        super();
        this.config = config;
        this.initializeMetrics();
        
        console.log(`✅ BrokerAdapter initialized for ${config.brokerName}`);
    }

    /**
     * Initialize broker metrics
     */
    private initializeMetrics(): void {
        this.metrics = {
            connectionUptime: 0,
            totalOrders: 0,
            successfulOrders: 0,
            rejectedOrders: 0,
            averageLatency: 0,
            lastHeartbeat: 0,
            errorCount: 0,
            reconnectionCount: 0
        };
    }

    /**
     * Connect to broker
     */
    public async connect(): Promise<boolean> {
        try {
            this.connectionStatus = ConnectionStatus.CONNECTING;
            this.emit('connectionStatusChanged', this.connectionStatus);
            
            const connected = await this.doConnect();
            
            if (connected) {
                this.connectionStatus = ConnectionStatus.CONNECTED;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
                this.emit('connected');
            } else {
                this.connectionStatus = ConnectionStatus.ERROR;
                this.emit('connectionError', 'Failed to connect');
            }
            
            this.emit('connectionStatusChanged', this.connectionStatus);
            return connected;
            
        } catch (error) {
            this.connectionStatus = ConnectionStatus.ERROR;
            this.metrics.errorCount++;
            this.emit('connectionError', error);
            this.emit('connectionStatusChanged', this.connectionStatus);
            return false;
        }
    }

    /**
     * Disconnect from broker
     */
    public async disconnect(): Promise<void> {
        try {
            this.stopHeartbeat();
            await this.doDisconnect();
            this.connectionStatus = ConnectionStatus.DISCONNECTED;
            this.emit('disconnected');
            this.emit('connectionStatusChanged', this.connectionStatus);
            
        } catch (error) {
            this.metrics.errorCount++;
            this.emit('error', error);
        }
    }

    /**
     * Submit order to broker
     */
    public async submitOrder(orderRequest: OrderRequest): Promise<OrderResponse> {
        const startTime = performance.now();
        
        try {
            // Check connection
            if (this.connectionStatus !== ConnectionStatus.CONNECTED) {
                throw new Error('Broker not connected');
            }
            
            // Check rate limits
            if (!this.checkRateLimit('order')) {
                throw new Error('Rate limit exceeded');
            }
            
            // Validate order
            this.validateOrder(orderRequest);
            
            // Submit order
            const response = await this.doSubmitOrder(orderRequest);
            
            // Update metrics
            this.metrics.totalOrders++;
            if (response.status !== OrderStatus.REJECTED) {
                this.metrics.successfulOrders++;
            } else {
                this.metrics.rejectedOrders++;
            }
            
            const latency = performance.now() - startTime;
            this.updateLatencyMetrics(latency);
            
            this.emit('orderResponse', response);
            return response;
            
        } catch (error) {
            this.metrics.rejectedOrders++;
            this.metrics.errorCount++;
            
            const errorResponse: OrderResponse = {
                brokerOrderId: '',
                clientOrderId: orderRequest.clientOrderId,
                status: OrderStatus.REJECTED,
                symbol: orderRequest.symbol,
                side: orderRequest.side,
                type: orderRequest.type,
                quantity: orderRequest.quantity,
                executedQuantity: 0,
                remainingQuantity: orderRequest.quantity,
                timestamp: Date.now(),
                errorMessage: error.message
            };
            
            this.emit('orderError', errorResponse);
            return errorResponse;
        }
    }

    /**
     * Cancel order
     */
    public async cancelOrder(brokerOrderId: string): Promise<boolean> {
        try {
            if (this.connectionStatus !== ConnectionStatus.CONNECTED) {
                throw new Error('Broker not connected');
            }
            
            if (!this.checkRateLimit('cancel')) {
                throw new Error('Rate limit exceeded');
            }
            
            const success = await this.doCancelOrder(brokerOrderId);
            this.emit('orderCancelled', { brokerOrderId, success });
            
            return success;
            
        } catch (error) {
            this.metrics.errorCount++;
            this.emit('error', error);
            return false;
        }
    }

    /**
     * Get account information
     */
    public async getAccountInfo(): Promise<AccountInfo | null> {
        try {
            if (this.connectionStatus !== ConnectionStatus.CONNECTED) {
                throw new Error('Broker not connected');
            }
            
            if (!this.checkRateLimit('account')) {
                throw new Error('Rate limit exceeded');
            }
            
            return await this.doGetAccountInfo();
            
        } catch (error) {
            this.metrics.errorCount++;
            this.emit('error', error);
            return null;
        }
    }

    /**
     * Get positions
     */
    public async getPositions(): Promise<Position[]> {
        try {
            if (this.connectionStatus !== ConnectionStatus.CONNECTED) {
                throw new Error('Broker not connected');
            }
            
            if (!this.checkRateLimit('positions')) {
                throw new Error('Rate limit exceeded');
            }
            
            return await this.doGetPositions();
            
        } catch (error) {
            this.metrics.errorCount++;
            this.emit('error', error);
            return [];
        }
    }

    /**
     * Subscribe to market data
     */
    public async subscribeMarketData(symbols: string[]): Promise<boolean> {
        try {
            if (this.connectionStatus !== ConnectionStatus.CONNECTED) {
                throw new Error('Broker not connected');
            }
            
            const success = await this.doSubscribeMarketData(symbols);
            if (success) {
                this.emit('marketDataSubscribed', symbols);
            }
            
            return success;
            
        } catch (error) {
            this.metrics.errorCount++;
            this.emit('error', error);
            return false;
        }
    }

    /**
     * Unsubscribe from market data
     */
    public async unsubscribeMarketData(symbols: string[]): Promise<boolean> {
        try {
            const success = await this.doUnsubscribeMarketData(symbols);
            if (success) {
                this.emit('marketDataUnsubscribed', symbols);
            }
            
            return success;
            
        } catch (error) {
            this.metrics.errorCount++;
            this.emit('error', error);
            return false;
        }
    }

    /**
     * Get broker metrics
     */
    public getMetrics(): BrokerMetrics {
        return { ...this.metrics };
    }

    /**
     * Get connection status
     */
    public getConnectionStatus(): ConnectionStatus {
        return this.connectionStatus;
    }

    /**
     * Get broker configuration
     */
    public getConfig(): BrokerConfig {
        return { ...this.config };
    }

    // Abstract methods to be implemented by specific broker adapters
    protected abstract doConnect(): Promise<boolean>;
    protected abstract doDisconnect(): Promise<void>;
    protected abstract doSubmitOrder(orderRequest: OrderRequest): Promise<OrderResponse>;
    protected abstract doCancelOrder(brokerOrderId: string): Promise<boolean>;
    protected abstract doGetAccountInfo(): Promise<AccountInfo>;
    protected abstract doGetPositions(): Promise<Position[]>;
    protected abstract doSubscribeMarketData(symbols: string[]): Promise<boolean>;
    protected abstract doUnsubscribeMarketData(symbols: string[]): Promise<boolean>;
    protected abstract doHeartbeat(): Promise<boolean>;

    /**
     * Validate order request
     */
    protected validateOrder(orderRequest: OrderRequest): void {
        if (!orderRequest.clientOrderId) {
            throw new Error('Client order ID is required');
        }
        
        if (!orderRequest.symbol) {
            throw new Error('Symbol is required');
        }
        
        if (orderRequest.quantity <= 0) {
            throw new Error('Quantity must be positive');
        }
        
        if (orderRequest.type === OrderType.LIMIT && !orderRequest.price) {
            throw new Error('Price is required for limit orders');
        }
        
        if (orderRequest.type === OrderType.STOP && !orderRequest.stopPrice) {
            throw new Error('Stop price is required for stop orders');
        }
    }

    /**
     * Check rate limits
     */
    protected checkRateLimit(type: string): boolean {
        const now = Date.now();
        
        // Clean old entries
        this.rateLimitQueue = this.rateLimitQueue.filter(
            entry => now - entry.timestamp < 60000 // Keep last minute
        );
        
        // Check orders per second
        if (type === 'order') {
            const recentOrders = this.rateLimitQueue.filter(
                entry => entry.type === 'order' && now - entry.timestamp < 1000
            );
            
            if (recentOrders.length >= this.config.rateLimits.ordersPerSecond) {
                return false;
            }
        }
        
        // Check requests per minute
        if (this.rateLimitQueue.length >= this.config.rateLimits.requestsPerMinute) {
            return false;
        }
        
        // Add current request
        this.rateLimitQueue.push({ timestamp: now, type });
        return true;
    }

    /**
     * Update latency metrics
     */
    protected updateLatencyMetrics(latency: number): void {
        const totalRequests = this.metrics.totalOrders;
        this.metrics.averageLatency = 
            (this.metrics.averageLatency * (totalRequests - 1) + latency) / totalRequests;
    }

    /**
     * Start heartbeat monitoring
     */
    protected startHeartbeat(): void {
        this.stopHeartbeat();
        
        this.heartbeatInterval = setInterval(async () => {
            try {
                const success = await this.doHeartbeat();
                
                if (success) {
                    this.metrics.lastHeartbeat = Date.now();
                    this.lastHeartbeat = Date.now();
                } else {
                    this.handleHeartbeatFailure();
                }
                
            } catch (error) {
                this.handleHeartbeatFailure();
            }
        }, this.config.heartbeatInterval);
    }

    /**
     * Stop heartbeat monitoring
     */
    protected stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = undefined;
        }
    }

    /**
     * Handle heartbeat failure
     */
    protected handleHeartbeatFailure(): void {
        this.metrics.errorCount++;
        
        if (this.connectionStatus === ConnectionStatus.CONNECTED) {
            this.connectionStatus = ConnectionStatus.ERROR;
            this.emit('connectionStatusChanged', this.connectionStatus);
            this.attemptReconnection();
        }
    }

    /**
     * Attempt reconnection
     */
    protected async attemptReconnection(): Promise<void> {
        if (this.reconnectAttempts >= this.config.maxRetries) {
            this.emit('reconnectionFailed', 'Max reconnection attempts reached');
            return;
        }
        
        this.connectionStatus = ConnectionStatus.RECONNECTING;
        this.emit('connectionStatusChanged', this.connectionStatus);
        this.reconnectAttempts++;
        this.metrics.reconnectionCount++;
        
        setTimeout(async () => {
            try {
                const connected = await this.connect();
                if (!connected) {
                    this.attemptReconnection();
                }
            } catch (error) {
                this.attemptReconnection();
            }
        }, this.config.retryDelay);
    }

    /**
     * Handle market data tick
     */
    protected handleMarketDataTick(tick: MarketDataTick): void {
        this.emit('marketDataTick', tick);
    }

    /**
     * Handle order update
     */
    protected handleOrderUpdate(orderUpdate: OrderResponse): void {
        this.emit('orderUpdate', orderUpdate);
    }

    /**
     * Handle position update
     */
    protected handlePositionUpdate(position: Position): void {
        this.emit('positionUpdate', position);
    }

    /**
     * Handle account update
     */
    protected handleAccountUpdate(accountInfo: AccountInfo): void {
        this.emit('accountUpdate', accountInfo);
    }
}

export default BrokerAdapter;
