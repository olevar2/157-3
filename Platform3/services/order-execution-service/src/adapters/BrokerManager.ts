/**
 * Centralized Broker Manager
 * Manages multiple broker connections and provides intelligent order routing
 * 
 * Features:
 * - Multi-broker connectivity and management
 * - Intelligent order routing based on spreads, latency, and liquidity
 * - Automatic failover and load balancing
 * - Real-time performance monitoring
 * - Risk management and position aggregation
 * - Unified API for all broker operations
 */

import { EventEmitter } from 'events';
import { BrokerAdapter, BrokerConfig, OrderRequest, OrderResponse, MarketDataRequest, MarketDataResponse, AccountInfo, Position } from './BrokerAdapter';
import { MetaTraderAdapter, MetaTraderConfig } from './MetaTraderAdapter';
import { OANDAAdapter, OANDAConfig } from './OANDAAdapter';
import { InteractiveBrokersAdapter, IBKRConfig } from './InteractiveBrokersAdapter';
import { cTraderAdapter, cTraderConfig } from './cTraderAdapter';

export interface BrokerManagerConfig {
  brokers: BrokerConfiguration[];
  routingStrategy: 'best_spread' | 'lowest_latency' | 'round_robin' | 'load_balanced';
  enableFailover: boolean;
  maxRetries: number;
  healthCheckInterval: number;
  performanceWindow: number; // milliseconds
}

export interface BrokerConfiguration {
  id: string;
  type: 'metatrader' | 'oanda' | 'interactive_brokers' | 'ctrader';
  config: MetaTraderConfig | OANDAConfig | IBKRConfig | cTraderConfig;
  priority: number; // 1-10, 10 being highest priority
  enabled: boolean;
  maxOrdersPerSecond: number;
  supportedSymbols: string[];
  features: BrokerFeatures;
}

export interface BrokerFeatures {
  supportsMarketOrders: boolean;
  supportsLimitOrders: boolean;
  supportsStopOrders: boolean;
  supportsOCOOrders: boolean;
  supportsBracketOrders: boolean;
  supportsTrailingStops: boolean;
  supportsPartialFills: boolean;
  supportsHedging: boolean;
  minOrderSize: number;
  maxOrderSize: number;
  maxPositions: number;
}

export interface BrokerStatus {
  id: string;
  connected: boolean;
  lastHeartbeat: Date;
  latency: number;
  errorRate: number;
  ordersPerSecond: number;
  availableBalance: number;
  usedMargin: number;
  openPositions: number;
  health: 'healthy' | 'degraded' | 'unhealthy';
}

export interface RoutingDecision {
  brokerId: string;
  reason: string;
  confidence: number;
  alternativeBrokers: string[];
}

export class BrokerManager extends EventEmitter {
  private config: BrokerManagerConfig;
  private brokers: Map<string, BrokerAdapter> = new Map();
  private brokerStatus: Map<string, BrokerStatus> = new Map();
  private marketData: Map<string, Map<string, MarketDataResponse>> = new Map(); // symbol -> broker -> data
  private performanceMetrics: Map<string, any> = new Map();
  private healthCheckTimer: NodeJS.Timeout | null = null;
  private logger: any;

  constructor(config: BrokerManagerConfig) {
    super();
    this.config = config;
    this.logger = console; // Replace with proper logger
    
    this.initializeBrokers();
    this.startHealthChecks();
  }

  async connect(): Promise<void> {
    this.logger.info('Connecting to all configured brokers...');
    
    const connectionPromises = Array.from(this.brokers.values()).map(async (broker) => {
      try {
        await broker.connect();
        this.updateBrokerStatus(broker.getId(), { connected: true, health: 'healthy' });
        this.logger.info(`✅ Connected to broker: ${broker.getId()}`);
      } catch (error) {
        this.updateBrokerStatus(broker.getId(), { connected: false, health: 'unhealthy' });
        this.logger.error(`❌ Failed to connect to broker ${broker.getId()}: ${error}`);
      }
    });
    
    await Promise.allSettled(connectionPromises);
    
    const connectedBrokers = Array.from(this.brokerStatus.values()).filter(status => status.connected).length;
    this.logger.info(`Connected to ${connectedBrokers}/${this.brokers.size} brokers`);
    
    if (connectedBrokers === 0) {
      throw new Error('Failed to connect to any brokers');
    }
    
    this.emit('connected', { connectedBrokers, totalBrokers: this.brokers.size });
  }

  async disconnect(): Promise<void> {
    this.logger.info('Disconnecting from all brokers...');
    
    // Stop health checks
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }
    
    // Disconnect all brokers
    const disconnectionPromises = Array.from(this.brokers.values()).map(async (broker) => {
      try {
        await broker.disconnect();
        this.updateBrokerStatus(broker.getId(), { connected: false });
      } catch (error) {
        this.logger.error(`Error disconnecting from broker ${broker.getId()}: ${error}`);
      }
    });
    
    await Promise.allSettled(disconnectionPromises);
    
    this.emit('disconnected');
    this.logger.info('✅ Disconnected from all brokers');
  }

  async placeOrder(request: OrderRequest): Promise<OrderResponse> {
    const startTime = Date.now();
    
    try {
      // Select best broker for this order
      const routing = await this.selectBroker(request);
      const broker = this.brokers.get(routing.brokerId);
      
      if (!broker) {
        throw new Error(`Selected broker not available: ${routing.brokerId}`);
      }
      
      this.logger.info(`Routing order to ${routing.brokerId}: ${routing.reason}`);
      
      // Place order with selected broker
      const response = await broker.placeOrder(request);
      
      // Update performance metrics
      const latency = Date.now() - startTime;
      this.updatePerformanceMetrics(routing.brokerId, 'placeOrder', latency, true);
      
      this.emit('orderPlaced', {
        brokerId: routing.brokerId,
        orderId: response.orderId,
        latency,
        routing
      });
      
      return response;
      
    } catch (error) {
      const latency = Date.now() - startTime;
      
      // Try failover if enabled
      if (this.config.enableFailover) {
        return await this.handleOrderFailover(request, error, startTime);
      }
      
      this.logger.error(`❌ Failed to place order: ${error}`);
      throw error;
    }
  }

  async cancelOrder(orderId: string, brokerId?: string): Promise<boolean> {
    try {
      if (brokerId) {
        const broker = this.brokers.get(brokerId);
        if (broker) {
          return await broker.cancelOrder(orderId);
        }
      }
      
      // Try all brokers if brokerId not specified
      for (const [id, broker] of this.brokers) {
        try {
          const result = await broker.cancelOrder(orderId);
          if (result) {
            this.logger.info(`✅ Order cancelled via broker ${id}: ${orderId}`);
            return true;
          }
        } catch (error) {
          // Continue to next broker
        }
      }
      
      return false;
      
    } catch (error) {
      this.logger.error(`❌ Failed to cancel order ${orderId}: ${error}`);
      return false;
    }
  }

  async getAggregatedAccountInfo(): Promise<AccountInfo> {
    const accountInfos: AccountInfo[] = [];
    
    for (const [brokerId, broker] of this.brokers) {
      try {
        if (this.isBrokerHealthy(brokerId)) {
          const accountInfo = await broker.getAccountInfo();
          accountInfos.push(accountInfo);
        }
      } catch (error) {
        this.logger.error(`Failed to get account info from ${brokerId}: ${error}`);
      }
    }
    
    if (accountInfos.length === 0) {
      throw new Error('No account information available from any broker');
    }
    
    // Aggregate account information
    return {
      accountId: 'aggregated',
      balance: accountInfos.reduce((sum, acc) => sum + acc.balance, 0),
      equity: accountInfos.reduce((sum, acc) => sum + acc.equity, 0),
      margin: accountInfos.reduce((sum, acc) => sum + acc.margin, 0),
      freeMargin: accountInfos.reduce((sum, acc) => sum + acc.freeMargin, 0),
      marginLevel: accountInfos.reduce((sum, acc) => sum + acc.marginLevel, 0) / accountInfos.length,
      currency: accountInfos[0].currency, // Assume same currency
      leverage: accountInfos.reduce((sum, acc) => sum + acc.leverage, 0) / accountInfos.length,
      profit: accountInfos.reduce((sum, acc) => sum + acc.profit, 0)
    };
  }

  async getAggregatedPositions(): Promise<Position[]> {
    const allPositions: Position[] = [];
    
    for (const [brokerId, broker] of this.brokers) {
      try {
        if (this.isBrokerHealthy(brokerId)) {
          const positions = await broker.getPositions();
          // Add broker ID to each position
          positions.forEach(pos => {
            pos.positionId = `${brokerId}_${pos.positionId}`;
          });
          allPositions.push(...positions);
        }
      } catch (error) {
        this.logger.error(`Failed to get positions from ${brokerId}: ${error}`);
      }
    }
    
    return allPositions;
  }

  async getBestMarketData(request: MarketDataRequest): Promise<MarketDataResponse> {
    const marketDataPromises = Array.from(this.brokers.entries()).map(async ([brokerId, broker]) => {
      try {
        if (this.isBrokerHealthy(brokerId)) {
          const data = await broker.getMarketData(request);
          return { brokerId, data };
        }
      } catch (error) {
        // Ignore errors from individual brokers
      }
      return null;
    });
    
    const results = await Promise.allSettled(marketDataPromises);
    const validResults = results
      .filter(result => result.status === 'fulfilled' && result.value)
      .map(result => (result as PromiseFulfilledResult<any>).value);
    
    if (validResults.length === 0) {
      throw new Error(`No market data available for ${request.symbol}`);
    }
    
    // Select best spread
    const bestData = validResults.reduce((best, current) => {
      const currentSpread = current.data.ask - current.data.bid;
      const bestSpread = best.data.ask - best.data.bid;
      return currentSpread < bestSpread ? current : best;
    });
    
    return bestData.data;
  }

  getBrokerStatus(brokerId?: string): BrokerStatus | BrokerStatus[] {
    if (brokerId) {
      return this.brokerStatus.get(brokerId) || this.createDefaultStatus(brokerId);
    }
    
    return Array.from(this.brokerStatus.values());
  }

  getPerformanceMetrics(): Map<string, any> {
    return this.performanceMetrics;
  }

  // Private helper methods
  private initializeBrokers(): void {
    for (const brokerConfig of this.config.brokers) {
      if (!brokerConfig.enabled) continue;
      
      let broker: BrokerAdapter;
      
      switch (brokerConfig.type) {
        case 'metatrader':
          broker = new MetaTraderAdapter(brokerConfig.config as MetaTraderConfig);
          break;
        case 'oanda':
          broker = new OANDAAdapter(brokerConfig.config as OANDAConfig);
          break;
        case 'interactive_brokers':
          broker = new InteractiveBrokersAdapter(brokerConfig.config as IBKRConfig);
          break;
        case 'ctrader':
          broker = new cTraderAdapter(brokerConfig.config as cTraderConfig);
          break;
        default:
          this.logger.error(`Unknown broker type: ${brokerConfig.type}`);
          continue;
      }
      
      // Set broker ID
      (broker as any).id = brokerConfig.id;
      
      // Setup event listeners
      this.setupBrokerEventListeners(broker, brokerConfig.id);
      
      this.brokers.set(brokerConfig.id, broker);
      this.brokerStatus.set(brokerConfig.id, this.createDefaultStatus(brokerConfig.id));
      
      this.logger.info(`Initialized broker: ${brokerConfig.id} (${brokerConfig.type})`);
    }
  }

  private setupBrokerEventListeners(broker: BrokerAdapter, brokerId: string): void {
    broker.on('connected', () => {
      this.updateBrokerStatus(brokerId, { connected: true, health: 'healthy' });
      this.emit('brokerConnected', brokerId);
    });
    
    broker.on('disconnected', () => {
      this.updateBrokerStatus(brokerId, { connected: false, health: 'unhealthy' });
      this.emit('brokerDisconnected', brokerId);
    });
    
    broker.on('error', (error) => {
      this.updateBrokerStatus(brokerId, { health: 'degraded' });
      this.emit('brokerError', { brokerId, error });
    });
    
    broker.on('marketData', (data) => {
      this.updateMarketData(brokerId, data.symbol, data);
    });
  }

  private async selectBroker(request: OrderRequest): Promise<RoutingDecision> {
    const availableBrokers = this.getAvailableBrokers(request.symbol);
    
    if (availableBrokers.length === 0) {
      throw new Error(`No available brokers for symbol: ${request.symbol}`);
    }
    
    switch (this.config.routingStrategy) {
      case 'best_spread':
        return await this.selectByBestSpread(request, availableBrokers);
      case 'lowest_latency':
        return this.selectByLowestLatency(availableBrokers);
      case 'round_robin':
        return this.selectRoundRobin(availableBrokers);
      case 'load_balanced':
        return this.selectLoadBalanced(availableBrokers);
      default:
        return {
          brokerId: availableBrokers[0],
          reason: 'default selection',
          confidence: 0.5,
          alternativeBrokers: availableBrokers.slice(1)
        };
    }
  }

  private getAvailableBrokers(symbol: string): string[] {
    return Array.from(this.brokers.keys()).filter(brokerId => {
      const status = this.brokerStatus.get(brokerId);
      const config = this.config.brokers.find(b => b.id === brokerId);
      
      return status?.connected && 
             status.health !== 'unhealthy' &&
             config?.enabled &&
             config.supportedSymbols.includes(symbol);
    });
  }

  private async selectByBestSpread(request: OrderRequest, brokers: string[]): Promise<RoutingDecision> {
    const spreadPromises = brokers.map(async (brokerId) => {
      try {
        const broker = this.brokers.get(brokerId);
        const marketData = await broker?.getMarketData({ symbol: request.symbol });
        const spread = marketData ? marketData.ask - marketData.bid : Infinity;
        return { brokerId, spread };
      } catch {
        return { brokerId, spread: Infinity };
      }
    });
    
    const spreads = await Promise.all(spreadPromises);
    const bestSpread = spreads.reduce((best, current) => 
      current.spread < best.spread ? current : best
    );
    
    return {
      brokerId: bestSpread.brokerId,
      reason: `best spread: ${bestSpread.spread.toFixed(5)}`,
      confidence: 0.9,
      alternativeBrokers: spreads.filter(s => s.brokerId !== bestSpread.brokerId).map(s => s.brokerId)
    };
  }

  private selectByLowestLatency(brokers: string[]): RoutingDecision {
    const latencies = brokers.map(brokerId => ({
      brokerId,
      latency: this.brokerStatus.get(brokerId)?.latency || Infinity
    }));
    
    const lowestLatency = latencies.reduce((best, current) =>
      current.latency < best.latency ? current : best
    );
    
    return {
      brokerId: lowestLatency.brokerId,
      reason: `lowest latency: ${lowestLatency.latency}ms`,
      confidence: 0.8,
      alternativeBrokers: latencies.filter(l => l.brokerId !== lowestLatency.brokerId).map(l => l.brokerId)
    };
  }

  private selectRoundRobin(brokers: string[]): RoutingDecision {
    // Simple round-robin implementation
    const index = Date.now() % brokers.length;
    const selectedBroker = brokers[index];
    
    return {
      brokerId: selectedBroker,
      reason: 'round-robin selection',
      confidence: 0.6,
      alternativeBrokers: brokers.filter(b => b !== selectedBroker)
    };
  }

  private selectLoadBalanced(brokers: string[]): RoutingDecision {
    const loads = brokers.map(brokerId => ({
      brokerId,
      load: this.brokerStatus.get(brokerId)?.ordersPerSecond || 0
    }));
    
    const leastLoaded = loads.reduce((best, current) =>
      current.load < best.load ? current : best
    );
    
    return {
      brokerId: leastLoaded.brokerId,
      reason: `least loaded: ${leastLoaded.load} orders/sec`,
      confidence: 0.7,
      alternativeBrokers: loads.filter(l => l.brokerId !== leastLoaded.brokerId).map(l => l.brokerId)
    };
  }

  private async handleOrderFailover(request: OrderRequest, originalError: any, startTime: number): Promise<OrderResponse> {
    this.logger.warn(`Attempting order failover due to: ${originalError}`);
    
    const routing = await this.selectBroker(request);
    const fallbackBrokers = routing.alternativeBrokers;
    
    for (const brokerId of fallbackBrokers) {
      try {
        const broker = this.brokers.get(brokerId);
        if (broker && this.isBrokerHealthy(brokerId)) {
          this.logger.info(`Failover attempt with broker: ${brokerId}`);
          
          const response = await broker.placeOrder(request);
          const latency = Date.now() - startTime;
          
          this.updatePerformanceMetrics(brokerId, 'placeOrder', latency, true);
          
          this.emit('orderFailover', {
            originalError,
            successfulBrokerId: brokerId,
            orderId: response.orderId,
            latency
          });
          
          return response;
        }
      } catch (error) {
        this.logger.warn(`Failover failed with broker ${brokerId}: ${error}`);
      }
    }
    
    throw new Error(`Order failover exhausted. Original error: ${originalError}`);
  }

  private isBrokerHealthy(brokerId: string): boolean {
    const status = this.brokerStatus.get(brokerId);
    return status?.connected === true && status.health !== 'unhealthy';
  }

  private updateBrokerStatus(brokerId: string, updates: Partial<BrokerStatus>): void {
    const currentStatus = this.brokerStatus.get(brokerId) || this.createDefaultStatus(brokerId);
    const newStatus = { ...currentStatus, ...updates, lastHeartbeat: new Date() };
    this.brokerStatus.set(brokerId, newStatus);
  }

  private createDefaultStatus(brokerId: string): BrokerStatus {
    return {
      id: brokerId,
      connected: false,
      lastHeartbeat: new Date(),
      latency: 0,
      errorRate: 0,
      ordersPerSecond: 0,
      availableBalance: 0,
      usedMargin: 0,
      openPositions: 0,
      health: 'unhealthy'
    };
  }

  private updateMarketData(brokerId: string, symbol: string, data: MarketDataResponse): void {
    if (!this.marketData.has(symbol)) {
      this.marketData.set(symbol, new Map());
    }
    this.marketData.get(symbol)!.set(brokerId, data);
  }

  private updatePerformanceMetrics(brokerId: string, operation: string, latency: number, success: boolean): void {
    const key = `${brokerId}_${operation}`;
    const current = this.performanceMetrics.get(key) || {
      totalRequests: 0,
      successfulRequests: 0,
      totalLatency: 0,
      averageLatency: 0,
      errorRate: 0
    };
    
    current.totalRequests++;
    if (success) {
      current.successfulRequests++;
      current.totalLatency += latency;
      current.averageLatency = current.totalLatency / current.successfulRequests;
    }
    current.errorRate = (current.totalRequests - current.successfulRequests) / current.totalRequests;
    
    this.performanceMetrics.set(key, current);
  }

  private startHealthChecks(): void {
    this.healthCheckTimer = setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);
  }

  private async performHealthChecks(): Promise<void> {
    for (const [brokerId, broker] of this.brokers) {
      try {
        const startTime = Date.now();
        
        // Simple health check - get account info
        await broker.getAccountInfo();
        
        const latency = Date.now() - startTime;
        this.updateBrokerStatus(brokerId, {
          latency,
          health: latency < 1000 ? 'healthy' : 'degraded'
        });
        
      } catch (error) {
        this.updateBrokerStatus(brokerId, {
          health: 'unhealthy',
          connected: false
        });
      }
    }
  }
}
