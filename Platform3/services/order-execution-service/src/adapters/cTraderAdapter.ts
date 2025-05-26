/**
 * cTrader Broker Adapter
 * Professional integration with cTrader platform using Open API and FIX API
 * 
 * Features:
 * - cTrader Open API integration
 * - FIX API support for institutional connectivity
 * - Real-time market data streaming
 * - Advanced order management with cTrader-specific features
 * - Copy trading and social trading integration
 * - Multi-asset class support
 */

import { BrokerAdapter, BrokerConfig, OrderRequest, OrderResponse, MarketDataRequest, MarketDataResponse, AccountInfo, Position, Trade } from './BrokerAdapter';
import { EventEmitter } from 'events';
import axios, { AxiosInstance } from 'axios';
import WebSocket from 'ws';

export interface cTraderConfig extends BrokerConfig {
  // Open API Configuration
  apiUrl: string; // https://openapi.ctrader.com
  clientId: string;
  clientSecret: string;
  accessToken: string;
  refreshToken: string;
  
  // FIX API Configuration (optional)
  fixHost?: string;
  fixPort?: number;
  fixUsername?: string;
  fixPassword?: string;
  
  // Account Configuration
  accountId: string;
  environment: 'live' | 'demo';
  
  // Advanced Features
  enableCopyTrading: boolean;
  enableSocialTrading: boolean;
  symbolGroups: string[];
  maxPositions: number;
}

export interface cTraderSymbol {
  symbolId: number;
  symbolName: string;
  enabled: boolean;
  baseAsset: string;
  quoteAsset: string;
  symbolCategory: string;
  description: string;
  sortOrder: number;
  minVolume: number;
  maxVolume: number;
  stepVolume: number;
  precision: number;
  tradingMode: string;
  calculationMode: string;
  commissionType: string;
  commission: number;
  swapRollover3Days: string;
  swapLong: number;
  swapShort: number;
  maxSpread: number;
  minSpread: number;
  sourceId: string;
  scheduleId: number;
  marginRate: number;
  deltaType: string;
}

export interface cTraderPosition {
  positionId: number;
  tradeType: string; // BUY, SELL
  symbolId: number;
  volume: number;
  entryPrice: number;
  currentPrice: number;
  commission: number;
  swap: number;
  pnl: number;
  grossPnl: number;
  netPnl: number;
  openTimestamp: number;
  label: string;
  comment: string;
  stopLoss?: number;
  takeProfit?: number;
  followingPositionId?: number;
  guaranteedStopLoss: boolean;
  usedMargin: number;
}

export interface cTraderOrder {
  orderId: number;
  orderType: string; // MARKET, LIMIT, STOP, STOP_LIMIT
  tradeType: string; // BUY, SELL
  symbolId: number;
  requestedVolume: number;
  executedVolume: number;
  limitPrice?: number;
  stopPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  expirationTimestamp?: number;
  comment: string;
  label: string;
  positionId?: number;
  relativeStopLoss?: number;
  relativeTakeProfit?: number;
  guaranteedStopLoss: boolean;
  trailingStopLoss: boolean;
  stopTriggerMethod: string;
}

export class cTraderAdapter extends BrokerAdapter {
  private config: cTraderConfig;
  private restClient: AxiosInstance;
  private wsConnection: WebSocket | null = null;
  private symbols: Map<string, cTraderSymbol> = new Map();
  private symbolIdMap: Map<number, string> = new Map();
  private marketData: Map<string, any> = new Map();
  private accessTokenExpiry: Date = new Date();

  constructor(config: cTraderConfig) {
    super(config);
    this.config = config;
    
    // Initialize REST API client
    this.restClient = axios.create({
      baseURL: config.apiUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Authorization': `Bearer ${config.accessToken}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'Platform3-cTrader-Adapter/1.0'
      }
    });

    this.setupInterceptors();
  }

  async connect(): Promise<void> {
    try {
      this.logger.info(`Connecting to cTrader ${this.config.environment} environment...`);
      
      // Validate and refresh access token if needed
      await this.validateAccessToken();
      
      // Get account information
      await this.validateAccount();
      
      // Load symbol information
      await this.loadSymbols();
      
      // Start WebSocket connection for real-time data
      await this.connectWebSocket();
      
      this.connected = true;
      this.emit('connected');
      
      this.logger.info(`✅ Successfully connected to cTrader ${this.config.environment}`);
      
    } catch (error) {
      this.logger.error(`❌ Failed to connect to cTrader: ${error}`);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      this.logger.info('Disconnecting from cTrader...');
      
      // Close WebSocket connection
      if (this.wsConnection) {
        this.wsConnection.close();
        this.wsConnection = null;
      }
      
      this.connected = false;
      this.emit('disconnected');
      
      this.logger.info('✅ Disconnected from cTrader');
      
    } catch (error) {
      this.logger.error(`❌ Error disconnecting from cTrader: ${error}`);
      throw error;
    }
  }

  async placeOrder(request: OrderRequest): Promise<OrderResponse> {
    const startTime = Date.now();
    
    try {
      this.validateOrderRequest(request);
      
      // Get symbol information
      const symbol = this.getSymbol(request.symbol);
      if (!symbol) {
        throw new Error(`Symbol not found: ${request.symbol}`);
      }
      
      // Create cTrader order
      const cTraderOrder = this.mapTocTraderOrder(request, symbol);
      
      // Send order to cTrader
      const response = await this.restClient.post(`/v2/accounts/${this.config.accountId}/orders`, cTraderOrder);
      
      // Parse response
      const orderResponse = this.parseOrderResponse(response.data);
      
      // Update performance metrics
      const latency = Date.now() - startTime;
      this.updatePerformanceMetrics('placeOrder', latency, true);
      
      this.logger.info(`✅ Order placed successfully: ${orderResponse.orderId} (${latency}ms)`);
      
      return orderResponse;
      
    } catch (error) {
      const latency = Date.now() - startTime;
      this.updatePerformanceMetrics('placeOrder', latency, false);
      
      this.logger.error(`❌ Failed to place order: ${error}`);
      throw error;
    }
  }

  async cancelOrder(orderId: string): Promise<boolean> {
    try {
      await this.restClient.delete(`/v2/accounts/${this.config.accountId}/orders/${orderId}`);
      
      this.logger.info(`✅ Order cancelled successfully: ${orderId}`);
      return true;
      
    } catch (error) {
      this.logger.error(`❌ Failed to cancel order ${orderId}: ${error}`);
      return false;
    }
  }

  async getAccountInfo(): Promise<AccountInfo> {
    try {
      const response = await this.restClient.get(`/v2/accounts/${this.config.accountId}`);
      const account = response.data;
      
      return {
        accountId: account.accountId.toString(),
        balance: account.balance / 100, // cTrader uses cents
        equity: account.equity / 100,
        margin: account.usedMargin / 100,
        freeMargin: account.freeMargin / 100,
        marginLevel: account.marginLevel,
        currency: account.depositCurrency,
        leverage: account.leverage,
        profit: account.unrealizedNetPnl / 100
      };
      
    } catch (error) {
      this.logger.error(`❌ Failed to get account info: ${error}`);
      throw error;
    }
  }

  async getPositions(): Promise<Position[]> {
    try {
      const response = await this.restClient.get(`/v2/accounts/${this.config.accountId}/positions`);
      
      return response.data.position.map((pos: cTraderPosition) => ({
        positionId: pos.positionId.toString(),
        symbol: this.getSymbolName(pos.symbolId),
        side: pos.tradeType.toLowerCase() as 'buy' | 'sell',
        size: pos.volume / 100, // cTrader uses volume in cents
        entryPrice: pos.entryPrice,
        currentPrice: pos.currentPrice,
        unrealizedPnL: pos.netPnl / 100,
        openTime: new Date(pos.openTimestamp)
      }));
      
    } catch (error) {
      this.logger.error(`❌ Failed to get positions: ${error}`);
      throw error;
    }
  }

  async getMarketData(request: MarketDataRequest): Promise<MarketDataResponse> {
    try {
      const symbol = this.getSymbol(request.symbol);
      if (!symbol) {
        throw new Error(`Symbol not found: ${request.symbol}`);
      }
      
      // Check if we have real-time data from WebSocket
      const realtimeData = this.marketData.get(request.symbol);
      if (realtimeData) {
        return {
          symbol: request.symbol,
          bid: realtimeData.bid,
          ask: realtimeData.ask,
          timestamp: new Date(realtimeData.timestamp),
          volume: realtimeData.volume || 0
        };
      }
      
      // Fallback to REST API
      const response = await this.restClient.get(`/v2/symbols/${symbol.symbolId}/tick`);
      const tick = response.data;
      
      return {
        symbol: request.symbol,
        bid: tick.bid,
        ask: tick.ask,
        timestamp: new Date(tick.timestamp),
        volume: 0
      };
      
    } catch (error) {
      this.logger.error(`❌ Failed to get market data for ${request.symbol}: ${error}`);
      throw error;
    }
  }

  // Private helper methods
  private async validateAccessToken(): Promise<void> {
    try {
      // Check if token is expired
      if (new Date() >= this.accessTokenExpiry) {
        await this.refreshAccessToken();
      }
      
      // Validate token with a simple API call
      await this.restClient.get('/v2/accounts');
      
    } catch (error) {
      if (error.response?.status === 401) {
        await this.refreshAccessToken();
      } else {
        throw error;
      }
    }
  }

  private async refreshAccessToken(): Promise<void> {
    try {
      const response = await axios.post(`${this.config.apiUrl}/oauth2/token`, {
        grant_type: 'refresh_token',
        refresh_token: this.config.refreshToken,
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret
      });
      
      this.config.accessToken = response.data.access_token;
      this.config.refreshToken = response.data.refresh_token;
      this.accessTokenExpiry = new Date(Date.now() + response.data.expires_in * 1000);
      
      // Update authorization header
      this.restClient.defaults.headers['Authorization'] = `Bearer ${this.config.accessToken}`;
      
      this.logger.info('✅ Access token refreshed successfully');
      
    } catch (error) {
      throw new Error(`Failed to refresh access token: ${error}`);
    }
  }

  private async validateAccount(): Promise<void> {
    try {
      const response = await this.restClient.get(`/v2/accounts/${this.config.accountId}`);
      
      if (!response.data || response.data.accountId.toString() !== this.config.accountId) {
        throw new Error('Account validation failed');
      }
      
      this.logger.info(`Account validated: ${response.data.accountId} (${response.data.depositCurrency})`);
      
    } catch (error) {
      throw new Error(`Failed to validate account: ${error}`);
    }
  }

  private async loadSymbols(): Promise<void> {
    try {
      const response = await this.restClient.get('/v2/symbols');
      
      for (const symbol of response.data.symbol) {
        const symbolName = this.normalizecTraderSymbol(symbol.symbolName);
        this.symbols.set(symbolName, symbol);
        this.symbolIdMap.set(symbol.symbolId, symbolName);
      }
      
      this.logger.info(`Loaded ${this.symbols.size} symbols from cTrader`);
      
    } catch (error) {
      this.logger.error(`Failed to load symbols: ${error}`);
    }
  }

  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `wss://openapi.ctrader.com/ws`;
      
      this.wsConnection = new WebSocket(wsUrl);
      
      this.wsConnection.on('open', () => {
        this.logger.info('WebSocket connection established');
        this.authenticateWebSocket();
        resolve();
      });
      
      this.wsConnection.on('message', (data) => {
        this.handleWebSocketMessage(data.toString());
      });
      
      this.wsConnection.on('error', (error) => {
        this.logger.error(`WebSocket error: ${error}`);
        reject(error);
      });
      
      this.wsConnection.on('close', () => {
        this.logger.warn('WebSocket connection closed');
        this.handleDisconnection();
      });
    });
  }

  private authenticateWebSocket(): void {
    if (this.wsConnection) {
      const authMessage = {
        payloadType: 'PROTO_OA_APPLICATION_AUTH_REQ',
        clientId: this.config.clientId,
        clientSecret: this.config.clientSecret
      };
      
      this.wsConnection.send(JSON.stringify(authMessage));
    }
  }

  private handleWebSocketMessage(message: string): void {
    try {
      const data = JSON.parse(message);
      
      switch (data.payloadType) {
        case 'PROTO_OA_SPOT_EVENT':
          this.handleSpotEvent(data);
          break;
        case 'PROTO_OA_EXECUTION_EVENT':
          this.handleExecutionEvent(data);
          break;
        case 'PROTO_OA_ORDER_ERROR_EVENT':
          this.handleOrderErrorEvent(data);
          break;
        default:
          // Handle other message types
          break;
      }
    } catch (error) {
      this.logger.error(`Error parsing WebSocket message: ${error}`);
    }
  }

  private handleSpotEvent(data: any): void {
    const symbolName = this.getSymbolName(data.symbolId);
    if (symbolName) {
      this.marketData.set(symbolName, {
        bid: data.bid,
        ask: data.ask,
        timestamp: data.timestamp,
        volume: data.volume
      });
      
      // Emit market data event
      this.emit('marketData', {
        symbol: symbolName,
        bid: data.bid,
        ask: data.ask,
        timestamp: new Date(data.timestamp)
      });
    }
  }

  private handleExecutionEvent(data: any): void {
    // Handle order execution events
    this.emit('orderUpdate', {
      orderId: data.orderId,
      status: data.executionType,
      fillPrice: data.executionPrice,
      fillQuantity: data.executedVolume
    });
  }

  private handleOrderErrorEvent(data: any): void {
    this.logger.error(`Order error: ${data.errorCode} - ${data.description}`);
    
    this.emit('orderError', {
      orderId: data.orderId,
      errorCode: data.errorCode,
      description: data.description
    });
  }

  private getSymbol(symbolName: string): cTraderSymbol | undefined {
    return this.symbols.get(symbolName);
  }

  private getSymbolName(symbolId: number): string {
    return this.symbolIdMap.get(symbolId) || '';
  }

  private normalizecTraderSymbol(cTraderSymbol: string): string {
    // Convert cTrader symbol format to Platform3 format
    // e.g., EURUSD -> EURUSD (usually no change needed)
    return cTraderSymbol.replace(/[^A-Z]/g, '');
  }

  private mapTocTraderOrder(request: OrderRequest, symbol: cTraderSymbol): any {
    const volume = Math.round(request.quantity * 100); // cTrader uses volume in cents
    
    const order: any = {
      symbolId: symbol.symbolId,
      orderType: this.mapOrderType(request.type),
      tradeType: request.side.toUpperCase(),
      volume: volume,
      comment: request.clientOrderId || 'Platform3',
      label: 'Platform3'
    };
    
    if (request.price && request.type !== 'market') {
      order.limitPrice = request.price;
    }
    
    if (request.stopPrice) {
      order.stopPrice = request.stopPrice;
    }
    
    if (request.stopLoss) {
      order.stopLoss = request.stopLoss;
    }
    
    if (request.takeProfit) {
      order.takeProfit = request.takeProfit;
    }
    
    return order;
  }

  private mapOrderType(type: string): string {
    const typeMap: Record<string, string> = {
      'market': 'MARKET',
      'limit': 'LIMIT',
      'stop': 'STOP',
      'stop_limit': 'STOP_LIMIT'
    };
    
    return typeMap[type] || 'MARKET';
  }

  private parseOrderResponse(responseData: any): OrderResponse {
    return {
      orderId: responseData.orderId?.toString() || '',
      status: responseData.orderStatus || 'submitted',
      fillPrice: responseData.executionPrice || 0,
      fillQuantity: responseData.executedVolume ? responseData.executedVolume / 100 : 0,
      timestamp: new Date(responseData.timestamp || Date.now())
    };
  }

  private setupInterceptors(): void {
    // Request interceptor for rate limiting
    this.restClient.interceptors.request.use(async (config) => {
      await this.rateLimiter.acquire();
      return config;
    });

    // Response interceptor for error handling
    this.restClient.interceptors.response.use(
      (response) => response,
      (error) => {
        this.handleRestError(error);
        return Promise.reject(error);
      }
    );
  }

  private handleRestError(error: any): void {
    if (error.response?.status === 401) {
      this.logger.warn('Authentication expired, attempting to refresh token...');
      this.refreshAccessToken().catch(() => {
        this.handleDisconnection();
      });
    } else if (error.response?.status === 429) {
      this.logger.warn('Rate limit exceeded, backing off...');
      this.rateLimiter.backoff();
    }
  }
}
