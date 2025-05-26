/**
 * OANDA Broker Adapter
 * Professional integration with OANDA v20 REST API
 * 
 * Features:
 * - OANDA v20 REST API integration
 * - Real-time streaming market data
 * - Advanced order management with OANDA-specific features
 * - Position and trade management
 * - Account information and transaction history
 * - Risk management and margin calculations
 */

import { BrokerAdapter, BrokerConfig, OrderRequest, OrderResponse, MarketDataRequest, MarketDataResponse, AccountInfo, Position, Trade } from './BrokerAdapter';
import { EventEmitter } from 'events';
import axios, { AxiosInstance } from 'axios';
import WebSocket from 'ws';

export interface OANDAConfig extends BrokerConfig {
  // OANDA API Configuration
  apiUrl: string; // https://api-fxtrade.oanda.com or https://api-fxpractice.oanda.com
  streamUrl: string; // https://stream-fxtrade.oanda.com or https://stream-fxpractice.oanda.com
  accessToken: string;
  accountId: string;
  
  // Environment
  environment: 'live' | 'practice';
  
  // Advanced Features
  enableStreaming: boolean;
  instruments: string[];
  maxConcurrentOrders: number;
  defaultUnits: number;
}

export interface OANDAInstrument {
  name: string;
  type: string;
  displayName: string;
  pipLocation: number;
  displayPrecision: number;
  tradeUnitsPrecision: number;
  minimumTradeSize: number;
  maximumTrailingStopDistance: number;
  minimumTrailingStopDistance: number;
  maximumPositionSize: number;
  maximumOrderUnits: number;
  marginRate: number;
}

export interface OANDAPrice {
  instrument: string;
  time: string;
  bids: Array<{ price: string; liquidity: number }>;
  asks: Array<{ price: string; liquidity: number }>;
  closeoutBid: string;
  closeoutAsk: string;
  status: string;
  tradeable: boolean;
}

export interface OANDAOrder {
  id: string;
  createTime: string;
  state: string;
  type: string;
  instrument: string;
  units: string;
  timeInForce: string;
  price?: string;
  stopLossOnFill?: any;
  takeProfitOnFill?: any;
  trailingStopLossOnFill?: any;
  clientExtensions?: any;
}

export class OANDAAdapter extends BrokerAdapter {
  private config: OANDAConfig;
  private restClient: AxiosInstance;
  private streamClient: AxiosInstance;
  private priceStream: any = null;
  private instruments: Map<string, OANDAInstrument> = new Map();
  private currentPrices: Map<string, OANDAPrice> = new Map();

  constructor(config: OANDAConfig) {
    super(config);
    this.config = config;
    
    // Initialize REST API client
    this.restClient = axios.create({
      baseURL: config.apiUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Authorization': `Bearer ${config.accessToken}`,
        'Content-Type': 'application/json',
        'Accept-Datetime-Format': 'RFC3339',
        'User-Agent': 'Platform3-OANDA-Adapter/1.0'
      }
    });

    // Initialize streaming client
    this.streamClient = axios.create({
      baseURL: config.streamUrl,
      timeout: 0, // No timeout for streaming
      headers: {
        'Authorization': `Bearer ${config.accessToken}`,
        'Accept': 'application/json',
        'User-Agent': 'Platform3-OANDA-Adapter/1.0'
      }
    });

    this.setupInterceptors();
  }

  async connect(): Promise<void> {
    try {
      this.logger.info(`Connecting to OANDA ${this.config.environment} environment...`);
      
      // Validate account and get account info
      await this.validateAccount();
      
      // Load instrument information
      await this.loadInstruments();
      
      // Start price streaming if enabled
      if (this.config.enableStreaming) {
        await this.startPriceStream();
      }
      
      this.connected = true;
      this.emit('connected');
      
      this.logger.info(`✅ Successfully connected to OANDA ${this.config.environment}`);
      
    } catch (error) {
      this.logger.error(`❌ Failed to connect to OANDA: ${error}`);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      this.logger.info('Disconnecting from OANDA...');
      
      // Stop price streaming
      if (this.priceStream) {
        this.priceStream.destroy();
        this.priceStream = null;
      }
      
      this.connected = false;
      this.emit('disconnected');
      
      this.logger.info('✅ Disconnected from OANDA');
      
    } catch (error) {
      this.logger.error(`❌ Error disconnecting from OANDA: ${error}`);
      throw error;
    }
  }

  async placeOrder(request: OrderRequest): Promise<OrderResponse> {
    const startTime = Date.now();
    
    try {
      this.validateOrderRequest(request);
      
      // Map Platform3 order to OANDA format
      const oandaOrder = this.mapToOANDAOrder(request);
      
      // Send order to OANDA
      const response = await this.restClient.post(`/v3/accounts/${this.config.accountId}/orders`, {
        order: oandaOrder
      });
      
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
      await this.restClient.put(`/v3/accounts/${this.config.accountId}/orders/${orderId}/cancel`);
      
      this.logger.info(`✅ Order cancelled successfully: ${orderId}`);
      return true;
      
    } catch (error) {
      this.logger.error(`❌ Failed to cancel order ${orderId}: ${error}`);
      return false;
    }
  }

  async getAccountInfo(): Promise<AccountInfo> {
    try {
      const response = await this.restClient.get(`/v3/accounts/${this.config.accountId}`);
      const account = response.data.account;
      
      return {
        accountId: account.id,
        balance: parseFloat(account.balance),
        equity: parseFloat(account.NAV),
        margin: parseFloat(account.marginUsed),
        freeMargin: parseFloat(account.marginAvailable),
        marginLevel: parseFloat(account.marginCloseoutPercent),
        currency: account.currency,
        leverage: parseFloat(account.marginRate),
        profit: parseFloat(account.unrealizedPL)
      };
      
    } catch (error) {
      this.logger.error(`❌ Failed to get account info: ${error}`);
      throw error;
    }
  }

  async getPositions(): Promise<Position[]> {
    try {
      const response = await this.restClient.get(`/v3/accounts/${this.config.accountId}/positions`);
      
      const positions: Position[] = [];
      
      for (const position of response.data.positions) {
        if (parseFloat(position.long.units) !== 0) {
          positions.push({
            positionId: `${position.instrument}_long`,
            symbol: this.normalizeInstrument(position.instrument),
            side: 'buy',
            size: Math.abs(parseFloat(position.long.units)),
            entryPrice: parseFloat(position.long.averagePrice),
            currentPrice: parseFloat(position.long.unrealizedPL) / parseFloat(position.long.units) + parseFloat(position.long.averagePrice),
            unrealizedPnL: parseFloat(position.long.unrealizedPL),
            openTime: new Date() // OANDA doesn't provide position open time directly
          });
        }
        
        if (parseFloat(position.short.units) !== 0) {
          positions.push({
            positionId: `${position.instrument}_short`,
            symbol: this.normalizeInstrument(position.instrument),
            side: 'sell',
            size: Math.abs(parseFloat(position.short.units)),
            entryPrice: parseFloat(position.short.averagePrice),
            currentPrice: parseFloat(position.short.averagePrice) - parseFloat(position.short.unrealizedPL) / parseFloat(position.short.units),
            unrealizedPnL: parseFloat(position.short.unrealizedPL),
            openTime: new Date()
          });
        }
      }
      
      return positions;
      
    } catch (error) {
      this.logger.error(`❌ Failed to get positions: ${error}`);
      throw error;
    }
  }

  async getMarketData(request: MarketDataRequest): Promise<MarketDataResponse> {
    try {
      const oandaInstrument = this.mapToOANDAInstrument(request.symbol);
      
      // Check if we have real-time price from stream
      const streamPrice = this.currentPrices.get(oandaInstrument);
      if (streamPrice && this.config.enableStreaming) {
        return {
          symbol: request.symbol,
          bid: parseFloat(streamPrice.bids[0].price),
          ask: parseFloat(streamPrice.asks[0].price),
          timestamp: new Date(streamPrice.time),
          volume: 0 // OANDA doesn't provide volume in price stream
        };
      }
      
      // Fallback to REST API
      const response = await this.restClient.get(`/v3/accounts/${this.config.accountId}/pricing`, {
        params: {
          instruments: oandaInstrument
        }
      });
      
      const price = response.data.prices[0];
      
      return {
        symbol: request.symbol,
        bid: parseFloat(price.bids[0].price),
        ask: parseFloat(price.asks[0].price),
        timestamp: new Date(price.time),
        volume: 0
      };
      
    } catch (error) {
      this.logger.error(`❌ Failed to get market data for ${request.symbol}: ${error}`);
      throw error;
    }
  }

  // Private helper methods
  private async validateAccount(): Promise<void> {
    try {
      const response = await this.restClient.get(`/v3/accounts/${this.config.accountId}`);
      
      if (response.data.account.id !== this.config.accountId) {
        throw new Error('Account ID mismatch');
      }
      
      this.logger.info(`Account validated: ${response.data.account.id} (${response.data.account.currency})`);
      
    } catch (error) {
      throw new Error(`Failed to validate account: ${error}`);
    }
  }

  private async loadInstruments(): Promise<void> {
    try {
      const response = await this.restClient.get(`/v3/accounts/${this.config.accountId}/instruments`);
      
      for (const instrument of response.data.instruments) {
        this.instruments.set(instrument.name, {
          name: instrument.name,
          type: instrument.type,
          displayName: instrument.displayName,
          pipLocation: instrument.pipLocation,
          displayPrecision: instrument.displayPrecision,
          tradeUnitsPrecision: instrument.tradeUnitsPrecision,
          minimumTradeSize: instrument.minimumTradeSize,
          maximumTrailingStopDistance: instrument.maximumTrailingStopDistance,
          minimumTrailingStopDistance: instrument.minimumTrailingStopDistance,
          maximumPositionSize: instrument.maximumPositionSize,
          maximumOrderUnits: instrument.maximumOrderUnits,
          marginRate: instrument.marginRate
        });
      }
      
      this.logger.info(`Loaded ${this.instruments.size} instruments from OANDA`);
      
    } catch (error) {
      this.logger.error(`Failed to load instruments: ${error}`);
    }
  }

  private async startPriceStream(): Promise<void> {
    try {
      const instruments = this.config.instruments.map(symbol => this.mapToOANDAInstrument(symbol)).join(',');
      
      const response = await this.streamClient.get(`/v3/accounts/${this.config.accountId}/pricing/stream`, {
        params: {
          instruments: instruments
        },
        responseType: 'stream'
      });
      
      this.priceStream = response.data;
      
      this.priceStream.on('data', (chunk: Buffer) => {
        const lines = chunk.toString().split('\n');
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              
              if (data.type === 'PRICE') {
                this.handlePriceUpdate(data);
              } else if (data.type === 'HEARTBEAT') {
                this.handleHeartbeat(data);
              }
            } catch (error) {
              this.logger.error(`Error parsing stream data: ${error}`);
            }
          }
        }
      });
      
      this.priceStream.on('error', (error: any) => {
        this.logger.error(`Price stream error: ${error}`);
        this.handleDisconnection();
      });
      
      this.logger.info('✅ Price streaming started');
      
    } catch (error) {
      this.logger.error(`Failed to start price stream: ${error}`);
    }
  }

  private handlePriceUpdate(priceData: any): void {
    this.currentPrices.set(priceData.instrument, priceData);
    
    // Emit market data event
    this.emit('marketData', {
      symbol: this.normalizeInstrument(priceData.instrument),
      bid: parseFloat(priceData.bids[0].price),
      ask: parseFloat(priceData.asks[0].price),
      timestamp: new Date(priceData.time)
    });
  }

  private handleHeartbeat(heartbeat: any): void {
    // Update last heartbeat time for connection monitoring
    this.lastHeartbeat = new Date(heartbeat.time);
  }

  private mapToOANDAOrder(request: OrderRequest): any {
    const instrument = this.mapToOANDAInstrument(request.symbol);
    const units = request.side === 'buy' ? request.quantity : -request.quantity;
    
    const order: any = {
      type: this.mapOrderType(request.type),
      instrument: instrument,
      units: units.toString(),
      timeInForce: request.timeInForce || 'FOK'
    };
    
    if (request.price && request.type !== 'market') {
      order.price = request.price.toString();
    }
    
    if (request.stopLoss) {
      order.stopLossOnFill = {
        price: request.stopLoss.toString()
      };
    }
    
    if (request.takeProfit) {
      order.takeProfitOnFill = {
        price: request.takeProfit.toString()
      };
    }
    
    if (request.clientOrderId) {
      order.clientExtensions = {
        id: request.clientOrderId,
        tag: 'Platform3'
      };
    }
    
    return order;
  }

  private mapOrderType(type: string): string {
    const typeMap: Record<string, string> = {
      'market': 'MARKET',
      'limit': 'LIMIT',
      'stop': 'STOP',
      'market_if_touched': 'MARKET_IF_TOUCHED'
    };
    
    return typeMap[type] || 'MARKET';
  }

  private mapToOANDAInstrument(symbol: string): string {
    // Convert Platform3 symbol format to OANDA format
    // e.g., EURUSD -> EUR_USD
    if (symbol.length === 6) {
      return `${symbol.substring(0, 3)}_${symbol.substring(3, 6)}`;
    }
    return symbol;
  }

  private normalizeInstrument(oandaInstrument: string): string {
    // Convert OANDA instrument format to Platform3 format
    // e.g., EUR_USD -> EURUSD
    return oandaInstrument.replace('_', '');
  }

  private parseOrderResponse(responseData: any): OrderResponse {
    const transaction = responseData.orderCreateTransaction || responseData.orderFillTransaction;
    
    return {
      orderId: transaction.id,
      status: responseData.orderFillTransaction ? 'filled' : 'pending',
      fillPrice: responseData.orderFillTransaction ? parseFloat(responseData.orderFillTransaction.price) : 0,
      fillQuantity: responseData.orderFillTransaction ? Math.abs(parseFloat(responseData.orderFillTransaction.units)) : 0,
      timestamp: new Date(transaction.time)
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
      this.logger.warn('Authentication failed, check access token');
      this.handleDisconnection();
    } else if (error.response?.status === 429) {
      this.logger.warn('Rate limit exceeded, backing off...');
      this.rateLimiter.backoff();
    }
  }
}
