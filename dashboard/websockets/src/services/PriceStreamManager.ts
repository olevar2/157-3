// Price Stream Manager - Handles real-time price updates from Market Data Service

import { Server as SocketIOServer, Socket } from 'socket.io';
import { Logger } from 'winston';
import axios from 'axios';

export interface PriceUpdate {
  symbol: string;
  bid: number;
  ask: number;
  timestamp: number;
  change: number;
  changePercent: number;
  volume?: number;
  high?: number;
  low?: number;
}

export interface PriceSubscription {
  socketId: string;
  userId: string;
  symbols: string[];
  lastUpdate: number;
}

export class PriceStreamManager {
  private io: SocketIOServer;
  private logger: Logger;
  private subscriptions: Map<string, PriceSubscription> = new Map();
  private priceCache: Map<string, PriceUpdate> = new Map();
  private updateInterval: NodeJS.Timeout | null = null;
  private marketDataServiceUrl: string;

  constructor(io: SocketIOServer, logger: Logger) {
    this.io = io;
    this.logger = logger;
    this.marketDataServiceUrl = process.env.MARKET_DATA_SERVICE_URL || 'http://localhost:3004';
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing Price Stream Manager...');
    
    // Start price update polling
    this.startPriceUpdates();
    
    // Test connection to Market Data Service
    try {
      await this.testMarketDataConnection();
      this.logger.info('✅ Connected to Market Data Service');
    } catch (error) {
      this.logger.warn('⚠️ Market Data Service not available, using mock data');
    }
  }

  private async testMarketDataConnection(): Promise<void> {
    const response = await axios.get(`${this.marketDataServiceUrl}/health`, { timeout: 5000 });
    if (response.status !== 200) {
      throw new Error('Market Data Service health check failed');
    }
  }

  async subscribeToPrices(socket: Socket, symbols: string[]): Promise<void> {
    const userId = socket.data.userId;
    const subscription: PriceSubscription = {
      socketId: socket.id,
      userId,
      symbols,
      lastUpdate: Date.now()
    };

    this.subscriptions.set(socket.id, subscription);
    
    // Join symbol-specific rooms
    symbols.forEach(symbol => {
      socket.join(`prices:${symbol}`);
    });

    // Send current prices immediately
    await this.sendCurrentPrices(socket, symbols);
    
    this.logger.info(`Price subscription added for user ${userId}: ${symbols.join(', ')}`);
  }

  async unsubscribeFromPrices(socket: Socket, symbols: string[]): Promise<void> {
    const subscription = this.subscriptions.get(socket.id);
    if (!subscription) return;

    // Remove symbols from subscription
    subscription.symbols = subscription.symbols.filter(s => !symbols.includes(s));
    
    // Leave symbol-specific rooms
    symbols.forEach(symbol => {
      socket.leave(`prices:${symbol}`);
    });

    if (subscription.symbols.length === 0) {
      this.subscriptions.delete(socket.id);
    }

    this.logger.info(`Price unsubscription for socket ${socket.id}: ${symbols.join(', ')}`);
  }

  cleanupUserSubscriptions(socket: Socket): void {
    this.subscriptions.delete(socket.id);
    this.logger.info(`Cleaned up price subscriptions for socket ${socket.id}`);
  }

  private startPriceUpdates(): void {
    // Update prices every 1 second
    this.updateInterval = setInterval(async () => {
      try {
        await this.fetchAndBroadcastPrices();
      } catch (error) {
        this.logger.error('Error in price update cycle:', error);
      }
    }, 1000);

    this.logger.info('Price update cycle started (1 second interval)');
  }

  private async fetchAndBroadcastPrices(): Promise<void> {
    // Get all unique symbols from subscriptions
    const allSymbols = new Set<string>();
    this.subscriptions.forEach(sub => {
      sub.symbols.forEach(symbol => allSymbols.add(symbol));
    });

    if (allSymbols.size === 0) return;

    // Fetch prices from Market Data Service or generate mock data
    const prices = await this.fetchPrices(Array.from(allSymbols));
    
    // Update cache and broadcast
    prices.forEach(price => {
      this.priceCache.set(price.symbol, price);
      this.broadcastPriceUpdate(price);
    });
  }

  private async fetchPrices(symbols: string[]): Promise<PriceUpdate[]> {
    try {
      // Try to fetch from Market Data Service
      const response = await axios.get(`${this.marketDataServiceUrl}/api/market-data/prices`, {
        params: { symbols: symbols.join(',') },
        timeout: 2000
      });

      if (response.data && response.data.prices) {
        return response.data.prices.map((price: any) => this.formatPriceUpdate(price));
      }
    } catch (error) {
      this.logger.debug('Market Data Service unavailable, using mock data');
    }

    // Fallback to mock data
    return this.generateMockPrices(symbols);
  }

  private formatPriceUpdate(rawPrice: any): PriceUpdate {
    return {
      symbol: rawPrice.symbol,
      bid: parseFloat(rawPrice.bid),
      ask: parseFloat(rawPrice.ask),
      timestamp: Date.now(),
      change: parseFloat(rawPrice.change || 0),
      changePercent: parseFloat(rawPrice.changePercent || 0),
      volume: rawPrice.volume ? parseFloat(rawPrice.volume) : undefined,
      high: rawPrice.high ? parseFloat(rawPrice.high) : undefined,
      low: rawPrice.low ? parseFloat(rawPrice.low) : undefined
    };
  }

  private generateMockPrices(symbols: string[]): PriceUpdate[] {
    const mockPrices: { [key: string]: { base: number, spread: number } } = {
      'EURUSD': { base: 1.0850, spread: 0.0002 },
      'GBPUSD': { base: 1.2650, spread: 0.0003 },
      'USDJPY': { base: 149.50, spread: 0.02 },
      'AUDUSD': { base: 0.6750, spread: 0.0002 },
      'USDCAD': { base: 1.3580, spread: 0.0003 },
      'USDCHF': { base: 0.8920, spread: 0.0002 },
      'NZDUSD': { base: 0.6150, spread: 0.0003 },
      'EURGBP': { base: 0.8580, spread: 0.0002 }
    };

    return symbols.map(symbol => {
      const mock = mockPrices[symbol] || { base: 1.0000, spread: 0.0001 };
      
      // Add some realistic price movement
      const volatility = 0.0001;
      const movement = (Math.random() - 0.5) * volatility;
      const currentPrice = mock.base + movement;
      
      const bid = currentPrice - mock.spread / 2;
      const ask = currentPrice + mock.spread / 2;
      
      // Calculate change (mock)
      const previousPrice = this.priceCache.get(symbol)?.bid || currentPrice;
      const change = bid - previousPrice;
      const changePercent = (change / previousPrice) * 100;

      return {
        symbol,
        bid: parseFloat(bid.toFixed(5)),
        ask: parseFloat(ask.toFixed(5)),
        timestamp: Date.now(),
        change: parseFloat(change.toFixed(5)),
        changePercent: parseFloat(changePercent.toFixed(3)),
        volume: Math.floor(Math.random() * 1000000),
        high: parseFloat((currentPrice + Math.random() * 0.001).toFixed(5)),
        low: parseFloat((currentPrice - Math.random() * 0.001).toFixed(5))
      };
    });
  }

  private broadcastPriceUpdate(price: PriceUpdate): void {
    // Broadcast to all subscribers of this symbol
    this.io.to(`prices:${price.symbol}`).emit('price:update', price);
    
    // Also broadcast to global price feed
    this.io.to('global').emit('price:tick', {
      symbol: price.symbol,
      bid: price.bid,
      ask: price.ask,
      change: price.change,
      timestamp: price.timestamp
    });
  }

  private async sendCurrentPrices(socket: Socket, symbols: string[]): Promise<void> {
    const currentPrices: PriceUpdate[] = [];
    
    for (const symbol of symbols) {
      const cachedPrice = this.priceCache.get(symbol);
      if (cachedPrice) {
        currentPrices.push(cachedPrice);
      } else {
        // Generate fresh price if not in cache
        const mockPrices = this.generateMockPrices([symbol]);
        currentPrices.push(...mockPrices);
      }
    }

    socket.emit('prices:initial', {
      prices: currentPrices,
      timestamp: Date.now()
    });
  }

  // Public method to get current prices (for HTTP API)
  getCurrentPrices(symbols?: string[]): PriceUpdate[] {
    if (!symbols) {
      return Array.from(this.priceCache.values());
    }
    
    return symbols
      .map(symbol => this.priceCache.get(symbol))
      .filter((price): price is PriceUpdate => price !== undefined);
  }

  // Get subscription statistics
  getSubscriptionStats(): any {
    const stats = {
      totalSubscriptions: this.subscriptions.size,
      uniqueUsers: new Set(Array.from(this.subscriptions.values()).map(s => s.userId)).size,
      symbolCounts: {} as { [symbol: string]: number }
    };

    this.subscriptions.forEach(sub => {
      sub.symbols.forEach(symbol => {
        stats.symbolCounts[symbol] = (stats.symbolCounts[symbol] || 0) + 1;
      });
    });

    return stats;
  }

  // Cleanup on shutdown
  destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.subscriptions.clear();
    this.priceCache.clear();
    this.logger.info('Price Stream Manager destroyed');
  }
}
