/**
 * Platform3 WebSocket Real-Time Communication Client
 * High-performance WebSocket client for real-time data streaming
 * Optimized for <1ms latency trading data
 */

import WebSocket from 'ws';
import { EventEmitter } from 'events';

export interface WebSocketConfig {
  url: string;
  reconnectDelay: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  enableLogging: boolean;
}

export interface MarketDataUpdate {
  symbol: string;
  bid: number;
  ask: number;
  timestamp: string;
  volume?: number;
  change?: number;
}

export interface TradingSignalUpdate {
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  timestamp: string;
  reasoning: string;
}

export interface RealTimeMessage {
  type: 'market_data' | 'trading_signal' | 'risk_alert' | 'system_status';
  data: any;
  timestamp: string;
  correlation_id?: string;
}

/**
 * High-performance WebSocket client for real-time Platform3 communication
 */
export class Platform3WebSocketClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageQueue: RealTimeMessage[] = [];
  private subscriptions: Set<string> = new Set();
  private performanceMetrics = {
    messagesReceived: 0,
    messagesPerSecond: 0,
    averageLatency: 0,
    lastMessageTime: 0
  };

  constructor(config: WebSocketConfig) {
    super();
    this.config = config;
    this.connect();
  }

  /**
   * Establish WebSocket connection
   */
  private connect(): void {
    try {
      this.ws = new WebSocket(this.config.url);
      
      this.ws.on('open', () => {
        this.log('✅ WebSocket connected to Platform3 Python engine');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        this.processMessageQueue();
        this.emit('connected');
      });

      this.ws.on('message', (data: WebSocket.RawData) => {
        this.handleMessage(data);
      });

      this.ws.on('close', (code: number, reason: Buffer) => {
        this.log(`WebSocket disconnected: ${code} - ${reason.toString()}`);
        this.isConnected = false;
        this.stopHeartbeat();
        this.emit('disconnected', { code, reason: reason.toString() });
        this.scheduleReconnect();
      });

      this.ws.on('error', (error: Error) => {
        this.log(`WebSocket error: ${error.message}`);
        this.emit('error', error);
      });

    } catch (error) {
      this.log(`WebSocket connection failed: ${error.message}`);
      this.scheduleReconnect();
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: WebSocket.RawData): void {
    const startTime = Date.now();
    
    try {
      const message: RealTimeMessage = JSON.parse(data.toString());
      
      // Calculate latency
      const messageTimestamp = new Date(message.timestamp).getTime();
      const latency = startTime - messageTimestamp;
      
      // Update performance metrics
      this.updatePerformanceMetrics(latency);
      
      // Emit specific events based on message type
      switch (message.type) {
        case 'market_data':
          this.emit('marketData', message.data as MarketDataUpdate);
          break;
        case 'trading_signal':
          this.emit('tradingSignal', message.data as TradingSignalUpdate);
          break;
        case 'risk_alert':
          this.emit('riskAlert', message.data);
          break;
        case 'system_status':
          this.emit('systemStatus', message.data);
          break;
        default:
          this.emit('message', message);
      }
      
      // Emit raw message for debugging
      this.emit('rawMessage', message);
      
    } catch (error) {
      this.log(`Message parsing error: ${error.message}`);
      this.emit('parseError', error);
    }
  }

  /**
   * Update performance metrics
   */
  private updatePerformanceMetrics(latency: number): void {
    this.performanceMetrics.messagesReceived++;
    this.performanceMetrics.lastMessageTime = Date.now();
    
    // Calculate running average latency
    const currentAvg = this.performanceMetrics.averageLatency;
    const newAvg = (currentAvg + latency) / 2;
    this.performanceMetrics.averageLatency = newAvg;
    
    // Calculate messages per second (simple approximation)
    const now = Date.now();
    if (this.performanceMetrics.lastMessageTime > 0) {
      const timeDiff = now - this.performanceMetrics.lastMessageTime;
      if (timeDiff > 0) {
        this.performanceMetrics.messagesPerSecond = 1000 / timeDiff;
      }
    }
  }

  /**
   * Subscribe to real-time data for specific symbols
   */
  public subscribe(symbols: string[]): void {
    symbols.forEach(symbol => this.subscriptions.add(symbol));
    
    const message: RealTimeMessage = {
      type: 'market_data',
      data: {
        action: 'subscribe',
        symbols: symbols
      },
      timestamp: new Date().toISOString()
    };
    
    this.sendMessage(message);
  }

  /**
   * Unsubscribe from real-time data
   */
  public unsubscribe(symbols: string[]): void {
    symbols.forEach(symbol => this.subscriptions.delete(symbol));
    
    const message: RealTimeMessage = {
      type: 'market_data',
      data: {
        action: 'unsubscribe',
        symbols: symbols
      },
      timestamp: new Date().toISOString()
    };
    
    this.sendMessage(message);
  }

  /**
   * Send market data update to Python engines
   */
  public sendMarketData(marketData: MarketDataUpdate): void {
    const message: RealTimeMessage = {
      type: 'market_data',
      data: marketData,
      timestamp: new Date().toISOString()
    };
    
    this.sendMessage(message);
  }

  /**
   * Request real-time trading signals
   */
  public requestTradingSignals(symbols: string[]): void {
    const message: RealTimeMessage = {
      type: 'trading_signal',
      data: {
        action: 'request',
        symbols: symbols
      },
      timestamp: new Date().toISOString()
    };
    
    this.sendMessage(message);
  }

  /**
   * Send message to WebSocket
   */
  private sendMessage(message: RealTimeMessage): void {
    if (this.isConnected && this.ws) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        this.log(`Send message error: ${error.message}`);
        this.messageQueue.push(message);
      }
    } else {
      this.messageQueue.push(message);
    }
  }

  /**
   * Process queued messages when connection is restored
   */
  private processMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
    }
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.ws.ping();
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat timer
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.log('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
      return;
    }

    const delay = Math.min(
      this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      30000 // Max 30 seconds
    );
    
    this.reconnectAttempts++;
    this.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Get current connection status
   */
  public getStatus(): {
    connected: boolean;
    subscriptions: string[];
    performance: typeof this.performanceMetrics;
    queueSize: number;
  } {
    return {
      connected: this.isConnected,
      subscriptions: Array.from(this.subscriptions),
      performance: { ...this.performanceMetrics },
      queueSize: this.messageQueue.length
    };
  }

  /**
   * Force close connection
   */
  public close(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
    }
    this.isConnected = false;
    this.emit('closed');
  }

  /**
   * Log message with timestamp
   */
  private log(message: string): void {
    if (this.config.enableLogging) {
      console.log(`[${new Date().toISOString()}] WebSocket: ${message}`);
    }
  }
}

/**
 * Factory function to create WebSocket client with default config
 */
export function createWebSocketClient(url: string, options: Partial<WebSocketConfig> = {}): Platform3WebSocketClient {
  const defaultConfig: WebSocketConfig = {
    url,
    reconnectDelay: 1000,
    maxReconnectAttempts: 10,
    heartbeatInterval: 30000,
    enableLogging: true
  };

  const config = { ...defaultConfig, ...options };
  return new Platform3WebSocketClient(config);
}

export default Platform3WebSocketClient;