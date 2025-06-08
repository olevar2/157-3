/**
 * Platform3 WebSocket Client for Real-Time Python-TypeScript Communication
 * High-performance WebSocket client for real-time data streaming from Python engines
 */

import WebSocket from 'ws';
import { EventEmitter } from 'events';

// Type definitions
export interface MarketDataUpdate {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  timestamp: string;
  volume: number;
  change: number;
  change_percent: number;
}

export interface TradingSignalUpdate {
  symbol: string;
  signal: string; // buy, sell, hold
  confidence: number;
  price_target?: number;
  stop_loss?: number;
  risk_score: number;
  reasoning: string;
  timestamp: string;
}

export interface SystemAlert {
  level: string; // info, warning, error, critical
  message: string;
  source: string;
  timestamp: string;
  data?: Record<string, any>;
}

export interface WebSocketMessage {
  type: string;
  data?: any;
  topic?: string;
  timestamp: string;
}

export interface WebSocketClientConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  pingInterval: number;
  enableHeartbeat: boolean;
}

/**
 * Real-time WebSocket client for Platform3 Python-TypeScript communication
 */
export class PythonWebSocketClient extends EventEmitter {
  private ws?: WebSocket;
  private config: WebSocketClientConfig;
  private reconnectAttempts: number = 0;
  private isConnecting: boolean = false;
  private subscriptions: Set<string> = new Set();
  private heartbeatTimer?: NodeJS.Timeout;
  private reconnectTimer?: NodeJS.Timeout;
  private messageCount: number = 0;
  private lastPingTime?: number;
  private connectionStartTime?: number;

  constructor(config: Partial<WebSocketClientConfig> = {}) {
    super();

    this.config = {
      url: 'ws://localhost:8001',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      pingInterval: 30000,
      enableHeartbeat: true,
      ...config
    };
  }

  /**
   * Connect to WebSocket server
   */
  public async connect(): Promise<void> {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isConnecting = true;
    this.connectionStartTime = Date.now();

    try {
      this.ws = new WebSocket(this.config.url);

      this.ws.on('open', () => {
        console.log('✅ WebSocket connected to Python bridge');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        
        this.emit('connected', {
          timestamp: new Date().toISOString(),
          connectionTime: Date.now() - (this.connectionStartTime || 0)
        });

        // Start heartbeat
        if (this.config.enableHeartbeat) {
          this.startHeartbeat();
        }

        // Re-subscribe to previously subscribed topics
        this.resubscribeAll();
      });

      this.ws.on('message', (data: WebSocket.Data) => {
        try {
          const message: WebSocketMessage = JSON.parse(data.toString());
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      });

      this.ws.on('close', (code: number, reason: string) => {
        console.log(`WebSocket disconnected: ${code} - ${reason}`);
        this.stopHeartbeat();
        
        this.emit('disconnected', {
          code,
          reason,
          timestamp: new Date().toISOString()
        });

        // Attempt reconnection
        if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
          this.scheduleReconnect();
        } else {
          this.emit('max_reconnect_attempts_reached');
        }
      });

      this.ws.on('error', (error: Error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      });

      this.ws.on('ping', () => {
        this.ws?.pong();
      });

      this.ws.on('pong', () => {
        if (this.lastPingTime) {
          const latency = Date.now() - this.lastPingTime;
          this.emit('latency', { latency_ms: latency });
        }
      });

    } catch (error) {
      this.isConnecting = false;
      throw error;
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = undefined;
    }

    this.emit('disconnected', {
      reason: 'client_disconnect',
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Subscribe to a topic
   */
  public subscribe(topic: string): void {
    this.subscriptions.add(topic);
    
    if (this.isConnected()) {
      this.sendMessage({
        type: 'subscribe',
        topic,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Unsubscribe from a topic
   */
  public unsubscribe(topic: string): void {
    this.subscriptions.delete(topic);
    
    if (this.isConnected()) {
      this.sendMessage({
        type: 'unsubscribe',
        topic,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Subscribe to market data for a symbol
   */
  public subscribeToMarketData(symbol: string): void {
    this.subscribe(`market_data.${symbol}`);
  }

  /**
   * Subscribe to trading signals for a symbol
   */
  public subscribeToTradingSignals(symbol: string): void {
    this.subscribe(`trading_signals.${symbol}`);
  }

  /**
   * Subscribe to system alerts
   */
  public subscribeToSystemAlerts(): void {
    this.subscribe('system_alerts');
  }

  /**
   * Send ping to server
   */
  public ping(): void {
    if (this.isConnected()) {
      this.lastPingTime = Date.now();
      this.sendMessage({
        type: 'ping',
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Get connection statistics
   */
  public getStats(): void {
    if (this.isConnected()) {
      this.sendMessage({
        type: 'stats',
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Check if WebSocket is connected
   */
  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get client metrics
   */
  public getMetrics(): {
    connected: boolean;
    messageCount: number;
    subscriptions: string[];
    reconnectAttempts: number;
    uptime?: number;
  } {
    return {
      connected: this.isConnected(),
      messageCount: this.messageCount,
      subscriptions: Array.from(this.subscriptions),
      reconnectAttempts: this.reconnectAttempts,
      uptime: this.connectionStartTime ? Date.now() - this.connectionStartTime : undefined
    };
  }

  // Private methods

  private handleMessage(message: WebSocketMessage): void {
    this.messageCount++;

    switch (message.type) {
      case 'connection_established':
        this.emit('connection_established', message.data);
        break;

      case 'market_data_update':
        this.emit('market_data', message.data as MarketDataUpdate);
        break;

      case 'trading_signal_update':
        this.emit('trading_signal', message.data as TradingSignalUpdate);
        break;

      case 'system_alert':
        this.emit('system_alert', message.data as SystemAlert);
        break;

      case 'subscription_confirmed':
        this.emit('subscription_confirmed', { topic: message.topic });
        break;

      case 'unsubscription_confirmed':
        this.emit('unsubscription_confirmed', { topic: message.topic });
        break;

      case 'pong':
        if (this.lastPingTime) {
          const latency = Date.now() - this.lastPingTime;
          this.emit('pong', { latency_ms: latency });
        }
        break;

      case 'stats_response':
        this.emit('stats', message.data);
        break;

      case 'error':
        this.emit('message_error', message.data);
        break;

      default:
        this.emit('unknown_message', message);
    }
  }

  private sendMessage(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.ping();
    }, this.config.pingInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.config.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1);
    
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnect failed:', error);
      });
    }, delay);
  }

  private resubscribeAll(): void {
    for (const topic of this.subscriptions) {
      this.sendMessage({
        type: 'subscribe',
        topic,
        timestamp: new Date().toISOString()
      });
    }
  }
}

// Export singleton instance
export const pythonWebSocketClient = new PythonWebSocketClient();

export default PythonWebSocketClient;