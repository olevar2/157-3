/**
 * Platform3 Python Engine Client
 * High-performance communication bridge between TypeScript services and Python AI engines
 * Optimized for <1ms latency and 24/7 operation
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import http from 'http';
import https from 'https';
import WebSocket from 'ws';
import { EventEmitter } from 'events';

export interface PythonEngineConfig {
  baseURL: string;
  timeout: number;
  maxRetries: number;
  enableHealthChecks: boolean;
  websocketURL?: string;
  apiKey?: string;
  // New connection pooling options
  maxSockets?: number;
  keepAlive?: boolean;
  keepAliveMsecs?: number;
  enableHttp2?: boolean;
  requestConcurrency?: number;
}

export interface TradingSignalRequest {
  symbol: string;
  timeframe: string;
  current_price: number;
  risk_level: string;
  market_conditions?: any;
}

export interface TradingSignalResponse {
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  risk_level: string;
  entry_price?: number;
  stop_loss?: number;
  take_profit?: number;
  position_size?: number;
  reasoning: string;
  timestamp: string;
}

export interface MarketAnalysisRequest {
  symbol: string;
  timeframe: string;
  indicators: string[];
  depth?: number;
}

export interface MarketAnalysisResponse {
  symbol: string;
  timeframe: string;
  technical_indicators: Record<string, any>;
  trend_analysis: {
    direction: 'bullish' | 'bearish' | 'sideways';
    strength: number;
    duration: string;
  };
  support_resistance: {
    support_levels: number[];
    resistance_levels: number[];
  };
  volatility: {
    current: number;
    percentile: number;
    classification: 'low' | 'medium' | 'high' | 'extreme';
  };
  timestamp: string;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  engines: Record<string, boolean>;
  latency_ms: number;
  timestamp: string;
  version: string;
}

export interface RiskAssessmentRequest {
  symbol: string;
  position_size: number;
  account_balance: number;
  existing_positions: any[];
  market_conditions: any;
}

export interface RiskAssessmentResponse {
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'extreme';
  max_position_size: number;
  warnings: string[];
  recommendations: string[];
  timestamp: string;
}

/**
 * High-performance Python Engine Client with WebSocket support
 */
export class PythonEngineClient extends EventEmitter {
  private httpClient: AxiosInstance;
  private wsClient: WebSocket | null = null;
  private config: PythonEngineConfig;
  private isConnected: boolean = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private httpAgent: http.Agent | https.Agent;
  private httpsAgent: http.Agent | https.Agent;
  private requestQueue: Array<() => Promise<any>> = [];
  private processingQueue: boolean = false;
  private performanceMetrics = {
    totalRequests: 0,
    averageLatency: 0,
    connectionPoolHits: 0,
    connectionPoolMisses: 0
  };

  constructor(config: PythonEngineConfig) {
    super();
    this.config = {
      maxSockets: 50,
      keepAlive: true,
      keepAliveMsecs: 1000,
      enableHttp2: false, // Enable when server supports it
      requestConcurrency: 10,
      ...config
    };
    
    // Create optimized HTTP agents with connection pooling
    this.httpAgent = new http.Agent({
      keepAlive: this.config.keepAlive,
      keepAliveMsecs: this.config.keepAliveMsecs,
      maxSockets: this.config.maxSockets,
      maxFreeSockets: Math.floor(this.config.maxSockets! / 2),
      timeout: this.config.timeout / 2, // Connection timeout
      scheduling: 'fifo' // First-in-first-out for predictable performance
    });

    this.httpsAgent = new https.Agent({
      keepAlive: this.config.keepAlive,
      keepAliveMsecs: this.config.keepAliveMsecs,
      maxSockets: this.config.maxSockets,
      maxFreeSockets: Math.floor(this.config.maxSockets! / 2),
      timeout: this.config.timeout / 2,
      scheduling: 'fifo'
    });
    
    // Initialize HTTP client with performance optimizations
    this.httpClient = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'Platform3-TypeScript-Client/1.0.0',
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate',
        ...(config.apiKey && { 'Authorization': `Bearer ${config.apiKey}` })
      },
      // Performance optimizations
      maxRedirects: 0,
      validateStatus: (status) => status < 500,
      // Use optimized HTTP agents
      httpAgent: this.httpAgent,
      httpsAgent: this.httpsAgent,
      // Additional optimizations
      decompress: true, // Automatic response decompression
      maxContentLength: 50 * 1024 * 1024, // 50MB max response
      maxBodyLength: 50 * 1024 * 1024, // 50MB max request body
    });

    // Setup request/response interceptors for monitoring
    this.setupInterceptors();
    
    // Initialize WebSocket if URL provided
    if (config.websocketURL) {
      this.initializeWebSocket();
    }
    
    // Start health checks if enabled
    if (config.enableHealthChecks) {
      this.startHealthChecks();
    }
  }

  /**
   * Setup axios interceptors for performance monitoring
   */
  private setupInterceptors(): void {
    // Request interceptor
    this.httpClient.interceptors.request.use(
      (config) => {
        config.metadata = { startTime: Date.now() };
        return config;
      },
      (error) => {
        this.emit('request_error', { error: error.message, timestamp: new Date().toISOString() });
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.httpClient.interceptors.response.use(
      (response) => {
        const duration = Date.now() - response.config.metadata.startTime;
        this.emit('request_completed', {
          method: response.config.method?.toUpperCase(),
          url: response.config.url,
          status: response.status,
          duration_ms: duration,
          timestamp: new Date().toISOString()
        });
        return response;
      },
      (error) => {
        const duration = error.config ? Date.now() - error.config.metadata.startTime : 0;
        this.emit('request_error', {
          method: error.config?.method?.toUpperCase(),
          url: error.config?.url,
          error: error.message,
          duration_ms: duration,
          timestamp: new Date().toISOString()
        });
        return Promise.reject(error);
      }
    );
  }

  /**
   * Initialize WebSocket connection for real-time data
   */
  private initializeWebSocket(): void {
    if (!this.config.websocketURL) return;

    try {
      this.wsClient = new WebSocket(this.config.websocketURL);
      
      this.wsClient.on('open', () => {
        console.log('✅ WebSocket connected to Python engines');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('websocket_connected');
      });
      
      this.wsClient.on('message', (data: WebSocket.RawData) => {
        try {
          const message = JSON.parse(data.toString());
          this.emit('real_time_data', message);
        } catch (error) {
          console.error('WebSocket message parse error:', error);
        }
      });
      
      this.wsClient.on('close', () => {
        console.log('WebSocket disconnected from Python engines');
        this.isConnected = false;
        this.emit('websocket_disconnected');
        this.attemptReconnect();
      });
      
      this.wsClient.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.emit('websocket_error', error);
      });
    } catch (error) {
      console.error('WebSocket initialization error:', error);
    }
  }

  /**
   * Attempt to reconnect WebSocket with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max WebSocket reconnection attempts reached');
      return;
    }

    const backoffDelay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    
    setTimeout(() => {
      console.log(`Attempting WebSocket reconnection (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.initializeWebSocket();
    }, backoffDelay);
  }

  /**
   * Start periodic health checks
   */
  private startHealthChecks(): void {
    this.healthCheckInterval = setInterval(async () => {
      try {
        const health = await this.checkHealth();
        this.emit('health_check', { healthy: health.status === 'healthy', data: health });
      } catch (error) {
        this.emit('health_check', { healthy: false, error: error.message });
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Test connection to Python engines
   */
  async testConnection(): Promise<boolean> {
    try {
      const response = await this.httpClient.get('/health', { timeout: 5000 });
      return response.status === 200;
    } catch (error) {
      console.warn('Python engine connection test failed:', error.message);
      return false;
    }
  }

  /**
   * Get health status from Python engines
   */
  async checkHealth(): Promise<HealthCheckResponse> {
    const response: AxiosResponse<HealthCheckResponse> = await this.httpClient.get('/health');
    return response.data;
  }

  /**
   * Get AI trading signals from Python engines
   */
  async getTradingSignals(request: TradingSignalRequest): Promise<TradingSignalResponse> {
    const response: AxiosResponse<TradingSignalResponse> = await this.httpClient.post('/api/v1/trading/signals', request);
    return response.data;
  }

  /**
   * Get comprehensive market analysis from Python engines
   */
  async getMarketAnalysis(request: MarketAnalysisRequest): Promise<MarketAnalysisResponse> {
    const response: AxiosResponse<MarketAnalysisResponse> = await this.httpClient.post('/api/v1/analysis/market', request);
    return response.data;
  }

  /**
   * Get risk assessment from Python engines
   */
  async getRiskAssessment(request: RiskAssessmentRequest): Promise<RiskAssessmentResponse> {
    const response: AxiosResponse<RiskAssessmentResponse> = await this.httpClient.post('/api/v1/risk/assess', request);
    return response.data;
  }

  /**
   * Execute ML prediction request
   */
  async getMLPredictions(symbol: string, timeframe: string, horizon: string): Promise<any> {
    const response = await this.httpClient.post('/api/v1/ml/predict', {
      symbol,
      timeframe,
      horizon
    });
    return response.data;
  }

  /**
   * Get pattern recognition results
   */
  async getPatternAnalysis(symbol: string, timeframe: string): Promise<any> {
    const response = await this.httpClient.post('/api/v1/patterns/detect', {
      symbol,
      timeframe
    });
    return response.data;
  }

  /**
   * Send real-time market data via WebSocket
   */
  sendMarketData(data: any): void {
    if (this.wsClient && this.isConnected) {
      this.wsClient.send(JSON.stringify({
        type: 'market_data',
        data,
        timestamp: new Date().toISOString()
      }));
    }
  }

  /**
   * Subscribe to real-time signals via WebSocket
   */
  subscribeToSignals(symbols: string[]): void {
    if (this.wsClient && this.isConnected) {
      this.wsClient.send(JSON.stringify({
        type: 'subscribe',
        symbols,
        timestamp: new Date().toISOString()
      }));
    }
  }

  /**
   * Unsubscribe from real-time signals
   */
  unsubscribeFromSignals(symbols: string[]): void {
    if (this.wsClient && this.isConnected) {
      this.wsClient.send(JSON.stringify({
        type: 'unsubscribe',
        symbols,
        timestamp: new Date().toISOString()
      }));
    }
  }

  /**
   * Get current connection status
   */
  getConnectionStatus(): { http: boolean; websocket: boolean } {
    return {
      http: true, // HTTP client is always available
      websocket: this.isConnected
    };
  }

  /**
   * Close all connections gracefully
   */
  close(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    
    if (this.wsClient) {
      this.wsClient.close();
      this.wsClient = null;
    }
    
    this.isConnected = false;
    this.emit('client_closed');
  }

  /**
   * Retry wrapper with exponential backoff
   */
  private async withRetry<T>(operation: () => Promise<T>): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        if (attempt === this.config.maxRetries) {
          break;
        }
        
        // Exponential backoff with jitter
        const baseDelay = Math.min(100 * Math.pow(2, attempt - 1), 5000);
        const jitter = Math.random() * 100;
        const delay = baseDelay + jitter;
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }
}

export default PythonEngineClient;