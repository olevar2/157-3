/**
 * Session Cache Manager for Forex Trading Platform
 * Manages trading session state with Redis for ultra-fast access
 * Optimized for scalping and day trading operations
 */

import Redis from 'ioredis';
import { EventEmitter } from 'events';

interface TradingSession {
  sessionId: string;
  traderId: string;
  activePositions: number;
  totalPnL: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  maxTrades: number;
  maxLoss: number;
  sessionStart: number;
  lastActivity: number;
  tradingPairs: string[];
  strategy: string;
  timeframe: string;
}

interface PositionData {
  positionId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  stopLoss?: number;
  takeProfit?: number;
  unrealizedPnL: number;
  entryTime: number;
  lastUpdate: number;
}

interface TradingSignal {
  signalId: string;
  symbol: string;
  signalType: 'BUY' | 'SELL' | 'CLOSE';
  confidence: number;
  price: number;
  timeframe: string;
  strategy: string;
  timestamp: number;
  expirationTime: number;
}

export class SessionCacheManager extends EventEmitter {
  private redis: Redis.Cluster;
  private pubSubRedis: Redis.Cluster;
  private isConnected: boolean = false;

  constructor() {
    super();
    this.initializeRedis();
  }

  private initializeRedis(): void {
    // Initialize Redis cluster connection
    this.redis = new Redis.Cluster([
      { port: 7000, host: '127.0.0.1' },
      { port: 7001, host: '127.0.0.1' },
      { port: 7002, host: '127.0.0.1' },
    ], {
      redisOptions: {
        password: 'ScalpingRedis2025!',
        lazyConnect: true,
        maxRetriesPerRequest: 3,
        retryDelayOnFailover: 100,
        connectTimeout: 10000,
        commandTimeout: 5000,
      },
      enableOfflineQueue: false,
      scaleReads: 'slave',
    });

    // Separate connection for pub/sub
    this.pubSubRedis = new Redis.Cluster([
      { port: 7000, host: '127.0.0.1' },
      { port: 7001, host: '127.0.0.1' },
      { port: 7002, host: '127.0.0.1' },
    ], {
      redisOptions: {
        password: 'ScalpingRedis2025!',
        lazyConnect: true,
      },
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redis.on('ready', () => {
      this.isConnected = true;
      console.log('✅ Redis cluster connected for session management');
      this.emit('connected');
    });

    this.redis.on('error', (error) => {
      console.error('❌ Redis cluster error:', error);
      this.isConnected = false;
      this.emit('error', error);
    });

    this.redis.on('close', () => {
      this.isConnected = false;
      console.log('⚠️ Redis cluster connection closed');
      this.emit('disconnected');
    });
  }

  /**
   * Initialize trading session
   */
  async createSession(session: TradingSession): Promise<void> {
    const sessionKey = `session:${session.sessionId}`;
    
    const sessionData = {
      trader_id: session.traderId,
      active_positions: session.activePositions,
      total_pnl: session.totalPnL,
      risk_level: session.riskLevel,
      max_trades: session.maxTrades,
      max_loss: session.maxLoss,
      session_start: session.sessionStart,
      last_activity: session.lastActivity,
      trading_pairs: session.tradingPairs.join(','),
      strategy: session.strategy,
      timeframe: session.timeframe,
      status: 'ACTIVE',
    };

    await this.redis.hset(sessionKey, sessionData);
    await this.redis.expire(sessionKey, 14400); // 4 hours expiration

    // Add to active sessions index
    await this.redis.zadd('active_sessions', session.lastActivity, session.sessionId);

    console.log(`📊 Trading session created: ${session.sessionId}`);
  }

  /**
   * Update session state using Lua script for atomicity
   */
  async updateSession(
    sessionId: string,
    activePositions: number,
    totalPnL: number,
    riskLevel: string
  ): Promise<{ alerts?: string[] }> {
    const sessionKey = `session:${sessionId}`;
    const timestamp = Date.now();

    // Use Lua script for atomic session update
    const result = await this.redis.eval(`
      local session_key = KEYS[1]
      local active_trades = tonumber(ARGV[1])
      local total_pnl = tonumber(ARGV[2])
      local risk_level = ARGV[3]
      local timestamp = tonumber(ARGV[4])
      
      -- Update session data
      redis.call('HSET', session_key,
          'active_trades', active_trades,
          'total_pnl', total_pnl,
          'risk_level', risk_level,
          'last_activity', timestamp
      )
      redis.call('EXPIRE', session_key, 14400)
      
      -- Update session statistics
      redis.call('HINCRBY', session_key, 'total_updates', 1)
      
      -- Check risk limits
      local max_trades = tonumber(redis.call('HGET', session_key, 'max_trades') or 10)
      local max_loss = tonumber(redis.call('HGET', session_key, 'max_loss') or -1000)
      
      local alerts = {}
      if active_trades > max_trades then
          table.insert(alerts, 'MAX_TRADES_EXCEEDED')
      end
      if total_pnl < max_loss then
          table.insert(alerts, 'MAX_LOSS_EXCEEDED')
      end
      
      return cjson.encode({alerts = alerts, updated = true})
    `, 1, sessionKey, activePositions, totalPnL, riskLevel, timestamp);

    return JSON.parse(result as string);
  }

  /**
   * Store trading position with expiration
   */
  async storePosition(position: PositionData): Promise<void> {
    const positionKey = `position:${position.positionId}`;
    
    const positionData = {
      symbol: position.symbol,
      side: position.side,
      quantity: position.quantity,
      entry_price: position.entryPrice,
      current_price: position.currentPrice,
      stop_loss: position.stopLoss || 0,
      take_profit: position.takeProfit || 0,
      unrealized_pnl: position.unrealizedPnL,
      entry_time: position.entryTime,
      last_update: position.lastUpdate,
      status: 'OPEN',
    };

    await this.redis.hset(positionKey, positionData);
    await this.redis.expire(positionKey, 86400); // 24 hours

    // Add to active positions index by symbol
    const activeKey = `active_positions:${position.symbol}`;
    await this.redis.zadd(activeKey, position.lastUpdate, position.positionId);
    await this.redis.expire(activeKey, 86400);
  }

  /**
   * Update position with price and P&L using Lua script
   */
  async updatePosition(
    positionId: string,
    currentPrice: number,
    unrealizedPnL: number
  ): Promise<{ exitTriggered?: boolean; reason?: string; price?: number }> {
    const positionKey = `position:${positionId}`;
    const timestamp = Date.now();

    const result = await this.redis.eval(`
      local position_key = KEYS[1]
      local current_price = tonumber(ARGV[1])
      local unrealized_pnl = tonumber(ARGV[2])
      local timestamp = tonumber(ARGV[3])
      
      -- Check if position exists
      if redis.call('EXISTS', position_key) == 0 then
          return cjson.encode({error = 'Position not found'})
      end
      
      -- Get current position data
      local position = redis.call('HGETALL', position_key)
      local pos_data = {}
      for i = 1, #position, 2 do
          pos_data[position[i]] = position[i + 1]
      end
      
      -- Update position fields
      redis.call('HSET', position_key,
          'current_price', current_price,
          'unrealized_pnl', unrealized_pnl,
          'last_update', timestamp
      )
      
      -- Check for stop loss or take profit
      local entry_price = tonumber(pos_data.entry_price or 0)
      local stop_loss = tonumber(pos_data.stop_loss or 0)
      local take_profit = tonumber(pos_data.take_profit or 0)
      local side = pos_data.side or 'BUY'
      
      local trigger_exit = false
      local exit_reason = ''
      
      if side == 'BUY' then
          if stop_loss > 0 and current_price <= stop_loss then
              trigger_exit = true
              exit_reason = 'STOP_LOSS'
          elseif take_profit > 0 and current_price >= take_profit then
              trigger_exit = true
              exit_reason = 'TAKE_PROFIT'
          end
      else -- SELL
          if stop_loss > 0 and current_price >= stop_loss then
              trigger_exit = true
              exit_reason = 'STOP_LOSS'
          elseif take_profit > 0 and current_price <= take_profit then
              trigger_exit = true
              exit_reason = 'TAKE_PROFIT'
          end
      end
      
      if trigger_exit then
          -- Add to exit queue
          local exit_data = position_key .. '|' .. exit_reason .. '|' .. current_price .. '|' .. timestamp
          redis.call('LPUSH', 'exit_queue', exit_data)
          redis.call('EXPIRE', 'exit_queue', 300)
          
          return cjson.encode({exitTriggered = true, reason = exit_reason, price = current_price})
      end
      
      return cjson.encode({updated = true, pnl = unrealized_pnl})
    `, 1, positionKey, currentPrice, unrealizedPnL, timestamp);

    return JSON.parse(result as string);
  }

  /**
   * Store trading signal with conflict detection
   */
  async storeSignal(signal: TradingSignal): Promise<{ stored: boolean; conflict?: string }> {
    const signalKey = `signal:${signal.signalId}`;
    const indexKey = `signals_index:${signal.symbol}`;

    const result = await this.redis.eval(`
      local signal_key = KEYS[1]
      local index_key = KEYS[2]
      local signal_type = ARGV[1]
      local confidence = tonumber(ARGV[2])
      local price = tonumber(ARGV[3])
      local symbol = ARGV[4]
      local timestamp = tonumber(ARGV[5])
      
      -- Validate signal confidence
      if confidence < 50 then
          return cjson.encode({error = 'Signal confidence too low'})
      end
      
      -- Check for conflicting signals in last 30 seconds
      local recent_signals = redis.call('ZRANGEBYSCORE', index_key, timestamp - 30, timestamp)
      for i = 1, #recent_signals do
          local signal_data = redis.call('HGETALL', recent_signals[i])
          if signal_data and #signal_data > 0 then
              for j = 1, #signal_data, 2 do
                  if signal_data[j] == 'signal_type' and signal_data[j + 1] ~= signal_type then
                      return cjson.encode({error = 'Conflicting signal detected', conflict_with = signal_data[j + 1]})
                  end
              end
          end
      end
      
      -- Store signal data
      redis.call('HSET', signal_key,
          'signal_type', signal_type,
          'confidence', confidence,
          'price', price,
          'symbol', symbol,
          'timestamp', timestamp,
          'status', 'ACTIVE'
      )
      redis.call('EXPIRE', signal_key, 300)
      
      -- Add to signals index
      redis.call('ZADD', index_key, timestamp, signal_key)
      
      -- Keep only last 100 signals
      local count = redis.call('ZCARD', index_key)
      if count > 100 then
          redis.call('ZREMRANGEBYRANK', index_key, 0, count - 101)
      end
      
      -- Publish signal to subscribers
      local signal_channel = 'signals:' .. symbol
      local signal_message = signal_type .. '|' .. confidence .. '|' .. price .. '|' .. timestamp
      redis.call('PUBLISH', signal_channel, signal_message)
      
      return cjson.encode({stored = true, published = true, confidence = confidence})
    `, 2, signalKey, indexKey, signal.signalType, signal.confidence, signal.price, signal.symbol, signal.timestamp);

    return JSON.parse(result as string);
  }

  /**
   * Get active trading session
   */
  async getSession(sessionId: string): Promise<TradingSession | null> {
    const sessionKey = `session:${sessionId}`;
    const sessionData = await this.redis.hgetall(sessionKey);

    if (Object.keys(sessionData).length === 0) {
      return null;
    }

    return {
      sessionId,
      traderId: sessionData.trader_id,
      activePositions: parseInt(sessionData.active_positions),
      totalPnL: parseFloat(sessionData.total_pnl),
      riskLevel: sessionData.risk_level as any,
      maxTrades: parseInt(sessionData.max_trades),
      maxLoss: parseFloat(sessionData.max_loss),
      sessionStart: parseInt(sessionData.session_start),
      lastActivity: parseInt(sessionData.last_activity),
      tradingPairs: sessionData.trading_pairs.split(','),
      strategy: sessionData.strategy,
      timeframe: sessionData.timeframe,
    };
  }

  /**
   * Get active positions for a symbol
   */
  async getActivePositions(symbol: string): Promise<PositionData[]> {
    const activeKey = `active_positions:${symbol}`;
    const positionIds = await this.redis.zrevrange(activeKey, 0, -1);

    const positions: PositionData[] = [];
    for (const positionId of positionIds) {
      const positionKey = `position:${positionId}`;
      const positionData = await this.redis.hgetall(positionKey);

      if (Object.keys(positionData).length > 0) {
        positions.push({
          positionId,
          symbol: positionData.symbol,
          side: positionData.side as 'BUY' | 'SELL',
          quantity: parseFloat(positionData.quantity),
          entryPrice: parseFloat(positionData.entry_price),
          currentPrice: parseFloat(positionData.current_price),
          stopLoss: parseFloat(positionData.stop_loss) || undefined,
          takeProfit: parseFloat(positionData.take_profit) || undefined,
          unrealizedPnL: parseFloat(positionData.unrealized_pnl),
          entryTime: parseInt(positionData.entry_time),
          lastUpdate: parseInt(positionData.last_update),
        });
      }
    }

    return positions;
  }

  /**
   * Subscribe to real-time signals
   */
  async subscribeToSignals(symbol: string, callback: (signal: any) => void): Promise<void> {
    const channel = `signals:${symbol}`;
    
    await this.pubSubRedis.subscribe(channel);
    
    this.pubSubRedis.on('message', (receivedChannel, message) => {
      if (receivedChannel === channel) {
        const [signalType, confidence, price, timestamp] = message.split('|');
        callback({
          symbol,
          signalType,
          confidence: parseFloat(confidence),
          price: parseFloat(price),
          timestamp: parseInt(timestamp),
        });
      }
    });

    console.log(`📡 Subscribed to signals for ${symbol}`);
  }

  /**
   * Get exit queue for processing
   */
  async getExitQueue(): Promise<any[]> {
    const exitData = await this.redis.lrange('exit_queue', 0, -1);
    const exits = [];

    for (const data of exitData) {
      const [positionKey, reason, price, timestamp] = data.split('|');
      const positionId = positionKey.replace('position:', '');
      
      exits.push({
        positionId,
        reason,
        price: parseFloat(price),
        timestamp: parseInt(timestamp),
      });
    }

    return exits;
  }

  /**
   * Clear processed exits from queue
   */
  async clearExitQueue(): Promise<void> {
    await this.redis.del('exit_queue');
  }

  /**
   * Get performance metrics
   */
  async getPerformanceMetrics(symbol: string): Promise<any> {
    const perfKey = `performance:${symbol}`;
    const metrics = await this.redis.hgetall(perfKey);

    if (Object.keys(metrics).length === 0) {
      return null;
    }

    return {
      totalTrades: parseInt(metrics.total_trades || '0'),
      winningTrades: parseInt(metrics.winning_trades || '0'),
      losingTrades: parseInt(metrics.losing_trades || '0'),
      winRate: parseFloat(metrics.win_rate || '0'),
      totalPnL: parseFloat(metrics.total_pnl || '0'),
      avgDuration: parseFloat(metrics.avg_duration || '0'),
    };
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; latency: number }> {
    const start = Date.now();
    
    try {
      await this.redis.ping();
      const latency = Date.now() - start;
      return { status: 'healthy', latency };
    } catch (error) {
      return { status: 'unhealthy', latency: -1 };
    }
  }

  /**
   * Close connections
   */
  async close(): Promise<void> {
    await this.redis.quit();
    await this.pubSubRedis.quit();
    this.isConnected = false;
    console.log('🔌 Redis connections closed');
  }
}
