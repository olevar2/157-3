import express, { Request, Response, NextFunction } from 'express';
import { Pool } from 'pg';
import * as redis from 'redis';
import winston from 'winston';
import { v4 as uuidv4 } from 'uuid';
import jwt from 'jsonwebtoken';
import Joi from 'joi';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import cors from 'cors';
import dotenv from 'dotenv';
import Decimal from 'decimal.js';
import axios from 'axios';
import { PythonEngineClient } from '../../../shared/PythonEngineClient';
// Python Engine Communication Interface
interface PythonEngineResponse {
  success: boolean;
  data?: any;
  error?: string;
}

interface TradingSignal {
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  riskLevel: 'low' | 'medium' | 'high';
  indicators: any[];
}

// Initialize Python Engine Client
const pythonEngine = new PythonEngineClient({
  baseURL: process.env.PYTHON_ENGINE_URL || 'http://localhost:8000',
  timeout: 5000,
  maxRetries: 3,
  enableHealthChecks: true,
  websocketURL: process.env.PYTHON_WS_URL || 'ws://localhost:8000/ws'
});

dotenv.config();

// Types
interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
}

interface AuthRequest extends Request {
  user?: User;
  sessionId?: string;
}

interface CurrencyPair {
  id: string;
  symbol: string;
  baseSymbol: string;
  quoteSymbol: string;
  pipSize: number;
  minTradeSize: number;
  maxTradeSize: number;
  marginRequirement: number;
  isActive: boolean;
}

interface Position {
  id: string;
  userId: string;
  accountId: string;
  pairId: string;
  symbol: string;
  type: 'buy' | 'sell';
  size: number;
  openPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  stopLoss?: number;
  takeProfit?: number;
  margin: number;
  commission: number;
  swap: number;
  openTime: Date;
  status: 'open' | 'closed' | 'pending';
}

interface Order {
  id: string;
  userId: string;
  accountId: string;
  pairId: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  side: 'buy' | 'sell';
  size: number;
  price?: number;
  stopPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  filledSize: number;
  filledPrice?: number;
  createdAt: Date;
  updatedAt: Date;
  expiresAt?: Date;
}

interface TradingAccount {
  id: string;
  userId: string;
  accountNumber: string;
  accountType: 'demo' | 'live' | 'paper';
  baseCurrency: string;
  balance: number;
  equity: number;
  marginAvailable: number;
  marginUsed: number;
  leverage: number;
  createdAt: Date;
}

interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  timestamp: Date;
}

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ 
      filename: 'logs/trading-service-error.log', 
      level: 'error' 
    }),
    new winston.transports.File({ 
      filename: 'logs/trading-service-combined.log' 
    }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME || 'forex_platform',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

pool.on('error', (err) => {
  logger.error('Unexpected error on idle client:', err);
  process.exit(-1);
});

// Redis connection
const redisConfig: any = {
  url: process.env.REDIS_URL || 'redis://localhost:6379',
  socket: {
    connectTimeout: 5000,
  }
};

if (process.env.REDIS_PASSWORD) {
  redisConfig.password = process.env.REDIS_PASSWORD;
}

const redisClient = redis.createClient(redisConfig);

redisClient.on('error', (err: any) => {
  logger.error('Redis connection error:', err);
});

redisClient.on('connect', () => {
  logger.info('Connected to Redis for trading cache');
});

// Connect to Redis
redisClient.connect().catch((err: any) => {
  logger.error('Failed to connect to Redis:', err);
});

// Express app setup
const app = express();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const tradingLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 10, // Limit each IP to 10 trading requests per minute
  message: {
    error: 'Too many trading requests, please try again later.',
    retryAfter: '1 minute'
  },
  standardHeaders: true,
  legacyHeaders: false,
});

// Validation schemas
const createOrderSchema = Joi.object({
  pairId: Joi.string().uuid().required(),
  type: Joi.string().valid('market', 'limit', 'stop', 'stop_limit').required(),
  side: Joi.string().valid('buy', 'sell').required(),
  size: Joi.number().positive().required(),
  price: Joi.number().positive().when('type', {
    is: Joi.string().valid('limit', 'stop_limit'),
    then: Joi.required(),
    otherwise: Joi.optional()
  }),
  stopPrice: Joi.number().positive().when('type', {
    is: Joi.string().valid('stop', 'stop_limit'),
    then: Joi.required(),
    otherwise: Joi.optional()
  }),
  stopLoss: Joi.number().positive().optional(),
  takeProfit: Joi.number().positive().optional(),
  expiresAt: Joi.date().greater('now').optional()
});

const modifyPositionSchema = Joi.object({
  stopLoss: Joi.number().positive().allow(null).optional(),
  takeProfit: Joi.number().positive().allow(null).optional()
});

// Authentication middleware
const authenticateToken = async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      res.status(401).json({ error: 'Access token required' });
      return;
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key') as any;
    
    if (decoded.type !== 'access') {
      res.status(401).json({ error: 'Invalid token type' });
      return;
    }

    // Get user from database
    const userResult = await pool.query(
      'SELECT id, email, first_name, last_name FROM users WHERE id = $1 AND status = $2',
      [decoded.userId, 'active']
    );

    if (userResult.rows.length === 0) {
      res.status(401).json({ error: 'User not found or inactive' });
      return;
    }

    const user = userResult.rows[0];
    req.user = {
      id: user.id,
      email: user.email,
      firstName: user.first_name,
      lastName: user.last_name
    };
    req.sessionId = decoded.sessionId;

    next();
  } catch (error) {
    logger.error('Token authentication error:', error);
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Market data functions
const getMarketData = async (symbol: string): Promise<MarketData | null> => {
  try {
    // Try to get from cache first
    const cached = await redisClient.get(`market:${symbol}`);
    if (cached) {
      return JSON.parse(cached);
    }

    // Fetch from external API (placeholder for now)
    // In production, this would connect to a real market data provider
    const mockData: MarketData = {
      symbol,
      bid: Math.random() * 1.5 + 0.5,
      ask: Math.random() * 1.5 + 0.5,
      timestamp: new Date()
    };

    // Cache for 1 second
    await redisClient.setEx(`market:${symbol}`, 1, JSON.stringify(mockData));
    
    return mockData;
  } catch (error) {
    logger.error('Failed to get market data:', error);
    return null;
  }
};

// Trading functions
const calculateMargin = (size: number, price: number, leverage: number, marginRequirement: number): number => {
  const notionalValue = new Decimal(size).mul(price);
  const requiredMargin = notionalValue.mul(marginRequirement).div(leverage);
  return requiredMargin.toNumber();
};

const calculatePnL = (side: 'buy' | 'sell', openPrice: number, currentPrice: number, size: number): number => {
  const priceDiff = new Decimal(currentPrice).minus(openPrice);
  const multiplier = side === 'buy' ? 1 : -1;
  return priceDiff.mul(size).mul(multiplier).toNumber();
};

// Routes

// Health check
app.get('/health', (req: Request, res: Response): void => {
  res.json({
    status: 'healthy',
    service: 'trading-service',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// Get user's trading accounts
app.get('/api/v1/accounts', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const result = await pool.query(
      `SELECT id, account_number, account_type, base_currency, balance, equity, 
              margin_available, margin_used, leverage, created_at 
       FROM trading_accounts 
       WHERE user_id = $1 
       ORDER BY created_at DESC`,
      [req.user.id]
    );

    res.json({
      success: true,
      data: result.rows
    });

  } catch (error) {
    logger.error('Get accounts error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get currency pairs
app.get('/api/v1/pairs', async (req: Request, res: Response): Promise<void> => {
  try {
    const result = await pool.query(
      `SELECT id, symbol, base_symbol, quote_symbol, pip_size, 
              min_trade_size, max_trade_size, margin_requirement, is_active 
       FROM currency_pairs 
       WHERE is_active = true 
       ORDER BY symbol`
    );

    res.json({
      success: true,
      data: result.rows
    });

  } catch (error) {
    logger.error('Get pairs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get market data for a symbol
app.get('/api/v1/market/:symbol', async (req: Request, res: Response): Promise<void> => {
  try {
    const { symbol } = req.params;
    
    if (!symbol) {
      res.status(400).json({ error: 'Symbol is required' });
      return;
    }
    
    const marketData = await getMarketData(symbol);
    if (!marketData) {
      res.status(404).json({ error: 'Market data not available' });
      return;
    }

    res.json({
      success: true,
      data: marketData
    });

  } catch (error) {
    logger.error('Get market data error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Create order with AI assistance
app.post('/api/v1/orders', authenticateToken, tradingLimiter, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const { error, value } = createOrderSchema.validate(req.body);
    if (error) {
      res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details.map(d => d.message) 
      });
      return;
    }

    const { pairId, type, side, size, price, stopPrice, stopLoss, takeProfit, expiresAt } = value;

    // Get user's default trading account
    const accountResult = await pool.query(
      'SELECT id, balance, margin_available, leverage FROM trading_accounts WHERE user_id = $1 AND account_type = $2',
      [req.user.id, 'demo']
    );

    if (accountResult.rows.length === 0) {
      res.status(404).json({ error: 'Trading account not found' });
      return;
    }

    const account = accountResult.rows[0];

    // Get currency pair info
    const pairResult = await pool.query(
      'SELECT symbol, margin_requirement FROM currency_pairs WHERE id = $1 AND is_active = true',
      [pairId]
    );

    if (pairResult.rows.length === 0) {
      res.status(404).json({ error: 'Currency pair not found' });
      return;
    }

    const pair = pairResult.rows[0];

    // Get AI trading signal for validation
    try {
      const aiSignal = await pythonEngine.getTradingSignals({
        symbol: pair.symbol,
        timeframe: '1h',
        current_price: price || 0,
        risk_level: 'medium'
      });
      
      if (aiSignal && aiSignal.action !== 'hold') {
        logger.info(`AI Signal received: ${aiSignal.action} ${pair.symbol} (confidence: ${aiSignal.confidence})`);
        
        // If AI signal conflicts with user order, warn but don't block
        if (aiSignal.action !== side) {
          logger.warn(`User order conflicts with AI signal: User wants ${side}, AI suggests ${aiSignal.action}`);
        }
      }
    } catch (error) {
      logger.warn('Failed to get AI signal:', error.message);
    }

    // Validate risk with Python engine
    try {
      const riskAssessment = await pythonEngine.getRiskAssessment({
        symbol: pair.symbol,
        position_size: size,
        account_balance: parseFloat(account.balance),
        existing_positions: [],
        market_conditions: {}
      });
      
      if (riskAssessment.risk_level === 'extreme') {
        res.status(400).json({ 
          error: 'Trade rejected by risk management system',
          risk_level: riskAssessment.risk_level,
          warnings: riskAssessment.warnings
        });
        return;
      }
    } catch (error) {
      logger.warn('Risk assessment failed, proceeding with caution:', error.message);
    }
        aiSignal: aiSignal ? { action: aiSignal.action, confidence: aiSignal.confidence } : null
      });
      return;
    }

    // Get current market price
    const marketData = await getMarketData(pair.symbol);
    if (!marketData) {
      res.status(503).json({ error: 'Market data unavailable' });
      return;
    }

    // For market orders, use current market price
    let orderPrice = price;
    if (type === 'market') {
      orderPrice = side === 'buy' ? marketData.ask : marketData.bid;
    }

    // Calculate required margin
    const requiredMargin = calculateMargin(size, orderPrice || marketData.ask, account.leverage, pair.margin_requirement);

    // Check if sufficient margin available
    if (requiredMargin > account.margin_available) {
      res.status(400).json({ 
        error: 'Insufficient margin',
        required: requiredMargin,
        available: account.margin_available 
      });
      return;
    }

    // Create order
    const orderId = uuidv4();
    const orderResult = await pool.query(
      `INSERT INTO orders (id, user_id, account_id, pair_id, symbol, type, side, size, price, stop_price, stop_loss, take_profit, status, expires_at) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) 
       RETURNING *`,
      [orderId, req.user.id, account.id, pairId, pair.symbol, type, side, size, orderPrice, stopPrice, stopLoss, takeProfit, 'pending', expiresAt]
    );

    const newOrder = orderResult.rows[0];

    // For market orders, execute immediately
    if (type === 'market') {
      // Create position
      const positionId = uuidv4();
      await pool.query(
        `INSERT INTO positions (id, user_id, account_id, pair_id, symbol, type, size, open_price, current_price, stop_loss, take_profit, margin, status) 
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`,
        [positionId, req.user.id, account.id, pairId, pair.symbol, side, size, orderPrice, orderPrice, stopLoss, takeProfit, requiredMargin, 'open']
      );

      // Update order status
      await pool.query(
        'UPDATE orders SET status = $1, filled_size = $2, filled_price = $3, updated_at = NOW() WHERE id = $4',
        ['filled', size, orderPrice, orderId]
      );

      // Update account margin
      await pool.query(
        'UPDATE trading_accounts SET margin_used = margin_used + $1, margin_available = margin_available - $1 WHERE id = $2',
        [requiredMargin, account.id]
      );

      newOrder.status = 'filled';
      newOrder.filled_size = size;
      newOrder.filled_price = orderPrice;
    }

    // Log the trade
    await pool.query(
      'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [req.user.id, 'order_created', 'order', orderId, JSON.stringify(newOrder), req.ip, req.get('User-Agent')]
    );

    logger.info('Order created with AI validation', { 
      userId: req.user.id, 
      orderId, 
      symbol: pair.symbol,
      type,
      side,
      size,
      aiSignal: aiSignal ? { action: aiSignal.action, confidence: aiSignal.confidence } : null,
      riskApproved,
      ip: req.ip 
    });

    res.status(201).json({
      success: true,
      data: newOrder,
      aiSignal: aiSignal ? { action: aiSignal.action, confidence: aiSignal.confidence, riskLevel: aiSignal.riskLevel } : null,
      message: 'Order created successfully'
    });

  } catch (error) {
    logger.error('Create order error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get user's orders
app.get('/api/v1/orders', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const limit = Math.min(parseInt(req.query.limit as string) || 50, 100);
    const offset = parseInt(req.query.offset as string) || 0;
    const status = req.query.status as string;

    let query = `
      SELECT o.*, cp.base_symbol, cp.quote_symbol 
      FROM orders o
      JOIN currency_pairs cp ON o.pair_id = cp.id
      WHERE o.user_id = $1
    `;
    const params: any[] = [req.user.id];

    if (status) {
      query += ' AND o.status = $2';
      params.push(status);
    }

    query += ' ORDER BY o.created_at DESC LIMIT $' + (params.length + 1) + ' OFFSET $' + (params.length + 2);
    params.push(limit, offset);

    const result = await pool.query(query, params);

    res.json({
      success: true,
      data: result.rows,
      pagination: {
        limit,
        offset,
        total: result.rowCount
      }
    });

  } catch (error) {
    logger.error('Get orders error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get user's positions
app.get('/api/v1/positions', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const result = await pool.query(
      `SELECT p.*, cp.base_symbol, cp.quote_symbol 
       FROM positions p
       JOIN currency_pairs cp ON p.pair_id = cp.id
       WHERE p.user_id = $1 AND p.status = 'open'
       ORDER BY p.open_time DESC`,
      [req.user.id]
    );

    // Update current prices and P&L for each position
    for (const position of result.rows) {
      const marketData = await getMarketData(position.symbol);
      if (marketData) {
        const currentPrice = position.type === 'buy' ? marketData.bid : marketData.ask;
        const unrealizedPnl = calculatePnL(position.type, position.open_price, currentPrice, position.size);
        
        position.current_price = currentPrice;
        position.unrealized_pnl = unrealizedPnl;

        // Update in database
        await pool.query(
          'UPDATE positions SET current_price = $1, unrealized_pnl = $2 WHERE id = $3',
          [currentPrice, unrealizedPnl, position.id]
        );
      }
    }

    res.json({
      success: true,
      data: result.rows
    });

  } catch (error) {
    logger.error('Get positions error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Close position
app.post('/api/v1/positions/:positionId/close', authenticateToken, tradingLimiter, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const { positionId } = req.params;

    // Get position
    const positionResult = await pool.query(
      'SELECT * FROM positions WHERE id = $1 AND user_id = $2 AND status = $3',
      [positionId, req.user.id, 'open']
    );

    if (positionResult.rows.length === 0) {
      res.status(404).json({ error: 'Position not found' });
      return;
    }

    const position = positionResult.rows[0];

    // Get current market price
    const marketData = await getMarketData(position.symbol);
    if (!marketData) {
      res.status(503).json({ error: 'Market data unavailable' });
      return;
    }

    const closePrice = position.type === 'buy' ? marketData.bid : marketData.ask;
    const realizedPnl = calculatePnL(position.type, position.open_price, closePrice, position.size);

    // Close position
    await pool.query(
      'UPDATE positions SET status = $1, close_price = $2, close_time = NOW(), realized_pnl = $3 WHERE id = $4',
      ['closed', closePrice, realizedPnl, positionId]
    );

    // Update account balance and margin
    await pool.query(
      'UPDATE trading_accounts SET balance = balance + $1, margin_used = margin_used - $2, margin_available = margin_available + $2 WHERE id = $3',
      [realizedPnl, position.margin, position.account_id]
    );

    // Log the trade
    await pool.query(
      'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [req.user.id, 'position_closed', 'position', positionId, JSON.stringify({ closePrice, realizedPnl }), req.ip, req.get('User-Agent')]
    );

    logger.info('Position closed', { 
      userId: req.user.id, 
      positionId, 
      symbol: position.symbol,
      closePrice,
      realizedPnl,
      ip: req.ip 
    });

    res.json({
      success: true,
      data: {
        positionId,
        closePrice,
        realizedPnl
      },
      message: 'Position closed successfully'
    });

  } catch (error) {
    logger.error('Close position error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Modify position (stop loss / take profit)
app.put('/api/v1/positions/:positionId', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const { positionId } = req.params;
    const { error, value } = modifyPositionSchema.validate(req.body);
    
    if (error) {
      res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details.map(d => d.message) 
      });
      return;
    }

    const { stopLoss, takeProfit } = value;

    // Check if position exists and belongs to user
    const positionResult = await pool.query(
      'SELECT * FROM positions WHERE id = $1 AND user_id = $2 AND status = $3',
      [positionId, req.user.id, 'open']
    );

    if (positionResult.rows.length === 0) {
      res.status(404).json({ error: 'Position not found' });
      return;
    }

    // Update position
    await pool.query(
      'UPDATE positions SET stop_loss = $1, take_profit = $2, updated_at = NOW() WHERE id = $3',
      [stopLoss, takeProfit, positionId]
    );

    // Log the modification
    await pool.query(
      'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [req.user.id, 'position_modified', 'position', positionId, JSON.stringify({ stopLoss, takeProfit }), req.ip, req.get('User-Agent')]
    );

    logger.info('Position modified', { 
      userId: req.user.id, 
      positionId, 
      stopLoss,
      takeProfit,
      ip: req.ip 
    });

    res.json({
      success: true,
      message: 'Position modified successfully'
    });

  } catch (error) {
    logger.error('Modify position error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get AI trading signals
app.get('/api/v1/signals/:symbol', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const { symbol } = req.params;
    
    // Get user's default trading account
    const accountResult = await pool.query(
      'SELECT id FROM trading_accounts WHERE user_id = $1 AND account_type = $2',
      [req.user.id, 'demo']
    );

    if (accountResult.rows.length === 0) {
      res.status(404).json({ error: 'Trading account not found' });
      return;
    }

    const account = accountResult.rows[0];

    // Get AI trading signal
    try {
      const aiSignal = await pythonEngine.getTradingSignals({
        symbol,
        timeframe: '1h',
        current_price: 0, // Would be populated from market data
        risk_level: 'medium'
      });
      
      if (!aiSignal) {
        res.status(503).json({ error: 'AI signal unavailable' });
        return;
      }

      // Get additional AI analysis
      const aiAnalysis = await pythonEngine.getMarketAnalysis({
        symbol,
        timeframe: '1h',
        indicators: ['rsi', 'macd', 'bollinger', 'sma', 'ema']
      });

      res.json({
        success: true,
        data: {
          signal: aiSignal,
          analysis: aiAnalysis,
          timestamp: new Date().toISOString()
        }
      });
    } catch (error) {
      logger.error('Failed to get AI signals:', error);
      res.status(503).json({ error: 'AI signal service unavailable' });
    }
      }
    });

  } catch (error) {
    logger.error('Get AI signals error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get trading history
app.get('/api/v1/history', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const limit = Math.min(parseInt(req.query.limit as string) || 50, 100);
    const offset = parseInt(req.query.offset as string) || 0;

    const result = await pool.query(
      `SELECT p.*, cp.base_symbol, cp.quote_symbol 
       FROM positions p
       JOIN currency_pairs cp ON p.pair_id = cp.id
       WHERE p.user_id = $1 AND p.status = 'closed'
       ORDER BY p.close_time DESC 
       LIMIT $2 OFFSET $3`,
      [req.user.id, limit, offset]
    );

    res.json({
      success: true,
      data: result.rows,
      pagination: {
        limit,
        offset,
        total: result.rowCount
      }
    });

  } catch (error) {
    logger.error('Get history error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction): void => {
  logger.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use('*', (req: Request, res: Response): void => {
  res.status(404).json({ error: 'Not found' });
});

// Start server
const PORT = process.env.PORT || 3002;

const startServer = async (): Promise<void> => {
  try {
    // Test database connection
    await pool.query('SELECT NOW()');
    logger.info('Database connection established');

    app.listen(PORT, () => {
      logger.info(`Trading Service running on port ${PORT}`);
      logger.info('Available endpoints:');
      logger.info('- GET /health');
      logger.info('- GET /api/v1/accounts');
      logger.info('- GET /api/v1/pairs');
      logger.info('- GET /api/v1/market/:symbol');
      logger.info('- POST /api/v1/orders');
      logger.info('- GET /api/v1/orders');
      logger.info('- GET /api/v1/positions');
      logger.info('- POST /api/v1/positions/:id/close');
      logger.info('- PUT /api/v1/positions/:id');
      logger.info('- GET /api/v1/history');
    });

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
};

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('Shutting down gracefully...');
  await pool.end();
  await redisClient.quit();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('Shutting down gracefully...');
  await pool.end();
  await redisClient.quit();
  process.exit(0);
});

startServer();
