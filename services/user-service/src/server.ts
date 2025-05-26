import express, { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { Pool } from 'pg';
import * as redis from 'redis';
import winston from 'winston';
import { v4 as uuidv4 } from 'uuid';
import speakeasy from 'speakeasy';
import QRCode from 'qrcode';
import Joi from 'joi';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

// Types
interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  phone?: string;
  country?: string;
  timezone: string;
  language: string;
  status: 'pending' | 'active' | 'suspended' | 'closed';
  kycStatus: string;
  twoFactorEnabled: boolean;
  emailVerified: boolean;
  phoneVerified: boolean;
  createdAt: Date;
  updatedAt: Date;
  lastLogin?: Date;
  preferences: any;
}

interface AuthRequest extends Request {
  user?: User;
  sessionId?: string;
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
      filename: 'logs/user-service-error.log', 
      level: 'error' 
    }),
    new winston.transports.File({ 
      filename: 'logs/user-service-combined.log' 
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

redisClient.on('error', (err) => {
  logger.error('Redis connection error:', err);
});

redisClient.on('connect', () => {
  logger.info('Connected to Redis for user sessions');
});

// Connect to Redis
redisClient.connect().catch((err) => {
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
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Limit each IP to 5 requests per windowMs
  message: {
    error: 'Too many authentication attempts, please try again later.',
    retryAfter: '15 minutes'
  },
  standardHeaders: true,
  legacyHeaders: false,
});

// Validation schemas
const registerSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().min(8).pattern(new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]')).required(),
  firstName: Joi.string().min(2).max(50).required(),
  lastName: Joi.string().min(2).max(50).required(),
  phone: Joi.string().optional(),
  country: Joi.string().length(2).required(),
  timezone: Joi.string().optional(),
  language: Joi.string().length(2).optional(),
});

const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().required(),
  twoFactorCode: Joi.string().length(6).optional(),
});

const updateProfileSchema = Joi.object({
  firstName: Joi.string().min(2).max(50).optional(),
  lastName: Joi.string().min(2).max(50).optional(),
  phone: Joi.string().optional(),
  country: Joi.string().length(2).optional(),
  timezone: Joi.string().optional(),
  language: Joi.string().length(2).optional(),
  preferences: Joi.object().optional(),
});

// Token generation
const generateTokens = (userId: string, sessionId: string): { accessToken: string; refreshToken: string } => {
  const accessToken = jwt.sign(
    { userId, sessionId, type: 'access' },
    process.env.JWT_SECRET || 'fallback-secret-key',
    { expiresIn: process.env.JWT_ACCESS_EXPIRY || '15m' } as jwt.SignOptions
  );

  const refreshToken = jwt.sign(
    { userId, sessionId, type: 'refresh' },
    process.env.JWT_REFRESH_SECRET || 'fallback-refresh-secret',
    { expiresIn: process.env.JWT_REFRESH_EXPIRY || '7d' } as jwt.SignOptions
  );

  return { accessToken, refreshToken };
};

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
      'SELECT id, email, first_name, last_name, phone, country, timezone, language, status, kyc_status, two_factor_enabled, email_verified, phone_verified, created_at, updated_at, last_login, preferences FROM users WHERE id = $1 AND status = $2',
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
      lastName: user.last_name,
      phone: user.phone,
      country: user.country,
      timezone: user.timezone,
      language: user.language,
      status: user.status,
      kycStatus: user.kyc_status,
      twoFactorEnabled: user.two_factor_enabled,
      emailVerified: user.email_verified,
      phoneVerified: user.phone_verified,
      createdAt: user.created_at,
      updatedAt: user.updated_at,
      lastLogin: user.last_login,
      preferences: user.preferences
    };
    req.sessionId = decoded.sessionId;

    next();
  } catch (error) {
    logger.error('Token authentication error:', error);
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Health check endpoint
app.get('/health', (req: Request, res: Response): void => {
  res.json({
    status: 'healthy',
    service: 'user-service',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// User registration
app.post('/api/v1/auth/register', authLimiter, async (req: Request, res: Response): Promise<void> => {
  try {
    const { error, value } = registerSchema.validate(req.body);
    if (error) {
      res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details.map(d => d.message) 
      });
      return;
    }

    const { email, password, firstName, lastName, phone, country, timezone, language } = value;

    // Check if user already exists
    const existingUser = await pool.query('SELECT id FROM users WHERE email = $1', [email]);
    if (existingUser.rows.length > 0) {
      res.status(409).json({ error: 'User already exists with this email' });
      return;
    }

    // Hash password
    const saltRounds = 12;
    const passwordHash = await bcrypt.hash(password, saltRounds);

    // Create user
    const userId = uuidv4();
    const userResult = await pool.query(
      `INSERT INTO users (id, email, password_hash, first_name, last_name, phone, country, timezone, language, status, kyc_status) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) 
       RETURNING id, email, first_name, last_name, created_at`,
      [userId, email, passwordHash, firstName, lastName, phone, country, timezone || 'UTC', language || 'en', 'pending', 'not_started']
    );

    const newUser = userResult.rows[0];

    // Generate account number and create demo trading account
    const accountResult = await pool.query('SELECT generate_account_number() as account_number');
    const accountNumber = accountResult.rows[0].account_number;

    await pool.query(
      `INSERT INTO trading_accounts (id, user_id, account_number, account_type, base_currency, balance, equity, margin_available, leverage) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
      [uuidv4(), userId, accountNumber, 'demo', 'USD', 10000, 10000, 10000, 100]
    );

    // Log the registration
    await pool.query(
      'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [userId, 'user_registered', 'user', userId, JSON.stringify({ email, account_number: accountNumber }), req.ip, req.get('User-Agent')]
    );

    logger.info('User registered successfully', { 
      userId, 
      email, 
      accountNumber,
      ip: req.ip 
    });

    res.status(201).json({
      message: 'User registered successfully',
      user: {
        id: newUser.id,
        email: newUser.email,
        firstName: newUser.first_name,
        lastName: newUser.last_name,
        createdAt: newUser.created_at
      },
      accountNumber
    });

  } catch (error) {
    logger.error('Registration error:', error);
    res.status(500).json({ error: 'Internal server error during registration' });
  }
});

// User login
app.post('/api/v1/auth/login', authLimiter, async (req: Request, res: Response): Promise<void> => {
  try {
    const { error, value } = loginSchema.validate(req.body);
    if (error) {
      res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details.map(d => d.message) 
      });
      return;
    }

    const { email, password, twoFactorCode } = value;

    // Get user from database
    const userResult = await pool.query(
      'SELECT id, email, password_hash, first_name, last_name, phone, country, timezone, language, status, kyc_status, two_factor_enabled, two_factor_secret, email_verified, phone_verified, created_at, updated_at, last_login, preferences, login_attempts, locked_until FROM users WHERE email = $1',
      [email]
    );

    if (userResult.rows.length === 0) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    const user = userResult.rows[0];

    // Check if account is locked
    if (user.locked_until && new Date() < new Date(user.locked_until)) {
      res.status(423).json({ 
        error: 'Account is temporarily locked due to multiple failed attempts',
        lockedUntil: user.locked_until
      });
      return;
    }

    // Check if account is active
    if (user.status !== 'active') {
      res.status(401).json({ error: 'Account is not active' });
      return;
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.password_hash);
    if (!isValidPassword) {
      // Increment login attempts
      await pool.query(
        'UPDATE users SET login_attempts = COALESCE(login_attempts, 0) + 1, locked_until = CASE WHEN COALESCE(login_attempts, 0) + 1 >= 5 THEN NOW() + INTERVAL \'15 minutes\' ELSE NULL END WHERE id = $1',
        [user.id]
      );
      
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    // Check 2FA if enabled
    if (user.two_factor_enabled) {
      if (!twoFactorCode) {
        res.status(200).json({ 
          message: 'Two-factor authentication required',
          requiresTwoFactor: true,
          tempToken: jwt.sign({ userId: user.id, type: 'temp_2fa' }, process.env.JWT_SECRET || 'fallback-secret-key', { expiresIn: '5m' })
        });
        return;
      }

      const isValid2FA = speakeasy.totp.verify({
        secret: user.two_factor_secret,
        encoding: 'base32',
        token: twoFactorCode,
        window: 2
      });

      if (!isValid2FA) {
        res.status(401).json({ error: 'Invalid two-factor authentication code' });
        return;
      }
    }

    // Reset login attempts on successful login
    await pool.query(
      'UPDATE users SET login_attempts = 0, locked_until = NULL, last_login = NOW() WHERE id = $1',
      [user.id]
    );

    // Create session
    const sessionId = uuidv4();
    const sessionData = {
      userId: user.id,
      email: user.email,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      createdAt: new Date().toISOString()
    };

    // Store session in Redis (expires in 7 days)
    await redisClient.setEx(`session:${sessionId}`, 7 * 24 * 60 * 60, JSON.stringify(sessionData));

    // Generate tokens
    const { accessToken, refreshToken } = generateTokens(user.id, sessionId);

    // Log successful login
    await pool.query(
      'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [user.id, 'user_login', 'session', sessionId, JSON.stringify({ success: true }), req.ip, req.get('User-Agent')]
    );

    logger.info('User logged in successfully', { 
      userId: user.id, 
      email: user.email, 
      sessionId,
      ip: req.ip 
    });

    res.json({
      message: 'Login successful',
      user: {
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
        status: user.status,
        emailVerified: user.email_verified,
        twoFactorEnabled: user.two_factor_enabled
      },
      accessToken,
      refreshToken,
      expiresIn: process.env.JWT_ACCESS_EXPIRY || '15m'
    });

  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error during login' });
  }
});

// Get user profile
app.get('/api/v1/auth/profile', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    // Get user's trading accounts
    const accountsResult = await pool.query(
      'SELECT account_number, account_type, base_currency, balance, equity, margin_available, leverage, created_at FROM trading_accounts WHERE user_id = $1',
      [req.user.id]
    );

    res.json({
      user: req.user,
      accounts: accountsResult.rows
    });

  } catch (error) {
    logger.error('Profile fetch error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update user profile
app.put('/api/v1/auth/profile', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'User not authenticated' });
      return;
    }

    const { error, value } = updateProfileSchema.validate(req.body);
    if (error) {
      res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details.map(d => d.message) 
      });
      return;
    }

    const updateFields: string[] = [];
    const updateValues: any[] = [];
    let paramIndex = 1;

    // Build dynamic update query
    Object.entries(value).forEach(([key, val]) => {
      if (val !== undefined) {
        updateFields.push(`${key === 'firstName' ? 'first_name' : key === 'lastName' ? 'last_name' : key} = $${paramIndex}`);
        updateValues.push(val);
        paramIndex++;
      }
    });

    if (updateFields.length === 0) {
      res.status(400).json({ error: 'No fields to update' });
      return;
    }

    updateFields.push(`updated_at = NOW()`);
    updateValues.push(req.user.id);

    const updateQuery = `
      UPDATE users 
      SET ${updateFields.join(', ')} 
      WHERE id = $${paramIndex} 
      RETURNING id, email, first_name, last_name, phone, country, timezone, language, updated_at
    `;

    const result = await pool.query(updateQuery, updateValues);
    const updatedUser = result.rows[0];

    // Log the update
    await pool.query(
      'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [req.user.id, 'profile_updated', 'user', req.user.id, JSON.stringify(value), req.ip, req.get('User-Agent')]
    );

    logger.info('User profile updated', { 
      userId: req.user.id, 
      updates: Object.keys(value),
      ip: req.ip 
    });

    res.json({
      message: 'Profile updated successfully',
      user: {
        id: updatedUser.id,
        email: updatedUser.email,
        firstName: updatedUser.first_name,
        lastName: updatedUser.last_name,
        phone: updatedUser.phone,
        country: updatedUser.country,
        timezone: updatedUser.timezone,
        language: updatedUser.language,
        updatedAt: updatedUser.updated_at
      }
    });

  } catch (error) {
    logger.error('Profile update error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Refresh token
app.post('/api/v1/auth/refresh', async (req: Request, res: Response): Promise<void> => {
  try {
    const { refreshToken } = req.body;

    if (!refreshToken) {
      res.status(401).json({ error: 'Refresh token required' });
      return;
    }

    const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET || 'fallback-refresh-secret') as any;
    
    if (decoded.type !== 'refresh') {
      res.status(401).json({ error: 'Invalid token type' });
      return;
    }

    // Check if session exists
    const sessionData = await redisClient.get(`session:${decoded.sessionId}`);
    if (!sessionData) {
      res.status(401).json({ error: 'Session expired or invalid' });
      return;
    }

    // Generate new access token
    const { accessToken } = generateTokens(decoded.userId, decoded.sessionId);

    res.json({
      accessToken,
      expiresIn: process.env.JWT_ACCESS_EXPIRY || '15m'
    });

  } catch (error) {
    logger.error('Token refresh error:', error);
    res.status(401).json({ error: 'Invalid refresh token' });
  }
});

// Logout
app.post('/api/v1/auth/logout', authenticateToken, async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    if (req.sessionId) {
      // Remove session from Redis
      await redisClient.del(`session:${req.sessionId}`);
      
      // Log logout
      await pool.query(
        'INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, user_agent) VALUES ($1, $2, $3, $4, $5, $6, $7)',
        [req.user?.id, 'user_logout', 'session', req.sessionId, JSON.stringify({ success: true }), req.ip, req.get('User-Agent')]
      );

      logger.info('User logged out', { 
        userId: req.user?.id, 
        sessionId: req.sessionId,
        ip: req.ip 
      });
    }

    res.json({ message: 'Logged out successfully' });

  } catch (error) {
    logger.error('Logout error:', error);
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
const PORT = process.env.PORT || 3003;

const startServer = async (): Promise<void> => {
  try {
    // Test database connection
    await pool.query('SELECT NOW()');
    logger.info('Database connection established');

    app.listen(PORT, () => {
      logger.info(`User Service running on port ${PORT}`);
      logger.info('Available endpoints:');
      logger.info('- GET /health');
      logger.info('- POST /api/v1/auth/register');
      logger.info('- POST /api/v1/auth/login');
      logger.info('- GET /api/v1/auth/profile');
      logger.info('- PUT /api/v1/auth/profile');
      logger.info('- POST /api/v1/auth/refresh');
      logger.info('- POST /api/v1/auth/logout');
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
