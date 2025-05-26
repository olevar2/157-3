const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');
const Redis = require('redis');
const winston = require('winston');
const { v4: uuidv4 } = require('uuid');
const speakeasy = require('speakeasy');
const QRCode = require('qrcode');
require('dotenv').config();

// Logger setup
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

const app = express();
const PORT = process.env.PORT || 3001;

// Database connection
const db = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

// Redis connection
const redis = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://redis:6379',
  password: process.env.REDIS_PASSWORD || 'RedisSecure2025!'
});

redis.on('error', (err) => logger.error('Redis error:', err));
redis.on('connect', () => logger.info('Connected to Redis'));

// Connect to Redis
redis.connect().catch(err => logger.error('Failed to connect to Redis:', err));

app.use(express.json());

// Test database connection
db.query('SELECT NOW()', (err, res) => {
  if (err) {
    logger.error('Database connection failed:', err);
  } else {
    logger.info('Database connected successfully');
  }
});

// Helper function to generate JWT
const generateToken = (user) => {
  return jwt.sign(
    { 
      userId: user.id,
      email: user.email,
      role: 'owner'
    },
    process.env.JWT_SECRET || 'forex-secret-key',
    { expiresIn: '24h' }
  );
};

// Helper function to hash password
const hashPassword = async (password) => {
  return await bcrypt.hash(password, 12);
};

// Helper function to verify password
const verifyPassword = async (password, hashedPassword) => {
  return await bcrypt.compare(password, hashedPassword);
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'user-service',
    timestamp: new Date().toISOString()
  });
});

// Owner registration (one-time setup)
app.post('/api/v1/auth/register-owner', async (req, res) => {
  try {
    const { email, password, fullName, phone } = req.body;

    // Validate input
    if (!email || !password || !fullName) {
      return res.status(400).json({
        error: 'Email, password, and full name are required'
      });
    }

    // Check if owner already exists
    const existingOwner = await db.query(
      'SELECT id FROM users WHERE role = $1 LIMIT 1',
      ['owner']
    );

    if (existingOwner.rows.length > 0) {
      return res.status(409).json({
        error: 'Owner account already exists. Only one owner allowed.'
      });
    }

    // Validate password strength
    if (password.length < 8) {
      return res.status(400).json({
        error: 'Password must be at least 8 characters long'
      });
    }

    // Hash password
    const hashedPassword = await hashPassword(password);
    const userId = uuidv4();

    // Generate 2FA secret
    const secret = speakeasy.generateSecret({
      name: `Forex Platform (${email})`,
      issuer: 'Personal Forex Trading Platform'
    });

    // Insert owner into database
    const result = await db.query(`
      INSERT INTO users (
        id, email, password_hash, full_name, phone, role, 
        status, two_factor_secret, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
      RETURNING id, email, full_name, phone, status, created_at
    `, [
      userId, email.toLowerCase(), hashedPassword, fullName, 
      phone || null, 'owner', 'active', secret.base32
    ]);

    const user = result.rows[0];

    // Generate QR code for 2FA setup
    const qrCodeUrl = await QRCode.toDataURL(secret.otpauth_url);

    // Log successful registration
    await db.query(`
      INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, NOW())
    `, [
      userId, 'USER_REGISTERED', 'user', userId,
      JSON.stringify({ email, full_name: fullName }),
      req.ip
    ]);

    logger.info(`Owner account created: ${email}`);

    res.status(201).json({
      message: 'Owner account created successfully',
      user: {
        id: user.id,
        email: user.email,
        fullName: user.full_name,
        phone: user.phone,
        status: user.status,
        createdAt: user.created_at
      },
      twoFactorSetup: {
        secret: secret.base32,
        qrCode: qrCodeUrl,
        manualEntry: secret.base32
      }
    });

  } catch (error) {
    logger.error('Owner registration error:', error);
    res.status(500).json({
      error: 'Failed to create owner account',
      message: error.message
    });
  }
});

// Owner login
app.post('/api/v1/auth/login', async (req, res) => {
  try {
    const { email, password, twoFactorCode } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        error: 'Email and password are required'
      });
    }

    // Find user
    const result = await db.query(
      'SELECT * FROM users WHERE email = $1 AND role = $2',
      [email.toLowerCase(), 'owner']
    );

    if (result.rows.length === 0) {
      return res.status(401).json({
        error: 'Invalid credentials'
      });
    }

    const user = result.rows[0];

    // Check if account is active
    if (user.status !== 'active') {
      return res.status(401).json({
        error: 'Account is not active'
      });
    }

    // Verify password
    const passwordValid = await verifyPassword(password, user.password_hash);
    if (!passwordValid) {
      return res.status(401).json({
        error: 'Invalid credentials'
      });
    }

    // Verify 2FA if secret exists
    if (user.two_factor_secret) {
      if (!twoFactorCode) {
        return res.status(400).json({
          error: 'Two-factor authentication code required',
          requiresTwoFactor: true
        });
      }

      const verified = speakeasy.totp.verify({
        secret: user.two_factor_secret,
        encoding: 'base32',
        token: twoFactorCode,
        window: 2
      });

      if (!verified) {
        return res.status(401).json({
          error: 'Invalid two-factor authentication code'
        });
      }
    }

    // Generate token
    const token = generateToken(user);
    const sessionId = uuidv4();

    // Store session in Redis
    await redis.setex(`session:${sessionId}`, 86400, JSON.stringify({
      userId: user.id,
      email: user.email,
      loginTime: new Date().toISOString(),
      ipAddress: req.ip
    }));

    // Update last login
    await db.query(
      'UPDATE users SET last_login_at = NOW(), last_login_ip = $1 WHERE id = $2',
      [req.ip, user.id]
    );

    // Log successful login
    await db.query(`
      INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, NOW())
    `, [
      user.id, 'USER_LOGIN', 'session', sessionId,
      JSON.stringify({ success: true, two_factor_used: !!user.two_factor_secret }),
      req.ip
    ]);

    logger.info(`Owner login successful: ${email}`);

    res.json({
      message: 'Login successful',
      token,
      sessionId,
      user: {
        id: user.id,
        email: user.email,
        fullName: user.full_name,
        phone: user.phone,
        lastLogin: user.last_login_at
      }
    });

  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({
      error: 'Login failed',
      message: error.message
    });
  }
});

// Get owner profile
app.get('/api/v1/users/profile', async (req, res) => {
  try {
    const userId = req.headers['x-user-id'];
    
    if (!userId) {
      return res.status(401).json({ error: 'User not authenticated' });
    }

    const result = await db.query(`
      SELECT id, email, full_name, phone, status, created_at, last_login_at,
             two_factor_enabled, email_notifications, sms_notifications
      FROM users WHERE id = $1 AND role = 'owner'
    `, [userId]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const user = result.rows[0];

    res.json({
      user: {
        id: user.id,
        email: user.email,
        fullName: user.full_name,
        phone: user.phone,
        status: user.status,
        createdAt: user.created_at,
        lastLogin: user.last_login_at,
        twoFactorEnabled: user.two_factor_enabled,
        notifications: {
          email: user.email_notifications,
          sms: user.sms_notifications
        }
      }
    });

  } catch (error) {
    logger.error('Profile fetch error:', error);
    res.status(500).json({
      error: 'Failed to fetch profile',
      message: error.message
    });
  }
});

// Update owner profile
app.put('/api/v1/users/profile', async (req, res) => {
  try {
    const userId = req.headers['x-user-id'];
    const { fullName, phone, emailNotifications, smsNotifications } = req.body;

    if (!userId) {
      return res.status(401).json({ error: 'User not authenticated' });
    }

    // Update profile
    const result = await db.query(`
      UPDATE users 
      SET full_name = COALESCE($1, full_name),
          phone = COALESCE($2, phone),
          email_notifications = COALESCE($3, email_notifications),
          sms_notifications = COALESCE($4, sms_notifications),
          updated_at = NOW()
      WHERE id = $5 AND role = 'owner'
      RETURNING id, email, full_name, phone, email_notifications, sms_notifications
    `, [fullName, phone, emailNotifications, smsNotifications, userId]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const user = result.rows[0];

    // Log profile update
    await db.query(`
      INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, NOW())
    `, [
      userId, 'PROFILE_UPDATED', 'user', userId,
      JSON.stringify({ full_name: fullName, phone, email_notifications: emailNotifications, sms_notifications: smsNotifications }),
      req.ip
    ]);

    res.json({
      message: 'Profile updated successfully',
      user: {
        id: user.id,
        email: user.email,
        fullName: user.full_name,
        phone: user.phone,
        notifications: {
          email: user.email_notifications,
          sms: user.sms_notifications
        }
      }
    });

  } catch (error) {
    logger.error('Profile update error:', error);
    res.status(500).json({
      error: 'Failed to update profile',
      message: error.message
    });
  }
});

// Logout
app.post('/api/v1/auth/logout', async (req, res) => {
  try {
    const sessionId = req.headers['x-session-id'];
    const userId = req.headers['x-user-id'];

    if (sessionId) {
      await redis.del(`session:${sessionId}`);
    }

    if (userId) {
      // Log logout
      await db.query(`
        INSERT INTO audit_logs (user_id, action, resource_type, resource_id, new_values, ip_address, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
      `, [
        userId, 'USER_LOGOUT', 'session', sessionId || 'unknown',
        JSON.stringify({ success: true }),
        req.ip
      ]);
    }

    res.json({ message: 'Logout successful' });

  } catch (error) {
    logger.error('Logout error:', error);
    res.status(500).json({
      error: 'Logout failed',
      message: error.message
    });
  }
});

// Global error handler
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    timestamp: new Date().toISOString()
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await db.end();
  await redis.quit();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  await db.end();
  await redis.quit();
  process.exit(0);
});

app.listen(PORT, '0.0.0.0', () => {
  logger.info(`🚀 User Service running on port ${PORT}`);
});

module.exports = app;
