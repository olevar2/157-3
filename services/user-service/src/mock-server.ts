import express, { Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import winston from 'winston';
import dotenv from 'dotenv';

dotenv.config();

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'logs/user-service-mock.log' })
  ]
});

const app = express();
const PORT = process.env.PORT || 3002;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// In-memory user storage for demo
const users = new Map();
const sessions = new Map();

// Mock data
const demoUser = {
  id: 'demo-user-1',
  email: 'demo@example.com',
  firstName: 'Demo',
  lastName: 'User',
  phone: '+1234567890',
  country: 'US',
  timezone: 'UTC',
  language: 'en',
  status: 'active',
  kycStatus: 'verified',
  twoFactorEnabled: false,
  emailVerified: true,
  phoneVerified: true,
  createdAt: new Date(),
  updatedAt: new Date(),
  lastLogin: new Date(),
  preferences: {
    theme: 'dark',
    notifications: true
  }
};

users.set('demo@example.com', {
  ...demoUser,
  passwordHash: '$2a$10$dummy.hash.for.demo.purposes.only'
});

// Routes
app.post('/api/v1/auth/login', async (req: Request, res: Response): Promise<void> => {
  try {
    const { email, password } = req.body;
    
    if (email === 'demo@example.com' && password === 'demo123') {
      const sessionId = 'demo-session-' + Date.now();
      const token = 'demo-jwt-token-' + Date.now();
      
      sessions.set(sessionId, {
        userId: demoUser.id,
        createdAt: new Date()
      });
      
      res.json({
        success: true,
        token,
        user: demoUser,
        sessionId
      });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/auth/profile', async (req: Request, res: Response): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      res.status(401).json({ error: 'No token provided' });
      return;
    }
    
    res.json({
      success: true,
      user: demoUser
    });
  } catch (error) {
    logger.error('Profile error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/v1/auth/logout', async (req: Request, res: Response): Promise<void> => {
  try {
    res.json({ success: true, message: 'Logged out successfully' });
  } catch (error) {
    logger.error('Logout error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/users/account', async (req: Request, res: Response): Promise<void> => {
  try {
    res.json({
      success: true,
      account: {
        id: 'demo-account-1',
        userId: demoUser.id,
        balance: 10000.00,
        equity: 10000.00,
        margin: 0.00,
        freeMargin: 10000.00,
        marginLevel: 0,
        currency: 'USD',
        leverage: 100,
        createdAt: new Date(),
        updatedAt: new Date()
      }
    });
  } catch (error) {
    logger.error('Account error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/health', (req: Request, res: Response) => {
  res.json({ 
    status: 'ok', 
    service: 'user-service-mock',
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`Mock User Service running on port ${PORT}`);
  logger.info('Demo credentials: email: demo@example.com, password: demo123');
});

export default app;
