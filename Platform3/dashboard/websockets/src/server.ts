// Real-time WebSocket Service for Forex Trading Platform
// Handles live price updates, order notifications, and AI chat messaging

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import winston from 'winston';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';

import { PriceStreamManager } from './services/PriceStreamManager';
import { OrderNotificationManager } from './services/OrderNotificationManager';
import { ChatMessageManager } from './services/ChatMessageManager';
import { EventBroadcaster } from './services/EventBroadcaster';
import { AuthenticationMiddleware } from './middleware/AuthenticationMiddleware';
import { RateLimitManager } from './services/RateLimitManager';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local' });

// Initialize logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
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
    }),
    new winston.transports.File({ 
      filename: 'logs/websocket-service.log',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ]
});

// Create Express app and HTTP server
const app = express();
const server = createServer(app);

// Configure Socket.IO with CORS
const io = new SocketIOServer(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"],
    credentials: true
  },
  transports: ['websocket', 'polling'],
  pingTimeout: 60000,
  pingInterval: 25000
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || "http://localhost:3000",
  credentials: true
}));
app.use(express.json());

// Initialize service managers
const priceStreamManager = new PriceStreamManager(io, logger);
const orderNotificationManager = new OrderNotificationManager(io, logger);
const chatMessageManager = new ChatMessageManager(io, logger);
const eventBroadcaster = new EventBroadcaster(io, logger);
const authMiddleware = new AuthenticationMiddleware(logger);
const rateLimitManager = new RateLimitManager(logger);

// Socket.IO connection handling
io.use(authMiddleware.authenticate.bind(authMiddleware));
io.use(rateLimitManager.limitConnections.bind(rateLimitManager));

io.on('connection', (socket) => {
  const userId = socket.data.userId;
  const userEmail = socket.data.userEmail;
  const connectionId = uuidv4();
  
  logger.info(`User connected: ${userEmail} (${userId}) - Connection: ${connectionId}`);

  // Join user-specific room
  socket.join(`user:${userId}`);
  socket.join('global');

  // Handle price subscription
  socket.on('subscribe:prices', async (data) => {
    try {
      await priceStreamManager.subscribeToPrices(socket, data.symbols || []);
      logger.info(`User ${userEmail} subscribed to price updates: ${data.symbols?.join(', ')}`);
    } catch (error) {
      logger.error('Price subscription error:', error);
      socket.emit('error', { type: 'PRICE_SUBSCRIPTION_ERROR', message: 'Failed to subscribe to prices' });
    }
  });

  // Handle order notifications subscription
  socket.on('subscribe:orders', async () => {
    try {
      await orderNotificationManager.subscribeToOrders(socket, userId);
      logger.info(`User ${userEmail} subscribed to order notifications`);
    } catch (error) {
      logger.error('Order subscription error:', error);
      socket.emit('error', { type: 'ORDER_SUBSCRIPTION_ERROR', message: 'Failed to subscribe to orders' });
    }
  });

  // Handle AI chat messages
  socket.on('chat:message', async (data) => {
    try {
      await rateLimitManager.checkMessageRate(userId);
      await chatMessageManager.handleChatMessage(socket, userId, data);
      logger.info(`Chat message from ${userEmail}: ${data.message?.substring(0, 50)}...`);
    } catch (error) {
      logger.error('Chat message error:', error);
      socket.emit('error', { type: 'CHAT_ERROR', message: 'Failed to process chat message' });
    }
  });

  // Handle event subscriptions
  socket.on('subscribe:events', async (data) => {
    try {
      await eventBroadcaster.subscribeToEvents(socket, userId, data.eventTypes || []);
      logger.info(`User ${userEmail} subscribed to events: ${data.eventTypes?.join(', ')}`);
    } catch (error) {
      logger.error('Event subscription error:', error);
      socket.emit('error', { type: 'EVENT_SUBSCRIPTION_ERROR', message: 'Failed to subscribe to events' });
    }
  });

  // Handle unsubscriptions
  socket.on('unsubscribe:prices', async (data) => {
    await priceStreamManager.unsubscribeFromPrices(socket, data.symbols || []);
  });

  socket.on('unsubscribe:orders', async () => {
    await orderNotificationManager.unsubscribeFromOrders(socket, userId);
  });

  socket.on('unsubscribe:events', async (data) => {
    await eventBroadcaster.unsubscribeFromEvents(socket, userId, data.eventTypes || []);
  });

  // Handle ping/pong for connection health
  socket.on('ping', () => {
    socket.emit('pong', { timestamp: Date.now() });
  });

  // Handle disconnection
  socket.on('disconnect', (reason) => {
    logger.info(`User disconnected: ${userEmail} (${userId}) - Reason: ${reason}`);
    
    // Cleanup subscriptions
    priceStreamManager.cleanupUserSubscriptions(socket);
    orderNotificationManager.cleanupUserSubscriptions(socket, userId);
    chatMessageManager.cleanupUserSubscriptions(socket, userId);
    eventBroadcaster.cleanupUserSubscriptions(socket, userId);
  });

  // Send welcome message
  socket.emit('connected', {
    connectionId,
    timestamp: Date.now(),
    message: 'Connected to Forex Trading Platform WebSocket'
  });
});

// REST API endpoints for health and status
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'websocket-service',
    timestamp: new Date().toISOString(),
    connections: io.engine.clientsCount,
    uptime: process.uptime()
  });
});

app.get('/api/status', (req, res) => {
  res.json({
    service: 'websocket-service',
    version: '1.0.0',
    connections: {
      total: io.engine.clientsCount,
      authenticated: io.sockets.sockets.size
    },
    features: [
      'real-time-prices',
      'order-notifications', 
      'ai-chat-messaging',
      'event-broadcasting'
    ],
    timestamp: new Date().toISOString()
  });
});

app.get('/api/connections', authMiddleware.authenticateHTTP, (req, res) => {
  const connections = Array.from(io.sockets.sockets.values()).map(socket => ({
    id: socket.id,
    userId: socket.data.userId,
    userEmail: socket.data.userEmail,
    connected: socket.connected,
    rooms: Array.from(socket.rooms)
  }));

  res.json({
    total: connections.length,
    connections
  });
});

// Error handling
app.use((error: any, req: any, res: any, next: any) => {
  logger.error('HTTP error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const PORT = process.env.PORT || 3006;

server.listen(PORT, () => {
  logger.info(`🚀 WebSocket Service running on port ${PORT}`);
  logger.info(`📡 Socket.IO server ready for connections`);
  logger.info(`🔗 Frontend URL: ${process.env.FRONTEND_URL || 'http://localhost:3000'}`);
  
  // Initialize service managers
  priceStreamManager.initialize();
  orderNotificationManager.initialize();
  chatMessageManager.initialize();
  eventBroadcaster.initialize();
  
  logger.info('✅ All WebSocket service managers initialized');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('WebSocket service stopped');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    logger.info('WebSocket service stopped');
    process.exit(0);
  });
});

export default server;
