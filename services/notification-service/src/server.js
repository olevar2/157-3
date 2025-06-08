/**
 * Notification Service
 * Handles real-time notifications, alerts, and communication
 * 
 * Author: Platform3 Development Team
 * Date: December 2024
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const winston = require('winston');
const redis = require('redis');
const { Pool } = require('pg');
const nodemailer = require('nodemailer');
const WebSocket = require('ws');
const http = require('http');

const ServiceDiscoveryMiddleware = require('../../../shared/communication/service_discovery_middleware');
const Platform3MessageQueue = require('../../../shared/communication/redis_message_queue');
const HealthCheckEndpoint = require('../../../shared/communication/health_check_endpoint');
const logger = require('../../../shared/logging/platform3_logger');


// Platform3 Service Mesh Integration
const correlationMiddleware = require('../../shared/middleware/correlation_middleware');
const { circuitBreakerMiddleware } = require('../../shared/middleware/circuit_breaker_middleware');
// Initialize Express app
const app = express();
// Apply service mesh middleware
app.use(correlationMiddleware);
app.use(circuitBreakerMiddleware('notification-service'));


// Platform3 Microservices Integration
const serviceDiscovery = new ServiceDiscoveryMiddleware('services', PORT || 3000);
const messageQueue = new Platform3MessageQueue();
const healthCheck = new HealthCheckEndpoint('services', [
    {
        name: 'redis',
        check: async () => {
            return { healthy: true, responseTime: 0 };
        }
    }
]);

// Apply service discovery middleware
app.use(serviceDiscovery.middleware());

// Add health check endpoints
app.use('/api', healthCheck.getRouter());

// Register service with Consul on startup
serviceDiscovery.registerService().catch(err => {
    logger.error('Failed to register service', { error: err.message });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
});

process.on('SIGINT', async () => {
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
});

const server = http.createServer(app);
const PORT = process.env.PORT || 3010;

// Configure logging
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'notification-service' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Database configuration
const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'trading_platform',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
};

const pool = new Pool(dbConfig);

// Redis configuration
const redisClient = redis.createClient({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD || undefined
});

redisClient.on('error', (err) => {
  logger.error('Redis Client Error:', err);
});

redisClient.on('connect', () => {
  logger.info('Connected to Redis');
});

// Email configuration
const emailTransporter = nodemailer.createTransporter({
  host: process.env.SMTP_HOST || 'smtp.gmail.com',
  port: process.env.SMTP_PORT || 587,
  secure: false,
  auth: {
    user: process.env.SMTP_USER || 'your-email@gmail.com',
    pass: process.env.SMTP_PASS || 'your-app-password'
  }
});

// WebSocket server
const wss = new WebSocket.Server({ server });
const connectedClients = new Map();

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use(limiter);

// Notification manager class
class NotificationManager {
  constructor() {
    this.notifications = [];
    this.templates = {
      TRADE_EXECUTED: {
        subject: 'Trade Executed',
        template: 'Your {{side}} order for {{symbol}} has been executed at {{price}}'
      },
      RISK_VIOLATION: {
        subject: 'Risk Violation Alert',
        template: 'Risk violation detected: {{type}}. Current value: {{actual}}, Limit: {{limit}}'
      },
      ACCOUNT_ALERT: {
        subject: 'Account Alert',
        template: 'Account alert: {{message}}'
      },
      SYSTEM_ALERT: {
        subject: 'System Alert',
        template: 'System alert: {{message}}'
      }
    };
    
    this.subscribeToAlerts();
  }

  // Subscribe to Redis channels for alerts
  async subscribeToAlerts() {
    try {
      const subscriber = redisClient.duplicate();
      await subscriber.connect();
      
      // Subscribe to various alert channels
      await subscriber.subscribe('trade_alerts', (message) => {
        this.handleTradeAlert(JSON.parse(message));
      });
      
      await subscriber.subscribe('risk_alerts', (message) => {
        this.handleRiskAlert(JSON.parse(message));
      });
      
      await subscriber.subscribe('compliance_alerts', (message) => {
        this.handleComplianceAlert(JSON.parse(message));
      });
      
      await subscriber.subscribe('system_alerts', (message) => {
        this.handleSystemAlert(JSON.parse(message));
      });
      
      logger.info('Subscribed to alert channels');
    } catch (error) {
      logger.error('Failed to subscribe to alerts:', error);
    }
  }

  // Handle trade alerts
  async handleTradeAlert(alert) {
    try {
      const notification = {
        id: `TRADE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'TRADE_EXECUTED',
        userId: alert.userId,
        title: 'Trade Executed',
        message: this.formatMessage('TRADE_EXECUTED', alert.data),
        data: alert.data,
        priority: 'MEDIUM',
        channels: ['websocket', 'email'],
        timestamp: new Date().toISOString(),
        status: 'PENDING'
      };

      await this.sendNotification(notification);
    } catch (error) {
      logger.error('Failed to handle trade alert:', error);
    }
  }

  // Handle risk alerts
  async handleRiskAlert(alert) {
    try {
      const notification = {
        id: `RISK_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'RISK_VIOLATION',
        userId: alert.userId,
        title: 'Risk Violation Alert',
        message: this.formatMessage('RISK_VIOLATION', alert.data),
        data: alert.data,
        priority: 'HIGH',
        channels: ['websocket', 'email', 'sms'],
        timestamp: new Date().toISOString(),
        status: 'PENDING'
      };

      await this.sendNotification(notification);
    } catch (error) {
      logger.error('Failed to handle risk alert:', error);
    }
  }

  // Handle compliance alerts
  async handleComplianceAlert(alert) {
    try {
      const notification = {
        id: `COMPLIANCE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'RISK_VIOLATION',
        userId: alert.data.userId,
        title: 'Compliance Alert',
        message: alert.message,
        data: alert.data,
        priority: 'CRITICAL',
        channels: ['websocket', 'email'],
        timestamp: new Date().toISOString(),
        status: 'PENDING'
      };

      await this.sendNotification(notification);
    } catch (error) {
      logger.error('Failed to handle compliance alert:', error);
    }
  }

  // Handle system alerts
  async handleSystemAlert(alert) {
    try {
      const notification = {
        id: `SYSTEM_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'SYSTEM_ALERT',
        userId: null, // System-wide alert
        title: 'System Alert',
        message: alert.message,
        data: alert.data,
        priority: 'HIGH',
        channels: ['websocket', 'email'],
        timestamp: new Date().toISOString(),
        status: 'PENDING'
      };

      await this.sendNotification(notification);
    } catch (error) {
      logger.error('Failed to handle system alert:', error);
    }
  }

  // Send notification through multiple channels
  async sendNotification(notification) {
    try {
      // Store notification in database
      await this.storeNotification(notification);

      // Send through specified channels
      const promises = notification.channels.map(channel => {
        switch (channel) {
          case 'websocket':
            return this.sendWebSocketNotification(notification);
          case 'email':
            return this.sendEmailNotification(notification);
          case 'sms':
            return this.sendSMSNotification(notification);
          default:
            logger.warn(`Unknown notification channel: ${channel}`);
            return Promise.resolve();
        }
      });

      await Promise.allSettled(promises);

      // Update notification status
      notification.status = 'SENT';
      await this.updateNotificationStatus(notification.id, 'SENT');

      logger.info(`Notification sent: ${notification.id}`);
    } catch (error) {
      logger.error('Failed to send notification:', error);
      await this.updateNotificationStatus(notification.id, 'FAILED');
    }
  }

  // Store notification in database
  async storeNotification(notification) {
    try {
      await pool.query(
        'INSERT INTO notifications (id, type, user_id, title, message, data, priority, channels, timestamp, status) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)',
        [
          notification.id,
          notification.type,
          notification.userId,
          notification.title,
          notification.message,
          JSON.stringify(notification.data),
          notification.priority,
          JSON.stringify(notification.channels),
          notification.timestamp,
          notification.status
        ]
      );
    } catch (error) {
      logger.error('Failed to store notification:', error);
      throw error;
    }
  }

  // Update notification status
  async updateNotificationStatus(notificationId, status) {
    try {
      await pool.query(
        'UPDATE notifications SET status = $1, updated_at = NOW() WHERE id = $2',
        [status, notificationId]
      );
    } catch (error) {
      logger.error('Failed to update notification status:', error);
    }
  }

  // Send WebSocket notification
  async sendWebSocketNotification(notification) {
    try {
      const message = JSON.stringify({
        type: 'notification',
        data: notification
      });

      if (notification.userId) {
        // Send to specific user
        const userConnections = connectedClients.get(notification.userId);
        if (userConnections) {
          userConnections.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(message);
            }
          });
        }
      } else {
        // Broadcast to all connected clients
        wss.clients.forEach(ws => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(message);
          }
        });
      }

      logger.info(`WebSocket notification sent: ${notification.id}`);
    } catch (error) {
      logger.error('Failed to send WebSocket notification:', error);
    }
  }

  // Send email notification
  async sendEmailNotification(notification) {
    try {
      if (!notification.userId) {
        logger.warn('Cannot send email notification without userId');
        return;
      }

      // Get user email from database
      const userResult = await pool.query(
        'SELECT email FROM users WHERE id = $1',
        [notification.userId]
      );

      if (userResult.rows.length === 0) {
        logger.warn(`User not found for notification: ${notification.userId}`);
        return;
      }

      const userEmail = userResult.rows[0].email;

      const mailOptions = {
        from: process.env.SMTP_FROM || 'noreply@platform3.com',
        to: userEmail,
        subject: notification.title,
        html: this.generateEmailHTML(notification),
        text: notification.message
      };

      await emailTransporter.sendMail(mailOptions);
      logger.info(`Email notification sent: ${notification.id} to ${userEmail}`);
    } catch (error) {
      logger.error('Failed to send email notification:', error);
    }
  }

  // Send SMS notification (placeholder)
  async sendSMSNotification(notification) {
    try {
      // SMS implementation would go here
      // For now, just log the notification
      logger.info(`SMS notification (placeholder): ${notification.id}`);
    } catch (error) {
      logger.error('Failed to send SMS notification:', error);
    }
  }

  // Format message using template
  formatMessage(templateType, data) {
    try {
      const template = this.templates[templateType];
      if (!template) {
        return JSON.stringify(data);
      }

      let message = template.template;
      Object.keys(data).forEach(key => {
        message = message.replace(new RegExp(`{{${key}}}`, 'g'), data[key]);
      });

      return message;
    } catch (error) {
      logger.error('Failed to format message:', error);
      return JSON.stringify(data);
    }
  }

  // Generate email HTML
  generateEmailHTML(notification) {
    return `
      <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
          <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #333; margin-bottom: 20px;">${notification.title}</h2>
            <p style="color: #666; line-height: 1.6; margin-bottom: 20px;">${notification.message}</p>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
              <strong>Priority:</strong> ${notification.priority}<br>
              <strong>Time:</strong> ${new Date(notification.timestamp).toLocaleString()}
            </div>
            <p style="color: #999; font-size: 12px; margin-top: 30px;">
              This is an automated message from Platform3 Trading Platform.
            </p>
          </div>
        </body>
      </html>
    `;
  }

  // Get user notifications
  async getUserNotifications(userId, limit = 50, offset = 0) {
    try {
      const result = await pool.query(
        'SELECT * FROM notifications WHERE user_id = $1 ORDER BY timestamp DESC LIMIT $2 OFFSET $3',
        [userId, limit, offset]
      );
      return result.rows;
    } catch (error) {
      logger.error('Failed to get user notifications:', error);
      throw error;
    }
  }

  // Mark notification as read
  async markAsRead(notificationId, userId) {
    try {
      await pool.query(
        'UPDATE notifications SET read_at = NOW() WHERE id = $1 AND user_id = $2',
        [notificationId, userId]
      );
      logger.info(`Notification marked as read: ${notificationId}`);
    } catch (error) {
      logger.error('Failed to mark notification as read:', error);
      throw error;
    }
  }
}

// Initialize notification manager
const notificationManager = new NotificationManager();

// WebSocket connection handling
wss.on('connection', (ws, req) => {
  logger.info('New WebSocket connection');

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);
      
      if (data.type === 'auth' && data.userId) {
        // Associate connection with user
        if (!connectedClients.has(data.userId)) {
          connectedClients.set(data.userId, new Set());
        }
        connectedClients.get(data.userId).add(ws);
        
        ws.userId = data.userId;
        ws.send(JSON.stringify({ type: 'auth_success', message: 'Authenticated successfully' }));
        logger.info(`User ${data.userId} authenticated via WebSocket`);
      }
    } catch (error) {
      logger.error('Failed to handle WebSocket message:', error);
    }
  });

  ws.on('close', () => {
    if (ws.userId) {
      const userConnections = connectedClients.get(ws.userId);
      if (userConnections) {
        userConnections.delete(ws);
        if (userConnections.size === 0) {
          connectedClients.delete(ws.userId);
        }
      }
      logger.info(`User ${ws.userId} disconnected from WebSocket`);
    }
  });
});

// Routes

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'notification-service',
    connectedClients: connectedClients.size
  });
});

// Send custom notification
app.post('/send', async (req, res) => {
  try {
    const notification = {
      id: `CUSTOM_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      ...req.body,
      timestamp: new Date().toISOString(),
      status: 'PENDING'
    };

    await notificationManager.sendNotification(notification);
    res.json({ success: true, notificationId: notification.id });
  } catch (error) {
    logger.error('Failed to send custom notification:', error);
    res.status(500).json({ error: 'Failed to send notification' });
  }
});

// Get user notifications
app.get('/notifications/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { limit = 50, offset = 0 } = req.query;
    
    const notifications = await notificationManager.getUserNotifications(
      userId, 
      parseInt(limit), 
      parseInt(offset)
    );
    
    res.json({ notifications });
  } catch (error) {
    logger.error('Failed to get user notifications:', error);
    res.status(500).json({ error: 'Failed to get notifications' });
  }
});

// Mark notification as read
app.put('/notifications/:notificationId/read', async (req, res) => {
  try {
    const { notificationId } = req.params;
    const { userId } = req.body;
    
    await notificationManager.markAsRead(notificationId, userId);
    res.json({ success: true });
  } catch (error) {
    logger.error('Failed to mark notification as read:', error);
    res.status(500).json({ error: 'Failed to mark as read' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
server.listen(PORT, () => {
  logger.info(`Notification service running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  wss.close();
  await pool.end();
  await redisClient.quit();
  process.exit(0);
});

module.exports = app;
