/**
 * Compliance Service
 * Handles regulatory compliance, audit trails, and risk monitoring
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

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3009;

// Configure logging
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'compliance-service' },
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

// Compliance monitoring class
class ComplianceMonitor {
  constructor() {
    this.riskLimits = {
      maxDailyLoss: 50000,
      maxPositionSize: 100000,
      maxLeverage: 100,
      maxDrawdown: 0.10,
      maxExposure: 1000000
    };
    
    this.auditTrail = [];
    this.violations = [];
    this.reports = [];
  }

  // Log compliance event
  async logEvent(eventType, data, userId = null) {
    try {
      const event = {
        id: `EVENT_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        eventType,
        data,
        userId,
        severity: this.determineSeverity(eventType, data)
      };

      this.auditTrail.push(event);

      // Store in database
      await pool.query(
        'INSERT INTO compliance_events (id, timestamp, event_type, data, user_id, severity) VALUES ($1, $2, $3, $4, $5, $6)',
        [event.id, event.timestamp, eventType, JSON.stringify(data), userId, event.severity]
      );

      // Cache in Redis
      await redisClient.setex(`compliance:event:${event.id}`, 86400, JSON.stringify(event));

      logger.info(`Compliance event logged: ${eventType}`, { eventId: event.id });
      return event;
    } catch (error) {
      logger.error('Failed to log compliance event:', error);
      throw error;
    }
  }

  // Check for risk violations
  async checkRiskViolations(tradeData) {
    try {
      const violations = [];

      // Check position size limit
      if (tradeData.positionSize > this.riskLimits.maxPositionSize) {
        violations.push({
          type: 'POSITION_SIZE_VIOLATION',
          limit: this.riskLimits.maxPositionSize,
          actual: tradeData.positionSize,
          severity: 'HIGH'
        });
      }

      // Check leverage limit
      if (tradeData.leverage > this.riskLimits.maxLeverage) {
        violations.push({
          type: 'LEVERAGE_VIOLATION',
          limit: this.riskLimits.maxLeverage,
          actual: tradeData.leverage,
          severity: 'CRITICAL'
        });
      }

      // Check daily loss limit
      const dailyPnL = await this.getDailyPnL(tradeData.userId);
      if (dailyPnL < -this.riskLimits.maxDailyLoss) {
        violations.push({
          type: 'DAILY_LOSS_VIOLATION',
          limit: this.riskLimits.maxDailyLoss,
          actual: Math.abs(dailyPnL),
          severity: 'CRITICAL'
        });
      }

      // Log violations
      for (const violation of violations) {
        await this.logViolation(violation, tradeData.userId);
      }

      return violations;
    } catch (error) {
      logger.error('Failed to check risk violations:', error);
      throw error;
    }
  }

  // Log risk violation
  async logViolation(violation, userId) {
    try {
      const violationRecord = {
        id: `VIOLATION_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        type: violation.type,
        severity: violation.severity,
        limit: violation.limit,
        actual: violation.actual,
        userId,
        status: 'ACTIVE'
      };

      this.violations.push(violationRecord);

      // Store in database
      await pool.query(
        'INSERT INTO risk_violations (id, timestamp, type, severity, limit_value, actual_value, user_id, status) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)',
        [violationRecord.id, violationRecord.timestamp, violation.type, violation.severity, violation.limit, violation.actual, userId, 'ACTIVE']
      );

      // Send alert
      await this.sendViolationAlert(violationRecord);

      logger.warn(`Risk violation detected: ${violation.type}`, violationRecord);
      return violationRecord;
    } catch (error) {
      logger.error('Failed to log violation:', error);
      throw error;
    }
  }

  // Send violation alert
  async sendViolationAlert(violation) {
    try {
      const alert = {
        type: 'RISK_VIOLATION',
        severity: violation.severity,
        message: `Risk violation detected: ${violation.type}`,
        data: violation,
        timestamp: new Date().toISOString()
      };

      // Publish to Redis for notification service
      await redisClient.publish('compliance_alerts', JSON.stringify(alert));

      logger.info(`Violation alert sent: ${violation.id}`);
    } catch (error) {
      logger.error('Failed to send violation alert:', error);
    }
  }

  // Get daily P&L for user
  async getDailyPnL(userId) {
    try {
      const today = new Date().toISOString().split('T')[0];
      const result = await pool.query(
        'SELECT COALESCE(SUM(pnl), 0) as daily_pnl FROM trades WHERE user_id = $1 AND DATE(created_at) = $2',
        [userId, today]
      );
      return parseFloat(result.rows[0].daily_pnl);
    } catch (error) {
      logger.error('Failed to get daily P&L:', error);
      return 0;
    }
  }

  // Generate compliance report
  async generateReport(reportType, startDate, endDate) {
    try {
      const report = {
        id: `REPORT_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: reportType,
        startDate,
        endDate,
        generatedAt: new Date().toISOString(),
        data: {}
      };

      switch (reportType) {
        case 'VIOLATIONS':
          report.data = await this.getViolationsReport(startDate, endDate);
          break;
        case 'AUDIT_TRAIL':
          report.data = await this.getAuditTrailReport(startDate, endDate);
          break;
        case 'RISK_SUMMARY':
          report.data = await this.getRiskSummaryReport(startDate, endDate);
          break;
        default:
          throw new Error(`Unknown report type: ${reportType}`);
      }

      this.reports.push(report);

      // Store in database
      await pool.query(
        'INSERT INTO compliance_reports (id, type, start_date, end_date, generated_at, data) VALUES ($1, $2, $3, $4, $5, $6)',
        [report.id, reportType, startDate, endDate, report.generatedAt, JSON.stringify(report.data)]
      );

      logger.info(`Compliance report generated: ${reportType}`, { reportId: report.id });
      return report;
    } catch (error) {
      logger.error('Failed to generate compliance report:', error);
      throw error;
    }
  }

  // Get violations report
  async getViolationsReport(startDate, endDate) {
    try {
      const result = await pool.query(
        'SELECT * FROM risk_violations WHERE timestamp BETWEEN $1 AND $2 ORDER BY timestamp DESC',
        [startDate, endDate]
      );
      
      return {
        totalViolations: result.rows.length,
        violationsByType: this.groupBy(result.rows, 'type'),
        violationsBySeverity: this.groupBy(result.rows, 'severity'),
        violations: result.rows
      };
    } catch (error) {
      logger.error('Failed to get violations report:', error);
      throw error;
    }
  }

  // Get audit trail report
  async getAuditTrailReport(startDate, endDate) {
    try {
      const result = await pool.query(
        'SELECT * FROM compliance_events WHERE timestamp BETWEEN $1 AND $2 ORDER BY timestamp DESC',
        [startDate, endDate]
      );
      
      return {
        totalEvents: result.rows.length,
        eventsByType: this.groupBy(result.rows, 'event_type'),
        eventsBySeverity: this.groupBy(result.rows, 'severity'),
        events: result.rows
      };
    } catch (error) {
      logger.error('Failed to get audit trail report:', error);
      throw error;
    }
  }

  // Get risk summary report
  async getRiskSummaryReport(startDate, endDate) {
    try {
      const [tradesResult, violationsResult, exposureResult] = await Promise.all([
        pool.query('SELECT COUNT(*) as total_trades, AVG(pnl) as avg_pnl FROM trades WHERE created_at BETWEEN $1 AND $2', [startDate, endDate]),
        pool.query('SELECT COUNT(*) as total_violations FROM risk_violations WHERE timestamp BETWEEN $1 AND $2', [startDate, endDate]),
        pool.query('SELECT MAX(position_size) as max_exposure FROM trades WHERE created_at BETWEEN $1 AND $2', [startDate, endDate])
      ]);

      return {
        totalTrades: parseInt(tradesResult.rows[0].total_trades),
        averagePnL: parseFloat(tradesResult.rows[0].avg_pnl) || 0,
        totalViolations: parseInt(violationsResult.rows[0].total_violations),
        maxExposure: parseFloat(exposureResult.rows[0].max_exposure) || 0,
        complianceScore: this.calculateComplianceScore(violationsResult.rows[0].total_violations, tradesResult.rows[0].total_trades)
      };
    } catch (error) {
      logger.error('Failed to get risk summary report:', error);
      throw error;
    }
  }

  // Calculate compliance score
  calculateComplianceScore(violations, totalTrades) {
    if (totalTrades === 0) return 100;
    const violationRate = violations / totalTrades;
    return Math.max(0, Math.min(100, 100 - (violationRate * 100)));
  }

  // Utility function to group array by property
  groupBy(array, property) {
    return array.reduce((groups, item) => {
      const key = item[property];
      groups[key] = groups[key] || [];
      groups[key].push(item);
      return groups;
    }, {});
  }

  // Determine event severity
  determineSeverity(eventType, data) {
    const highSeverityEvents = ['TRADE_EXECUTION', 'RISK_VIOLATION', 'SYSTEM_ERROR'];
    const mediumSeverityEvents = ['USER_LOGIN', 'POSITION_CHANGE', 'SETTINGS_UPDATE'];
    
    if (highSeverityEvents.includes(eventType)) return 'HIGH';
    if (mediumSeverityEvents.includes(eventType)) return 'MEDIUM';
    return 'LOW';
  }
}

// Initialize compliance monitor
const complianceMonitor = new ComplianceMonitor();

// Routes

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'compliance-service'
  });
});

// Log compliance event
app.post('/events', async (req, res) => {
  try {
    const { eventType, data, userId } = req.body;
    const event = await complianceMonitor.logEvent(eventType, data, userId);
    res.json({ success: true, event });
  } catch (error) {
    logger.error('Failed to log event:', error);
    res.status(500).json({ error: 'Failed to log event' });
  }
});

// Check risk violations
app.post('/check-violations', async (req, res) => {
  try {
    const violations = await complianceMonitor.checkRiskViolations(req.body);
    res.json({ violations });
  } catch (error) {
    logger.error('Failed to check violations:', error);
    res.status(500).json({ error: 'Failed to check violations' });
  }
});

// Generate compliance report
app.post('/reports', async (req, res) => {
  try {
    const { reportType, startDate, endDate } = req.body;
    const report = await complianceMonitor.generateReport(reportType, startDate, endDate);
    res.json({ success: true, report });
  } catch (error) {
    logger.error('Failed to generate report:', error);
    res.status(500).json({ error: 'Failed to generate report' });
  }
});

// Get violations
app.get('/violations', async (req, res) => {
  try {
    const { startDate, endDate, severity, type } = req.query;
    let query = 'SELECT * FROM risk_violations WHERE 1=1';
    const params = [];
    let paramCount = 0;

    if (startDate && endDate) {
      query += ` AND timestamp BETWEEN $${++paramCount} AND $${++paramCount}`;
      params.push(startDate, endDate);
    }

    if (severity) {
      query += ` AND severity = $${++paramCount}`;
      params.push(severity);
    }

    if (type) {
      query += ` AND type = $${++paramCount}`;
      params.push(type);
    }

    query += ' ORDER BY timestamp DESC LIMIT 100';

    const result = await pool.query(query, params);
    res.json({ violations: result.rows });
  } catch (error) {
    logger.error('Failed to get violations:', error);
    res.status(500).json({ error: 'Failed to get violations' });
  }
});

// Get audit trail
app.get('/audit-trail', async (req, res) => {
  try {
    const { startDate, endDate, eventType, userId } = req.query;
    let query = 'SELECT * FROM compliance_events WHERE 1=1';
    const params = [];
    let paramCount = 0;

    if (startDate && endDate) {
      query += ` AND timestamp BETWEEN $${++paramCount} AND $${++paramCount}`;
      params.push(startDate, endDate);
    }

    if (eventType) {
      query += ` AND event_type = $${++paramCount}`;
      params.push(eventType);
    }

    if (userId) {
      query += ` AND user_id = $${++paramCount}`;
      params.push(userId);
    }

    query += ' ORDER BY timestamp DESC LIMIT 100';

    const result = await pool.query(query, params);
    res.json({ events: result.rows });
  } catch (error) {
    logger.error('Failed to get audit trail:', error);
    res.status(500).json({ error: 'Failed to get audit trail' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
  logger.info(`Compliance service running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await pool.end();
  await redisClient.quit();
  process.exit(0);
});

module.exports = app;
