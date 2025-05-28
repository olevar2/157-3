import winston from 'winston';

// Create logger instance for Configuration Service
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    winston.format.json(),
    winston.format.printf(({ timestamp, level, message, service = 'config-service', ...meta }) => {
      return JSON.stringify({
        timestamp,
        level,
        service,
        message,
        ...meta
      });
    })
  ),
  defaultMeta: { 
    service: 'config-service',
    version: '1.0.0'
  },
  transports: [
    // Console transport for development
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    
    // File transport for all logs
    new winston.transports.File({ 
      filename: 'logs/config-service.log',
      maxsize: 10485760, // 10MB
      maxFiles: 5,
      tailable: true
    }),
    
    // Error file transport
    new winston.transports.File({ 
      filename: 'logs/config-service-error.log',
      level: 'error',
      maxsize: 10485760, // 10MB
      maxFiles: 5,
      tailable: true
    })
  ],
  
  // Handle exceptions and rejections
  exceptionHandlers: [
    new winston.transports.File({ filename: 'logs/config-service-exceptions.log' })
  ],
  rejectionHandlers: [
    new winston.transports.File({ filename: 'logs/config-service-rejections.log' })
  ]
});

// Create logs directory if it doesn't exist
import { mkdirSync } from 'fs';
try {
  mkdirSync('logs', { recursive: true });
} catch (error) {
  // Directory already exists or permission error
}

// Export logger with additional utility methods
export const createLogger = (serviceName: string) => {
  return logger.child({ service: serviceName });
};

export default logger;
