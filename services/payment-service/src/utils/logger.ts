import winston from 'winston';
import { serverConfig } from '@/config';

const logFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json(),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    return JSON.stringify({
      timestamp,
      level,
      message,
      service: 'payment-service',
      ...meta
    });
  })
);

export const logger = winston.createLogger({
  level: serverConfig.nodeEnv === 'production' ? 'info' : 'debug',
  format: logFormat,
  defaultMeta: { service: 'payment-service' },
  transports: [
    new winston.transports.File({ 
      filename: 'logs/payment-service-error.log', 
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    new winston.transports.File({ 
      filename: 'logs/payment-service-combined.log',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ]
});

// Add console transport for development
if (serverConfig.nodeEnv !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.combine(
      winston.format.colorize(),
      winston.format.simple(),
      winston.format.printf(({ timestamp, level, message, ...meta }) => {
        const metaStr = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : '';
        return `${timestamp} [${level}]: ${message} ${metaStr}`;
      })
    )
  }));
}

export default logger;
