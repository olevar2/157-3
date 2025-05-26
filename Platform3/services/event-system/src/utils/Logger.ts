import winston, { Logger as WinstonLogger, format, transports } from 'winston';
import path from 'path';

export interface LoggerConfig {
  level?: string;
  format?: 'json' | 'simple';
  logToFile?: boolean;
  logDirectory?: string;
  maxFileSize?: string;
  maxFiles?: string;
  colorize?: boolean;
}

export class Logger {
  private logger: WinstonLogger;
  private context: string;

  constructor(context: string = 'Application', config?: LoggerConfig) {
    this.context = context;
    this.logger = this.createLogger(config || {});
  }

  private createLogger(config: LoggerConfig): WinstonLogger {
    const {
      level = process.env.LOG_LEVEL || 'info',
      format: logFormat = 'json',
      logToFile = true,
      logDirectory = './logs',
      maxFileSize = '20m',
      maxFiles = '14d',
      colorize = process.env.NODE_ENV !== 'production'
    } = config;

    // Create custom format
    const customFormat = format.combine(
      format.timestamp({
        format: 'YYYY-MM-DD HH:mm:ss.SSS'
      }),
      format.errors({ stack: true }),
      format.printf(({ timestamp, level, message, context, ...meta }) => {
        const logObject = {
          timestamp,
          level: level.toUpperCase(),
          context: context || this.context,
          message,
          ...meta
        };

        if (logFormat === 'json') {
          return JSON.stringify(logObject);
        } else {
          const metaStr = Object.keys(meta).length > 0 ? ` ${JSON.stringify(meta)}` : '';
          return `${timestamp} [${level.toUpperCase()}] [${context || this.context}] ${message}${metaStr}`;
        }
      })
    );

    const loggerTransports: winston.transport[] = [
      // Console transport
      new transports.Console({
        level,
        format: colorize
          ? format.combine(
              format.colorize(),
              customFormat
            )
          : customFormat
      })
    ];

    // File transports (if enabled)
    if (logToFile) {
      // Combined logs
      loggerTransports.push(        new transports.File({
          filename: path.join(logDirectory, 'combined.log'),
          level: 'info',
          format: customFormat,
          maxsize: this.parseFileSize(maxFileSize),
          maxFiles: parseInt(maxFiles, 10),
          tailable: true,
          zippedArchive: true
        })
      );      // Error logs
      loggerTransports.push(
        new transports.File({
          filename: path.join(logDirectory, 'error.log'),
          level: 'error',
          format: customFormat,
          maxsize: this.parseFileSize(maxFileSize),
          maxFiles: parseInt(maxFiles, 10),
          tailable: true,
          zippedArchive: true
        })
      );

      // Debug logs (only in development)
      if (process.env.NODE_ENV !== 'production') {
        loggerTransports.push(
          new transports.File({
            filename: path.join(logDirectory, 'debug.log'),
            level: 'debug',
            format: customFormat,
            maxsize: this.parseFileSize(maxFileSize),
            maxFiles: 7,
            tailable: true,
            zippedArchive: true
          })
        );
      }
    }

    return winston.createLogger({
      level,
      format: customFormat,
      transports: loggerTransports,      // Handle uncaught exceptions and rejections
      exceptionHandlers: logToFile ? [
        new transports.File({
          filename: path.join(logDirectory, 'exceptions.log'),
          maxsize: this.parseFileSize(maxFileSize),
          maxFiles: parseInt(maxFiles, 10)
        })
      ] : [],
      rejectionHandlers: logToFile ? [
        new transports.File({
          filename: path.join(logDirectory, 'rejections.log'),
          maxsize: this.parseFileSize(maxFileSize),
          maxFiles: parseInt(maxFiles, 10)
        })
      ] : []
    });
  }

  private parseFileSize(sizeStr: string): number {
    const units: { [key: string]: number } = {
      'b': 1,
      'k': 1024,
      'm': 1024 * 1024,
      'g': 1024 * 1024 * 1024
    };

    const match = sizeStr.toLowerCase().match(/^(\d+)([bkmg])?$/);
    if (!match) {
      return 20 * 1024 * 1024; // Default 20MB
    }

    const size = parseInt(match[1]);
    const unit = match[2] || 'b';
    return size * units[unit];
  }

  // Core logging methods
  public error(message: string, meta?: any): void {
    this.logger.error(message, { context: this.context, ...meta });
  }

  public warn(message: string, meta?: any): void {
    this.logger.warn(message, { context: this.context, ...meta });
  }

  public info(message: string, meta?: any): void {
    this.logger.info(message, { context: this.context, ...meta });
  }

  public debug(message: string, meta?: any): void {
    this.logger.debug(message, { context: this.context, ...meta });
  }

  public verbose(message: string, meta?: any): void {
    this.logger.verbose(message, { context: this.context, ...meta });
  }

  // Trading-specific logging methods
  public logTrade(message: string, tradeData: any): void {
    this.info(message, {
      type: 'TRADE',
      trade: tradeData,
      timestamp: new Date().toISOString()
    });
  }

  public logMarketData(message: string, marketData: any): void {
    this.debug(message, {
      type: 'MARKET_DATA',
      market: marketData,
      timestamp: new Date().toISOString()
    });
  }

  public logEvent(message: string, eventData: any): void {
    this.info(message, {
      type: 'EVENT',
      event: eventData,
      timestamp: new Date().toISOString()
    });
  }

  public logPerformance(message: string, performanceData: any): void {
    this.debug(message, {
      type: 'PERFORMANCE',
      performance: performanceData,
      timestamp: new Date().toISOString()
    });
  }

  public logSecurity(message: string, securityData: any): void {
    this.warn(message, {
      type: 'SECURITY',
      security: securityData,
      timestamp: new Date().toISOString()
    });
  }

  public logAudit(message: string, auditData: any): void {
    this.info(message, {
      type: 'AUDIT',
      audit: auditData,
      timestamp: new Date().toISOString()
    });
  }

  // HTTP request logging
  public logRequest(method: string, url: string, statusCode: number, responseTime: number, meta?: any): void {
    const level = statusCode >= 400 ? 'error' : statusCode >= 300 ? 'warn' : 'info';
    
    this.logger.log(level, `${method} ${url} ${statusCode}`, {
      context: this.context,
      type: 'HTTP_REQUEST',
      method,
      url,
      statusCode,
      responseTime,
      timestamp: new Date().toISOString(),
      ...meta
    });
  }

  // Database operation logging
  public logDbOperation(operation: string, table: string, duration: number, meta?: any): void {
    this.debug(`Database ${operation} on ${table}`, {
      type: 'DATABASE',
      operation,
      table,
      duration,
      timestamp: new Date().toISOString(),
      ...meta
    });
  }

  // Create child logger with additional context
  public createChild(childContext: string): Logger {
    const fullContext = `${this.context}:${childContext}`;
    return new Logger(fullContext);
  }

  // Get the underlying Winston logger (for advanced usage)
  public getWinstonLogger(): WinstonLogger {
    return this.logger;
  }

  // Update log level dynamically
  public setLevel(level: string): void {
    this.logger.level = level;
    this.logger.transports.forEach(transport => {
      transport.level = level;
    });
  }

  // Close logger and clean up
  public close(): Promise<void> {
    return new Promise((resolve) => {
      this.logger.end(() => {
        resolve();
      });
    });
  }
}

// Export singleton instance for global use
export const globalLogger = new Logger('Global', {
  level: process.env.LOG_LEVEL || 'info',
  format: process.env.LOG_FORMAT as 'json' | 'simple' || 'json',
  logToFile: process.env.LOG_TO_FILE !== 'false',
  logDirectory: process.env.LOG_DIRECTORY || './logs',
  colorize: process.env.NODE_ENV !== 'production'
});

// Helper function to create logger instances
export function createLogger(context: string, config?: LoggerConfig): Logger {
  return new Logger(context, config);
}
