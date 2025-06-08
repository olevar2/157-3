/**
 * Log levels for the logging system
 */
export enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2, 
    ERROR = 3
}

/**
 * Log output formats
 */
export enum LogFormat {
    TEXT = 'text',
    JSON = 'json'
}

/**
 * Log output destinations
 */
export enum LogDestination {
    CONSOLE = 'console',
    FILE = 'file'
}

/**
 * Configuration for the logging system
 */
export interface LogConfig {
    /** Minimum log level to output */
    minLevel: LogLevel;
    
    /** Log format (text or JSON) */
    format: LogFormat;
    
    /** Log destinations (console, file) */
    destinations: LogDestination[];
    
    /** Path to the log file (if file destination is enabled) */
    filePath: string;
    
    /** Maximum log file size in bytes before rotation */
    maxFileSize: number;
    
    /** Maximum number of rotated log files to keep */
    maxFiles: number;
}

/**
 * Interface for components that need to log
 */
export interface ILogger {
    debug(message: string, context?: Record<string, any>): void;
    info(message: string, context?: Record<string, any>): void;
    warn(message: string, context?: Record<string, any>): void;
    error(message: string, error?: Error, context?: Record<string, any>): void;
}
