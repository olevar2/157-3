import * as fs from 'fs';
import * as path from 'path';
import { LogLevel, LogFormat, LogDestination, LogConfig } from './types';
import { LogManager } from './log-manager';

/**
 * Production-grade Logger for the TypeScript-Python bridge.
 * 
 * Features:
 * - Multiple log levels (DEBUG, INFO, WARN, ERROR)
 * - Context information (timestamp, component name, correlation IDs)
 * - Multiple output destinations (console, file)
 * - Log rotation
 * - Thread-safe for concurrent logging
 */
export class Logger {
    private component: string;
    private logConfig: LogConfig;
    private static instanceMap: Map<string, Logger> = new Map();

    /**
     * Get a logger instance for a specific component
     * @param component The component name to use in log messages
     */
    public static getLogger(component: string): Logger {
        if (!Logger.instanceMap.has(component)) {
            Logger.instanceMap.set(component, new Logger(component));
        }
        return Logger.instanceMap.get(component)!;
    }

    /**
     * Constructor for the Logger class
     * @param component The component name to use in log messages
     */
    constructor(component: string) {
        this.component = component;
        this.logConfig = LogManager.getInstance().getConfig();
        
        // Ensure log directory exists
        if (this.logConfig.destinations.includes(LogDestination.FILE)) {
            const logDir = path.dirname(this.logConfig.filePath);
            if (!fs.existsSync(logDir)) {
                fs.mkdirSync(logDir, { recursive: true });
            }
        }
    }

    /**
     * Log a debug message
     * @param message The message to log
     * @param context Additional context data to include
     */
    public debug(message: string, context: Record<string, any> = {}): void {
        this.log(LogLevel.DEBUG, message, context);
    }

    /**
     * Log an info message
     * @param message The message to log
     * @param context Additional context data to include
     */
    public info(message: string, context: Record<string, any> = {}): void {
        this.log(LogLevel.INFO, message, context);
    }

    /**
     * Log a warning message
     * @param message The message to log
     * @param context Additional context data to include
     */
    public warn(message: string, context: Record<string, any> = {}): void {
        this.log(LogLevel.WARN, message, context);
    }

    /**
     * Log an error message
     * @param message The message to log
     * @param context Additional context data to include
     */
    public error(message: string, error?: Error, context: Record<string, any> = {}): void {
        // Add error details to context
        if (error) {
            context = {
                ...context,
                errorName: error.name,
                errorMessage: error.message,
                stack: error.stack
            };
        }
        this.log(LogLevel.ERROR, message, context);
    }

    /**
     * Internal method to handle log message formatting and output
     * @param level The log level
     * @param message The message to log
     * @param context Additional context data to include
     */
    private log(level: LogLevel, message: string, context: Record<string, any> = {}): void {
        // Check if we should log this level
        if (level < this.logConfig.minLevel) {
            return;
        }

        // Add standard fields to context
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level: LogLevel[level],
            component: this.component,
            message,
            ...context
        };

        // Format the log entry
        let formattedLog: string;
        switch (this.logConfig.format) {
            case LogFormat.JSON:
                formattedLog = JSON.stringify(logEntry);
                break;
            case LogFormat.TEXT:
            default:
                formattedLog = `[${timestamp}] [${LogLevel[level]}] [${this.component}] ${message}`;
                if (Object.keys(context).length > 0) {
                    formattedLog += ` ${JSON.stringify(context)}`;
                }
        }

        // Output to configured destinations
        this.output(formattedLog, level);
    }

    /**
     * Output log entry to all configured destinations
     * @param formattedLog The formatted log message
     * @param level The log level
     */
    private output(formattedLog: string, level: LogLevel): void {
        // Use lock mechanism for thread-safety
        LogManager.getInstance().acquireLock();

        try {
            // Output to console if configured
            if (this.logConfig.destinations.includes(LogDestination.CONSOLE)) {
                switch (level) {
                    case LogLevel.ERROR:
                        console.error(formattedLog);
                        break;
                    case LogLevel.WARN:
                        console.warn(formattedLog);
                        break;
                    case LogLevel.INFO:
                        console.info(formattedLog);
                        break;
                    case LogLevel.DEBUG:
                    default:
                        console.debug(formattedLog);
                        break;
                }
            }

            // Output to file if configured
            if (this.logConfig.destinations.includes(LogDestination.FILE)) {
                this.writeToFile(formattedLog + '\n');
            }
        } finally {
            LogManager.getInstance().releaseLock();
        }
    }

    /**
     * Write a log entry to the configured file
     * @param formattedLog The formatted log message
     */
    private writeToFile(formattedLog: string): void {
        try {
            // Check if log rotation is needed
            this.checkRotation();

            // Append to log file
            fs.appendFileSync(this.logConfig.filePath, formattedLog);
        } catch (error) {
            // If file logging fails, output to console as fallback
            console.error(`Failed to write to log file: ${error instanceof Error ? error.message : String(error)}`);
            console.error(formattedLog);
        }
    }

    /**
     * Check if log rotation is needed and rotate if necessary
     */
    private checkRotation(): void {
        if (!this.logConfig.maxFileSize || !this.logConfig.filePath) {
            return;
        }

        try {
            // Check current file size
            if (fs.existsSync(this.logConfig.filePath)) {
                const stats = fs.statSync(this.logConfig.filePath);

                // If file exceeds max size, rotate
                if (stats.size >= this.logConfig.maxFileSize) {
                    this.rotateLog();
                }
            }
        } catch (error) {
            console.error(`Failed to check log file size: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    /**
     * Rotate log files
     */
    private rotateLog(): void {
        try {
            const basePath = this.logConfig.filePath;
            const ext = path.extname(basePath);
            const baseWithoutExt = basePath.substring(0, basePath.length - ext.length);
            const timestamp = new Date().toISOString().replace(/:/g, '-').replace(/\./g, '-');
            
            // Rotate the current log file
            const rotatedPath = `${baseWithoutExt}.${timestamp}${ext}`;
            fs.renameSync(basePath, rotatedPath);

            // Remove old log files if maxFiles is configured
            if (this.logConfig.maxFiles > 0) {
                const logDir = path.dirname(basePath);
                const baseFileName = path.basename(baseWithoutExt);
                
                // Find all log files with this base name
                const files = fs.readdirSync(logDir)
                    .filter(file => file.startsWith(baseFileName) && file !== path.basename(basePath))
                    .map(file => path.join(logDir, file))
                    .sort((a, b) => {
                        return fs.statSync(b).mtime.getTime() - fs.statSync(a).mtime.getTime();
                    });

                // Delete old files beyond maxFiles
                if (files.length >= this.logConfig.maxFiles) {
                    files.slice(this.logConfig.maxFiles - 1).forEach(file => {
                        fs.unlinkSync(file);
                    });
                }
            }
        } catch (error) {
            console.error(`Failed to rotate log file: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
}
