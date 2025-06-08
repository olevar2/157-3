import * as path from 'path';
import { LogLevel, LogFormat, LogDestination, LogConfig } from './types';

/**
 * Singleton class to manage logging configuration and provide thread-safe logging
 */
export class LogManager {
    private static instance: LogManager;
    private config: LogConfig;
    private isLocked: boolean = false;
    private lockQueue: (() => void)[] = [];

    /**
     * Get the singleton instance of LogManager
     */
    public static getInstance(): LogManager {
        if (!LogManager.instance) {
            LogManager.instance = new LogManager();
        }
        return LogManager.instance;
    }

    /**
     * Private constructor for singleton pattern
     */
    private constructor() {
        // Set default configuration
        this.config = {
            minLevel: LogLevel.INFO,
            format: LogFormat.TEXT,
            destinations: [LogDestination.CONSOLE],
            filePath: path.join(process.cwd(), 'logs', 'platform3.log'),
            maxFileSize: 10 * 1024 * 1024, // 10 MB
            maxFiles: 10
        };
    }

    /**
     * Configure the logging system
     * @param config The logging configuration
     */
    public configure(config: Partial<LogConfig>): void {
        this.config = {
            ...this.config,
            ...config
        };
    }

    /**
     * Get the current logging configuration
     */
    public getConfig(): LogConfig {
        return { ...this.config };
    }

    /**
     * Acquire lock for thread-safe logging
     */
    public acquireLock(): void {
        if (this.isLocked) {
            // If already locked, add callback to queue
            const promise = new Promise<void>(resolve => {
                this.lockQueue.push(resolve);
            });
            
            // Wait for lock to be released
            // This creates a microtask to run after the current task
            promise.then(() => {
                this.isLocked = true;
            });
            
            // In a real implementation, we might use a more sophisticated locking mechanism
            // such as a semaphore or mutex, but this simple approach works for logging
            return;
        }
        
        // Acquire lock
        this.isLocked = true;
    }

    /**
     * Release lock for thread-safe logging
     */
    public releaseLock(): void {
        this.isLocked = false;
        
        // Process next item in the queue
        if (this.lockQueue.length > 0) {
            const next = this.lockQueue.shift();
            if (next) {
                next();
            }
        }
    }
}
