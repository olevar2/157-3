import { Logger } from './logger';
import { LogManager } from './log-manager';
import { LogLevel, LogFormat, LogDestination, LogConfig, ILogger } from './types';

// Re-export all logging components
export {
    Logger,
    LogManager,
    LogLevel,
    LogFormat,
    LogDestination,
    LogConfig,
    ILogger
};

// Export default getLogger function for convenience
export default function getLogger(component: string): Logger {
    return Logger.getLogger(component);
}
