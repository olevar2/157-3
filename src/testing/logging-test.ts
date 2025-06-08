import * as path from 'path';
import * as fs from 'fs';
import { Logger, LogManager, LogLevel, LogFormat, LogDestination } from '../shared/logging';

// Configure the logger
LogManager.getInstance().configure({
    minLevel: LogLevel.DEBUG,
    format: LogFormat.JSON,
    destinations: [LogDestination.CONSOLE, LogDestination.FILE],
    filePath: path.join(process.cwd(), 'logs', 'test-logging.log')
});

// Get some loggers for different components
const apiLogger = Logger.getLogger('API');
const bridgeLogger = Logger.getLogger('Bridge');
const dbLogger = Logger.getLogger('Database');

// Log some test messages
apiLogger.debug('API debug message', { endpoint: '/users', method: 'GET' });
apiLogger.info('API request received', { endpoint: '/users', method: 'GET', userId: '123' });
bridgeLogger.info('Bridge message sent', { messageId: 'abc123', destination: 'python-service' });
dbLogger.warn('Database connection pool at 80% capacity', { poolSize: 10, activeConnections: 8 });

// Log an error with stack trace
try {
    throw new Error('Something went wrong');
} catch (error) {
    if (error instanceof Error) {
        bridgeLogger.error('Bridge communication failed', error, { 
            messageId: 'def456', 
            retryCount: 3 
        });
    }
}

// Check if log file was created
const logFilePath = path.join(process.cwd(), 'logs', 'test-logging.log');
console.log(`\nChecking log file at: ${logFilePath}`);
if (fs.existsSync(logFilePath)) {
    console.log('✅ Log file successfully created');
    
    // Print first few lines of log file
    const logContent = fs.readFileSync(logFilePath, 'utf-8');
    const lines = logContent.split('\n').filter(line => line.trim());
    
    console.log('\nLog file contents (first 3 entries):');
    lines.slice(0, 3).forEach((line, i) => {
        console.log(`${i + 1}: ${line}`);
    });
    
    // Test JSON parsing
    try {
        const firstLogJson = JSON.parse(lines[0]);
        console.log('\n✅ JSON parsing successful');
        console.log('Sample parsed entry:');
        console.log(`  Level: ${firstLogJson.level}`);
        console.log(`  Component: ${firstLogJson.component}`);
        console.log(`  Message: ${firstLogJson.message}`);
    } catch (err) {
        console.error('❌ JSON parsing failed:', err);
    }
} else {
    console.error('❌ Log file not created');
}
