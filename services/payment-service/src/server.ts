/**
 * Payment Service - Main Server
 * Comprehensive payment processing and broker account integration service.
 * 
 * Features:
 * - Multi-broker account integration
 * - Secure payment processing (Stripe, PayPal)
 * - Real-time balance synchronization
 * - Transaction audit trails
 * - Compliance and security
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { createLogger, format, transports } from 'winston';
import dotenv from 'dotenv';

// Import middleware and utilities
import { authMiddleware } from './middleware/auth';
import { errorHandler } from './middleware/errorHandler';
import { securityMiddleware } from './middleware/security';
import { validationMiddleware } from './middleware/validation';
import { DatabaseManager } from './utils/database';
import { RedisManager } from './utils/redis';
import { Logger } from './utils/logger';

// Import route handlers
import { BrokerAccountRoutes } from './routes/brokerAccounts';
import { PaymentRoutes } from './routes/payments';
import { TransactionRoutes } from './routes/transactions';
import { BalanceRoutes } from './routes/balances';
import { ComplianceRoutes } from './routes/compliance';

// Import services
import { BrokerIntegrationService } from './services/BrokerIntegrationService';
import { PaymentProcessorService } from './services/PaymentProcessorService';
import { BalanceSyncService } from './services/BalanceSyncService';
import { ComplianceService } from './services/ComplianceService';
import { AuditService } from './services/AuditService';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local' });

class PaymentServer {
    private app: express.Application;
    private logger: any;
    private dbManager: DatabaseManager;
    private redisManager: RedisManager;
    
    // Core services
    private brokerIntegrationService: BrokerIntegrationService;
    private paymentProcessorService: PaymentProcessorService;
    private balanceSyncService: BalanceSyncService;
    private complianceService: ComplianceService;
    private auditService: AuditService;
    
    constructor() {
        this.app = express();
        this.logger = Logger.getInstance();
        this.initializeServices();
        this.setupMiddleware();
        this.setupRoutes();
        this.setupErrorHandling();
    }
    
    private async initializeServices(): Promise<void> {
        try {
            // Initialize database and Redis connections
            this.dbManager = new DatabaseManager();
            await this.dbManager.connect();
            
            this.redisManager = new RedisManager();
            await this.redisManager.connect();
            
            // Initialize core services
            this.brokerIntegrationService = new BrokerIntegrationService({
                database: this.dbManager,
                redis: this.redisManager,
                logger: this.logger
            });
            
            this.paymentProcessorService = new PaymentProcessorService({
                stripeKey: process.env.STRIPE_SECRET_KEY,
                paypalConfig: {
                    clientId: process.env.PAYPAL_CLIENT_ID,
                    clientSecret: process.env.PAYPAL_CLIENT_SECRET,
                    environment: process.env.PAYPAL_ENVIRONMENT || 'sandbox'
                },
                database: this.dbManager,
                logger: this.logger
            });
            
            this.balanceSyncService = new BalanceSyncService({
                brokerService: this.brokerIntegrationService,
                database: this.dbManager,
                redis: this.redisManager,
                logger: this.logger
            });
            
            this.complianceService = new ComplianceService({
                database: this.dbManager,
                logger: this.logger
            });
            
            this.auditService = new AuditService({
                database: this.dbManager,
                logger: this.logger
            });
            
            this.logger.info('Payment service core services initialized successfully');
            
        } catch (error) {
            this.logger.error('Failed to initialize payment services:', error);
            throw error;
        }
    }
    
    private setupMiddleware(): void {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    scriptSrc: ["'self'"],
                    imgSrc: ["'self'", "data:", "https:"],
                },
            },
            hsts: {
                maxAge: 31536000,
                includeSubDomains: true,
                preload: true
            }
        }));
        
        // CORS configuration
        this.app.use(cors({
            origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
        }));
        
        // Rate limiting
        const limiter = rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 100, // Limit each IP to 100 requests per windowMs
            message: 'Too many requests from this IP, please try again later.',
            standardHeaders: true,
            legacyHeaders: false,
        });
        this.app.use('/api/', limiter);
        
        // Stricter rate limiting for sensitive endpoints
        const strictLimiter = rateLimit({
            windowMs: 15 * 60 * 1000,
            max: 10,
            message: 'Too many sensitive requests, please try again later.'
        });
        this.app.use('/api/v1/payments', strictLimiter);
        this.app.use('/api/v1/broker-accounts', strictLimiter);
        
        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
        
        // Compression
        this.app.use(compression());
        
        // Custom security middleware
        this.app.use(securityMiddleware);
        
        // Request logging
        this.app.use((req, res, next) => {
            this.logger.info(`${req.method} ${req.path}`, {
                ip: req.ip,
                userAgent: req.get('User-Agent'),
                timestamp: new Date().toISOString()
            });
            next();
        });
    }
    
    private setupRoutes(): void {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.status(200).json({
                status: 'healthy',
                service: 'payment-service',
                timestamp: new Date().toISOString(),
                version: process.env.npm_package_version || '1.0.0'
            });
        });
        
        // Readiness check
        this.app.get('/ready', async (req, res) => {
            try {
                // Check database connection
                await this.dbManager.healthCheck();
                
                // Check Redis connection
                await this.redisManager.healthCheck();
                
                res.status(200).json({
                    status: 'ready',
                    services: {
                        database: 'connected',
                        redis: 'connected',
                        brokerIntegration: 'ready',
                        paymentProcessor: 'ready'
                    },
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                res.status(503).json({
                    status: 'not ready',
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        // Metrics endpoint
        this.app.get('/metrics', async (req, res) => {
            try {
                const metrics = await this.getServiceMetrics();
                res.status(200).json(metrics);
            } catch (error) {
                res.status(500).json({ error: 'Failed to retrieve metrics' });
            }
        });
        
        // API routes with authentication
        this.app.use('/api/v1/broker-accounts', authMiddleware, new BrokerAccountRoutes({
            brokerService: this.brokerIntegrationService,
            auditService: this.auditService,
            logger: this.logger
        }).getRouter());
        
        this.app.use('/api/v1/payments', authMiddleware, new PaymentRoutes({
            paymentService: this.paymentProcessorService,
            complianceService: this.complianceService,
            auditService: this.auditService,
            logger: this.logger
        }).getRouter());
        
        this.app.use('/api/v1/transactions', authMiddleware, new TransactionRoutes({
            database: this.dbManager,
            auditService: this.auditService,
            logger: this.logger
        }).getRouter());
        
        this.app.use('/api/v1/balances', authMiddleware, new BalanceRoutes({
            balanceService: this.balanceSyncService,
            brokerService: this.brokerIntegrationService,
            logger: this.logger
        }).getRouter());
        
        this.app.use('/api/v1/compliance', authMiddleware, new ComplianceRoutes({
            complianceService: this.complianceService,
            auditService: this.auditService,
            logger: this.logger
        }).getRouter());
        
        // 404 handler
        this.app.use('*', (req, res) => {
            res.status(404).json({
                error: 'Endpoint not found',
                path: req.originalUrl,
                method: req.method,
                timestamp: new Date().toISOString()
            });
        });
    }
    
    private setupErrorHandling(): void {
        // Global error handler
        this.app.use(errorHandler);
        
        // Unhandled promise rejection handler
        process.on('unhandledRejection', (reason, promise) => {
            this.logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
        });
        
        // Uncaught exception handler
        process.on('uncaughtException', (error) => {
            this.logger.error('Uncaught Exception:', error);
            process.exit(1);
        });
        
        // Graceful shutdown
        process.on('SIGTERM', () => {
            this.logger.info('SIGTERM received, shutting down gracefully');
            this.shutdown();
        });
        
        process.on('SIGINT', () => {
            this.logger.info('SIGINT received, shutting down gracefully');
            this.shutdown();
        });
    }
    
    private async getServiceMetrics(): Promise<any> {
        try {
            const dbStats = await this.dbManager.getConnectionStats();
            const redisStats = await this.redisManager.getStats();
            
            return {
                service: 'payment-service',
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                database: dbStats,
                redis: redisStats,
                activeConnections: {
                    brokers: await this.brokerIntegrationService.getActiveConnectionCount(),
                    payments: await this.paymentProcessorService.getActiveTransactionCount()
                }
            };
        } catch (error) {
            throw new Error(`Failed to collect metrics: ${error.message}`);
        }
    }
    
    private async shutdown(): Promise<void> {
        try {
            this.logger.info('Starting graceful shutdown...');
            
            // Stop accepting new connections
            if (this.server) {
                this.server.close();
            }
            
            // Stop services
            await this.balanceSyncService.stop();
            await this.brokerIntegrationService.disconnect();
            
            // Close database connections
            await this.dbManager.disconnect();
            await this.redisManager.disconnect();
            
            this.logger.info('Graceful shutdown completed');
            process.exit(0);
        } catch (error) {
            this.logger.error('Error during shutdown:', error);
            process.exit(1);
        }
    }
    
    private server: any;
    
    public start(): void {
        const port = process.env.PORT || 3006;
        
        this.server = this.app.listen(port, () => {
            this.logger.info(`Payment service started on port ${port}`);
            this.logger.info('Environment:', process.env.NODE_ENV || 'development');
            
            // Start background services
            this.startBackgroundServices();
        });
        
        this.server.on('error', (error: any) => {
            if (error.syscall !== 'listen') {
                throw error;
            }
            
            const bind = typeof port === 'string' ? 'Pipe ' + port : 'Port ' + port;
            
            switch (error.code) {
                case 'EACCES':
                    this.logger.error(bind + ' requires elevated privileges');
                    process.exit(1);
                    break;
                case 'EADDRINUSE':
                    this.logger.error(bind + ' is already in use');
                    process.exit(1);
                    break;
                default:
                    throw error;
            }
        });
    }
    
    private async startBackgroundServices(): Promise<void> {
        try {
            // Start balance synchronization service
            await this.balanceSyncService.start();
            
            // Start compliance monitoring
            await this.complianceService.startMonitoring();
            
            this.logger.info('Background services started successfully');
        } catch (error) {
            this.logger.error('Failed to start background services:', error);
        }
    }
    
    public getApp(): express.Application {
        return this.app;
    }
}

// Start the server if this file is run directly
if (require.main === module) {
    const server = new PaymentServer();
    server.start();
}

export { PaymentServer };
