/**
 * Broker Account Routes
 * API endpoints for managing broker account integrations
 */

import { Router, Request, Response } from 'express';
import { body, param, query, validationResult } from 'express-validator';

interface BrokerAccountRoutesConfig {
    brokerService: any;
    auditService: any;
    logger: any;
}

export class BrokerAccountRoutes {
    private router: Router;
    private brokerService: any;
    private auditService: any;
    private logger: any;

    constructor(config: BrokerAccountRoutesConfig) {
        this.router = Router();
        this.brokerService = config.brokerService;
        this.auditService = config.auditService;
        this.logger = config.logger;
        this.setupRoutes();
    }

    private setupRoutes(): void {
        // Get all broker accounts for user
        this.router.get('/', this.getBrokerAccounts.bind(this));

        // Get specific broker account
        this.router.get('/:accountId', 
            param('accountId').isUUID(),
            this.getBrokerAccount.bind(this)
        );

        // Add new broker account
        this.router.post('/',
            [
                body('broker_name').notEmpty().isString(),
                body('account_number').notEmpty().isString(),
                body('api_key').notEmpty().isString(),
                body('api_secret').optional().isString(),
                body('account_type').isIn(['demo', 'live']),
                body('server_url').optional().isURL()
            ],
            this.addBrokerAccount.bind(this)
        );

        // Update broker account
        this.router.put('/:accountId',
            [
                param('accountId').isUUID(),
                body('broker_name').optional().isString(),
                body('account_number').optional().isString(),
                body('api_key').optional().isString(),
                body('api_secret').optional().isString(),
                body('account_type').optional().isIn(['demo', 'live']),
                body('server_url').optional().isURL(),
                body('status').optional().isIn(['active', 'inactive', 'suspended'])
            ],
            this.updateBrokerAccount.bind(this)
        );

        // Delete broker account
        this.router.delete('/:accountId',
            param('accountId').isUUID(),
            this.deleteBrokerAccount.bind(this)
        );

        // Test broker connection
        this.router.post('/:accountId/test-connection',
            param('accountId').isUUID(),
            this.testConnection.bind(this)
        );

        // Sync account balance
        this.router.post('/:accountId/sync-balance',
            param('accountId').isUUID(),
            this.syncBalance.bind(this)
        );
    }

    private async getBrokerAccounts(req: Request, res: Response): Promise<void> {
        try {
            const userId = req.user?.id;
            const { status, broker_name } = req.query;

            const accounts = await this.brokerService.getUserAccounts(userId, {
                status: status as string,
                brokerName: broker_name as string
            });

            res.json({
                success: true,
                data: accounts,
                count: accounts.length,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching broker accounts:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch broker accounts'
            });
        }
    }

    private async getBrokerAccount(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { accountId } = req.params;

            const account = await this.brokerService.getAccount(accountId, userId);

            if (!account) {
                return res.status(404).json({
                    success: false,
                    error: 'Broker account not found'
                });
            }

            res.json({
                success: true,
                data: account,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching broker account:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch broker account'
            });
        }
    }

    private async addBrokerAccount(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const accountData = {
                ...req.body,
                user_id: userId
            };

            // Test connection before saving
            const connectionTest = await this.brokerService.testConnection(accountData);
            if (!connectionTest.success) {
                return res.status(400).json({
                    success: false,
                    error: 'Failed to connect to broker',
                    details: connectionTest.error
                });
            }

            const account = await this.brokerService.createAccount(accountData);

            // Log audit event
            await this.auditService.logEvent({
                userId,
                action: 'broker_account_created',
                resourceType: 'broker_account',
                resourceId: account.id,
                details: {
                    broker_name: account.broker_name,
                    account_type: account.account_type
                }
            });

            res.status(201).json({
                success: true,
                data: account,
                message: 'Broker account added successfully'
            });

        } catch (error) {
            this.logger.error('Error adding broker account:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to add broker account'
            });
        }
    }

    private async updateBrokerAccount(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { accountId } = req.params;
            const updateData = req.body;

            const account = await this.brokerService.updateAccount(accountId, userId, updateData);

            if (!account) {
                return res.status(404).json({
                    success: false,
                    error: 'Broker account not found'
                });
            }

            // Log audit event
            await this.auditService.logEvent({
                userId,
                action: 'broker_account_updated',
                resourceType: 'broker_account',
                resourceId: accountId,
                details: updateData
            });

            res.json({
                success: true,
                data: account,
                message: 'Broker account updated successfully'
            });

        } catch (error) {
            this.logger.error('Error updating broker account:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to update broker account'
            });
        }
    }

    private async deleteBrokerAccount(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { accountId } = req.params;

            const deleted = await this.brokerService.deleteAccount(accountId, userId);

            if (!deleted) {
                return res.status(404).json({
                    success: false,
                    error: 'Broker account not found'
                });
            }

            // Log audit event
            await this.auditService.logEvent({
                userId,
                action: 'broker_account_deleted',
                resourceType: 'broker_account',
                resourceId: accountId
            });

            res.json({
                success: true,
                message: 'Broker account deleted successfully'
            });

        } catch (error) {
            this.logger.error('Error deleting broker account:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to delete broker account'
            });
        }
    }

    private async testConnection(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { accountId } = req.params;

            const account = await this.brokerService.getAccount(accountId, userId);
            if (!account) {
                return res.status(404).json({
                    success: false,
                    error: 'Broker account not found'
                });
            }

            const connectionTest = await this.brokerService.testConnection(account);

            res.json({
                success: true,
                data: {
                    connected: connectionTest.success,
                    latency: connectionTest.latency,
                    server_time: connectionTest.serverTime,
                    account_info: connectionTest.accountInfo
                },
                message: connectionTest.success ? 'Connection successful' : 'Connection failed',
                error: connectionTest.error
            });

        } catch (error) {
            this.logger.error('Error testing broker connection:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to test broker connection'
            });
        }
    }

    private async syncBalance(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { accountId } = req.params;

            const account = await this.brokerService.getAccount(accountId, userId);
            if (!account) {
                return res.status(404).json({
                    success: false,
                    error: 'Broker account not found'
                });
            }

            const balanceSync = await this.brokerService.syncAccountBalance(accountId);

            res.json({
                success: true,
                data: {
                    account_id: accountId,
                    balance: balanceSync.balance,
                    equity: balanceSync.equity,
                    margin: balanceSync.margin,
                    free_margin: balanceSync.freeMargin,
                    last_updated: balanceSync.lastUpdated
                },
                message: 'Balance synchronized successfully'
            });

        } catch (error) {
            this.logger.error('Error syncing account balance:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to sync account balance'
            });
        }
    }

    public getRouter(): Router {
        return this.router;
    }
}
