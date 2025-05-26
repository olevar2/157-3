/**
 * Transaction Routes
 * API endpoints for transaction history and management
 */

import { Router, Request, Response } from 'express';
import { param, query, validationResult } from 'express-validator';

interface TransactionRoutesConfig {
    database: any;
    auditService: any;
    logger: any;
}

export class TransactionRoutes {
    private router: Router;
    private database: any;
    private auditService: any;
    private logger: any;

    constructor(config: TransactionRoutesConfig) {
        this.router = Router();
        this.database = config.database;
        this.auditService = config.auditService;
        this.logger = config.logger;
        this.setupRoutes();
    }

    private setupRoutes(): void {
        // Get transaction history
        this.router.get('/', this.getTransactions.bind(this));

        // Get specific transaction
        this.router.get('/:transactionId',
            param('transactionId').isUUID(),
            this.getTransaction.bind(this)
        );

        // Get transaction summary/statistics
        this.router.get('/summary/stats', this.getTransactionSummary.bind(this));

        // Export transactions
        this.router.get('/export/:format',
            param('format').isIn(['csv', 'pdf', 'excel']),
            this.exportTransactions.bind(this)
        );
    }

    private async getTransactions(req: Request, res: Response): Promise<void> {
        try {
            const userId = req.user?.id;
            const {
                page = 1,
                limit = 50,
                type,
                status,
                start_date,
                end_date,
                broker_account_id,
                min_amount,
                max_amount,
                currency
            } = req.query;

            // Build query conditions
            let whereConditions = ['user_id = $1'];
            let queryParams: any[] = [userId];
            let paramIndex = 2;

            if (type) {
                whereConditions.push(`transaction_type = $${paramIndex}`);
                queryParams.push(type);
                paramIndex++;
            }

            if (status) {
                whereConditions.push(`status = $${paramIndex}`);
                queryParams.push(status);
                paramIndex++;
            }

            if (start_date) {
                whereConditions.push(`created_at >= $${paramIndex}`);
                queryParams.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                whereConditions.push(`created_at <= $${paramIndex}`);
                queryParams.push(end_date);
                paramIndex++;
            }

            if (broker_account_id) {
                whereConditions.push(`broker_account_id = $${paramIndex}`);
                queryParams.push(broker_account_id);
                paramIndex++;
            }

            if (min_amount) {
                whereConditions.push(`amount >= $${paramIndex}`);
                queryParams.push(min_amount);
                paramIndex++;
            }

            if (max_amount) {
                whereConditions.push(`amount <= $${paramIndex}`);
                queryParams.push(max_amount);
                paramIndex++;
            }

            if (currency) {
                whereConditions.push(`currency = $${paramIndex}`);
                queryParams.push(currency);
                paramIndex++;
            }

            // Count total records
            const countQuery = `
                SELECT COUNT(*) as total 
                FROM transactions 
                WHERE ${whereConditions.join(' AND ')}
            `;
            const countResult = await this.database.query(countQuery, queryParams);
            const total = parseInt(countResult.rows[0].total);

            // Get paginated results
            const offset = (Number(page) - 1) * Number(limit);
            const dataQuery = `
                SELECT 
                    t.*,
                    ba.broker_name,
                    ba.account_number
                FROM transactions t
                LEFT JOIN broker_accounts ba ON t.broker_account_id = ba.id
                WHERE ${whereConditions.join(' AND ')}
                ORDER BY t.created_at DESC
                LIMIT $${paramIndex} OFFSET $${paramIndex + 1}
            `;
            queryParams.push(Number(limit), offset);

            const result = await this.database.query(dataQuery, queryParams);

            res.json({
                success: true,
                data: result.rows,
                pagination: {
                    page: Number(page),
                    limit: Number(limit),
                    total,
                    pages: Math.ceil(total / Number(limit))
                },
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching transactions:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch transactions'
            });
        }
    }

    private async getTransaction(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { transactionId } = req.params;

            const query = `
                SELECT 
                    t.*,
                    ba.broker_name,
                    ba.account_number,
                    p.payment_method,
                    p.transaction_id as payment_transaction_id
                FROM transactions t
                LEFT JOIN broker_accounts ba ON t.broker_account_id = ba.id
                LEFT JOIN payments p ON t.payment_id = p.id
                WHERE t.id = $1 AND t.user_id = $2
            `;

            const result = await this.database.query(query, [transactionId, userId]);

            if (result.rows.length === 0) {
                return res.status(404).json({
                    success: false,
                    error: 'Transaction not found'
                });
            }

            res.json({
                success: true,
                data: result.rows[0],
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching transaction:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch transaction'
            });
        }
    }

    private async getTransactionSummary(req: Request, res: Response): Promise<void> {
        try {
            const userId = req.user?.id;
            const { period = '30d' } = req.query;

            // Calculate date range based on period
            let startDate: Date;
            const endDate = new Date();

            switch (period) {
                case '7d':
                    startDate = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
                    break;
                case '30d':
                    startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
                    break;
                case '90d':
                    startDate = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);
                    break;
                case '1y':
                    startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000);
                    break;
                default:
                    startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
            }

            // Get summary statistics
            const summaryQuery = `
                SELECT 
                    COUNT(*) as total_transactions,
                    COUNT(CASE WHEN transaction_type = 'deposit' THEN 1 END) as total_deposits,
                    COUNT(CASE WHEN transaction_type = 'withdrawal' THEN 1 END) as total_withdrawals,
                    COALESCE(SUM(CASE WHEN transaction_type = 'deposit' AND status = 'completed' THEN amount ELSE 0 END), 0) as total_deposited,
                    COALESCE(SUM(CASE WHEN transaction_type = 'withdrawal' AND status = 'completed' THEN amount ELSE 0 END), 0) as total_withdrawn,
                    COALESCE(AVG(CASE WHEN transaction_type = 'deposit' AND status = 'completed' THEN amount END), 0) as avg_deposit,
                    COALESCE(AVG(CASE WHEN transaction_type = 'withdrawal' AND status = 'completed' THEN amount END), 0) as avg_withdrawal
                FROM transactions 
                WHERE user_id = $1 
                AND created_at >= $2 
                AND created_at <= $3
            `;

            const summaryResult = await this.database.query(summaryQuery, [userId, startDate, endDate]);

            // Get transaction trends (daily aggregation)
            const trendsQuery = `
                SELECT 
                    DATE(created_at) as date,
                    transaction_type,
                    COUNT(*) as count,
                    COALESCE(SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END), 0) as total_amount
                FROM transactions 
                WHERE user_id = $1 
                AND created_at >= $2 
                AND created_at <= $3
                GROUP BY DATE(created_at), transaction_type
                ORDER BY date DESC
            `;

            const trendsResult = await this.database.query(trendsQuery, [userId, startDate, endDate]);

            // Get status breakdown
            const statusQuery = `
                SELECT 
                    status,
                    COUNT(*) as count,
                    COALESCE(SUM(amount), 0) as total_amount
                FROM transactions 
                WHERE user_id = $1 
                AND created_at >= $2 
                AND created_at <= $3
                GROUP BY status
            `;

            const statusResult = await this.database.query(statusQuery, [userId, startDate, endDate]);

            res.json({
                success: true,
                data: {
                    summary: summaryResult.rows[0],
                    trends: trendsResult.rows,
                    status_breakdown: statusResult.rows,
                    period,
                    date_range: {
                        start: startDate.toISOString(),
                        end: endDate.toISOString()
                    }
                },
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching transaction summary:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch transaction summary'
            });
        }
    }

    private async exportTransactions(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { format } = req.params;
            const { start_date, end_date, type } = req.query;

            // Build query for export
            let whereConditions = ['user_id = $1'];
            let queryParams: any[] = [userId];
            let paramIndex = 2;

            if (start_date) {
                whereConditions.push(`created_at >= $${paramIndex}`);
                queryParams.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                whereConditions.push(`created_at <= $${paramIndex}`);
                queryParams.push(end_date);
                paramIndex++;
            }

            if (type) {
                whereConditions.push(`transaction_type = $${paramIndex}`);
                queryParams.push(type);
                paramIndex++;
            }

            const query = `
                SELECT 
                    t.id,
                    t.transaction_type,
                    t.amount,
                    t.currency,
                    t.status,
                    t.description,
                    t.created_at,
                    t.updated_at,
                    ba.broker_name,
                    ba.account_number
                FROM transactions t
                LEFT JOIN broker_accounts ba ON t.broker_account_id = ba.id
                WHERE ${whereConditions.join(' AND ')}
                ORDER BY t.created_at DESC
            `;

            const result = await this.database.query(query, queryParams);

            // Log audit event
            await this.auditService.logEvent({
                userId,
                action: 'transactions_exported',
                resourceType: 'transaction',
                details: {
                    format,
                    record_count: result.rows.length,
                    filters: { start_date, end_date, type }
                }
            });

            // For now, return JSON data
            // In a real implementation, you would generate CSV/PDF/Excel files
            if (format === 'csv') {
                res.setHeader('Content-Type', 'text/csv');
                res.setHeader('Content-Disposition', 'attachment; filename=transactions.csv');
                
                // Simple CSV generation
                const csvHeader = 'ID,Type,Amount,Currency,Status,Description,Created At,Broker,Account\n';
                const csvData = result.rows.map(row => 
                    `${row.id},${row.transaction_type},${row.amount},${row.currency},${row.status},"${row.description}",${row.created_at},${row.broker_name || ''},${row.account_number || ''}`
                ).join('\n');
                
                res.send(csvHeader + csvData);
            } else {
                res.json({
                    success: true,
                    data: result.rows,
                    format,
                    exported_at: new Date().toISOString(),
                    record_count: result.rows.length
                });
            }

        } catch (error) {
            this.logger.error('Error exporting transactions:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to export transactions'
            });
        }
    }

    public getRouter(): Router {
        return this.router;
    }
}
