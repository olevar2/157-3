/**
 * Payment Routes
 * API endpoints for payment processing (Stripe, PayPal)
 */

import { Router, Request, Response } from 'express';
import { body, param, query, validationResult } from 'express-validator';

interface PaymentRoutesConfig {
    paymentService: any;
    complianceService: any;
    auditService: any;
    logger: any;
}

export class PaymentRoutes {
    private router: Router;
    private paymentService: any;
    private complianceService: any;
    private auditService: any;
    private logger: any;

    constructor(config: PaymentRoutesConfig) {
        this.router = Router();
        this.paymentService = config.paymentService;
        this.complianceService = config.complianceService;
        this.auditService = config.auditService;
        this.logger = config.logger;
        this.setupRoutes();
    }

    private setupRoutes(): void {
        // Process payment
        this.router.post('/process',
            [
                body('amount').isFloat({ min: 0.01 }),
                body('currency').isIn(['USD', 'EUR', 'GBP', 'JPY']),
                body('payment_method').isIn(['stripe', 'paypal']),
                body('description').optional().isString(),
                body('broker_account_id').optional().isUUID()
            ],
            this.processPayment.bind(this)
        );

        // Get payment status
        this.router.get('/:paymentId',
            param('paymentId').isUUID(),
            this.getPaymentStatus.bind(this)
        );

        // Get payment history
        this.router.get('/', this.getPaymentHistory.bind(this));

        // Refund payment
        this.router.post('/:paymentId/refund',
            [
                param('paymentId').isUUID(),
                body('amount').optional().isFloat({ min: 0.01 }),
                body('reason').notEmpty().isString()
            ],
            this.refundPayment.bind(this)
        );

        // Webhook endpoints for payment providers
        this.router.post('/webhooks/stripe', this.handleStripeWebhook.bind(this));
        this.router.post('/webhooks/paypal', this.handlePayPalWebhook.bind(this));

        // Create payment intent (for frontend)
        this.router.post('/create-intent',
            [
                body('amount').isFloat({ min: 0.01 }),
                body('currency').isIn(['USD', 'EUR', 'GBP', 'JPY']),
                body('payment_method').isIn(['stripe', 'paypal'])
            ],
            this.createPaymentIntent.bind(this)
        );
    }

    private async processPayment(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { amount, currency, payment_method, description, broker_account_id } = req.body;

            // Compliance check
            const complianceCheck = await this.complianceService.checkPayment({
                userId,
                amount,
                currency,
                paymentMethod: payment_method
            });

            if (!complianceCheck.approved) {
                return res.status(403).json({
                    success: false,
                    error: 'Payment not approved by compliance',
                    reason: complianceCheck.reason
                });
            }

            // Process payment
            const paymentResult = await this.paymentService.processPayment({
                userId,
                amount,
                currency,
                paymentMethod: payment_method,
                description,
                brokerAccountId: broker_account_id
            });

            // Log audit event
            await this.auditService.logEvent({
                userId,
                action: 'payment_processed',
                resourceType: 'payment',
                resourceId: paymentResult.paymentId,
                details: {
                    amount,
                    currency,
                    payment_method,
                    status: paymentResult.status
                }
            });

            res.json({
                success: true,
                data: {
                    payment_id: paymentResult.paymentId,
                    status: paymentResult.status,
                    transaction_id: paymentResult.transactionId,
                    amount,
                    currency,
                    created_at: paymentResult.createdAt
                },
                message: 'Payment processed successfully'
            });

        } catch (error) {
            this.logger.error('Error processing payment:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to process payment'
            });
        }
    }

    private async getPaymentStatus(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { paymentId } = req.params;

            const payment = await this.paymentService.getPayment(paymentId, userId);

            if (!payment) {
                return res.status(404).json({
                    success: false,
                    error: 'Payment not found'
                });
            }

            res.json({
                success: true,
                data: payment,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching payment status:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch payment status'
            });
        }
    }

    private async getPaymentHistory(req: Request, res: Response): Promise<void> {
        try {
            const userId = req.user?.id;
            const { 
                page = 1, 
                limit = 50, 
                status, 
                payment_method, 
                start_date, 
                end_date 
            } = req.query;

            const filters = {
                status: status as string,
                paymentMethod: payment_method as string,
                startDate: start_date as string,
                endDate: end_date as string
            };

            const payments = await this.paymentService.getPaymentHistory(userId, {
                page: Number(page),
                limit: Number(limit),
                filters
            });

            res.json({
                success: true,
                data: payments.data,
                pagination: {
                    page: Number(page),
                    limit: Number(limit),
                    total: payments.total,
                    pages: Math.ceil(payments.total / Number(limit))
                },
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.logger.error('Error fetching payment history:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to fetch payment history'
            });
        }
    }

    private async refundPayment(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { paymentId } = req.params;
            const { amount, reason } = req.body;

            // Check if payment exists and belongs to user
            const payment = await this.paymentService.getPayment(paymentId, userId);
            if (!payment) {
                return res.status(404).json({
                    success: false,
                    error: 'Payment not found'
                });
            }

            // Process refund
            const refundResult = await this.paymentService.refundPayment({
                paymentId,
                amount: amount || payment.amount,
                reason
            });

            // Log audit event
            await this.auditService.logEvent({
                userId,
                action: 'payment_refunded',
                resourceType: 'payment',
                resourceId: paymentId,
                details: {
                    refund_amount: refundResult.amount,
                    reason,
                    refund_id: refundResult.refundId
                }
            });

            res.json({
                success: true,
                data: {
                    refund_id: refundResult.refundId,
                    amount: refundResult.amount,
                    status: refundResult.status,
                    created_at: refundResult.createdAt
                },
                message: 'Refund processed successfully'
            });

        } catch (error) {
            this.logger.error('Error processing refund:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to process refund'
            });
        }
    }

    private async createPaymentIntent(req: Request, res: Response): Promise<void> {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    errors: errors.array()
                });
            }

            const userId = req.user?.id;
            const { amount, currency, payment_method } = req.body;

            const intent = await this.paymentService.createPaymentIntent({
                userId,
                amount,
                currency,
                paymentMethod: payment_method
            });

            res.json({
                success: true,
                data: {
                    client_secret: intent.clientSecret,
                    payment_intent_id: intent.id,
                    amount,
                    currency
                },
                message: 'Payment intent created successfully'
            });

        } catch (error) {
            this.logger.error('Error creating payment intent:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to create payment intent'
            });
        }
    }

    private async handleStripeWebhook(req: Request, res: Response): Promise<void> {
        try {
            const signature = req.headers['stripe-signature'] as string;
            const event = await this.paymentService.verifyStripeWebhook(req.body, signature);

            await this.paymentService.handleStripeEvent(event);

            res.json({ received: true });

        } catch (error) {
            this.logger.error('Error handling Stripe webhook:', error);
            res.status(400).json({
                success: false,
                error: 'Webhook verification failed'
            });
        }
    }

    private async handlePayPalWebhook(req: Request, res: Response): Promise<void> {
        try {
            const headers = req.headers;
            const event = await this.paymentService.verifyPayPalWebhook(req.body, headers);

            await this.paymentService.handlePayPalEvent(event);

            res.json({ received: true });

        } catch (error) {
            this.logger.error('Error handling PayPal webhook:', error);
            res.status(400).json({
                success: false,
                error: 'Webhook verification failed'
            });
        }
    }

    public getRouter(): Router {
        return this.router;
    }
}
