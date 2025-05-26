import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';
import { logger } from '@/utils/logger';

export const validateRequest = (schema: {
  body?: Joi.ObjectSchema;
  query?: Joi.ObjectSchema;
  params?: Joi.ObjectSchema;
}) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const errors: string[] = [];

    // Validate request body
    if (schema.body) {
      const { error } = schema.body.validate(req.body);
      if (error) {
        errors.push(`Body: ${error.details.map(d => d.message).join(', ')}`);
      }
    }

    // Validate query parameters
    if (schema.query) {
      const { error } = schema.query.validate(req.query);
      if (error) {
        errors.push(`Query: ${error.details.map(d => d.message).join(', ')}`);
      }
    }

    // Validate path parameters
    if (schema.params) {
      const { error } = schema.params.validate(req.params);
      if (error) {
        errors.push(`Params: ${error.details.map(d => d.message).join(', ')}`);
      }
    }

    if (errors.length > 0) {
      logger.warn('Request validation failed', { 
        errors, 
        path: req.path, 
        method: req.method 
      });
      
      res.status(400).json({
        error: 'Validation failed',
        details: errors,
        code: 'VALIDATION_ERROR'
      });
      return;
    }

    next();
  };
};

// Common validation schemas
export const depositSchema = {
  body: Joi.object({
    amount: Joi.number().positive().min(1).max(1000000).required()
      .messages({
        'number.positive': 'Amount must be positive',
        'number.min': 'Amount must be at least $1',
        'number.max': 'Amount cannot exceed $1,000,000'
      }),
    currency: Joi.string().valid('USD', 'EUR', 'GBP', 'JPY').default('USD'),
    paymentMethodId: Joi.string().uuid().optional(),
    gateway: Joi.string().valid('stripe', 'paypal').required(),
    metadata: Joi.object().optional()
  })
};

export const withdrawalSchema = {
  body: Joi.object({
    amount: Joi.number().positive().min(1).max(1000000).required()
      .messages({
        'number.positive': 'Amount must be positive',
        'number.min': 'Amount must be at least $1',
        'number.max': 'Amount cannot exceed $1,000,000'
      }),
    currency: Joi.string().valid('USD', 'EUR', 'GBP', 'JPY').default('USD'),
    destinationType: Joi.string().valid('bank_account', 'paypal').required(),
    destinationId: Joi.string().required(),
    metadata: Joi.object().optional()
  })
};

export const paymentMethodSchema = {
  body: Joi.object({
    type: Joi.string().valid('credit_card', 'debit_card', 'bank_account', 'paypal').required(),
    gateway: Joi.string().valid('stripe', 'paypal').required(),
    gatewayMethodId: Joi.string().required(),
    isDefault: Joi.boolean().default(false),
    metadata: Joi.object().optional()
  })
};

export const transactionQuerySchema = {
  query: Joi.object({
    page: Joi.number().integer().min(1).default(1),
    limit: Joi.number().integer().min(1).max(100).default(20),
    type: Joi.string().valid('deposit', 'withdrawal').optional(),
    status: Joi.string().valid('pending', 'processing', 'completed', 'failed', 'cancelled').optional(),
    gateway: Joi.string().valid('stripe', 'paypal', 'bank_transfer').optional(),
    startDate: Joi.date().iso().optional(),
    endDate: Joi.date().iso().min(Joi.ref('startDate')).optional(),
    sortBy: Joi.string().valid('createdAt', 'amount', 'status').default('createdAt'),
    sortOrder: Joi.string().valid('asc', 'desc').default('desc')
  })
};

export const uuidParamSchema = {
  params: Joi.object({
    id: Joi.string().uuid().required()
  })
};
