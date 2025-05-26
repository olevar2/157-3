import { Request, Response, NextFunction } from 'express';
import { logger } from '@/utils/logger';
import { serverConfig } from '@/config';

export interface PaymentError extends Error {
  statusCode?: number;
  code?: string;
  details?: any;
  isOperational?: boolean;
}

// Custom error classes
export class ValidationError extends Error implements PaymentError {
  statusCode = 400;
  code = 'VALIDATION_ERROR';
  isOperational = true;

  constructor(message: string, public details?: any) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class AuthenticationError extends Error implements PaymentError {
  statusCode = 401;
  code = 'AUTHENTICATION_ERROR';
  isOperational = true;

  constructor(message: string = 'Authentication failed') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class AuthorizationError extends Error implements PaymentError {
  statusCode = 403;
  code = 'AUTHORIZATION_ERROR';
  isOperational = true;

  constructor(message: string = 'Insufficient permissions') {
    super(message);
    this.name = 'AuthorizationError';
  }
}

export class NotFoundError extends Error implements PaymentError {
  statusCode = 404;
  code = 'NOT_FOUND';
  isOperational = true;

  constructor(message: string = 'Resource not found') {
    super(message);
    this.name = 'NotFoundError';
  }
}

export class PaymentProcessingError extends Error implements PaymentError {
  statusCode = 422;
  code = 'PAYMENT_PROCESSING_ERROR';
  isOperational = true;

  constructor(message: string, public details?: any) {
    super(message);
    this.name = 'PaymentProcessingError';
  }
}

export class InsufficientFundsError extends Error implements PaymentError {
  statusCode = 422;
  code = 'INSUFFICIENT_FUNDS';
  isOperational = true;

  constructor(message: string = 'Insufficient funds for this transaction') {
    super(message);
    this.name = 'InsufficientFundsError';
  }
}

export class RateLimitError extends Error implements PaymentError {
  statusCode = 429;
  code = 'RATE_LIMIT_EXCEEDED';
  isOperational = true;

  constructor(message: string = 'Rate limit exceeded') {
    super(message);
    this.name = 'RateLimitError';
  }
}

export class ExternalServiceError extends Error implements PaymentError {
  statusCode = 502;
  code = 'EXTERNAL_SERVICE_ERROR';
  isOperational = true;

  constructor(message: string, public service?: string, public details?: any) {
    super(message);
    this.name = 'ExternalServiceError';
  }
}

// Main error handling middleware
export const errorHandler = (
  err: PaymentError,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  // Log error details
  const errorInfo = {
    name: err.name,
    message: err.message,
    statusCode: err.statusCode,
    code: err.code,
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    userId: (req as any).user?.id,
    timestamp: new Date().toISOString()
  };

  if (err.statusCode && err.statusCode >= 500) {
    logger.error('Server error occurred', errorInfo);
  } else {
    logger.warn('Client error occurred', errorInfo);
  }

  // Handle specific error types
  let statusCode = err.statusCode || 500;
  let errorCode = err.code || 'INTERNAL_SERVER_ERROR';
  let message = err.message || 'An unexpected error occurred';
  let details = err.details;

  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    statusCode = 401;
    errorCode = 'INVALID_TOKEN';
    message = 'Invalid authentication token';
  } else if (err.name === 'TokenExpiredError') {
    statusCode = 401;
    errorCode = 'TOKEN_EXPIRED';
    message = 'Authentication token has expired';
  }

  // Database errors
  else if (err.name === 'SequelizeValidationError' || err.name === 'ValidationError') {
    statusCode = 400;
    errorCode = 'VALIDATION_ERROR';
    message = 'Validation failed';
  } else if ((err as any).code === '23505') { // PostgreSQL unique violation
    statusCode = 409;
    errorCode = 'DUPLICATE_RESOURCE';
    message = 'Resource already exists';
  } else if ((err as any).code === '23503') { // PostgreSQL foreign key violation
    statusCode = 400;
    errorCode = 'INVALID_REFERENCE';
    message = 'Invalid reference to related resource';
  } else if ((err as any).code === '23502') { // PostgreSQL not null violation
    statusCode = 400;
    errorCode = 'MISSING_REQUIRED_FIELD';
    message = 'Required field is missing';
  }

  // Stripe errors
  else if (err.name === 'StripeCardError') {
    statusCode = 422;
    errorCode = 'CARD_ERROR';
    message = 'Card was declined';
  } else if (err.name === 'StripeInvalidRequestError') {
    statusCode = 400;
    errorCode = 'INVALID_REQUEST';
    message = 'Invalid request to payment processor';
  } else if (err.name === 'StripeAPIError') {
    statusCode = 502;
    errorCode = 'PAYMENT_GATEWAY_ERROR';
    message = 'Payment gateway error';
  } else if (err.name === 'StripeConnectionError') {
    statusCode = 503;
    errorCode = 'PAYMENT_GATEWAY_UNAVAILABLE';
    message = 'Payment gateway temporarily unavailable';
  }

  // CORS errors
  else if (err.message === 'Not allowed by CORS') {
    statusCode = 403;
    errorCode = 'CORS_ERROR';
    message = 'CORS policy violation';
  }

  // Rate limiting errors
  else if (err.name === 'TooManyRequestsError' || statusCode === 429) {
    statusCode = 429;
    errorCode = 'RATE_LIMIT_EXCEEDED';
    message = 'Too many requests, please try again later';
  }

  // Prepare error response
  const errorResponse: any = {
    error: message,
    code: errorCode,
    timestamp: new Date().toISOString()
  };

  // Add details in development mode or for operational errors
  if (serverConfig.nodeEnv !== 'production' || err.isOperational) {
    if (details) {
      errorResponse.details = details;
    }
    
    // Add stack trace in development
    if (serverConfig.nodeEnv === 'development') {
      errorResponse.stack = err.stack;
    }
  }

  // Add request ID for tracking
  if (req.headers['x-request-id']) {
    errorResponse.requestId = req.headers['x-request-id'];
  }

  res.status(statusCode).json(errorResponse);
};

// 404 handler for unknown routes
export const notFoundHandler = (req: Request, res: Response): void => {
  logger.warn('Route not found', {
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  res.status(404).json({
    error: 'Endpoint not found',
    code: 'NOT_FOUND',
    path: req.path,
    method: req.method,
    availableEndpoints: [
      'GET /health',
      'GET /api/v1/payments/balance',
      'GET /api/v1/payments/history',
      'GET /api/v1/payments/methods',
      'POST /api/v1/payments/deposit',
      'POST /api/v1/payments/withdraw',
      'POST /api/v1/payments/methods',
      'DELETE /api/v1/payments/methods/:id'
    ],
    timestamp: new Date().toISOString()
  });
};

// Async error wrapper for route handlers
export const asyncHandler = (fn: Function) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// Global unhandled error handlers
export const setupGlobalErrorHandlers = (): void => {
  process.on('uncaughtException', (error: Error) => {
    logger.error('Uncaught Exception', {
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    });
    
    // Graceful shutdown
    process.exit(1);
  });

  process.on('unhandledRejection', (reason: any, promise: Promise<any>) => {
    logger.error('Unhandled Rejection', {
      reason: reason?.message || reason,
      stack: reason?.stack,
      promise: promise.toString(),
      timestamp: new Date().toISOString()
    });
  });
};
