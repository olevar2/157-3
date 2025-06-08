from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
const jwt = require('jsonwebtoken');

// Middleware to authenticate JWT tokens
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({
      success: false,
      message: 'Access token required'
    });
  }

  jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key', (err, user) => {
    if (err) {
      return res.status(403).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }
    req.user = user;
    next();
  });
};

// Middleware to validate trade data
const validateTrade = (req, res, next) => {
  const { pair_id, type, amount, price } = req.body;

  // Required field validation
  if (!pair_id || !type || !amount || !price) {
    return res.status(400).json({
      success: false,
      message: 'pair_id, type, amount, and price are required'
    });
  }

  // Type validation
  if (!['buy', 'sell'].includes(type)) {
    return res.status(400).json({
      success: false,
      message: 'type must be either "buy" or "sell"'
    });
  }

  // Numeric validation
  const numAmount = parseFloat(amount);
  const numPrice = parseFloat(price);

  if (isNaN(numAmount) || numAmount <= 0) {
    return res.status(400).json({
      success: false,
      message: 'amount must be a positive number'
    });
  }

  if (isNaN(numPrice) || numPrice <= 0) {
    return res.status(400).json({
      success: false,
      message: 'price must be a positive number'
    });
  }

  // Pair ID validation (should be an integer)
  const numPairId = parseInt(pair_id);
  if (isNaN(numPairId) || numPairId <= 0) {
    return res.status(400).json({
      success: false,
      message: 'pair_id must be a valid positive integer'
    });
  }

  next();
};

// Middleware to validate portfolio update data
const validatePortfolioUpdate = (req, res, next) => {
  const { asset_symbol, amount, operation } = req.body;

  if (!asset_symbol || amount === undefined || !operation) {
    return res.status(400).json({
      success: false,
      message: 'asset_symbol, amount, and operation are required'
    });
  }

  if (!['add', 'subtract'].includes(operation)) {
    return res.status(400).json({
      success: false,
      message: 'operation must be "add" or "subtract"'
    });
  }

  const numAmount = parseFloat(amount);
  if (isNaN(numAmount) || numAmount <= 0) {
    return res.status(400).json({
      success: false,
      message: 'amount must be a positive number'
    });
  }

  next();
};

// Middleware for rate limiting (basic implementation)
const rateLimitMap = new Map();

const rateLimit = (windowMs = 60000, maxRequests = 100) => {
  return (req, res, next) => {
    const clientId = req.ip || req.connection.remoteAddress;
    const now = Date.now();
    const windowStart = now - windowMs;

    if (!rateLimitMap.has(clientId)) {
      rateLimitMap.set(clientId, []);
    }

    const requests = rateLimitMap.get(clientId);
    
    // Remove old requests outside the window
    const validRequests = requests.filter(timestamp => timestamp > windowStart);
    
    if (validRequests.length >= maxRequests) {
      return res.status(429).json({
        success: false,
        message: 'Too many requests, please try again later'
      });
    }

    validRequests.push(now);
    rateLimitMap.set(clientId, validRequests);
    
    next();
  };
};

// Middleware to log requests
const requestLogger = (req, res, next) => {
  const timestamp = new Date().toISOString();
  const method = req.method;
  const url = req.url;
  const userAgent = req.get('User-Agent') || 'Unknown';
  
  console.log(`[${timestamp}] ${method} ${url} - ${userAgent}`);
  next();
};

// Error handling middleware
const errorHandler = (err, req, res, next) => {
  console.error('Error:', err);

  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    return res.status(401).json({
      success: false,
      message: 'Invalid token'
    });
  }

  if (err.name === 'TokenExpiredError') {
    return res.status(401).json({
      success: false,
      message: 'Token expired'
    });
  }

  // Database errors
  if (err.code === '23505') { // PostgreSQL unique violation
    return res.status(409).json({
      success: false,
      message: 'Resource already exists'
    });
  }

  if (err.code === '23503') { // PostgreSQL foreign key violation
    return res.status(400).json({
      success: false,
      message: 'Invalid reference to related resource'
    });
  }

  // Default error
  res.status(500).json({
    success: false,
    message: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : err.message
  });
};

module.exports = {
  authenticateToken,
  validateTrade,
  validatePortfolioUpdate,
  rateLimit,
  requestLogger,
  errorHandler
};
