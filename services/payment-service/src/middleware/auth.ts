import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { serverConfig } from '@/config';
import { logger } from '@/utils/logger';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

export const authenticateToken = (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    logger.warn('Authentication failed: No token provided', { 
      ip: req.ip, 
      userAgent: req.get('User-Agent') 
    });
    res.status(401).json({ 
      error: 'Access denied. No token provided.',
      code: 'NO_TOKEN'
    });
    return;
  }

  try {
    const decoded = jwt.verify(token, serverConfig.jwtSecret) as any;
    req.user = {
      id: decoded.userId || decoded.id,
      email: decoded.email,
      role: decoded.role || 'user'
    };
    
    logger.debug('User authenticated successfully', { 
      userId: req.user.id, 
      email: req.user.email 
    });
    
    next();
  } catch (error) {
    logger.warn('Authentication failed: Invalid token', { 
      error: error instanceof Error ? error.message : 'Unknown error',
      ip: req.ip,
      userAgent: req.get('User-Agent')
    });
    
    if (error instanceof jwt.TokenExpiredError) {
      res.status(401).json({ 
        error: 'Token expired.',
        code: 'TOKEN_EXPIRED'
      });
    } else if (error instanceof jwt.JsonWebTokenError) {
      res.status(401).json({ 
        error: 'Invalid token.',
        code: 'INVALID_TOKEN'
      });
    } else {
      res.status(401).json({ 
        error: 'Token verification failed.',
        code: 'TOKEN_VERIFICATION_FAILED'
      });
    }
  }
};

export const requireRole = (roles: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ 
        error: 'Authentication required.',
        code: 'AUTHENTICATION_REQUIRED'
      });
      return;
    }

    if (!roles.includes(req.user.role)) {
      logger.warn('Authorization failed: Insufficient permissions', { 
        userId: req.user.id,
        userRole: req.user.role,
        requiredRoles: roles
      });
      
      res.status(403).json({ 
        error: 'Insufficient permissions.',
        code: 'INSUFFICIENT_PERMISSIONS'
      });
      return;
    }

    next();
  };
};

export const optionalAuth = (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    next();
    return;
  }

  try {
    const decoded = jwt.verify(token, serverConfig.jwtSecret) as any;
    req.user = {
      id: decoded.userId || decoded.id,
      email: decoded.email,
      role: decoded.role || 'user'
    };
  } catch (error) {
    // Ignore token errors for optional auth
    logger.debug('Optional auth: Invalid token ignored', { 
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }

  next();
};
