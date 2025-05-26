// Authentication Middleware for WebSocket connections

import { Socket } from 'socket.io';
import { Logger } from 'winston';
import jwt from 'jsonwebtoken';
import axios from 'axios';

export interface AuthenticatedSocket extends Socket {
  data: {
    userId: string;
    userEmail: string;
    isAuthenticated: boolean;
  };
}

export class AuthenticationMiddleware {
  private logger: Logger;
  private userServiceUrl: string;
  private jwtSecret: string;

  constructor(logger: Logger) {
    this.logger = logger;
    this.userServiceUrl = process.env.USER_SERVICE_URL || 'http://localhost:3002';
    this.jwtSecret = process.env.JWT_SECRET || 'default-secret';
  }

  async authenticate(socket: Socket, next: (err?: Error) => void): Promise<void> {
    try {
      // Extract token from handshake auth or query
      const token = this.extractToken(socket);
      
      if (!token) {
        this.logger.warn('WebSocket connection attempted without token');
        return next(new Error('Authentication token required'));
      }

      // Verify JWT token
      const decoded = await this.verifyToken(token);
      
      if (!decoded || !decoded.userId) {
        this.logger.warn('WebSocket connection attempted with invalid token');
        return next(new Error('Invalid authentication token'));
      }

      // Validate user exists and is active
      const user = await this.validateUser(decoded.userId, token);
      
      if (!user) {
        this.logger.warn(`WebSocket connection attempted for non-existent user: ${decoded.userId}`);
        return next(new Error('User not found or inactive'));
      }

      // Attach user data to socket
      socket.data = {
        userId: user.id,
        userEmail: user.email,
        isAuthenticated: true
      };

      this.logger.info(`WebSocket authentication successful for user: ${user.email}`);
      next();

    } catch (error) {
      this.logger.error('WebSocket authentication error:', error);
      next(new Error('Authentication failed'));
    }
  }

  async authenticateHTTP(req: any, res: any, next: any): Promise<void> {
    try {
      const authHeader = req.headers['authorization'];
      const token = authHeader && authHeader.split(' ')[1];

      if (!token) {
        return res.status(401).json({ error: 'Access token required' });
      }

      const decoded = await this.verifyToken(token);
      
      if (!decoded || !decoded.userId) {
        return res.status(401).json({ error: 'Invalid token' });
      }

      const user = await this.validateUser(decoded.userId, token);
      
      if (!user) {
        return res.status(401).json({ error: 'User not found' });
      }

      req.user = user;
      next();

    } catch (error) {
      this.logger.error('HTTP authentication error:', error);
      res.status(403).json({ error: 'Authentication failed' });
    }
  }

  private extractToken(socket: Socket): string | null {
    // Try to get token from handshake auth
    const authHeader = socket.handshake.auth?.token;
    if (authHeader) {
      return authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader;
    }

    // Try to get token from query parameters
    const queryToken = socket.handshake.query?.token;
    if (queryToken && typeof queryToken === 'string') {
      return queryToken;
    }

    // Try to get token from headers
    const headerToken = socket.handshake.headers?.authorization;
    if (headerToken && typeof headerToken === 'string') {
      return headerToken.startsWith('Bearer ') ? headerToken.slice(7) : headerToken;
    }

    return null;
  }

  private async verifyToken(token: string): Promise<any> {
    try {
      const decoded = jwt.verify(token, this.jwtSecret) as any;
      
      // Check token type and expiration
      if (decoded.type !== 'access') {
        throw new Error('Invalid token type');
      }

      return decoded;
    } catch (error) {
      if (error instanceof jwt.TokenExpiredError) {
        throw new Error('Token expired');
      } else if (error instanceof jwt.JsonWebTokenError) {
        throw new Error('Invalid token');
      }
      throw error;
    }
  }

  private async validateUser(userId: string, token: string): Promise<any> {
    try {
      // First try to validate with User Service
      const response = await axios.get(`${this.userServiceUrl}/api/v1/users/profile`, {
        headers: {
          'Authorization': `Bearer ${token}`
        },
        timeout: 5000
      });

      if (response.data && response.data.user) {
        return response.data.user;
      }

      throw new Error('User validation failed');

    } catch (error) {
      // If User Service is unavailable, try basic JWT validation
      this.logger.warn('User Service unavailable, using token-only validation');
      
      try {
        const decoded = jwt.verify(token, this.jwtSecret) as any;
        
        if (decoded.userId === userId && decoded.email) {
          return {
            id: decoded.userId,
            email: decoded.email,
            status: 'active' // Assume active if token is valid
          };
        }
      } catch (jwtError) {
        this.logger.error('Token validation failed:', jwtError);
      }

      return null;
    }
  }

  // Method to create a test token for development
  createTestToken(userId: string, email: string): string {
    return jwt.sign(
      {
        userId,
        email,
        type: 'access'
      },
      this.jwtSecret,
      { expiresIn: '1h' }
    );
  }

  // Method to validate session token
  async validateSessionToken(token: string): Promise<boolean> {
    try {
      const decoded = await this.verifyToken(token);
      return !!decoded;
    } catch (error) {
      return false;
    }
  }

  // Method to refresh token (if needed)
  async refreshToken(refreshToken: string): Promise<string | null> {
    try {
      const decoded = jwt.verify(refreshToken, this.jwtSecret) as any;
      
      if (decoded.type !== 'refresh') {
        throw new Error('Invalid refresh token type');
      }

      // Generate new access token
      const newAccessToken = jwt.sign(
        {
          userId: decoded.userId,
          email: decoded.email,
          type: 'access'
        },
        this.jwtSecret,
        { expiresIn: '1h' }
      );

      return newAccessToken;

    } catch (error) {
      this.logger.error('Token refresh failed:', error);
      return null;
    }
  }

  // Get user info from token without validation
  decodeToken(token: string): any {
    try {
      return jwt.decode(token);
    } catch (error) {
      return null;
    }
  }

  // Check if token is expired
  isTokenExpired(token: string): boolean {
    try {
      const decoded = jwt.decode(token) as any;
      if (!decoded || !decoded.exp) return true;
      
      return Date.now() >= decoded.exp * 1000;
    } catch (error) {
      return true;
    }
  }

  // Get token expiration time
  getTokenExpiration(token: string): number | null {
    try {
      const decoded = jwt.decode(token) as any;
      return decoded?.exp ? decoded.exp * 1000 : null;
    } catch (error) {
      return null;
    }
  }
}
