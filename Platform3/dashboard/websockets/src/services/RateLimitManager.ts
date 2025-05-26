// Rate Limit Manager - Prevents abuse and manages connection limits

import { Socket } from 'socket.io';
import { Logger } from 'winston';

export interface RateLimitConfig {
  maxConnections: number;
  maxConnectionsPerUser: number;
  maxMessagesPerMinute: number;
  maxSubscriptionsPerUser: number;
  windowSizeMs: number;
}

export interface UserRateLimit {
  userId: string;
  connections: number;
  messages: { timestamp: number; count: number }[];
  subscriptions: number;
  lastActivity: number;
}

export class RateLimitManager {
  private logger: Logger;
  private config: RateLimitConfig;
  private userLimits: Map<string, UserRateLimit> = new Map();
  private connectionCount: number = 0;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(logger: Logger) {
    this.logger = logger;
    this.config = {
      maxConnections: parseInt(process.env.MAX_WEBSOCKET_CONNECTIONS || '1000'),
      maxConnectionsPerUser: parseInt(process.env.MAX_CONNECTIONS_PER_USER || '5'),
      maxMessagesPerMinute: parseInt(process.env.MAX_MESSAGES_PER_MINUTE || '60'),
      maxSubscriptionsPerUser: parseInt(process.env.MAX_SUBSCRIPTIONS_PER_USER || '20'),
      windowSizeMs: 60 * 1000 // 1 minute
    };

    this.startCleanupInterval();
  }

  async limitConnections(socket: Socket, next: (err?: Error) => void): Promise<void> {
    try {
      // Check global connection limit
      if (this.connectionCount >= this.config.maxConnections) {
        this.logger.warn(`Global connection limit reached: ${this.connectionCount}`);
        return next(new Error('Server connection limit reached'));
      }

      const userId = socket.data?.userId;
      if (!userId) {
        // If no user ID, allow connection but count it
        this.connectionCount++;
        return next();
      }

      // Get or create user rate limit
      const userLimit = this.getUserRateLimit(userId);

      // Check per-user connection limit
      if (userLimit.connections >= this.config.maxConnectionsPerUser) {
        this.logger.warn(`User connection limit reached for ${userId}: ${userLimit.connections}`);
        return next(new Error('User connection limit reached'));
      }

      // Update counters
      this.connectionCount++;
      userLimit.connections++;
      userLimit.lastActivity = Date.now();

      this.logger.debug(`Connection allowed for user ${userId}. Total: ${this.connectionCount}, User: ${userLimit.connections}`);
      
      // Set up disconnect handler to decrement counters
      socket.on('disconnect', () => {
        this.handleDisconnection(userId);
      });

      next();

    } catch (error) {
      this.logger.error('Rate limit check error:', error);
      next(new Error('Rate limit check failed'));
    }
  }

  async checkMessageRate(userId: string): Promise<void> {
    const userLimit = this.getUserRateLimit(userId);
    const now = Date.now();
    const windowStart = now - this.config.windowSizeMs;

    // Clean old message records
    userLimit.messages = userLimit.messages.filter(msg => msg.timestamp > windowStart);

    // Count messages in current window
    const messageCount = userLimit.messages.reduce((sum, msg) => sum + msg.count, 0);

    if (messageCount >= this.config.maxMessagesPerMinute) {
      this.logger.warn(`Message rate limit exceeded for user ${userId}: ${messageCount} messages`);
      throw new Error('Message rate limit exceeded');
    }

    // Add current message
    const lastMessage = userLimit.messages[userLimit.messages.length - 1];
    if (lastMessage && now - lastMessage.timestamp < 1000) {
      // Same second, increment count
      lastMessage.count++;
    } else {
      // New second, add new record
      userLimit.messages.push({ timestamp: now, count: 1 });
    }

    userLimit.lastActivity = now;
  }

  async checkSubscriptionLimit(userId: string, newSubscriptions: number = 1): Promise<void> {
    const userLimit = this.getUserRateLimit(userId);

    if (userLimit.subscriptions + newSubscriptions > this.config.maxSubscriptionsPerUser) {
      this.logger.warn(`Subscription limit exceeded for user ${userId}: ${userLimit.subscriptions + newSubscriptions}`);
      throw new Error('Subscription limit exceeded');
    }

    userLimit.subscriptions += newSubscriptions;
    userLimit.lastActivity = Date.now();
  }

  removeSubscriptions(userId: string, count: number = 1): void {
    const userLimit = this.userLimits.get(userId);
    if (userLimit) {
      userLimit.subscriptions = Math.max(0, userLimit.subscriptions - count);
      userLimit.lastActivity = Date.now();
    }
  }

  private getUserRateLimit(userId: string): UserRateLimit {
    let userLimit = this.userLimits.get(userId);
    
    if (!userLimit) {
      userLimit = {
        userId,
        connections: 0,
        messages: [],
        subscriptions: 0,
        lastActivity: Date.now()
      };
      this.userLimits.set(userId, userLimit);
    }

    return userLimit;
  }

  private handleDisconnection(userId: string): void {
    this.connectionCount = Math.max(0, this.connectionCount - 1);
    
    const userLimit = this.userLimits.get(userId);
    if (userLimit) {
      userLimit.connections = Math.max(0, userLimit.connections - 1);
      userLimit.lastActivity = Date.now();
    }

    this.logger.debug(`Connection closed for user ${userId}. Total: ${this.connectionCount}`);
  }

  private startCleanupInterval(): void {
    // Clean up inactive user limits every 5 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanupInactiveUsers();
    }, 5 * 60 * 1000);

    this.logger.info('Rate limit cleanup interval started');
  }

  private cleanupInactiveUsers(): void {
    const now = Date.now();
    const inactiveThreshold = 30 * 60 * 1000; // 30 minutes
    let cleanedCount = 0;

    for (const [userId, userLimit] of this.userLimits.entries()) {
      if (now - userLimit.lastActivity > inactiveThreshold && userLimit.connections === 0) {
        this.userLimits.delete(userId);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      this.logger.info(`Cleaned up ${cleanedCount} inactive user rate limits`);
    }
  }

  // Get current rate limit statistics
  getStatistics(): any {
    const now = Date.now();
    const activeUsers = Array.from(this.userLimits.values()).filter(
      limit => now - limit.lastActivity < 5 * 60 * 1000 // Active in last 5 minutes
    );

    return {
      globalConnections: this.connectionCount,
      maxGlobalConnections: this.config.maxConnections,
      activeUsers: activeUsers.length,
      totalTrackedUsers: this.userLimits.size,
      config: this.config,
      userStats: {
        totalConnections: activeUsers.reduce((sum, user) => sum + user.connections, 0),
        totalSubscriptions: activeUsers.reduce((sum, user) => sum + user.subscriptions, 0),
        averageConnectionsPerUser: activeUsers.length > 0 
          ? activeUsers.reduce((sum, user) => sum + user.connections, 0) / activeUsers.length 
          : 0
      }
    };
  }

  // Get rate limit info for a specific user
  getUserStatistics(userId: string): any {
    const userLimit = this.userLimits.get(userId);
    
    if (!userLimit) {
      return {
        userId,
        exists: false
      };
    }

    const now = Date.now();
    const windowStart = now - this.config.windowSizeMs;
    const recentMessages = userLimit.messages.filter(msg => msg.timestamp > windowStart);
    const messageCount = recentMessages.reduce((sum, msg) => sum + msg.count, 0);

    return {
      userId,
      exists: true,
      connections: userLimit.connections,
      maxConnections: this.config.maxConnectionsPerUser,
      subscriptions: userLimit.subscriptions,
      maxSubscriptions: this.config.maxSubscriptionsPerUser,
      messagesInWindow: messageCount,
      maxMessagesPerMinute: this.config.maxMessagesPerMinute,
      lastActivity: userLimit.lastActivity,
      isActive: now - userLimit.lastActivity < 5 * 60 * 1000
    };
  }

  // Check if user is within limits
  isUserWithinLimits(userId: string): boolean {
    const userLimit = this.userLimits.get(userId);
    if (!userLimit) return true;

    const now = Date.now();
    const windowStart = now - this.config.windowSizeMs;
    const recentMessages = userLimit.messages.filter(msg => msg.timestamp > windowStart);
    const messageCount = recentMessages.reduce((sum, msg) => sum + msg.count, 0);

    return (
      userLimit.connections <= this.config.maxConnectionsPerUser &&
      userLimit.subscriptions <= this.config.maxSubscriptionsPerUser &&
      messageCount <= this.config.maxMessagesPerMinute
    );
  }

  // Update rate limit configuration
  updateConfig(newConfig: Partial<RateLimitConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.logger.info('Rate limit configuration updated:', this.config);
  }

  // Reset rate limits for a user (admin function)
  resetUserLimits(userId: string): void {
    const userLimit = this.userLimits.get(userId);
    if (userLimit) {
      userLimit.messages = [];
      userLimit.subscriptions = 0;
      userLimit.lastActivity = Date.now();
      this.logger.info(`Rate limits reset for user ${userId}`);
    }
  }

  // Ban a user temporarily (admin function)
  banUser(userId: string, durationMs: number = 60 * 60 * 1000): void {
    const userLimit = this.getUserRateLimit(userId);
    
    // Set impossible limits to effectively ban the user
    userLimit.messages = Array(this.config.maxMessagesPerMinute + 1).fill({
      timestamp: Date.now(),
      count: 1
    });
    
    // Schedule unban
    setTimeout(() => {
      this.resetUserLimits(userId);
      this.logger.info(`User ${userId} unbanned after temporary ban`);
    }, durationMs);

    this.logger.warn(`User ${userId} temporarily banned for ${durationMs}ms`);
  }

  // Cleanup on shutdown
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.userLimits.clear();
    this.connectionCount = 0;
    this.logger.info('Rate Limit Manager destroyed');
  }
}
