// Order Notification Manager - Handles real-time order status updates

import { Server as SocketIOServer, Socket } from 'socket.io';
import { Logger } from 'winston';
import axios from 'axios';

export interface OrderNotification {
  orderId: string;
  userId: string;
  type: 'ORDER_CREATED' | 'ORDER_EXECUTED' | 'ORDER_CANCELLED' | 'ORDER_MODIFIED' | 'ORDER_FILLED' | 'ORDER_REJECTED';
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  status: string;
  timestamp: number;
  message?: string;
  executionPrice?: number;
  fillQuantity?: number;
  remainingQuantity?: number;
}

export interface PositionUpdate {
  positionId: string;
  userId: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  timestamp: number;
}

export interface OrderSubscription {
  socketId: string;
  userId: string;
  subscribed: boolean;
  lastUpdate: number;
}

export class OrderNotificationManager {
  private io: SocketIOServer;
  private logger: Logger;
  private subscriptions: Map<string, OrderSubscription> = new Map();
  private tradingServiceUrl: string;
  private pollInterval: NodeJS.Timeout | null = null;

  constructor(io: SocketIOServer, logger: Logger) {
    this.io = io;
    this.logger = logger;
    this.tradingServiceUrl = process.env.TRADING_SERVICE_URL || 'http://localhost:3003';
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing Order Notification Manager...');
    
    // Start polling for order updates
    this.startOrderPolling();
    
    // Test connection to Trading Service
    try {
      await this.testTradingServiceConnection();
      this.logger.info('✅ Connected to Trading Service');
    } catch (error) {
      this.logger.warn('⚠️ Trading Service not available, notifications will be limited');
    }
  }

  private async testTradingServiceConnection(): Promise<void> {
    const response = await axios.get(`${this.tradingServiceUrl}/health`, { timeout: 5000 });
    if (response.status !== 200) {
      throw new Error('Trading Service health check failed');
    }
  }

  async subscribeToOrders(socket: Socket, userId: string): Promise<void> {
    const subscription: OrderSubscription = {
      socketId: socket.id,
      userId,
      subscribed: true,
      lastUpdate: Date.now()
    };

    this.subscriptions.set(socket.id, subscription);
    
    // Join user-specific order room
    socket.join(`orders:${userId}`);
    socket.join('orders:global');

    // Send current orders and positions
    await this.sendCurrentOrdersAndPositions(socket, userId);
    
    this.logger.info(`Order subscription added for user ${userId}`);
  }

  async unsubscribeFromOrders(socket: Socket, userId: string): Promise<void> {
    this.subscriptions.delete(socket.id);
    
    // Leave order rooms
    socket.leave(`orders:${userId}`);
    socket.leave('orders:global');

    this.logger.info(`Order unsubscription for user ${userId}`);
  }

  cleanupUserSubscriptions(socket: Socket, userId: string): void {
    this.subscriptions.delete(socket.id);
    this.logger.info(`Cleaned up order subscriptions for user ${userId}`);
  }

  private startOrderPolling(): void {
    // Poll for order updates every 2 seconds
    this.pollInterval = setInterval(async () => {
      try {
        await this.pollOrderUpdates();
      } catch (error) {
        this.logger.error('Error in order polling cycle:', error);
      }
    }, 2000);

    this.logger.info('Order polling cycle started (2 second interval)');
  }

  private async pollOrderUpdates(): Promise<void> {
    // Get all subscribed users
    const subscribedUsers = new Set<string>();
    this.subscriptions.forEach(sub => {
      if (sub.subscribed) {
        subscribedUsers.add(sub.userId);
      }
    });

    if (subscribedUsers.size === 0) return;

    // Poll each user's orders
    for (const userId of subscribedUsers) {
      try {
        await this.checkUserOrderUpdates(userId);
      } catch (error) {
        this.logger.error(`Error checking orders for user ${userId}:`, error);
      }
    }
  }

  private async checkUserOrderUpdates(userId: string): Promise<void> {
    try {
      // Fetch recent orders from Trading Service
      const response = await axios.get(`${this.tradingServiceUrl}/api/v1/trades`, {
        headers: { 'X-User-ID': userId },
        params: { limit: 10, status: 'recent' },
        timeout: 3000
      });

      if (response.data && response.data.trades) {
        // Process order updates
        response.data.trades.forEach((trade: any) => {
          this.processOrderUpdate(userId, trade);
        });
      }

      // Fetch current positions
      const positionsResponse = await axios.get(`${this.tradingServiceUrl}/api/v1/positions`, {
        headers: { 'X-User-ID': userId },
        timeout: 3000
      });

      if (positionsResponse.data && positionsResponse.data.positions) {
        // Process position updates
        positionsResponse.data.positions.forEach((position: any) => {
          this.processPositionUpdate(userId, position);
        });
      }

    } catch (error) {
      // If Trading Service is unavailable, generate mock notifications
      this.generateMockOrderUpdate(userId);
    }
  }

  private processOrderUpdate(userId: string, trade: any): void {
    const notification: OrderNotification = {
      orderId: trade.id,
      userId,
      type: this.mapTradeStatusToNotificationType(trade.status),
      symbol: trade.symbol,
      side: trade.side,
      quantity: parseFloat(trade.quantity),
      price: trade.price ? parseFloat(trade.price) : undefined,
      status: trade.status,
      timestamp: new Date(trade.updatedAt || trade.createdAt).getTime(),
      message: this.generateOrderMessage(trade),
      executionPrice: trade.executionPrice ? parseFloat(trade.executionPrice) : undefined,
      fillQuantity: trade.fillQuantity ? parseFloat(trade.fillQuantity) : undefined,
      remainingQuantity: trade.remainingQuantity ? parseFloat(trade.remainingQuantity) : undefined
    };

    this.broadcastOrderNotification(notification);
  }

  private processPositionUpdate(userId: string, position: any): void {
    const update: PositionUpdate = {
      positionId: position.id,
      userId,
      symbol: position.symbol,
      side: position.side === 'buy' ? 'long' : 'short',
      quantity: parseFloat(position.quantity),
      entryPrice: parseFloat(position.entryPrice),
      currentPrice: parseFloat(position.currentPrice || position.entryPrice),
      unrealizedPnL: parseFloat(position.unrealizedPnL || 0),
      realizedPnL: parseFloat(position.realizedPnL || 0),
      timestamp: Date.now()
    };

    this.broadcastPositionUpdate(update);
  }

  private generateMockOrderUpdate(userId: string): void {
    // Generate occasional mock notifications for testing
    if (Math.random() > 0.95) { // 5% chance per poll
      const symbols = ['EURUSD', 'GBPUSD', 'USDJPY'];
      const types: OrderNotification['type'][] = ['ORDER_EXECUTED', 'ORDER_CREATED', 'ORDER_FILLED'];
      
      const notification: OrderNotification = {
        orderId: `mock-${Date.now()}`,
        userId,
        type: types[Math.floor(Math.random() * types.length)],
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        side: Math.random() > 0.5 ? 'buy' : 'sell',
        quantity: parseFloat((Math.random() * 10).toFixed(2)),
        price: parseFloat((1.0 + Math.random() * 0.5).toFixed(5)),
        status: 'executed',
        timestamp: Date.now(),
        message: 'Mock order notification for testing'
      };

      this.broadcastOrderNotification(notification);
    }
  }

  private mapTradeStatusToNotificationType(status: string): OrderNotification['type'] {
    switch (status.toLowerCase()) {
      case 'pending': return 'ORDER_CREATED';
      case 'executed': return 'ORDER_EXECUTED';
      case 'cancelled': return 'ORDER_CANCELLED';
      case 'filled': return 'ORDER_FILLED';
      case 'rejected': return 'ORDER_REJECTED';
      case 'modified': return 'ORDER_MODIFIED';
      default: return 'ORDER_CREATED';
    }
  }

  private generateOrderMessage(trade: any): string {
    const action = trade.side === 'buy' ? 'Buy' : 'Sell';
    const status = trade.status.charAt(0).toUpperCase() + trade.status.slice(1);
    return `${action} order for ${trade.quantity} ${trade.symbol} ${status.toLowerCase()}`;
  }

  private broadcastOrderNotification(notification: OrderNotification): void {
    // Send to specific user
    this.io.to(`orders:${notification.userId}`).emit('order:notification', notification);
    
    // Send to global order feed (for admin/monitoring)
    this.io.to('orders:global').emit('order:update', {
      orderId: notification.orderId,
      symbol: notification.symbol,
      type: notification.type,
      timestamp: notification.timestamp
    });

    this.logger.debug(`Order notification sent: ${notification.type} for ${notification.symbol}`);
  }

  private broadcastPositionUpdate(update: PositionUpdate): void {
    // Send to specific user
    this.io.to(`orders:${update.userId}`).emit('position:update', update);

    this.logger.debug(`Position update sent for ${update.symbol}: P&L ${update.unrealizedPnL}`);
  }

  private async sendCurrentOrdersAndPositions(socket: Socket, userId: string): Promise<void> {
    try {
      // Fetch current orders
      const ordersResponse = await axios.get(`${this.tradingServiceUrl}/api/v1/trades`, {
        headers: { 'X-User-ID': userId },
        params: { limit: 20 },
        timeout: 3000
      });

      if (ordersResponse.data && ordersResponse.data.trades) {
        socket.emit('orders:initial', {
          orders: ordersResponse.data.trades,
          timestamp: Date.now()
        });
      }

      // Fetch current positions
      const positionsResponse = await axios.get(`${this.tradingServiceUrl}/api/v1/positions`, {
        headers: { 'X-User-ID': userId },
        timeout: 3000
      });

      if (positionsResponse.data && positionsResponse.data.positions) {
        socket.emit('positions:initial', {
          positions: positionsResponse.data.positions,
          timestamp: Date.now()
        });
      }

    } catch (error) {
      this.logger.warn(`Could not fetch initial orders/positions for user ${userId}:`, error.message);
      
      // Send empty initial data
      socket.emit('orders:initial', { orders: [], timestamp: Date.now() });
      socket.emit('positions:initial', { positions: [], timestamp: Date.now() });
    }
  }

  // Public method to manually trigger order notification
  async triggerOrderNotification(notification: OrderNotification): Promise<void> {
    this.broadcastOrderNotification(notification);
  }

  // Get subscription statistics
  getSubscriptionStats(): any {
    return {
      totalSubscriptions: this.subscriptions.size,
      activeSubscriptions: Array.from(this.subscriptions.values()).filter(s => s.subscribed).length,
      uniqueUsers: new Set(Array.from(this.subscriptions.values()).map(s => s.userId)).size
    };
  }

  // Cleanup on shutdown
  destroy(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.subscriptions.clear();
    this.logger.info('Order Notification Manager destroyed');
  }
}
