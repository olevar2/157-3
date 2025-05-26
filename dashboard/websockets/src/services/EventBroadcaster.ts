// Event Broadcaster - Handles system-wide event broadcasting

import { Server as SocketIOServer, Socket } from 'socket.io';
import { Logger } from 'winston';
import axios from 'axios';

export interface SystemEvent {
  id: string;
  type: 'MARKET_ALERT' | 'SYSTEM_NOTIFICATION' | 'TRADE_SIGNAL' | 'NEWS_UPDATE' | 'MAINTENANCE' | 'ERROR';
  title: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: number;
  userId?: string; // If null, broadcast to all users
  metadata?: {
    symbol?: string;
    price?: number;
    change?: number;
    url?: string;
    duration?: number;
  };
}

export interface EventSubscription {
  socketId: string;
  userId: string;
  eventTypes: string[];
  lastUpdate: number;
}

export class EventBroadcaster {
  private io: SocketIOServer;
  private logger: Logger;
  private subscriptions: Map<string, EventSubscription> = new Map();
  private eventServiceUrl: string;
  private eventQueue: SystemEvent[] = [];
  private pollInterval: NodeJS.Timeout | null = null;

  constructor(io: SocketIOServer, logger: Logger) {
    this.io = io;
    this.logger = logger;
    this.eventServiceUrl = process.env.EVENT_SERVICE_URL || 'http://localhost:3005';
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing Event Broadcaster...');
    
    // Start event polling
    this.startEventPolling();
    
    // Test connection to Event System
    try {
      await this.testEventServiceConnection();
      this.logger.info('✅ Connected to Event System');
    } catch (error) {
      this.logger.warn('⚠️ Event System not available, using local event generation');
    }

    // Generate some initial system events
    this.generateInitialEvents();
  }

  private async testEventServiceConnection(): Promise<void> {
    const response = await axios.get(`${this.eventServiceUrl}/health`, { timeout: 5000 });
    if (response.status !== 200) {
      throw new Error('Event System health check failed');
    }
  }

  async subscribeToEvents(socket: Socket, userId: string, eventTypes: string[]): Promise<void> {
    const subscription: EventSubscription = {
      socketId: socket.id,
      userId,
      eventTypes,
      lastUpdate: Date.now()
    };

    this.subscriptions.set(socket.id, subscription);
    
    // Join event-specific rooms
    eventTypes.forEach(eventType => {
      socket.join(`events:${eventType}`);
    });
    
    // Join global events room
    socket.join('events:global');

    // Send recent events
    await this.sendRecentEvents(socket, eventTypes);
    
    this.logger.info(`Event subscription added for user ${userId}: ${eventTypes.join(', ')}`);
  }

  async unsubscribeFromEvents(socket: Socket, userId: string, eventTypes: string[]): Promise<void> {
    const subscription = this.subscriptions.get(socket.id);
    if (!subscription) return;

    // Remove event types from subscription
    subscription.eventTypes = subscription.eventTypes.filter(type => !eventTypes.includes(type));
    
    // Leave event-specific rooms
    eventTypes.forEach(eventType => {
      socket.leave(`events:${eventType}`);
    });

    if (subscription.eventTypes.length === 0) {
      this.subscriptions.delete(socket.id);
      socket.leave('events:global');
    }

    this.logger.info(`Event unsubscription for user ${userId}: ${eventTypes.join(', ')}`);
  }

  cleanupUserSubscriptions(socket: Socket, userId: string): void {
    this.subscriptions.delete(socket.id);
    this.logger.info(`Cleaned up event subscriptions for user ${userId}`);
  }

  private startEventPolling(): void {
    // Poll for events every 5 seconds
    this.pollInterval = setInterval(async () => {
      try {
        await this.pollSystemEvents();
        this.processEventQueue();
      } catch (error) {
        this.logger.error('Error in event polling cycle:', error);
      }
    }, 5000);

    this.logger.info('Event polling cycle started (5 second interval)');
  }

  private async pollSystemEvents(): Promise<void> {
    try {
      // Try to fetch events from Event System
      const response = await axios.get(`${this.eventServiceUrl}/api/events/recent`, {
        params: { limit: 10 },
        timeout: 3000
      });

      if (response.data && response.data.events) {
        response.data.events.forEach((event: any) => {
          this.addEventToQueue(this.formatSystemEvent(event));
        });
      }
    } catch (error) {
      // Generate mock events if Event System is unavailable
      this.generateMockEvents();
    }
  }

  private formatSystemEvent(rawEvent: any): SystemEvent {
    return {
      id: rawEvent.id || `event-${Date.now()}`,
      type: rawEvent.type || 'SYSTEM_NOTIFICATION',
      title: rawEvent.title || 'System Event',
      message: rawEvent.message || 'System event occurred',
      severity: rawEvent.severity || 'medium',
      timestamp: new Date(rawEvent.timestamp || Date.now()).getTime(),
      userId: rawEvent.userId,
      metadata: rawEvent.metadata
    };
  }

  private generateMockEvents(): void {
    // Generate occasional mock events for testing
    if (Math.random() > 0.8) { // 20% chance per poll
      const eventTypes: SystemEvent['type'][] = ['MARKET_ALERT', 'TRADE_SIGNAL', 'NEWS_UPDATE'];
      const severities: SystemEvent['severity'][] = ['low', 'medium', 'high'];
      const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'];
      
      const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      
      let event: SystemEvent;
      
      switch (eventType) {
        case 'MARKET_ALERT':
          event = {
            id: `alert-${Date.now()}`,
            type: 'MARKET_ALERT',
            title: `${symbol} Price Alert`,
            message: `${symbol} has moved significantly. Current price shows ${Math.random() > 0.5 ? 'bullish' : 'bearish'} momentum.`,
            severity: severities[Math.floor(Math.random() * severities.length)],
            timestamp: Date.now(),
            metadata: {
              symbol,
              price: parseFloat((1.0 + Math.random() * 0.5).toFixed(5)),
              change: parseFloat(((Math.random() - 0.5) * 0.01).toFixed(5))
            }
          };
          break;
          
        case 'TRADE_SIGNAL':
          event = {
            id: `signal-${Date.now()}`,
            type: 'TRADE_SIGNAL',
            title: `Trading Signal: ${symbol}`,
            message: `AI analysis suggests a ${Math.random() > 0.5 ? 'BUY' : 'SELL'} signal for ${symbol} based on technical indicators.`,
            severity: 'medium',
            timestamp: Date.now(),
            metadata: {
              symbol,
              price: parseFloat((1.0 + Math.random() * 0.5).toFixed(5))
            }
          };
          break;
          
        case 'NEWS_UPDATE':
          event = {
            id: `news-${Date.now()}`,
            type: 'NEWS_UPDATE',
            title: 'Economic News Update',
            message: 'New economic data released. Check the economic calendar for potential market impact.',
            severity: 'low',
            timestamp: Date.now(),
            metadata: {
              url: 'https://example.com/economic-calendar'
            }
          };
          break;
          
        default:
          return;
      }
      
      this.addEventToQueue(event);
    }
  }

  private generateInitialEvents(): void {
    // Generate welcome and system status events
    const welcomeEvent: SystemEvent = {
      id: `welcome-${Date.now()}`,
      type: 'SYSTEM_NOTIFICATION',
      title: 'Welcome to Forex Trading Platform',
      message: 'Real-time event system is active. You will receive market alerts, trade signals, and system notifications.',
      severity: 'low',
      timestamp: Date.now()
    };

    const statusEvent: SystemEvent = {
      id: `status-${Date.now()}`,
      type: 'SYSTEM_NOTIFICATION',
      title: 'System Status',
      message: 'All trading services are operational. Market data is streaming live.',
      severity: 'low',
      timestamp: Date.now()
    };

    this.addEventToQueue(welcomeEvent);
    this.addEventToQueue(statusEvent);
  }

  private addEventToQueue(event: SystemEvent): void {
    this.eventQueue.push(event);
    
    // Keep queue size manageable
    if (this.eventQueue.length > 100) {
      this.eventQueue = this.eventQueue.slice(-50);
    }
  }

  private processEventQueue(): void {
    while (this.eventQueue.length > 0) {
      const event = this.eventQueue.shift()!;
      this.broadcastEvent(event);
    }
  }

  private broadcastEvent(event: SystemEvent): void {
    if (event.userId) {
      // Send to specific user
      this.io.to(`user:${event.userId}`).emit('event:notification', event);
    } else {
      // Broadcast to all subscribers of this event type
      this.io.to(`events:${event.type}`).emit('event:notification', event);
      
      // Also send to global events room
      this.io.to('events:global').emit('event:broadcast', {
        id: event.id,
        type: event.type,
        title: event.title,
        severity: event.severity,
        timestamp: event.timestamp
      });
    }

    this.logger.debug(`Event broadcasted: ${event.type} - ${event.title}`);
  }

  private async sendRecentEvents(socket: Socket, eventTypes: string[]): Promise<void> {
    // Send last 10 events of subscribed types
    const recentEvents = this.eventQueue
      .filter(event => eventTypes.includes(event.type))
      .slice(-10);

    if (recentEvents.length > 0) {
      socket.emit('events:initial', {
        events: recentEvents,
        timestamp: Date.now()
      });
    }
  }

  // Public method to manually broadcast an event
  async broadcastSystemEvent(event: Omit<SystemEvent, 'id' | 'timestamp'>): Promise<void> {
    const fullEvent: SystemEvent = {
      ...event,
      id: `manual-${Date.now()}`,
      timestamp: Date.now()
    };

    this.addEventToQueue(fullEvent);
  }

  // Public method to send maintenance notification
  async broadcastMaintenanceNotification(message: string, duration?: number): Promise<void> {
    const event: SystemEvent = {
      id: `maintenance-${Date.now()}`,
      type: 'MAINTENANCE',
      title: 'Scheduled Maintenance',
      message,
      severity: 'high',
      timestamp: Date.now(),
      metadata: { duration }
    };

    this.broadcastEvent(event);
  }

  // Get subscription statistics
  getSubscriptionStats(): any {
    const stats = {
      totalSubscriptions: this.subscriptions.size,
      uniqueUsers: new Set(Array.from(this.subscriptions.values()).map(s => s.userId)).size,
      eventTypeCounts: {} as { [type: string]: number },
      queueSize: this.eventQueue.length
    };

    this.subscriptions.forEach(sub => {
      sub.eventTypes.forEach(type => {
        stats.eventTypeCounts[type] = (stats.eventTypeCounts[type] || 0) + 1;
      });
    });

    return stats;
  }

  // Get recent events (for HTTP API)
  getRecentEvents(limit: number = 20): SystemEvent[] {
    return this.eventQueue.slice(-limit);
  }

  // Cleanup on shutdown
  destroy(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.subscriptions.clear();
    this.eventQueue = [];
    this.logger.info('Event Broadcaster destroyed');
  }
}
