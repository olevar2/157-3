/**
 * Session Event Streams Manager
 * Handles real-time trading session lifecycle events and state management
 * Optimized for forex scalping and day trading operations
 */

import { Kafka, Producer, Consumer, EachMessagePayload } from 'kafkajs';
import { SchemaRegistry, SchemaType } from '@kafkajs/confluent-schema-registry';

// Trading session definitions
export enum TradingSession {
  ASIAN = 'ASIAN',
  LONDON = 'LONDON',
  NY = 'NY',
  OVERLAP_ASIAN_LONDON = 'OVERLAP_ASIAN_LONDON',
  OVERLAP_LONDON_NY = 'OVERLAP_LONDON_NY'
}

export enum SessionEventType {
  PRE_OPEN = 'PRE_OPEN',
  OPEN = 'OPEN',
  CLOSE = 'CLOSE',
  POST_CLOSE = 'POST_CLOSE',
  OVERLAP_START = 'OVERLAP_START',
  OVERLAP_END = 'OVERLAP_END'
}

export interface SessionEvent {
  eventId: string;
  timestamp: number;
  session: TradingSession;
  eventType: SessionEventType;
  timezone: string;
  metadata: Record<string, string>;
}

export interface SessionState {
  session: TradingSession;
  isActive: boolean;
  openTime: number;
  closeTime: number;
  timezone: string;
  overlaps: TradingSession[];
}

/**
 * Session configuration with timezone-aware scheduling
 */
const SESSION_CONFIG = {
  [TradingSession.ASIAN]: {
    timezone: 'Asia/Tokyo',
    openHour: 0,  // 00:00 JST
    closeHour: 9, // 09:00 JST
    overlaps: [TradingSession.OVERLAP_ASIAN_LONDON]
  },
  [TradingSession.LONDON]: {
    timezone: 'Europe/London',
    openHour: 8,  // 08:00 GMT
    closeHour: 17, // 17:00 GMT
    overlaps: [TradingSession.OVERLAP_ASIAN_LONDON, TradingSession.OVERLAP_LONDON_NY]
  },
  [TradingSession.NY]: {
    timezone: 'America/New_York',
    openHour: 8,  // 08:00 EST/EDT
    closeHour: 17, // 17:00 EST/EDT
    overlaps: [TradingSession.OVERLAP_LONDON_NY]
  }
};

export class SessionEventStreamsManager {
  private kafka: Kafka;
  private producer: Producer;
  private consumer: Consumer;
  private schemaRegistry: SchemaRegistry;
  private sessionStates: Map<TradingSession, SessionState> = new Map();
  private eventHandlers: Map<SessionEventType, Function[]> = new Map();
  private isRunning = false;

  constructor() {
    // Initialize Kafka client with optimized configuration
    this.kafka = new Kafka({
      clientId: 'session-event-streams-manager',
      brokers: [
        'localhost:9092',
        'localhost:9093',
        'localhost:9094'
      ],
      connectionTimeout: 3000,
      requestTimeout: 30000,
      retry: {
        initialRetryTime: 100,
        retries: 8
      }
    });

    // Initialize Schema Registry
    this.schemaRegistry = new SchemaRegistry({
      host: 'http://localhost:8081'
    });

    // Initialize producer with low-latency settings
    this.producer = this.kafka.producer({
      maxInFlightRequests: 1,
      idempotent: true,
      transactionTimeout: 30000,
      allowAutoTopicCreation: false
    });

    // Initialize consumer
    this.consumer = this.kafka.consumer({
      groupId: 'session-event-streams-manager',
      sessionTimeout: 30000,
      rebalanceTimeout: 60000,
      heartbeatInterval: 3000,
      maxBytesPerPartition: 1048576,
      maxBytes: 10485760,
      allowAutoTopicCreation: false
    });

    this.initializeEventHandlers();
    this.initializeSessionStates();
  }

  /**
   * Initialize default event handlers
   */
  private initializeEventHandlers(): void {
    // Session open handlers
    this.on(SessionEventType.OPEN, (event: SessionEvent) => {
      console.log(`🟢 Session ${event.session} opened at ${new Date(event.timestamp).toISOString()}`);
      this.updateSessionState(event.session, { isActive: true, openTime: event.timestamp });
    });

    // Session close handlers
    this.on(SessionEventType.CLOSE, (event: SessionEvent) => {
      console.log(`🔴 Session ${event.session} closed at ${new Date(event.timestamp).toISOString()}`);
      this.updateSessionState(event.session, { isActive: false, closeTime: event.timestamp });
    });

    // Overlap handlers
    this.on(SessionEventType.OVERLAP_START, (event: SessionEvent) => {
      console.log(`🔄 Session overlap started: ${event.metadata.overlap} at ${new Date(event.timestamp).toISOString()}`);
    });

    this.on(SessionEventType.OVERLAP_END, (event: SessionEvent) => {
      console.log(`⏹️ Session overlap ended: ${event.metadata.overlap} at ${new Date(event.timestamp).toISOString()}`);
    });
  }

  /**
   * Initialize session states
   */
  private initializeSessionStates(): void {
    Object.values(TradingSession).forEach(session => {
      if (session.includes('OVERLAP')) return; // Skip overlap sessions

      const config = SESSION_CONFIG[session as keyof typeof SESSION_CONFIG];
      this.sessionStates.set(session, {
        session,
        isActive: false,
        openTime: 0,
        closeTime: 0,
        timezone: config.timezone,
        overlaps: config.overlaps
      });
    });
  }

  /**
   * Start the session event streams manager
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Session Event Streams Manager is already running');
    }

    try {
      // Connect producer and consumer
      await this.producer.connect();
      await this.consumer.connect();

      // Subscribe to session events topic
      await this.consumer.subscribe({
        topic: 'forex.sessions.events',
        fromBeginning: false
      });

      // Start consuming events
      await this.consumer.run({
        eachMessage: this.handleSessionEvent.bind(this)
      });

      // Start session monitoring
      this.startSessionMonitoring();

      this.isRunning = true;
      console.log('✅ Session Event Streams Manager started successfully');

    } catch (error) {
      console.error('❌ Failed to start Session Event Streams Manager:', error);
      throw error;
    }
  }

  /**
   * Stop the session event streams manager
   */
  async stop(): Promise<void> {
    if (!this.isRunning) return;

    try {
      await this.consumer.disconnect();
      await this.producer.disconnect();
      this.isRunning = false;
      console.log('✅ Session Event Streams Manager stopped successfully');
    } catch (error) {
      console.error('❌ Error stopping Session Event Streams Manager:', error);
      throw error;
    }
  }

  /**
   * Handle incoming session events
   */
  private async handleSessionEvent(payload: EachMessagePayload): Promise<void> {
    try {
      // Deserialize the message using schema registry
      const event = await this.schemaRegistry.decode(payload.message.value!) as SessionEvent;
      
      // Execute registered handlers for this event type
      const handlers = this.eventHandlers.get(event.eventType) || [];
      await Promise.all(handlers.map(handler => handler(event)));

    } catch (error) {
      console.error('❌ Error handling session event:', error);
      // Send to dead letter queue
      await this.sendToDeadLetterQueue(payload.message, error);
    }
  }

  /**
   * Publish a session event
   */
  async publishSessionEvent(event: SessionEvent): Promise<void> {
    try {
      // Encode the event using schema registry
      const encodedValue = await this.schemaRegistry.encode(
        'forex.sessions.events-value',
        event
      );

      await this.producer.send({
        topic: 'forex.sessions.events',
        messages: [{
          key: `${event.session}-${event.eventType}`,
          value: encodedValue,
          timestamp: event.timestamp.toString(),
          partition: this.getPartitionForSession(event.session)
        }]
      });

      console.log(`📤 Published session event: ${event.session} ${event.eventType}`);

    } catch (error) {
      console.error('❌ Error publishing session event:', error);
      throw error;
    }
  }

  /**
   * Register an event handler
   */
  on(eventType: SessionEventType, handler: Function): void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    this.eventHandlers.get(eventType)!.push(handler);
  }

  /**
   * Get current session state
   */
  getSessionState(session: TradingSession): SessionState | undefined {
    return this.sessionStates.get(session);
  }

  /**
   * Get all active sessions
   */
  getActiveSessions(): TradingSession[] {
    return Array.from(this.sessionStates.values())
      .filter(state => state.isActive)
      .map(state => state.session);
  }

  /**
   * Check if a session is currently active
   */
  isSessionActive(session: TradingSession): boolean {
    const state = this.sessionStates.get(session);
    return state?.isActive || false;
  }

  /**
   * Get current session overlap
   */
  getCurrentOverlap(): TradingSession | null {
    const activeSessions = this.getActiveSessions();
    
    if (activeSessions.includes(TradingSession.ASIAN) && 
        activeSessions.includes(TradingSession.LONDON)) {
      return TradingSession.OVERLAP_ASIAN_LONDON;
    }
    
    if (activeSessions.includes(TradingSession.LONDON) && 
        activeSessions.includes(TradingSession.NY)) {
      return TradingSession.OVERLAP_LONDON_NY;
    }
    
    return null;
  }

  /**
   * Start monitoring session times and generate events
   */
  private startSessionMonitoring(): void {
    const checkInterval = 60000; // Check every minute

    setInterval(async () => {
      try {
        await this.checkSessionTransitions();
      } catch (error) {
        console.error('❌ Error in session monitoring:', error);
      }
    }, checkInterval);

    console.log('🕐 Session monitoring started (checking every minute)');
  }

  /**
   * Check for session transitions and generate events
   */
  private async checkSessionTransitions(): Promise<void> {
    const now = Date.now();
    
    for (const [session, config] of Object.entries(SESSION_CONFIG)) {
      const sessionEnum = session as TradingSession;
      const currentState = this.sessionStates.get(sessionEnum);
      
      if (!currentState) continue;

      const shouldBeActive = this.shouldSessionBeActive(sessionEnum, now);
      
      // Session opening
      if (shouldBeActive && !currentState.isActive) {
        await this.publishSessionEvent({
          eventId: `${sessionEnum}-open-${now}`,
          timestamp: now,
          session: sessionEnum,
          eventType: SessionEventType.OPEN,
          timezone: config.timezone,
          metadata: {}
        });
      }
      
      // Session closing
      if (!shouldBeActive && currentState.isActive) {
        await this.publishSessionEvent({
          eventId: `${sessionEnum}-close-${now}`,
          timestamp: now,
          session: sessionEnum,
          eventType: SessionEventType.CLOSE,
          timezone: config.timezone,
          metadata: {}
        });
      }
    }

    // Check for overlap events
    await this.checkOverlapTransitions(now);
  }

  /**
   * Check if a session should be active at a given time
   */
  private shouldSessionBeActive(session: TradingSession, timestamp: number): boolean {
    const config = SESSION_CONFIG[session as keyof typeof SESSION_CONFIG];
    if (!config) return false;

    const date = new Date(timestamp);
    const hour = date.getUTCHours(); // Simplified - should use proper timezone conversion
    
    return hour >= config.openHour && hour < config.closeHour;
  }

  /**
   * Check for session overlap transitions
   */
  private async checkOverlapTransitions(timestamp: number): Promise<void> {
    const currentOverlap = this.getCurrentOverlap();
    
    // Check if we need to generate overlap events
    // This is a simplified version - real implementation would track overlap states
    if (currentOverlap) {
      // Generate overlap start event if needed
      // Generate overlap end event if needed
    }
  }

  /**
   * Update session state
   */
  private updateSessionState(session: TradingSession, updates: Partial<SessionState>): void {
    const currentState = this.sessionStates.get(session);
    if (currentState) {
      this.sessionStates.set(session, { ...currentState, ...updates });
    }
  }

  /**
   * Get partition for session to ensure ordered processing
   */
  private getPartitionForSession(session: TradingSession): number {
    const sessionIndex = Object.values(TradingSession).indexOf(session);
    return sessionIndex % 4; // Assuming 4 partitions for session events topic
  }

  /**
   * Send failed messages to dead letter queue
   */
  private async sendToDeadLetterQueue(message: any, error: any): Promise<void> {
    try {
      await this.producer.send({
        topic: 'forex.deadletter.queue',
        messages: [{
          key: `session-event-error-${Date.now()}`,
          value: JSON.stringify({
            originalMessage: message.value?.toString(),
            error: error.message,
            timestamp: Date.now()
          })
        }]
      });
    } catch (dlqError) {
      console.error('❌ Failed to send to dead letter queue:', dlqError);
    }
  }
}

// Example usage and export
export default SessionEventStreamsManager;

// Factory function for easy instantiation
export function createSessionManager(): SessionEventStreamsManager {
  return new SessionEventStreamsManager();
}

// Utility functions
export class SessionUtils {
  /**
   * Get session for current time
   */
  static getCurrentSession(): TradingSession | null {
    const now = new Date();
    const hour = now.getUTCHours();
    
    // Simplified session detection (should use proper timezone handling)
    if (hour >= 0 && hour < 9) return TradingSession.ASIAN;
    if (hour >= 8 && hour < 17) return TradingSession.LONDON;
    if (hour >= 13 && hour < 22) return TradingSession.NY;
    
    return null;
  }

  /**
   * Get next session opening time
   */
  static getNextSessionOpen(): { session: TradingSession; timestamp: number } | null {
    // Implementation would calculate next session opening
    // This is a placeholder
    return null;
  }

  /**
   * Calculate session overlap duration
   */
  static getOverlapDuration(overlap: TradingSession): number {
    switch (overlap) {
      case TradingSession.OVERLAP_ASIAN_LONDON:
        return 1 * 60 * 60 * 1000; // 1 hour
      case TradingSession.OVERLAP_LONDON_NY:
        return 4 * 60 * 60 * 1000; // 4 hours
      default:
        return 0;
    }
  }
}
