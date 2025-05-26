// Chat Message Manager - Handles AI chat messaging and responses

import { Server as SocketIOServer, Socket } from 'socket.io';
import { Logger } from 'winston';
import { v4 as uuidv4 } from 'uuid';

export interface ChatMessage {
  id: string;
  userId: string;
  message: string;
  timestamp: number;
  type: 'user' | 'ai' | 'system';
  metadata?: {
    symbols?: string[];
    intent?: string;
    confidence?: number;
  };
}

export interface ChatSession {
  sessionId: string;
  userId: string;
  socketId: string;
  messages: ChatMessage[];
  startTime: number;
  lastActivity: number;
  context: {
    currentSymbol?: string;
    tradingMode?: 'demo' | 'live';
    preferences?: any;
  };
}

export class ChatMessageManager {
  private io: SocketIOServer;
  private logger: Logger;
  private sessions: Map<string, ChatSession> = new Map();
  private userSessions: Map<string, string> = new Map(); // userId -> sessionId

  constructor(io: SocketIOServer, logger: Logger) {
    this.io = io;
    this.logger = logger;
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing Chat Message Manager...');
    
    // Start session cleanup interval
    setInterval(() => {
      this.cleanupInactiveSessions();
    }, 300000); // 5 minutes

    this.logger.info('✅ Chat Message Manager initialized');
  }

  async handleChatMessage(socket: Socket, userId: string, data: any): Promise<void> {
    try {
      // Get or create chat session
      const session = this.getOrCreateSession(socket, userId);
      
      // Create user message
      const userMessage: ChatMessage = {
        id: uuidv4(),
        userId,
        message: data.message,
        timestamp: Date.now(),
        type: 'user',
        metadata: data.metadata
      };

      // Add to session
      session.messages.push(userMessage);
      session.lastActivity = Date.now();

      // Update context if provided
      if (data.context) {
        session.context = { ...session.context, ...data.context };
      }

      // Broadcast user message to user's room
      this.io.to(`chat:${userId}`).emit('chat:message', userMessage);

      // Process message and generate AI response
      const aiResponse = await this.generateAIResponse(userMessage, session);
      
      if (aiResponse) {
        session.messages.push(aiResponse);
        this.io.to(`chat:${userId}`).emit('chat:message', aiResponse);
      }

      this.logger.info(`Chat message processed for user ${userId}: ${data.message.substring(0, 50)}...`);

    } catch (error) {
      this.logger.error('Error handling chat message:', error);
      
      // Send error message to user
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        userId,
        message: 'Sorry, I encountered an error processing your message. Please try again.',
        timestamp: Date.now(),
        type: 'system'
      };

      this.io.to(`chat:${userId}`).emit('chat:message', errorMessage);
    }
  }

  private getOrCreateSession(socket: Socket, userId: string): ChatSession {
    let sessionId = this.userSessions.get(userId);
    
    if (!sessionId || !this.sessions.has(sessionId)) {
      // Create new session
      sessionId = uuidv4();
      const session: ChatSession = {
        sessionId,
        userId,
        socketId: socket.id,
        messages: [],
        startTime: Date.now(),
        lastActivity: Date.now(),
        context: {}
      };

      this.sessions.set(sessionId, session);
      this.userSessions.set(userId, sessionId);
      
      // Join chat room
      socket.join(`chat:${userId}`);
      
      // Send welcome message
      const welcomeMessage: ChatMessage = {
        id: uuidv4(),
        userId,
        message: 'Hello! I\'m your AI trading assistant. I can help you with market analysis, trading strategies, and platform navigation. How can I assist you today?',
        timestamp: Date.now(),
        type: 'ai'
      };

      session.messages.push(welcomeMessage);
      socket.emit('chat:message', welcomeMessage);
      
      this.logger.info(`New chat session created for user ${userId}`);
    }

    return this.sessions.get(sessionId)!;
  }

  private async generateAIResponse(userMessage: ChatMessage, session: ChatSession): Promise<ChatMessage | null> {
    try {
      // Analyze user intent
      const intent = this.analyzeIntent(userMessage.message);
      
      // Generate response based on intent
      const responseText = await this.generateResponseText(userMessage.message, intent, session);
      
      if (!responseText) return null;

      const aiMessage: ChatMessage = {
        id: uuidv4(),
        userId: userMessage.userId,
        message: responseText,
        timestamp: Date.now(),
        type: 'ai',
        metadata: {
          intent,
          confidence: 0.8
        }
      };

      return aiMessage;

    } catch (error) {
      this.logger.error('Error generating AI response:', error);
      return null;
    }
  }

  private analyzeIntent(message: string): string {
    const lowerMessage = message.toLowerCase();
    
    // Market analysis intents
    if (lowerMessage.includes('price') || lowerMessage.includes('chart') || lowerMessage.includes('analysis')) {
      return 'market_analysis';
    }
    
    // Trading intents
    if (lowerMessage.includes('buy') || lowerMessage.includes('sell') || lowerMessage.includes('trade')) {
      return 'trading_action';
    }
    
    // Portfolio intents
    if (lowerMessage.includes('portfolio') || lowerMessage.includes('balance') || lowerMessage.includes('profit')) {
      return 'portfolio_inquiry';
    }
    
    // Help intents
    if (lowerMessage.includes('help') || lowerMessage.includes('how') || lowerMessage.includes('what')) {
      return 'help_request';
    }
    
    // News/events intents
    if (lowerMessage.includes('news') || lowerMessage.includes('event') || lowerMessage.includes('economic')) {
      return 'news_inquiry';
    }

    return 'general_conversation';
  }

  private async generateResponseText(message: string, intent: string, session: ChatSession): Promise<string> {
    // This is a simplified AI response generator
    // In a real implementation, this would integrate with an AI service like OpenAI, Claude, etc.
    
    switch (intent) {
      case 'market_analysis':
        return this.generateMarketAnalysisResponse(message, session);
      
      case 'trading_action':
        return this.generateTradingActionResponse(message, session);
      
      case 'portfolio_inquiry':
        return this.generatePortfolioResponse(message, session);
      
      case 'help_request':
        return this.generateHelpResponse(message, session);
      
      case 'news_inquiry':
        return this.generateNewsResponse(message, session);
      
      default:
        return this.generateGeneralResponse(message, session);
    }
  }

  private generateMarketAnalysisResponse(message: string, session: ChatSession): string {
    const responses = [
      "Based on current market conditions, I can see some interesting patterns. Would you like me to analyze a specific currency pair?",
      "The forex market is showing mixed signals today. EUR/USD is consolidating while GBP/USD shows bullish momentum. What pair interests you?",
      "I notice you're asking about market analysis. I can provide technical indicators, support/resistance levels, and trend analysis. Which symbol would you like me to focus on?",
      "Market volatility is moderate today. The major pairs are trading within their daily ranges. Would you like a detailed analysis of any specific pair?"
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  private generateTradingActionResponse(message: string, session: ChatSession): string {
    const responses = [
      "I can help you with trade execution. Please note that all trades should be carefully considered. Would you like me to analyze the market conditions for your intended trade?",
      "Before placing any trade, let's review the current market conditions and your risk management strategy. What pair are you considering?",
      "I see you're interested in trading. Remember to always use proper risk management. What's your trading plan for this position?",
      "Trading decisions should be based on thorough analysis. Would you like me to help you evaluate the market conditions for your trade idea?"
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  private generatePortfolioResponse(message: string, session: ChatSession): string {
    const responses = [
      "I can help you review your portfolio performance. Your current positions and P&L are displayed in the trading dashboard. Would you like me to analyze any specific positions?",
      "Portfolio management is crucial for long-term success. I can help you review your risk exposure and position sizing. What would you like to know?",
      "Your portfolio metrics are available in the dashboard. I can help interpret the data and suggest optimization strategies. What specific aspect interests you?",
      "Let's review your portfolio together. I can analyze your risk-reward ratios and suggest improvements. What would you like to focus on?"
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  private generateHelpResponse(message: string, session: ChatSession): string {
    const responses = [
      "I'm here to help! I can assist with market analysis, trading strategies, platform navigation, and portfolio management. What would you like to learn about?",
      "Here are some things I can help you with:\n• Market analysis and price predictions\n• Trading strategy recommendations\n• Risk management advice\n• Platform feature explanations\n\nWhat interests you most?",
      "I can guide you through various aspects of forex trading. Whether you need technical analysis, fundamental insights, or platform help, I'm here for you. What's your question?",
      "Welcome to your AI trading assistant! I can help with analysis, trading decisions, and platform features. Feel free to ask me anything about forex trading."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  private generateNewsResponse(message: string, session: ChatSession): string {
    const responses = [
      "Economic news can significantly impact forex markets. I can help you understand how events affect currency pairs. Are you looking for information about a specific event or currency?",
      "Market-moving news includes central bank decisions, economic indicators, and geopolitical events. Would you like me to explain how these affect your trading pairs?",
      "Staying informed about economic events is crucial for forex trading. I can help you interpret news and its potential market impact. What news are you interested in?",
      "Economic calendars show upcoming events that may affect currency prices. I can help you prepare for these events and understand their potential impact."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  private generateGeneralResponse(message: string, session: ChatSession): string {
    const responses = [
      "I understand you're looking for assistance. I'm here to help with all aspects of forex trading. Could you be more specific about what you'd like to know?",
      "Thank you for your message. I can help with market analysis, trading strategies, and platform features. What would you like to explore?",
      "I'm your AI trading assistant, ready to help with any forex-related questions. What can I assist you with today?",
      "I'm here to support your trading journey. Whether you need analysis, strategy advice, or platform help, just let me know what you're looking for."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  cleanupUserSubscriptions(socket: Socket, userId: string): void {
    const sessionId = this.userSessions.get(userId);
    if (sessionId) {
      const session = this.sessions.get(sessionId);
      if (session && session.socketId === socket.id) {
        // Don't delete the session immediately, just update the socket
        session.socketId = '';
        this.logger.info(`Chat session updated for user ${userId} disconnect`);
      }
    }
  }

  private cleanupInactiveSessions(): void {
    const now = Date.now();
    const maxInactiveTime = 30 * 60 * 1000; // 30 minutes

    for (const [sessionId, session] of this.sessions.entries()) {
      if (now - session.lastActivity > maxInactiveTime) {
        this.sessions.delete(sessionId);
        this.userSessions.delete(session.userId);
        this.logger.info(`Cleaned up inactive chat session for user ${session.userId}`);
      }
    }
  }

  // Get chat history for a user
  getChatHistory(userId: string): ChatMessage[] {
    const sessionId = this.userSessions.get(userId);
    if (sessionId) {
      const session = this.sessions.get(sessionId);
      return session?.messages || [];
    }
    return [];
  }

  // Get session statistics
  getSessionStats(): any {
    return {
      activeSessions: this.sessions.size,
      totalMessages: Array.from(this.sessions.values()).reduce((sum, session) => sum + session.messages.length, 0),
      averageSessionLength: Array.from(this.sessions.values()).reduce((sum, session) => sum + session.messages.length, 0) / this.sessions.size || 0
    };
  }

  // Cleanup on shutdown
  destroy(): void {
    this.sessions.clear();
    this.userSessions.clear();
    this.logger.info('Chat Message Manager destroyed');
  }
}
