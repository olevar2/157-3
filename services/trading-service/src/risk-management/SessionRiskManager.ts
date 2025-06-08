import axios from 'axios';
import { EventEmitter } from 'events';

/**
 * Session Risk Manager
 * Manages risk parameters and exposure limits across trading sessions
 */

export enum TradingSession {
  ASIAN = 'ASIAN',
  LONDON = 'LONDON', 
  NEW_YORK = 'NEW_YORK',
  OVERLAP_LDN_NY = 'OVERLAP_LDN_NY'
}

export interface SessionRiskLimits {
  maxDailyLoss: number;           // Maximum loss per day
  maxSessionLoss: number;         // Maximum loss per session
  maxPositionSize: number;        // Maximum position size
  maxOpenPositions: number;       // Maximum number of concurrent positions
  maxCorrelatedExposure: number;  // Maximum exposure to correlated pairs
  maxLeverage: number;            // Maximum leverage allowed
  riskPerTrade: number;           // Risk per individual trade (% of account)
}

export interface SessionMetrics {
  session: TradingSession;
  startTime: Date;
  endTime?: Date;
  totalPnl: number;
  realizedPnl: number;
  unrealizedPnl: number;
  tradesCount: number;
  winningTrades: number;
  losingTrades: number;
  maxPositionSize: number;
  currentPositions: number;
  riskExposure: number;
}

export interface RiskViolation {
  type: 'DAILY_LOSS' | 'SESSION_LOSS' | 'POSITION_SIZE' | 'MAX_POSITIONS' | 'CORRELATION' | 'LEVERAGE';
  severity: 'WARNING' | 'CRITICAL';
  currentValue: number;
  limitValue: number;
  timestamp: Date;
  session: TradingSession;
}

export class SessionRiskManager extends EventEmitter {
  private riskLimits: Map<TradingSession, SessionRiskLimits> = new Map();
  private sessionMetrics: Map<TradingSession, SessionMetrics> = new Map();
  private currentSession: TradingSession;
  private pythonEngineUrl: string;
  private dailyMetrics: {
    totalPnl: number;
    startEquity: number;
    currentEquity: number;
    tradesCount: number;
  };

  constructor() {
    super();
    this.pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
    this.currentSession = this.getCurrentSession();
    
    this.dailyMetrics = {
      totalPnl: 0,
      startEquity: 0,
      currentEquity: 0,
      tradesCount: 0
    };

    this.initializeDefaultLimits();
    console.log(`SessionRiskManager initialized for ${this.currentSession} session`);
  }

  /**
   * Initialize default risk limits for all sessions
   */
  private initializeDefaultLimits(): void {
    const defaultLimits: SessionRiskLimits = {
      maxDailyLoss: 0.05,        // 5% of account
      maxSessionLoss: 0.02,      // 2% of account per session
      maxPositionSize: 0.10,     // 10% of account per position
      maxOpenPositions: 5,       // Max 5 concurrent positions
      maxCorrelatedExposure: 0.15, // 15% total correlation exposure
      maxLeverage: 100,          // 100:1 max leverage
      riskPerTrade: 0.01         // 1% risk per trade
    };

    // Different limits for different sessions
    this.riskLimits.set(TradingSession.ASIAN, { 
      ...defaultLimits,
      maxSessionLoss: 0.015, // Lower during Asian session
      maxOpenPositions: 3 
    });
    
    this.riskLimits.set(TradingSession.LONDON, {
      ...defaultLimits,
      maxSessionLoss: 0.025, // Higher during London session
      maxOpenPositions: 6
    });
    
    this.riskLimits.set(TradingSession.NEW_YORK, {
      ...defaultLimits,
      maxSessionLoss: 0.025, // Higher during NY session
      maxOpenPositions: 6
    });
    
    this.riskLimits.set(TradingSession.OVERLAP_LDN_NY, {
      ...defaultLimits,
      maxSessionLoss: 0.03,  // Highest during overlap
      maxOpenPositions: 8,
      maxCorrelatedExposure: 0.20
    });
  }

  /**
   * Determine current trading session based on UTC time
   */
  private getCurrentSession(): TradingSession {
    const now = new Date();
    const utcHour = now.getUTCHours();

    if (utcHour >= 0 && utcHour < 8) {
      return TradingSession.ASIAN;
    } else if (utcHour >= 8 && utcHour < 12) {
      return TradingSession.LONDON;
    } else if (utcHour >= 12 && utcHour < 16) {
      return TradingSession.OVERLAP_LDN_NY;
    } else if (utcHour >= 16 && utcHour < 21) {
      return TradingSession.NEW_YORK;
    } else {
      return TradingSession.ASIAN; // Late NY/Early Asian
    }
  }

  /**
   * Check if a new trade violates risk limits
   */
  public async validateTradeRisk(symbol: string, size: number, price: number, leverage: number = 1): Promise<{
    allowed: boolean;
    violations: RiskViolation[];
    adjustedSize?: number;
  }> {
    const violations: RiskViolation[] = [];
    const sessionLimits = this.riskLimits.get(this.currentSession)!;
    const sessionMetrics = this.getSessionMetrics(this.currentSession);
    
    const positionValue = size * price;
    const riskAmount = positionValue / this.dailyMetrics.currentEquity;

    // Check position size limit
    if (riskAmount > sessionLimits.maxPositionSize) {
      violations.push({
        type: 'POSITION_SIZE',
        severity: 'CRITICAL',
        currentValue: riskAmount,
        limitValue: sessionLimits.maxPositionSize,
        timestamp: new Date(),
        session: this.currentSession
      });
    }

    // Check maximum positions limit
    if (sessionMetrics.currentPositions >= sessionLimits.maxOpenPositions) {
      violations.push({
        type: 'MAX_POSITIONS',
        severity: 'CRITICAL',
        currentValue: sessionMetrics.currentPositions,
        limitValue: sessionLimits.maxOpenPositions,
        timestamp: new Date(),
        session: this.currentSession
      });
    }

    // Check leverage limit
    if (leverage > sessionLimits.maxLeverage) {
      violations.push({
        type: 'LEVERAGE',
        severity: 'WARNING',
        currentValue: leverage,
        limitValue: sessionLimits.maxLeverage,
        timestamp: new Date(),
        session: this.currentSession
      });
    }

    // Check daily loss limit
    const dailyLoss = (this.dailyMetrics.startEquity - this.dailyMetrics.currentEquity) / this.dailyMetrics.startEquity;
    if (dailyLoss > sessionLimits.maxDailyLoss) {
      violations.push({
        type: 'DAILY_LOSS',
        severity: 'CRITICAL',
        currentValue: dailyLoss,
        limitValue: sessionLimits.maxDailyLoss,
        timestamp: new Date(),
        session: this.currentSession
      });
    }

    // Check session loss limit
    const sessionLoss = Math.abs(sessionMetrics.totalPnl) / this.dailyMetrics.currentEquity;
    if (sessionLoss > sessionLimits.maxSessionLoss) {
      violations.push({
        type: 'SESSION_LOSS',
        severity: 'CRITICAL',
        currentValue: sessionLoss,
        limitValue: sessionLimits.maxSessionLoss,
        timestamp: new Date(),
        session: this.currentSession
      });
    }

    // Calculate adjusted size if position size violation
    let adjustedSize: number | undefined;
    if (violations.some(v => v.type === 'POSITION_SIZE')) {
      adjustedSize = (sessionLimits.maxPositionSize * this.dailyMetrics.currentEquity) / price;
    }

    const allowed = violations.filter(v => v.severity === 'CRITICAL').length === 0;

    // Log violations
    if (violations.length > 0) {
      console.warn(`Risk violations detected for ${symbol}:`, violations);
      
      // Emit risk violation event
      this.emit('riskViolation', {
        symbol,
        violations,
        session: this.currentSession,
        adjustedSize
      });
      
      // Notify Python engines
      await this.notifyPythonEngines('risk_violation', {
        symbol,
        violations,
        session: this.currentSession
      });
    }

    return { allowed, violations, adjustedSize };
  }

  /**
   * Update session metrics after a trade
   */
  public updateTradeMetrics(pnl: number, positionSize: number, isWin: boolean): void {
    const sessionMetrics = this.getSessionMetrics(this.currentSession);
    
    sessionMetrics.totalPnl += pnl;
    sessionMetrics.tradesCount++;
    
    if (isWin) {
      sessionMetrics.winningTrades++;
    } else {
      sessionMetrics.losingTrades++;
    }
    
    sessionMetrics.maxPositionSize = Math.max(sessionMetrics.maxPositionSize, positionSize);
    
    // Update daily metrics
    this.dailyMetrics.totalPnl += pnl;
    this.dailyMetrics.currentEquity += pnl;
    this.dailyMetrics.tradesCount++;

    console.log(`Trade metrics updated: Session PnL=${sessionMetrics.totalPnl}, Daily PnL=${this.dailyMetrics.totalPnl}`);
  }

  /**
   * Update position count
   */
  public updatePositionCount(change: number): void {
    const sessionMetrics = this.getSessionMetrics(this.currentSession);
    sessionMetrics.currentPositions = Math.max(0, sessionMetrics.currentPositions + change);
  }

  /**
   * Get session metrics, creating if not exists
   */
  private getSessionMetrics(session: TradingSession): SessionMetrics {
    if (!this.sessionMetrics.has(session)) {
      this.sessionMetrics.set(session, {
        session,
        startTime: new Date(),
        totalPnl: 0,
        realizedPnl: 0,
        unrealizedPnl: 0,
        tradesCount: 0,
        winningTrades: 0,
        losingTrades: 0,
        maxPositionSize: 0,
        currentPositions: 0,
        riskExposure: 0
      });
    }
    return this.sessionMetrics.get(session)!;
  }

  /**
   * Switch to new trading session
   */
  public switchSession(): void {
    const newSession = this.getCurrentSession();
    if (newSession !== this.currentSession) {
      const oldMetrics = this.getSessionMetrics(this.currentSession);
      oldMetrics.endTime = new Date();
      
      console.log(`Session changed from ${this.currentSession} to ${newSession}`);
      
      this.emit('sessionChanged', {
        oldSession: this.currentSession,
        newSession,
        oldMetrics
      });
      
      this.currentSession = newSession;
    }
  }

  /**
   * Set daily starting equity
   */
  public setDailyStartEquity(equity: number): void {
    this.dailyMetrics.startEquity = equity;
    this.dailyMetrics.currentEquity = equity;
    console.log(`Daily start equity set to: ${equity}`);
  }

  /**
   * Get current session risk limits
   */
  public getCurrentRiskLimits(): SessionRiskLimits {
    return this.riskLimits.get(this.currentSession)!;
  }

  /**
   * Update risk limits for a session
   */
  public updateRiskLimits(session: TradingSession, limits: Partial<SessionRiskLimits>): void {
    const currentLimits = this.riskLimits.get(session)!;
    this.riskLimits.set(session, { ...currentLimits, ...limits });
    console.log(`Risk limits updated for ${session} session:`, limits);
  }

  /**
   * Get session metrics
   */
  public getSessionMetricsForSession(session: TradingSession): SessionMetrics {
    return this.getSessionMetrics(session);
  }

  /**
   * Get daily metrics
   */
  public getDailyMetrics() {
    return { ...this.dailyMetrics };
  }

  /**
   * Reset daily metrics (call at start of new trading day)
   */
  public resetDailyMetrics(startEquity: number): void {
    this.dailyMetrics = {
      totalPnl: 0,
      startEquity,
      currentEquity: startEquity,
      tradesCount: 0
    };
    
    this.sessionMetrics.clear();
    console.log(`Daily metrics reset with start equity: ${startEquity}`);
  }

  /**
   * Notify Python engines about risk events
   */
  private async notifyPythonEngines(eventType: string, data: any): Promise<void> {
    try {
      await axios.post(`${this.pythonEngineUrl}/api/risk/session-event`, {
        eventType,
        data,
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      console.error(`Failed to notify Python engines about ${eventType}:`, error.message);
    }
  }
}
