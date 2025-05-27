/**
 * Rapid Drawdown Protection
 * Real-time drawdown monitoring and protection system
 * 
 * This module provides advanced drawdown protection including:
 * - Real-time drawdown calculation and monitoring
 * - Multi-timeframe drawdown analysis (intraday, daily, weekly)
 * - Automatic position reduction during drawdown periods
 * - Emergency stop-loss activation for severe drawdowns
 * - Recovery mode with reduced risk parameters
 * 
 * Expected Benefits:
 * - Capital preservation during adverse market conditions
 * - Automatic risk reduction to prevent catastrophic losses
 * - Systematic approach to drawdown management
 * - Enhanced long-term account survival
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';

// --- Types ---
export enum DrawdownSeverity {
  NONE = 'NONE',
  MINOR = 'MINOR',
  MODERATE = 'MODERATE',
  SEVERE = 'SEVERE',
  CRITICAL = 'CRITICAL',
  EMERGENCY = 'EMERGENCY',
}

export enum ProtectionAction {
  NONE = 'NONE',
  MONITOR = 'MONITOR',
  REDUCE_SIZE = 'REDUCE_SIZE',
  HALT_NEW_TRADES = 'HALT_NEW_TRADES',
  CLOSE_POSITIONS = 'CLOSE_POSITIONS',
  EMERGENCY_STOP = 'EMERGENCY_STOP',
}

export interface DrawdownMetrics {
  accountId: string;
  currentBalance: number;
  peakBalance: number;
  currentDrawdown: number;
  currentDrawdownPercent: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  drawdownDuration: number; // in milliseconds
  consecutiveLosses: number;
  severity: DrawdownSeverity;
  timestamp: Date;
}

export interface DrawdownConfig {
  minorThreshold: number; // 2%
  moderateThreshold: number; // 5%
  severeThreshold: number; // 10%
  criticalThreshold: number; // 15%
  emergencyThreshold: number; // 20%
  maxConsecutiveLosses: number;
  recoveryThreshold: number; // 50% recovery from max drawdown
  monitoringInterval: number; // milliseconds
}

export interface ProtectionRule {
  severity: DrawdownSeverity;
  action: ProtectionAction;
  positionSizeReduction: number; // percentage
  newTradeRestriction: boolean;
  emergencyCloseAll: boolean;
  recoveryPeriod: number; // milliseconds
}

export interface DrawdownEvent {
  accountId: string;
  eventType: 'DRAWDOWN_START' | 'DRAWDOWN_INCREASE' | 'DRAWDOWN_RECOVERY' | 'PROTECTION_TRIGGERED';
  metrics: DrawdownMetrics;
  action: ProtectionAction;
  message: string;
  timestamp: Date;
}

export interface AccountSnapshot {
  accountId: string;
  balance: number;
  equity: number;
  unrealizedPnL: number;
  realizedPnL: number;
  timestamp: Date;
}

/**
 * Real-time drawdown protection system
 */
export class RapidDrawdownProtection extends EventEmitter {
  private config: DrawdownConfig;
  private protectionRules: Map<DrawdownSeverity, ProtectionRule> = new Map();
  private accountMetrics: Map<string, DrawdownMetrics> = new Map();
  private accountPeaks: Map<string, number> = new Map();
  private drawdownStartTimes: Map<string, number> = new Map();
  private consecutiveLosses: Map<string, number> = new Map();
  private protectionActive: Map<string, ProtectionAction> = new Map();
  private logger: Logger;
  private monitoringInterval: NodeJS.Timeout | null = null;

  constructor(config: DrawdownConfig, logger: Logger) {
    super();
    this.config = config;
    this.logger = logger;
    this._initializeProtectionRules();
    this._startMonitoring();
    this.logger.info('RapidDrawdownProtection initialized for capital preservation');
  }

  /**
   * Updates account snapshot and checks for drawdown
   */
  public updateAccountSnapshot(snapshot: AccountSnapshot): DrawdownMetrics {
    const accountId = snapshot.accountId;
    const currentBalance = snapshot.equity; // Use equity for real-time calculation

    // Initialize or update peak balance
    const currentPeak = this.accountPeaks.get(accountId) || currentBalance;
    const newPeak = Math.max(currentPeak, currentBalance);
    this.accountPeaks.set(accountId, newPeak);

    // Calculate drawdown
    const currentDrawdown = newPeak - currentBalance;
    const currentDrawdownPercent = newPeak > 0 ? (currentDrawdown / newPeak) * 100 : 0;

    // Get or initialize existing metrics
    const existingMetrics = this.accountMetrics.get(accountId);
    const maxDrawdown = existingMetrics ? Math.max(existingMetrics.maxDrawdown, currentDrawdown) : currentDrawdown;
    const maxDrawdownPercent = existingMetrics ? Math.max(existingMetrics.maxDrawdownPercent, currentDrawdownPercent) : currentDrawdownPercent;

    // Calculate drawdown duration
    let drawdownDuration = 0;
    if (currentDrawdown > 0) {
      if (!this.drawdownStartTimes.has(accountId)) {
        this.drawdownStartTimes.set(accountId, Date.now());
      }
      drawdownDuration = Date.now() - this.drawdownStartTimes.get(accountId)!;
    } else {
      this.drawdownStartTimes.delete(accountId);
    }

    // Update consecutive losses
    const consecutiveLosses = this._updateConsecutiveLosses(accountId, snapshot);

    // Determine severity
    const severity = this._determineSeverity(currentDrawdownPercent, consecutiveLosses);

    const metrics: DrawdownMetrics = {
      accountId,
      currentBalance,
      peakBalance: newPeak,
      currentDrawdown,
      currentDrawdownPercent,
      maxDrawdown,
      maxDrawdownPercent,
      drawdownDuration,
      consecutiveLosses,
      severity,
      timestamp: snapshot.timestamp,
    };

    this.accountMetrics.set(accountId, metrics);

    // Check if protection action is needed
    this._evaluateProtectionAction(metrics);

    return metrics;
  }

  /**
   * Gets current drawdown metrics for an account
   */
  public getDrawdownMetrics(accountId: string): DrawdownMetrics | undefined {
    return this.accountMetrics.get(accountId);
  }

  /**
   * Gets current protection status for an account
   */
  public getProtectionStatus(accountId: string): ProtectionAction {
    return this.protectionActive.get(accountId) || ProtectionAction.NONE;
  }

  /**
   * Manually triggers protection action for an account
   */
  public triggerProtection(accountId: string, action: ProtectionAction, reason: string): void {
    this.protectionActive.set(accountId, action);
    
    const metrics = this.accountMetrics.get(accountId);
    if (metrics) {
      const event: DrawdownEvent = {
        accountId,
        eventType: 'PROTECTION_TRIGGERED',
        metrics,
        action,
        message: `Manual protection triggered: ${reason}`,
        timestamp: new Date(),
      };

      this.emit('protectionTriggered', event);
      this.logger.warn(`Manual protection triggered for ${accountId}`, { action, reason });
    }
  }

  /**
   * Resets protection for an account (use carefully)
   */
  public resetProtection(accountId: string): void {
    this.protectionActive.delete(accountId);
    this.consecutiveLosses.delete(accountId);
    
    this.logger.info(`Protection reset for account ${accountId}`);
    this.emit('protectionReset', { accountId, timestamp: new Date() });
  }

  // --- Private Methods ---

  private _initializeProtectionRules(): void {
    this.protectionRules.set(DrawdownSeverity.MINOR, {
      severity: DrawdownSeverity.MINOR,
      action: ProtectionAction.MONITOR,
      positionSizeReduction: 0,
      newTradeRestriction: false,
      emergencyCloseAll: false,
      recoveryPeriod: 30 * 60 * 1000, // 30 minutes
    });

    this.protectionRules.set(DrawdownSeverity.MODERATE, {
      severity: DrawdownSeverity.MODERATE,
      action: ProtectionAction.REDUCE_SIZE,
      positionSizeReduction: 25,
      newTradeRestriction: false,
      emergencyCloseAll: false,
      recoveryPeriod: 60 * 60 * 1000, // 1 hour
    });

    this.protectionRules.set(DrawdownSeverity.SEVERE, {
      severity: DrawdownSeverity.SEVERE,
      action: ProtectionAction.REDUCE_SIZE,
      positionSizeReduction: 50,
      newTradeRestriction: true,
      emergencyCloseAll: false,
      recoveryPeriod: 2 * 60 * 60 * 1000, // 2 hours
    });

    this.protectionRules.set(DrawdownSeverity.CRITICAL, {
      severity: DrawdownSeverity.CRITICAL,
      action: ProtectionAction.HALT_NEW_TRADES,
      positionSizeReduction: 75,
      newTradeRestriction: true,
      emergencyCloseAll: false,
      recoveryPeriod: 4 * 60 * 60 * 1000, // 4 hours
    });

    this.protectionRules.set(DrawdownSeverity.EMERGENCY, {
      severity: DrawdownSeverity.EMERGENCY,
      action: ProtectionAction.EMERGENCY_STOP,
      positionSizeReduction: 100,
      newTradeRestriction: true,
      emergencyCloseAll: true,
      recoveryPeriod: 24 * 60 * 60 * 1000, // 24 hours
    });
  }

  private _startMonitoring(): void {
    this.monitoringInterval = setInterval(() => {
      this._performPeriodicChecks();
    }, this.config.monitoringInterval);
  }

  private _performPeriodicChecks(): void {
    for (const [accountId, metrics] of this.accountMetrics) {
      // Check for recovery
      this._checkRecovery(accountId, metrics);
      
      // Check for escalation
      this._checkEscalation(accountId, metrics);
    }
  }

  private _updateConsecutiveLosses(accountId: string, snapshot: AccountSnapshot): number {
    const currentLosses = this.consecutiveLosses.get(accountId) || 0;
    
    // Simple heuristic: if realized P&L is negative, increment losses
    if (snapshot.realizedPnL < 0) {
      const newCount = currentLosses + 1;
      this.consecutiveLosses.set(accountId, newCount);
      return newCount;
    } else if (snapshot.realizedPnL > 0) {
      // Reset on profit
      this.consecutiveLosses.set(accountId, 0);
      return 0;
    }
    
    return currentLosses;
  }

  private _determineSeverity(drawdownPercent: number, consecutiveLosses: number): DrawdownSeverity {
    // Check emergency conditions first
    if (drawdownPercent >= this.config.emergencyThreshold || consecutiveLosses >= this.config.maxConsecutiveLosses * 2) {
      return DrawdownSeverity.EMERGENCY;
    }
    
    if (drawdownPercent >= this.config.criticalThreshold || consecutiveLosses >= this.config.maxConsecutiveLosses) {
      return DrawdownSeverity.CRITICAL;
    }
    
    if (drawdownPercent >= this.config.severeThreshold) {
      return DrawdownSeverity.SEVERE;
    }
    
    if (drawdownPercent >= this.config.moderateThreshold) {
      return DrawdownSeverity.MODERATE;
    }
    
    if (drawdownPercent >= this.config.minorThreshold) {
      return DrawdownSeverity.MINOR;
    }
    
    return DrawdownSeverity.NONE;
  }

  private _evaluateProtectionAction(metrics: DrawdownMetrics): void {
    const currentAction = this.protectionActive.get(metrics.accountId) || ProtectionAction.NONE;
    const rule = this.protectionRules.get(metrics.severity);
    
    if (!rule) return;

    // Only escalate protection, never downgrade automatically
    if (this._isActionEscalation(currentAction, rule.action)) {
      this.protectionActive.set(metrics.accountId, rule.action);
      
      const event: DrawdownEvent = {
        accountId: metrics.accountId,
        eventType: 'PROTECTION_TRIGGERED',
        metrics,
        action: rule.action,
        message: this._generateProtectionMessage(metrics, rule),
        timestamp: new Date(),
      };

      this.emit('protectionTriggered', event);
      this.logger.warn(`Drawdown protection activated for ${metrics.accountId}`, {
        severity: metrics.severity,
        action: rule.action,
        drawdownPercent: metrics.currentDrawdownPercent,
      });
    }
  }

  private _isActionEscalation(current: ProtectionAction, proposed: ProtectionAction): boolean {
    const actionLevels = {
      [ProtectionAction.NONE]: 0,
      [ProtectionAction.MONITOR]: 1,
      [ProtectionAction.REDUCE_SIZE]: 2,
      [ProtectionAction.HALT_NEW_TRADES]: 3,
      [ProtectionAction.CLOSE_POSITIONS]: 4,
      [ProtectionAction.EMERGENCY_STOP]: 5,
    };

    return actionLevels[proposed] > actionLevels[current];
  }

  private _checkRecovery(accountId: string, metrics: DrawdownMetrics): void {
    const currentAction = this.protectionActive.get(accountId);
    if (!currentAction || currentAction === ProtectionAction.NONE) return;

    // Check if drawdown has recovered sufficiently
    const recoveryPercent = (1 - metrics.currentDrawdownPercent / metrics.maxDrawdownPercent) * 100;
    
    if (recoveryPercent >= this.config.recoveryThreshold && metrics.severity === DrawdownSeverity.NONE) {
      this.protectionActive.delete(accountId);
      
      const event: DrawdownEvent = {
        accountId,
        eventType: 'DRAWDOWN_RECOVERY',
        metrics,
        action: ProtectionAction.NONE,
        message: `Drawdown recovery detected - protection lifted`,
        timestamp: new Date(),
      };

      this.emit('drawdownRecovery', event);
      this.logger.info(`Drawdown recovery for ${accountId}`, { recoveryPercent });
    }
  }

  private _checkEscalation(accountId: string, metrics: DrawdownMetrics): void {
    // Additional escalation logic can be added here
    // For example, time-based escalation or velocity-based escalation
  }

  private _generateProtectionMessage(metrics: DrawdownMetrics, rule: ProtectionRule): string {
    return `Drawdown protection activated: ${metrics.currentDrawdownPercent.toFixed(2)}% drawdown detected. ` +
           `Action: ${rule.action}. Position size reduction: ${rule.positionSizeReduction}%.`;
  }

  public destroy(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }
}
