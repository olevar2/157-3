/**
 * Risk Management Service
 * Comprehensive risk management for high-frequency trading operations
 * 
 * This module provides risk management including:
 * - Real-time position monitoring and limits
 * - Dynamic risk assessment and adjustment
 * - Automated risk controls and circuit breakers
 * - Portfolio-level risk management
 * - Compliance and regulatory risk checks
 * 
 * Expected Benefits:
 * - Automated risk control for all trading activities
 * - Real-time position and exposure monitoring
 * - Dynamic risk adjustment based on market conditions
 * - Regulatory compliance and audit trail
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';

export interface RiskLimits {
  userId: string;
  accountId: string;
  maxPositionSize: number;
  maxDailyLoss: number;
  maxDrawdown: number;
  maxLeverage: number;
  maxOrderSize: number;
  maxOrdersPerSecond: number;
  maxOrdersPerDay: number;
  allowedSymbols: string[];
  blockedSymbols: string[];
  tradingHours: {
    start: string;
    end: string;
    timezone: string;
  };
}

export interface RiskMetrics {
  accountId: string;
  currentExposure: number;
  dailyPnL: number;
  unrealizedPnL: number;
  realizedPnL: number;
  currentDrawdown: number;
  maxDrawdown: number;
  currentLeverage: number;
  marginUsed: number;
  marginAvailable: number;
  openPositions: number;
  ordersToday: number;
  lastOrderTime: Date;
  riskScore: number; // 0-100
}

export interface RiskCheck {
  checkId: string;
  accountId: string;
  checkType: 'pre_trade' | 'post_trade' | 'position' | 'portfolio';
  passed: boolean;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  violations: RiskViolation[];
  recommendations: string[];
  timestamp: Date;
}

export interface RiskViolation {
  type: string;
  severity: 'warning' | 'error' | 'critical';
  message: string;
  currentValue: number;
  limitValue: number;
  action: 'block' | 'warn' | 'reduce' | 'close';
}

export interface TradeRequest {
  userId: string;
  accountId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  orderType: string;
  metadata?: Record<string, any>;
}

export interface Position {
  accountId: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  marginUsed: number;
}

export class RiskManagementService extends EventEmitter {
  private logger: Logger;
  private riskLimits: Map<string, RiskLimits> = new Map();
  private riskMetrics: Map<string, RiskMetrics> = new Map();
  private orderCounts: Map<string, number[]> = new Map(); // Track orders per second
  private circuitBreakers: Map<string, boolean> = new Map();

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this.logger.info('Risk Management Service initialized');
    
    // Start periodic risk monitoring
    this.startRiskMonitoring();
  }

  /**
   * Perform pre-trade risk check
   */
  async performPreTradeCheck(request: TradeRequest): Promise<RiskCheck> {
    const checkId = `pre_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      const limits = this.riskLimits.get(request.accountId);
      const metrics = this.riskMetrics.get(request.accountId);
      
      if (!limits) {
        throw new Error(`No risk limits found for account ${request.accountId}`);
      }

      const violations: RiskViolation[] = [];

      // Check if account is circuit broken
      if (this.circuitBreakers.get(request.accountId)) {
        violations.push({
          type: 'circuit_breaker',
          severity: 'critical',
          message: 'Account is circuit broken due to excessive risk',
          currentValue: 1,
          limitValue: 0,
          action: 'block'
        });
      }

      // Check symbol restrictions
      if (limits.blockedSymbols.includes(request.symbol)) {
        violations.push({
          type: 'blocked_symbol',
          severity: 'error',
          message: `Symbol ${request.symbol} is blocked for trading`,
          currentValue: 1,
          limitValue: 0,
          action: 'block'
        });
      }

      if (limits.allowedSymbols.length > 0 && !limits.allowedSymbols.includes(request.symbol)) {
        violations.push({
          type: 'symbol_not_allowed',
          severity: 'error',
          message: `Symbol ${request.symbol} is not in allowed list`,
          currentValue: 1,
          limitValue: 0,
          action: 'block'
        });
      }

      // Check order size limits
      if (request.quantity > limits.maxOrderSize) {
        violations.push({
          type: 'order_size_exceeded',
          severity: 'error',
          message: 'Order size exceeds maximum allowed',
          currentValue: request.quantity,
          limitValue: limits.maxOrderSize,
          action: 'block'
        });
      }

      // Check position size limits
      const newPositionSize = await this.calculateNewPositionSize(request);
      if (Math.abs(newPositionSize) > limits.maxPositionSize) {
        violations.push({
          type: 'position_size_exceeded',
          severity: 'error',
          message: 'New position would exceed maximum position size',
          currentValue: Math.abs(newPositionSize),
          limitValue: limits.maxPositionSize,
          action: 'reduce'
        });
      }

      // Check leverage limits
      if (metrics) {
        const newLeverage = await this.calculateNewLeverage(request, metrics);
        if (newLeverage > limits.maxLeverage) {
          violations.push({
            type: 'leverage_exceeded',
            severity: 'warning',
            message: 'New trade would exceed maximum leverage',
            currentValue: newLeverage,
            limitValue: limits.maxLeverage,
            action: 'warn'
          });
        }

        // Check daily loss limits
        if (metrics.dailyPnL < -limits.maxDailyLoss) {
          violations.push({
            type: 'daily_loss_exceeded',
            severity: 'critical',
            message: 'Daily loss limit exceeded',
            currentValue: Math.abs(metrics.dailyPnL),
            limitValue: limits.maxDailyLoss,
            action: 'block'
          });
        }

        // Check drawdown limits
        if (metrics.currentDrawdown > limits.maxDrawdown) {
          violations.push({
            type: 'drawdown_exceeded',
            severity: 'critical',
            message: 'Maximum drawdown exceeded',
            currentValue: metrics.currentDrawdown,
            limitValue: limits.maxDrawdown,
            action: 'block'
          });
        }
      }

      // Check order frequency limits
      const orderFrequencyViolation = this.checkOrderFrequency(request.accountId, limits);
      if (orderFrequencyViolation) {
        violations.push(orderFrequencyViolation);
      }

      // Check trading hours
      const tradingHoursViolation = this.checkTradingHours(limits);
      if (tradingHoursViolation) {
        violations.push(tradingHoursViolation);
      }

      const riskLevel = this.calculateRiskLevel(violations);
      const passed = !violations.some(v => v.action === 'block');

      const riskCheck: RiskCheck = {
        checkId,
        accountId: request.accountId,
        checkType: 'pre_trade',
        passed,
        riskLevel,
        violations,
        recommendations: this.generateRecommendations(violations),
        timestamp: new Date()
      };

      this.logger.info(`Pre-trade risk check completed: ${checkId}, passed: ${passed}`);
      this.emit('riskCheckCompleted', riskCheck);

      return riskCheck;

    } catch (error) {
      this.logger.error(`Error in pre-trade risk check ${checkId}:`, error);
      throw error;
    }
  }

  /**
   * Update risk limits for an account
   */
  async updateRiskLimits(accountId: string, limits: RiskLimits): Promise<void> {
    this.riskLimits.set(accountId, limits);
    this.logger.info(`Risk limits updated for account: ${accountId}`);
    this.emit('riskLimitsUpdated', { accountId, limits });
  }

  /**
   * Update risk metrics for an account
   */
  async updateRiskMetrics(accountId: string, metrics: RiskMetrics): Promise<void> {
    this.riskMetrics.set(accountId, metrics);
    
    // Check for automatic circuit breaker triggers
    await this.checkCircuitBreakers(accountId, metrics);
    
    this.emit('riskMetricsUpdated', { accountId, metrics });
  }

  /**
   * Get current risk metrics for an account
   */
  getRiskMetrics(accountId: string): RiskMetrics | undefined {
    return this.riskMetrics.get(accountId);
  }

  /**
   * Get risk limits for an account
   */
  getRiskLimits(accountId: string): RiskLimits | undefined {
    return this.riskLimits.get(accountId);
  }

  /**
   * Trigger circuit breaker for an account
   */
  async triggerCircuitBreaker(accountId: string, reason: string): Promise<void> {
    this.circuitBreakers.set(accountId, true);
    this.logger.warn(`Circuit breaker triggered for account ${accountId}: ${reason}`);
    this.emit('circuitBreakerTriggered', { accountId, reason, timestamp: new Date() });
  }

  /**
   * Reset circuit breaker for an account
   */
  async resetCircuitBreaker(accountId: string): Promise<void> {
    this.circuitBreakers.delete(accountId);
    this.logger.info(`Circuit breaker reset for account: ${accountId}`);
    this.emit('circuitBreakerReset', { accountId, timestamp: new Date() });
  }

  /**
   * Calculate new position size after trade
   */
  private async calculateNewPositionSize(request: TradeRequest): Promise<number> {
    // This would integrate with position tracking service
    // For now, return the requested quantity
    return request.quantity * (request.side === 'sell' ? -1 : 1);
  }

  /**
   * Calculate new leverage after trade
   */
  private async calculateNewLeverage(request: TradeRequest, metrics: RiskMetrics): Promise<number> {
    // Simplified leverage calculation
    const tradeValue = request.quantity * (request.price || 1);
    const newExposure = metrics.currentExposure + tradeValue;
    const accountEquity = metrics.marginUsed + metrics.marginAvailable;
    
    return accountEquity > 0 ? newExposure / accountEquity : 0;
  }

  /**
   * Check order frequency limits
   */
  private checkOrderFrequency(accountId: string, limits: RiskLimits): RiskViolation | null {
    const now = Date.now();
    const orderTimes = this.orderCounts.get(accountId) || [];
    
    // Remove orders older than 1 second
    const recentOrders = orderTimes.filter(time => now - time < 1000);
    
    if (recentOrders.length >= limits.maxOrdersPerSecond) {
      return {
        type: 'order_frequency_exceeded',
        severity: 'warning',
        message: 'Order frequency limit exceeded',
        currentValue: recentOrders.length,
        limitValue: limits.maxOrdersPerSecond,
        action: 'warn'
      };
    }

    // Update order count
    recentOrders.push(now);
    this.orderCounts.set(accountId, recentOrders);

    return null;
  }

  /**
   * Check trading hours
   */
  private checkTradingHours(limits: RiskLimits): RiskViolation | null {
    const now = new Date();
    const currentHour = now.getHours();
    const startHour = parseInt(limits.tradingHours.start.split(':')[0]);
    const endHour = parseInt(limits.tradingHours.end.split(':')[0]);

    if (currentHour < startHour || currentHour >= endHour) {
      return {
        type: 'outside_trading_hours',
        severity: 'warning',
        message: 'Trading outside allowed hours',
        currentValue: currentHour,
        limitValue: startHour,
        action: 'warn'
      };
    }

    return null;
  }

  /**
   * Calculate overall risk level
   */
  private calculateRiskLevel(violations: RiskViolation[]): 'low' | 'medium' | 'high' | 'critical' {
    if (violations.some(v => v.severity === 'critical')) return 'critical';
    if (violations.some(v => v.severity === 'error')) return 'high';
    if (violations.some(v => v.severity === 'warning')) return 'medium';
    return 'low';
  }

  /**
   * Generate recommendations based on violations
   */
  private generateRecommendations(violations: RiskViolation[]): string[] {
    const recommendations = [];

    for (const violation of violations) {
      switch (violation.type) {
        case 'position_size_exceeded':
          recommendations.push('Consider reducing position size or closing existing positions');
          break;
        case 'leverage_exceeded':
          recommendations.push('Reduce leverage by closing positions or adding margin');
          break;
        case 'daily_loss_exceeded':
          recommendations.push('Stop trading for today to prevent further losses');
          break;
        case 'order_frequency_exceeded':
          recommendations.push('Reduce order frequency to comply with limits');
          break;
        default:
          recommendations.push(`Address ${violation.type} violation`);
      }
    }

    return recommendations;
  }

  /**
   * Check for automatic circuit breaker triggers
   */
  private async checkCircuitBreakers(accountId: string, metrics: RiskMetrics): Promise<void> {
    const limits = this.riskLimits.get(accountId);
    if (!limits) return;

    // Trigger circuit breaker for excessive daily loss
    if (metrics.dailyPnL < -limits.maxDailyLoss * 1.5) {
      await this.triggerCircuitBreaker(accountId, 'Excessive daily loss');
    }

    // Trigger circuit breaker for excessive drawdown
    if (metrics.currentDrawdown > limits.maxDrawdown * 1.2) {
      await this.triggerCircuitBreaker(accountId, 'Excessive drawdown');
    }

    // Trigger circuit breaker for high risk score
    if (metrics.riskScore > 90) {
      await this.triggerCircuitBreaker(accountId, 'High risk score');
    }
  }

  /**
   * Start periodic risk monitoring
   */
  private startRiskMonitoring(): void {
    setInterval(() => {
      this.performPeriodicRiskChecks();
    }, 5000); // Every 5 seconds
  }

  /**
   * Perform periodic risk checks
   */
  private async performPeriodicRiskChecks(): Promise<void> {
    for (const [accountId, metrics] of this.riskMetrics) {
      try {
        await this.checkCircuitBreakers(accountId, metrics);
      } catch (error) {
        this.logger.error(`Error in periodic risk check for ${accountId}:`, error);
      }
    }
  }
}
