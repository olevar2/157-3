import axios from 'axios';
import { EventEmitter } from 'events';

/**
 * Rapid Drawdown Protection System
 * Real-time monitoring and protection against sudden account value drops
 */

export interface DrawdownMetrics {
  currentDrawdown: number;
  maxDrawdown: number;
  peakEquity: number;
  currentEquity: number;
  drawdownStartTime?: Date;
  consecutiveLosses: number;
  rapidDrawdownThreshold: number;
}

export interface DrawdownConfig {
  maxDrawdownPercent: number;          // e.g., 5% = 0.05
  rapidDrawdownThreshold: number;      // e.g., 2% in short time = 0.02
  rapidTimeFrameMinutes: number;       // e.g., 15 minutes
  consecutiveLossLimit: number;        // e.g., 5 consecutive losses
  pauseTradingOnTrigger: boolean;      // Halt all trading when triggered
  alertThresholds: number[];           // e.g., [0.01, 0.02, 0.03] for 1%, 2%, 3%
}

export interface AccountPosition {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
}

export interface EquitySnapshot {
  timestamp: Date;
  totalEquity: number;
  realizedPnl: number;
  unrealizedPnl: number;
  positions: AccountPosition[];
}

export class RapidDrawdownProtection extends EventEmitter {
  private config: DrawdownConfig;
  private pythonEngineUrl: string;
  private currentMetrics: DrawdownMetrics;
  private equityHistory: EquitySnapshot[] = [];
  private tradingPaused = false;
  private lastAlertLevel = -1;

  constructor(config: DrawdownConfig) {
    super();
    this.config = config;
    this.pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
    
    this.currentMetrics = {
      currentDrawdown: 0,
      maxDrawdown: 0,
      peakEquity: 0,
      currentEquity: 0,
      consecutiveLosses: 0,
      rapidDrawdownThreshold: config.rapidDrawdownThreshold
    };

    console.log('RapidDrawdownProtection initialized with config:', config);
  }

  /**
   * Update equity and check for drawdown conditions
   */
  public async updateEquity(snapshot: EquitySnapshot): Promise<void> {
    this.equityHistory.push(snapshot);
    
    // Keep only recent history for rapid drawdown calculation
    const cutoffTime = new Date(snapshot.timestamp.getTime() - this.config.rapidTimeFrameMinutes * 60 * 1000);
    this.equityHistory = this.equityHistory.filter(s => s.timestamp > cutoffTime);

    // Update peak equity
    if (snapshot.totalEquity > this.currentMetrics.peakEquity) {
      this.currentMetrics.peakEquity = snapshot.totalEquity;
      this.currentMetrics.drawdownStartTime = undefined;
    }

    // Calculate current drawdown
    this.currentMetrics.currentEquity = snapshot.totalEquity;
    this.currentMetrics.currentDrawdown = (this.currentMetrics.peakEquity - snapshot.totalEquity) / this.currentMetrics.peakEquity;

    // Update max drawdown
    if (this.currentMetrics.currentDrawdown > this.currentMetrics.maxDrawdown) {
      this.currentMetrics.maxDrawdown = this.currentMetrics.currentDrawdown;
      if (!this.currentMetrics.drawdownStartTime) {
        this.currentMetrics.drawdownStartTime = snapshot.timestamp;
      }
    }

    // Check for rapid drawdown
    await this.checkRapidDrawdown();

    // Check alert thresholds
    this.checkAlertThresholds();

    // Check if trading should be paused
    this.checkTradingPause();

    console.log(`Drawdown update: Current=${(this.currentMetrics.currentDrawdown * 100).toFixed(2)}%, Max=${(this.currentMetrics.maxDrawdown * 100).toFixed(2)}%, Equity=${snapshot.totalEquity}`);
  }

  /**
   * Check for rapid drawdown within the specified time frame
   */
  private async checkRapidDrawdown(): Promise<void> {
    if (this.equityHistory.length < 2) return;

    const oldest = this.equityHistory[0];
    const newest = this.equityHistory[this.equityHistory.length - 1];
    
    const rapidDrawdown = (oldest.totalEquity - newest.totalEquity) / oldest.totalEquity;

    if (rapidDrawdown >= this.config.rapidDrawdownThreshold) {
      const timeSpan = (newest.timestamp.getTime() - oldest.timestamp.getTime()) / 60000; // minutes
      
      console.warn(`⚠️  RAPID DRAWDOWN DETECTED: ${(rapidDrawdown * 100).toFixed(2)}% in ${timeSpan.toFixed(1)} minutes`);
      
      this.emit('rapidDrawdownDetected', {
        drawdownPercent: rapidDrawdown,
        timeSpanMinutes: timeSpan,
        oldestEquity: oldest.totalEquity,
        newestEquity: newest.totalEquity,
        metrics: this.currentMetrics
      });

      // Notify Python engines about rapid drawdown
      await this.notifyPythonEngines('rapid_drawdown', {
        drawdown: rapidDrawdown,
        timeSpan: timeSpan,
        currentMetrics: this.currentMetrics
      });

      if (this.config.pauseTradingOnTrigger) {
        this.pauseTrading('Rapid drawdown protection triggered');
      }
    }
  }

  /**
   * Check alert thresholds and emit appropriate events
   */
  private checkAlertThresholds(): void {
    for (let i = 0; i < this.config.alertThresholds.length; i++) {
      const threshold = this.config.alertThresholds[i];
      if (this.currentMetrics.currentDrawdown >= threshold && i > this.lastAlertLevel) {
        this.lastAlertLevel = i;
        
        console.warn(`🚨 Drawdown Alert Level ${i + 1}: ${(threshold * 100).toFixed(1)}% reached`);
        
        this.emit('drawdownAlert', {
          level: i + 1,
          threshold: threshold,
          currentDrawdown: this.currentMetrics.currentDrawdown,
          metrics: this.currentMetrics
        });
        break;
      }
    }
  }

  /**
   * Check if trading should be paused based on drawdown limits
   */
  private checkTradingPause(): void {
    if (this.currentMetrics.currentDrawdown >= this.config.maxDrawdownPercent && !this.tradingPaused) {
      this.pauseTrading(`Maximum drawdown limit reached: ${(this.config.maxDrawdownPercent * 100).toFixed(1)}%`);
    }
  }

  /**
   * Pause all trading activities
   */
  public pauseTrading(reason: string): void {
    if (this.tradingPaused) return;

    this.tradingPaused = true;
    console.error(`🛑 TRADING PAUSED: ${reason}`);
    
    this.emit('tradingPaused', {
      reason,
      timestamp: new Date(),
      metrics: this.currentMetrics
    });
  }

  /**
   * Resume trading activities
   */
  public resumeTrading(reason: string): void {
    if (!this.tradingPaused) return;

    this.tradingPaused = false;
    this.lastAlertLevel = -1; // Reset alert level
    
    console.log(`✅ TRADING RESUMED: ${reason}`);
    
    this.emit('tradingResumed', {
      reason,
      timestamp: new Date(),
      metrics: this.currentMetrics
    });
  }

  /**
   * Update consecutive losses count
   */
  public recordLoss(): void {
    this.currentMetrics.consecutiveLosses++;
    
    if (this.currentMetrics.consecutiveLosses >= this.config.consecutiveLossLimit) {
      console.warn(`⚠️  Consecutive loss limit reached: ${this.currentMetrics.consecutiveLosses} losses`);
      
      this.emit('consecutiveLossLimitReached', {
        consecutiveLosses: this.currentMetrics.consecutiveLosses,
        limit: this.config.consecutiveLossLimit,
        metrics: this.currentMetrics
      });

      if (this.config.pauseTradingOnTrigger) {
        this.pauseTrading(`Consecutive loss limit reached: ${this.currentMetrics.consecutiveLosses} losses`);
      }
    }
  }

  /**
   * Reset consecutive losses count (call on winning trade)
   */
  public resetConsecutiveLosses(): void {
    this.currentMetrics.consecutiveLosses = 0;
  }

  /**
   * Notify Python engines about drawdown events
   */
  private async notifyPythonEngines(eventType: string, data: any): Promise<void> {
    try {
      await axios.post(`${this.pythonEngineUrl}/api/risk/drawdown-event`, {
        eventType,
        data,
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      console.error(`Failed to notify Python engines about ${eventType}:`, error.message);
    }
  }

  /**
   * Get current drawdown metrics
   */
  public getMetrics(): DrawdownMetrics {
    return { ...this.currentMetrics };
  }

  /**
   * Check if trading is currently paused
   */
  public isTradingPaused(): boolean {
    return this.tradingPaused;
  }

  /**
   * Reset all metrics (use carefully, typically at start of new trading session)
   */
  public resetMetrics(initialEquity: number): void {
    this.currentMetrics = {
      currentDrawdown: 0,
      maxDrawdown: 0,
      peakEquity: initialEquity,
      currentEquity: initialEquity,
      consecutiveLosses: 0,
      rapidDrawdownThreshold: this.config.rapidDrawdownThreshold
    };
    
    this.equityHistory = [];
    this.lastAlertLevel = -1;
    
    console.log(`Drawdown metrics reset with initial equity: ${initialEquity}`);
  }

  /**
   * Update configuration
   */
  public updateConfig(newConfig: Partial<DrawdownConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log('Drawdown protection config updated:', newConfig);
  }
}
