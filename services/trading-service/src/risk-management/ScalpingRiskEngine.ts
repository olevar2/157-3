import axios from 'axios';
import { EventEmitter } from 'events';

interface ScalpingRiskParams {
  maxPositionSize: number;
  maxDailyLoss: number;
  maxOpenPositions: number;
  minTimeBetweenTrades: number; // milliseconds
  volatilityThreshold: number;
  drawdownLimit: number;
}

interface TradeRiskAssessment {
  approved: boolean;
  riskScore: number;
  maxSize: number;
  warnings: string[];
  rejectionReason?: string;
}

interface PositionMetrics {
  totalExposure: number;
  dailyPnL: number;
  openPositions: number;
  recentTrades: number;
  drawdown: number;
}

export class ScalpingRiskEngine extends EventEmitter {
  private pythonEngineUrl: string;
  private riskParams: ScalpingRiskParams;
  private lastTradeTime: number = 0;
  private dailyTrades: Map<string, number> = new Map();

  constructor(params?: Partial<ScalpingRiskParams>) {
    super();
    
    this.pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
    
    this.riskParams = {
      maxPositionSize: 100000,
      maxDailyLoss: 5000,
      maxOpenPositions: 10,
      minTimeBetweenTrades: 1000, // 1 second
      volatilityThreshold: 0.05,
      drawdownLimit: 0.1,
      ...params
    };

    console.log("ScalpingRiskEngine initialized with params:", this.riskParams);
  }

  async assessRisk(trade: any, account: any): Promise<TradeRiskAssessment> {
    try {
      console.log("Assessing scalping risk for trade:", trade.symbol);

      const warnings: string[] = [];
      let riskScore = 0;
      let maxSize = this.riskParams.maxPositionSize;

      // Get current position metrics
      const metrics = await this.getPositionMetrics(account.id);
      
      // Time-based risk check
      const currentTime = Date.now();
      const timeSinceLastTrade = currentTime - this.lastTradeTime;
      
      if (timeSinceLastTrade < this.riskParams.minTimeBetweenTrades) {
        return {
          approved: false,
          riskScore: 100,
          maxSize: 0,
          warnings: [],
          rejectionReason: 'Minimum time between trades not met'
        };
      }

      // Position count check
      if (metrics.openPositions >= this.riskParams.maxOpenPositions) {
        return {
          approved: false,
          riskScore: 100,
          maxSize: 0,
          warnings: [],
          rejectionReason: 'Maximum open positions limit reached'
        };
      }

      // Daily loss check
      if (Math.abs(metrics.dailyPnL) >= this.riskParams.maxDailyLoss) {
        return {
          approved: false,
          riskScore: 100,
          maxSize: 0,
          warnings: [],
          rejectionReason: 'Daily loss limit exceeded'
        };
      }

      // Drawdown check
      if (metrics.drawdown >= this.riskParams.drawdownLimit) {
        riskScore += 30;
        maxSize *= 0.5;
        warnings.push('High drawdown detected - reducing position size');
      }

      // Get volatility assessment from Python engine
      const volatilityData = await this.getPythonVolatilityAssessment(trade.symbol);
      if (volatilityData && volatilityData.volatility > this.riskParams.volatilityThreshold) {
        riskScore += 25;
        maxSize *= 0.7;
        warnings.push('High volatility detected - reducing position size');
      }

      // Position size check
      if (trade.size > maxSize) {
        return {
          approved: false,
          riskScore: 100,
          maxSize,
          warnings,
          rejectionReason: `Position size ${trade.size} exceeds maximum allowed ${maxSize}`
        };
      }

      // Final risk assessment
      const approved = riskScore < 80;
      
      if (approved) {
        this.lastTradeTime = currentTime;
        this.emit('riskAssessed', { trade, approved, riskScore });
      }

      return {
        approved,
        riskScore,
        maxSize,
        warnings,
        rejectionReason: approved ? undefined : 'Risk score too high'
      };

    } catch (error) {
      console.error('Risk assessment error:', error);
      return {
        approved: false,
        riskScore: 100,
        maxSize: 0,
        warnings: [],
        rejectionReason: 'Risk assessment system error'
      };
    }
  }

  private async getPositionMetrics(accountId: string): Promise<PositionMetrics> {
    try {
      // This would normally query the database or cache
      // For now, return mock metrics
      return {
        totalExposure: 50000,
        dailyPnL: -200,
        openPositions: 3,
        recentTrades: 5,
        drawdown: 0.03
      };
    } catch (error) {
      console.error('Failed to get position metrics:', error);
      return {
        totalExposure: 0,
        dailyPnL: 0,
        openPositions: 0,
        recentTrades: 0,
        drawdown: 0
      };
    }
  }

  private async getPythonVolatilityAssessment(symbol: string): Promise<any> {
    try {
      const response = await axios.get(`${this.pythonEngineUrl}/api/v1/risk/volatility/${symbol}`);
      return response.data.success ? response.data.data : null;
    } catch (error) {
      console.error('Failed to get Python volatility assessment:', error);
      return null;
    }
  }

  public updateRiskParams(newParams: Partial<ScalpingRiskParams>): void {
    this.riskParams = { ...this.riskParams, ...newParams };
    console.log('Risk parameters updated:', this.riskParams);
  }

  public getRiskParams(): ScalpingRiskParams {
    return { ...this.riskParams };
  }

  public resetDailyCounters(): void {
    this.dailyTrades.clear();
    console.log('Daily risk counters reset');
  }
}
