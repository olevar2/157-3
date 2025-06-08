import axios from 'axios';

interface PositionSizingParams {
  accountBalance: number;
  riskPercentage: number;
  stopLossDistance: number;
  maxPositionSize: number;
  volatilityAdjustment: boolean;
}

interface PositionSizeResult {
  recommendedSize: number;
  maxSize: number;
  riskAmount: number;
  confidenceLevel: number;
  adjustments: string[];
}

export class DayTradingPositionSizer {
  private pythonEngineUrl: string;

  constructor() {
    this.pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
    console.log("DayTradingPositionSizer initialized");
  }

  async calculatePositionSize(params: PositionSizingParams, symbol: string): Promise<PositionSizeResult> {
    try {
      console.log("Calculating day trading position size for:", symbol);

      const adjustments: string[] = [];
      let recommendedSize = 0;
      let confidenceLevel = 0.8;

      // Basic position sizing using risk percentage
      const riskAmount = params.accountBalance * (params.riskPercentage / 100);
      const basicSize = Math.floor(riskAmount / params.stopLossDistance);

      // Get volatility data from Python engine
      const volatilityData = await this.getPythonVolatilityData(symbol);
      
      if (volatilityData && params.volatilityAdjustment) {
        const volatilityAdjustment = this.calculateVolatilityAdjustment(volatilityData.volatility);
        recommendedSize = Math.floor(basicSize * volatilityAdjustment);
        adjustments.push(`Volatility adjustment: ${(volatilityAdjustment * 100).toFixed(1)}%`);
      } else {
        recommendedSize = basicSize;
      }

      // Apply maximum position size limit
      const maxSize = Math.min(params.maxPositionSize, params.accountBalance * 0.1);
      if (recommendedSize > maxSize) {
        recommendedSize = maxSize;
        adjustments.push(`Capped at maximum position size: ${maxSize}`);
      }

      // Get market sentiment from Python AI
      const sentimentData = await this.getPythonSentimentData(symbol);
      if (sentimentData) {
        if (sentimentData.sentiment < -0.5) {
          recommendedSize *= 0.8;
          confidenceLevel *= 0.9;
          adjustments.push('Negative sentiment detected - reducing position size');
        } else if (sentimentData.sentiment > 0.5) {
          recommendedSize *= 1.1;
          confidenceLevel *= 1.05;
          adjustments.push('Positive sentiment detected - increasing position size');
        }
      }

      // Final safety check
      if (recommendedSize < 1000) {
        recommendedSize = 0;
        adjustments.push('Position size too small - trade rejected');
      }

      return {
        recommendedSize: Math.floor(recommendedSize),
        maxSize,
        riskAmount,
        confidenceLevel: Math.min(confidenceLevel, 1.0),
        adjustments
      };

    } catch (error) {
      console.error('Position sizing calculation error:', error);
      return {
        recommendedSize: 0,
        maxSize: 0,
        riskAmount: 0,
        confidenceLevel: 0,
        adjustments: ['Error in position sizing calculation']
      };
    }
  }

  private calculateVolatilityAdjustment(volatility: number): number {
    // Reduce position size for high volatility
    if (volatility > 0.05) return 0.6; // High volatility
    if (volatility > 0.03) return 0.8; // Medium volatility
    if (volatility > 0.01) return 1.0; // Normal volatility
    return 1.2; // Low volatility - can increase size slightly
  }

  private async getPythonVolatilityData(symbol: string): Promise<any> {
    try {
      const response = await axios.get(`${this.pythonEngineUrl}/api/v1/volatility/${symbol}`);
      return response.data.success ? response.data.data : null;
    } catch (error) {
      console.error('Failed to get Python volatility data:', error);
      return null;
    }
  }

  private async getPythonSentimentData(symbol: string): Promise<any> {
    try {
      const response = await axios.get(`${this.pythonEngineUrl}/api/v1/sentiment/${symbol}`);
      return response.data.success ? response.data.data : null;
    } catch (error) {
      console.error('Failed to get Python sentiment data:', error);
      return null;
    }
  }

  public calculateRiskReward(entryPrice: number, stopLoss: number, takeProfit: number): number {
    const risk = Math.abs(entryPrice - stopLoss);
    const reward = Math.abs(takeProfit - entryPrice);
    return risk > 0 ? reward / risk : 0;
  }
}
