// Risk Analysis Engine - Portfolio risk metrics and position sizing
// Provides comprehensive risk assessment and portfolio optimization
// Bridge to Python AI risk models for humanitarian forex trading platform

import { Logger } from 'winston';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import { EventEmitter } from 'events';
import { mean, standardDeviation, variance } from 'simple-statistics';

// Communication interfaces for Python engine integration
export interface PythonEngineInterface {
  sendCommand(command: string, data: any): Promise<any>;
  isConnected(): boolean;
  disconnect(): Promise<void>;
}

export interface Platform3EngineConnection {
  initialized: boolean;
  pythonProcess?: ChildProcess;
  communicationQueue: Map<string, any>;
  eventEmitter: EventEmitter;
}

// Python Risk Analysis Integration Interface
export interface PythonRiskInterface {
  calculatePortfolioRisk(positions: any[], marketData: any): Promise<PortfolioRisk>;
  assessPositionRisk(position: any, marketData: any): Promise<PositionRisk>;
  optimizePositionSizing(signals: any[], accountBalance: number): Promise<PositionSizeRecommendation[]>;
}

export interface PositionSizeRecommendation {
  symbol: string;
  recommendedSize: number;
  maxRisk: number;
  stopLoss: number;
  takeProfit: number;
  confidence: number;
}

export interface RiskAnalysisResult {
  timestamp: number;
  portfolioRisk: PortfolioRisk;
  positionRisks: PositionRisk[];
  recommendations: RiskRecommendation[];
  metrics: RiskMetrics;
  alerts: RiskAlert[];
}

export interface PortfolioRisk {
  totalExposure: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinRatio: number;
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number;
  correlationRisk: number;
  concentrationRisk: number;
  riskScore: number; // 0-100
}

export interface PositionRisk {
  symbol: string;
  positionSize: number;
  marketValue: number;
  unrealizedPnL: number;
  riskAmount: number;
  riskPercentage: number;
  stopLoss?: number;
  takeProfit?: number;
  riskRewardRatio?: number;
  positionScore: number; // 0-100
  recommendations: string[];
}

export interface RiskRecommendation {
  type: 'reduce_position' | 'add_stop_loss' | 'diversify' | 'hedge' | 'close_position' | 'rebalance';
  priority: 'low' | 'medium' | 'high' | 'critical';
  symbol?: string;
  description: string;
  action: string;
  impact: string;
  confidence: number;
}

export interface RiskMetrics {
  portfolioValue: number;
  totalRisk: number;
  riskPercentage: number;
  diversificationRatio: number;
  leverageRatio: number;
  marginUtilization: number;
  correlationMatrix: { [key: string]: { [key: string]: number } };
  sectorExposure: { [sector: string]: number };
}

export interface RiskAlert {
  level: 'info' | 'warning' | 'danger' | 'critical';
  type: string;
  message: string;
  symbol?: string;
  value: number;
  threshold: number;
  timestamp: number;
}

export interface Position {
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  stopLoss?: number;
  takeProfit?: number;
}

export class RiskAnalysisEngine {
  private logger: Logger;
  private ready: boolean = false;
  private pythonEngine: Platform3EngineConnection;
  private pythonInterface: PythonEngineInterface;
  private riskLimits = {
    maxPositionRisk: 0.02,    // 2% per position
    maxPortfolioRisk: 0.20,   // 20% total portfolio
    maxDrawdown: 0.15,        // 15% max drawdown
    maxLeverage: 10,          // 10:1 leverage
    maxCorrelation: 0.7,      // 70% correlation limit
    minDiversification: 0.3   // 30% minimum diversification
  };

  constructor(logger: Logger) {
    this.logger = logger;
    this.pythonEngine = {
      initialized: false,
      communicationQueue: new Map(),
      eventEmitter: new EventEmitter()
    };
    this.pythonInterface = this.createPythonInterface();
  }

  private createPythonInterface(): PythonEngineInterface {
    return {
      sendCommand: async (command: string, data: any) => {
        return this.sendToPythonEngine(command, data);
      },
      isConnected: () => {
        return this.pythonEngine.initialized && this.pythonEngine.pythonProcess !== undefined;
      },
      disconnect: async () => {
        await this.disconnectPythonEngine();
      }
    };
  }

  private async sendToPythonEngine(command: string, data: any): Promise<any> {
    if (!this.pythonEngine.initialized) {
      throw new Error('Python risk analysis engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const requestId = Math.random().toString(36).substr(2, 9);
      const message = JSON.stringify({
        id: requestId,
        command,
        data,
        timestamp: Date.now(),
        engine_type: 'risk_analysis'
      });

      // Set up response handler
      const timeout = setTimeout(() => {
        this.pythonEngine.communicationQueue.delete(requestId);
        reject(new Error(`Python risk engine timeout for command: ${command}`));
      }, 30000); // 30 second timeout

      this.pythonEngine.communicationQueue.set(requestId, {
        resolve,
        reject,
        timeout
      });

      // Send to Python process
      if (this.pythonEngine.pythonProcess && this.pythonEngine.pythonProcess.stdin) {
        this.pythonEngine.pythonProcess.stdin.write(message + '\n');
      } else {
        clearTimeout(timeout);
        this.pythonEngine.communicationQueue.delete(requestId);
        reject(new Error('Python risk process not available'));
      }
    });
  }

  private async disconnectPythonEngine(): Promise<void> {
    if (this.pythonEngine.pythonProcess) {
      this.pythonEngine.pythonProcess.kill();
      this.pythonEngine.pythonProcess = undefined;
    }
    this.pythonEngine.initialized = false;
    this.pythonEngine.communicationQueue.clear();
  }

  async initialize(): Promise<void> {
    this.logger.info('🚀 Initializing Risk Analysis Engine for humanitarian trading...');
    
    try {
      // Initialize Python risk analysis engine connection
      await this.initializePythonEngine();
      
      // Test risk calculations
      const testPositions = this.generateTestPositions();
      const testRisk = await this.analyzePortfolio(testPositions, 100000);
      
      if (testRisk.portfolioRisk.riskScore >= 0) {
        // Test Python risk engine integration
        await this.testPythonRiskEngineIntegration();
        
        this.ready = true;
        this.logger.info('✅ Risk Analysis Engine initialized with Python AI bridge');
      } else {
        throw new Error('Risk calculation test failed');
      }
    } catch (error) {
      this.logger.error('❌ Risk Analysis Engine initialization failed:', error);
      throw error;
    }
  }

  private async initializePythonEngine(): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonScriptPath = path.join(__dirname, '../../ai-platform/coordination/engine/platform3_engine.py');
      
      this.logger.info(`Starting Python risk analysis engine: ${pythonScriptPath}`);
      
      const pythonProcess = spawn('python', [pythonScriptPath, '--mode=risk-analysis'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.pythonEngine.pythonProcess = pythonProcess;

      // Handle Python process output
      pythonProcess.stdout?.on('data', (data) => {
        const lines = data.toString().split('\n').filter((line: string) => line.trim());
        
        for (const line of lines) {
          try {
            const response = JSON.parse(line);
            this.handlePythonResponse(response);
          } catch (error) {
            this.logger.debug('Python risk output:', line);
          }
        }
      });

      pythonProcess.stderr?.on('data', (data) => {
        this.logger.error('Python risk engine error:', data.toString());
      });

      pythonProcess.on('close', (code) => {
        this.logger.warn(`Python risk engine process closed with code ${code}`);
        this.pythonEngine.initialized = false;
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python risk engine process error:', error);
        reject(error);
      });

      // Wait for initialization confirmation
      setTimeout(() => {
        if (pythonProcess.pid) {
          this.pythonEngine.initialized = true;
          resolve();
        } else {
          reject(new Error('Python risk engine failed to start'));
        }
      }, 3000);
    });
  }

  private handlePythonResponse(response: any): void {
    if (response.id && this.pythonEngine.communicationQueue.has(response.id)) {
      const { resolve, reject, timeout } = this.pythonEngine.communicationQueue.get(response.id);
      clearTimeout(timeout);
      this.pythonEngine.communicationQueue.delete(response.id);

      if (response.error) {
        reject(new Error(response.error));
      } else {
        resolve(response.result);
      }
    }
  }

  private async testPythonRiskEngineIntegration(): Promise<void> {
    this.logger.info('🧪 Testing Python risk analysis engine integration...');
    
    try {
      // Test basic communication
      const pingResult = await this.pythonInterface.sendCommand('ping', { 
        message: 'risk_integration_test',
        engine_type: 'risk_analysis'
      });
      if (pingResult.status !== 'pong') {
        throw new Error('Python risk engine ping test failed');
      }

      // Test risk calculation
      const testPositions = this.generateTestPositions().slice(0, 3);
      const riskResult = await this.pythonInterface.sendCommand('calculate_ai_portfolio_risk', {
        positions: testPositions,
        balance: 100000,
        risk_limits: this.riskLimits
      });

      if (!riskResult || typeof riskResult.total_risk !== 'number') {
        throw new Error('Python risk calculation test failed');
      }

      this.logger.info('✅ Python risk analysis engine integration test passed');
    } catch (error) {
      this.logger.error('❌ Python risk engine integration test failed:', error);
      throw error;
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  async analyzePortfolio(positions: Position[], accountBalance: number): Promise<RiskAnalysisResult> {
    if (!this.ready) {
      throw new Error('Risk Analysis Engine not initialized');
    }

    this.logger.debug(`Analyzing portfolio risk for ${positions.length} positions with balance ${accountBalance}`);

    // Calculate individual position risks
    const positionRisks = positions.map(position => this.analyzePositionRisk(position, accountBalance));

    // Calculate portfolio-level risk metrics
    const portfolioRisk = this.calculatePortfolioRisk(positions, positionRisks, accountBalance);

    // Generate risk metrics
    const metrics = this.calculateRiskMetrics(positions, accountBalance);

    // Generate recommendations
    const recommendations = this.generateRiskRecommendations(portfolioRisk, positionRisks, metrics);

    // Generate risk alerts
    const alerts = this.generateRiskAlerts(portfolioRisk, positionRisks, metrics);

    return {
      timestamp: Date.now(),
      portfolioRisk,
      positionRisks,
      recommendations,
      metrics,
      alerts
    };
  }

  private analyzePositionRisk(position: Position, accountBalance: number): PositionRisk {
    const marketValue = Math.abs(position.marketValue);
    const riskAmount = position.stopLoss 
      ? Math.abs(position.currentPrice - position.stopLoss) * Math.abs(position.quantity)
      : marketValue * 0.02; // Default 2% risk if no stop loss

    const riskPercentage = riskAmount / accountBalance;
    
    // Calculate risk-reward ratio
    let riskRewardRatio: number | undefined;
    if (position.takeProfit && position.stopLoss) {
      const reward = Math.abs(position.takeProfit - position.currentPrice) * Math.abs(position.quantity);
      riskRewardRatio = reward / riskAmount;
    }

    // Calculate position score (0-100, higher is better)
    let positionScore = 100;
    
    // Penalize high risk
    if (riskPercentage > this.riskLimits.maxPositionRisk) {
      positionScore -= (riskPercentage - this.riskLimits.maxPositionRisk) * 1000;
    }
    
    // Reward good risk-reward ratio
    if (riskRewardRatio && riskRewardRatio >= 2) {
      positionScore += 10;
    } else if (riskRewardRatio && riskRewardRatio < 1) {
      positionScore -= 20;
    }
    
    // Penalize missing stop loss
    if (!position.stopLoss) {
      positionScore -= 30;
    }

    positionScore = Math.max(0, Math.min(100, positionScore));

    // Generate position-specific recommendations
    const recommendations: string[] = [];
    
    if (riskPercentage > this.riskLimits.maxPositionRisk) {
      recommendations.push('Reduce position size to limit risk');
    }
    
    if (!position.stopLoss) {
      recommendations.push('Add stop loss to limit downside risk');
    }
    
    if (riskRewardRatio && riskRewardRatio < 1.5) {
      recommendations.push('Consider adjusting take profit for better risk-reward ratio');
    }

    return {
      symbol: position.symbol,
      positionSize: position.quantity,
      marketValue,
      unrealizedPnL: position.unrealizedPnL,
      riskAmount,
      riskPercentage,
      stopLoss: position.stopLoss,
      takeProfit: position.takeProfit,
      riskRewardRatio,
      positionScore,
      recommendations
    };
  }

  private calculatePortfolioRisk(positions: Position[], positionRisks: PositionRisk[], accountBalance: number): PortfolioRisk {
    const totalExposure = positions.reduce((sum, pos) => sum + Math.abs(pos.marketValue), 0);
    const totalRisk = positionRisks.reduce((sum, risk) => sum + risk.riskAmount, 0);
    
    // Calculate Value at Risk (simplified Monte Carlo simulation)
    const returns = this.simulatePortfolioReturns(positions, 1000);
    returns.sort((a, b) => a - b);
    
    const var95 = Math.abs(returns[Math.floor(returns.length * 0.05)] * accountBalance);
    const var99 = Math.abs(returns[Math.floor(returns.length * 0.01)] * accountBalance);
    
    // Expected Shortfall (average of worst 5% scenarios)
    const worstReturns = returns.slice(0, Math.floor(returns.length * 0.05));
    const expectedShortfall = Math.abs(mean(worstReturns) * accountBalance);

    // Calculate Sharpe ratio (simplified)
    const avgReturn = mean(returns);
    const returnStd = standardDeviation(returns);
    const sharpeRatio = returnStd > 0 ? avgReturn / returnStd : 0;

    // Calculate Sortino ratio (downside deviation)
    const negativeReturns = returns.filter(r => r < 0);
    const downsideDeviation = negativeReturns.length > 0 ? standardDeviation(negativeReturns) : returnStd;
    const sortinRatio = downsideDeviation > 0 ? avgReturn / downsideDeviation : 0;

    // Calculate correlation risk
    const correlationRisk = this.calculateCorrelationRisk(positions);

    // Calculate concentration risk
    const concentrationRisk = this.calculateConcentrationRisk(positions, totalExposure);

    // Calculate max drawdown (simplified)
    const maxDrawdown = Math.max(...returns.map(r => Math.abs(Math.min(0, r))));

    // Overall risk score (0-100, lower is better)
    let riskScore = 0;
    
    riskScore += (totalRisk / accountBalance) * 100; // Risk percentage
    riskScore += correlationRisk * 50; // Correlation penalty
    riskScore += concentrationRisk * 30; // Concentration penalty
    riskScore += maxDrawdown * 100; // Drawdown penalty
    
    riskScore = Math.min(100, riskScore);

    return {
      totalExposure,
      maxDrawdown,
      sharpeRatio,
      sortinRatio,
      var95,
      var99,
      expectedShortfall,
      correlationRisk,
      concentrationRisk,
      riskScore
    };
  }

  private simulatePortfolioReturns(positions: Position[], simulations: number): number[] {
    const returns: number[] = [];
    
    for (let i = 0; i < simulations; i++) {
      let portfolioReturn = 0;
      
      for (const position of positions) {
        // Simulate price movement (normal distribution)
        const volatility = 0.02; // 2% daily volatility
        const priceChange = this.randomNormal(0, volatility);
        const positionReturn = priceChange * (position.marketValue / position.currentPrice);
        portfolioReturn += positionReturn;
      }
      
      returns.push(portfolioReturn);
    }
    
    return returns;
  }

  private randomNormal(mean: number, std: number): number {
    // Box-Muller transformation for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * std + mean;
  }

  private calculateCorrelationRisk(positions: Position[]): number {
    // Simplified correlation calculation based on currency pairs
    const currencies = new Set<string>();
    
    for (const position of positions) {
      const symbol = position.symbol;
      if (symbol.length === 6) { // Forex pair
        currencies.add(symbol.substring(0, 3)); // Base currency
        currencies.add(symbol.substring(3, 6)); // Quote currency
      }
    }
    
    // Higher correlation risk if many positions share the same currencies
    const uniqueCurrencies = currencies.size;
    const totalPositions = positions.length;
    
    if (totalPositions === 0) return 0;
    
    const diversificationRatio = uniqueCurrencies / (totalPositions * 2); // Each pair has 2 currencies
    return Math.max(0, 1 - diversificationRatio);
  }

  private calculateConcentrationRisk(positions: Position[], totalExposure: number): number {
    if (totalExposure === 0) return 0;
    
    // Calculate Herfindahl-Hirschman Index for concentration
    let hhi = 0;
    
    for (const position of positions) {
      const weight = Math.abs(position.marketValue) / totalExposure;
      hhi += weight * weight;
    }
    
    // Normalize HHI (0 = perfectly diversified, 1 = completely concentrated)
    const normalizedHHI = (hhi - 1/positions.length) / (1 - 1/positions.length);
    return Math.max(0, normalizedHHI);
  }

  private calculateRiskMetrics(positions: Position[], accountBalance: number): RiskMetrics {
    const portfolioValue = accountBalance + positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
    const totalExposure = positions.reduce((sum, pos) => sum + Math.abs(pos.marketValue), 0);
    const totalRisk = positions.reduce((sum, pos) => {
      const riskAmount = pos.stopLoss 
        ? Math.abs(pos.currentPrice - pos.stopLoss) * Math.abs(pos.quantity)
        : Math.abs(pos.marketValue) * 0.02;
      return sum + riskAmount;
    }, 0);

    const riskPercentage = totalRisk / accountBalance;
    const leverageRatio = totalExposure / accountBalance;
    const marginUtilization = Math.min(1, leverageRatio / 10); // Assuming 10:1 max leverage

    // Calculate diversification ratio
    const diversificationRatio = 1 - this.calculateConcentrationRisk(positions, totalExposure);

    // Build correlation matrix (simplified)
    const correlationMatrix: { [key: string]: { [key: string]: number } } = {};
    const symbols = positions.map(p => p.symbol);
    
    for (const symbol1 of symbols) {
      correlationMatrix[symbol1] = {};
      for (const symbol2 of symbols) {
        if (symbol1 === symbol2) {
          correlationMatrix[symbol1][symbol2] = 1;
        } else {
          // Simplified correlation based on shared currencies
          correlationMatrix[symbol1][symbol2] = this.calculatePairCorrelation(symbol1, symbol2);
        }
      }
    }

    // Calculate sector exposure (by currency)
    const sectorExposure: { [sector: string]: number } = {};
    for (const position of positions) {
      const symbol = position.symbol;
      if (symbol.length === 6) {
        const baseCurrency = symbol.substring(0, 3);
        const quoteCurrency = symbol.substring(3, 6);
        
        sectorExposure[baseCurrency] = (sectorExposure[baseCurrency] || 0) + Math.abs(position.marketValue);
        sectorExposure[quoteCurrency] = (sectorExposure[quoteCurrency] || 0) + Math.abs(position.marketValue);
      }
    }

    return {
      portfolioValue,
      totalRisk,
      riskPercentage,
      diversificationRatio,
      leverageRatio,
      marginUtilization,
      correlationMatrix,
      sectorExposure
    };
  }

  private calculatePairCorrelation(symbol1: string, symbol2: string): number {
    // Simplified correlation calculation
    if (symbol1.length !== 6 || symbol2.length !== 6) return 0;
    
    const currencies1 = [symbol1.substring(0, 3), symbol1.substring(3, 6)];
    const currencies2 = [symbol2.substring(0, 3), symbol2.substring(3, 6)];
    
    const sharedCurrencies = currencies1.filter(c => currencies2.includes(c)).length;
    
    switch (sharedCurrencies) {
      case 2: return 1.0;   // Same pair
      case 1: return 0.7;   // One shared currency
      default: return 0.1;  // No shared currencies
    }
  }

  private generateRiskRecommendations(
    portfolioRisk: PortfolioRisk, 
    positionRisks: PositionRisk[], 
    metrics: RiskMetrics
  ): RiskRecommendation[] {
    const recommendations: RiskRecommendation[] = [];

    // Portfolio-level recommendations
    if (portfolioRisk.riskScore > 70) {
      recommendations.push({
        type: 'reduce_position',
        priority: 'critical',
        description: 'Portfolio risk is critically high',
        action: 'Reduce overall position sizes or close high-risk positions',
        impact: 'Significantly reduce portfolio risk',
        confidence: 0.9
      });
    }

    if (portfolioRisk.concentrationRisk > 0.7) {
      recommendations.push({
        type: 'diversify',
        priority: 'high',
        description: 'Portfolio is highly concentrated',
        action: 'Diversify across different currency pairs and timeframes',
        impact: 'Reduce concentration risk and improve risk-adjusted returns',
        confidence: 0.8
      });
    }

    if (metrics.leverageRatio > this.riskLimits.maxLeverage) {
      recommendations.push({
        type: 'reduce_position',
        priority: 'high',
        description: 'Leverage ratio exceeds safe limits',
        action: 'Reduce position sizes to lower leverage',
        impact: 'Reduce margin risk and potential for large losses',
        confidence: 0.9
      });
    }

    // Position-level recommendations
    for (const positionRisk of positionRisks) {
      if (positionRisk.riskPercentage > this.riskLimits.maxPositionRisk) {
        recommendations.push({
          type: 'reduce_position',
          priority: 'medium',
          symbol: positionRisk.symbol,
          description: `Position risk exceeds ${this.riskLimits.maxPositionRisk * 100}% limit`,
          action: `Reduce ${positionRisk.symbol} position size`,
          impact: 'Lower individual position risk',
          confidence: 0.8
        });
      }

      if (!positionRisk.stopLoss) {
        recommendations.push({
          type: 'add_stop_loss',
          priority: 'medium',
          symbol: positionRisk.symbol,
          description: 'Position lacks stop loss protection',
          action: `Add stop loss to ${positionRisk.symbol} position`,
          impact: 'Limit potential losses',
          confidence: 0.9
        });
      }

      if (positionRisk.riskRewardRatio && positionRisk.riskRewardRatio < 1) {
        recommendations.push({
          type: 'close_position',
          priority: 'low',
          symbol: positionRisk.symbol,
          description: 'Poor risk-reward ratio',
          action: `Consider closing or adjusting ${positionRisk.symbol} position`,
          impact: 'Improve overall portfolio risk-reward profile',
          confidence: 0.6
        });
      }
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  private generateRiskAlerts(
    portfolioRisk: PortfolioRisk, 
    positionRisks: PositionRisk[], 
    metrics: RiskMetrics
  ): RiskAlert[] {
    const alerts: RiskAlert[] = [];
    const timestamp = Date.now();

    // Portfolio risk alerts
    if (portfolioRisk.riskScore > 80) {
      alerts.push({
        level: 'critical',
        type: 'Portfolio Risk',
        message: 'Portfolio risk score is critically high',
        value: portfolioRisk.riskScore,
        threshold: 80,
        timestamp
      });
    } else if (portfolioRisk.riskScore > 60) {
      alerts.push({
        level: 'warning',
        type: 'Portfolio Risk',
        message: 'Portfolio risk score is elevated',
        value: portfolioRisk.riskScore,
        threshold: 60,
        timestamp
      });
    }

    // Leverage alerts
    if (metrics.leverageRatio > this.riskLimits.maxLeverage) {
      alerts.push({
        level: 'danger',
        type: 'Leverage',
        message: 'Leverage ratio exceeds safe limits',
        value: metrics.leverageRatio,
        threshold: this.riskLimits.maxLeverage,
        timestamp
      });
    }

    // Concentration alerts
    if (portfolioRisk.concentrationRisk > 0.8) {
      alerts.push({
        level: 'warning',
        type: 'Concentration',
        message: 'Portfolio concentration is too high',
        value: portfolioRisk.concentrationRisk,
        threshold: 0.8,
        timestamp
      });
    }

    // Position-specific alerts
    for (const positionRisk of positionRisks) {
      if (positionRisk.riskPercentage > this.riskLimits.maxPositionRisk * 2) {
        alerts.push({
          level: 'danger',
          type: 'Position Risk',
          message: `${positionRisk.symbol} position risk is extremely high`,
          symbol: positionRisk.symbol,
          value: positionRisk.riskPercentage,
          threshold: this.riskLimits.maxPositionRisk * 2,
          timestamp
        });
      }
    }

    return alerts.sort((a, b) => {
      const levelOrder = { critical: 4, danger: 3, warning: 2, info: 1 };
      return levelOrder[b.level] - levelOrder[a.level];
    });
  }

  private generateTestPositions(): Position[] {
    return [
      {
        symbol: 'EURUSD',
        side: 'long',
        quantity: 10000,
        entryPrice: 1.0850,
        currentPrice: 1.0870,
        marketValue: 10870,
        unrealizedPnL: 200,
        stopLoss: 1.0800,
        takeProfit: 1.0950
      },
      {
        symbol: 'GBPUSD',
        side: 'short',
        quantity: -5000,
        entryPrice: 1.2650,
        currentPrice: 1.2630,
        marketValue: -6315,
        unrealizedPnL: 100,
        stopLoss: 1.2700
      }
    ];
  }
  
  // Integration testing methods for humanitarian mission validation
  async runIntegrationTests(): Promise<boolean> {
    this.logger.info('🧪 Running Risk Analysis Engine integration tests...');

    try {
      // Test 1: Python risk engine connectivity
      const pingTest = await this.pythonInterface.sendCommand('ping', { 
        test: 'risk_integration',
        engine_type: 'risk_analysis'
      });
      if (pingTest.status !== 'pong') {
        throw new Error('Python risk engine ping test failed');
      }

      // Test 2: Risk analysis with sample positions
      const samplePositions = this.generateTestPositions();
      const riskResult = await this.analyzePortfolio(samplePositions, 100000);
      
      if (!riskResult || !riskResult.portfolioRisk || !riskResult.positionRisks) {
        throw new Error('Risk analysis test failed');
      }

      // Test 3: Position sizing validation
      const sizingTest = await this.pythonInterface.sendCommand('optimize_position_sizing', {
        positions: samplePositions,
        account_balance: 100000,
        risk_limits: this.riskLimits,
        humanitarian_mode: true
      });
      
      if (!sizingTest.recommendations || !Array.isArray(sizingTest.recommendations)) {
        throw new Error('Position sizing optimization test failed');
      }

      // Test 4: AI risk enhancement validation
      const aiEnhancedTest = await this.pythonInterface.sendCommand('validate_ai_risk_enhancement', {
        test_type: 'portfolio_optimization',
        humanitarian_focus: true
      });
      
      if (!aiEnhancedTest.enabled) {
        throw new Error('AI risk enhancement validation failed');
      }

      // Test 5: Humanitarian mode validation
      const humanitarianTest = await this.pythonInterface.sendCommand('validate_humanitarian_mode', {
        module: 'risk_analysis'
      });
      if (!humanitarianTest.enabled) {
        throw new Error('Humanitarian mode not enabled in Python risk engine');
      }

      this.logger.info('✅ All Risk Analysis Engine integration tests passed');
      return true;
    } catch (error) {
      this.logger.error('❌ Risk integration tests failed:', error);
      return false;
    }
  }
}
