// Technical Analysis Engine - RSI, MACD, Bollinger Bands, Moving Averages
// Provides comprehensive technical indicator calculations and trend analysis
// Bridge to Python AI engines for humanitarian forex trading platform

import { Logger } from 'winston';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import { EventEmitter } from 'events';
import * as TI from 'technicalindicators';
import { mean, standardDeviation } from 'simple-statistics';

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

export interface MarketData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TechnicalAnalysisResult {
  symbol: string;
  timestamp: number;
  indicators: {
    rsi: RSIAnalysis;
    macd: MACDAnalysis;
    bollingerBands: BollingerBandsAnalysis;
    movingAverages: MovingAveragesAnalysis;
    stochastic: StochasticAnalysis;
    atr: ATRAnalysis;
  };
  signals: TradingSignal[];
  trend: TrendAnalysis;
  support: number[];
  resistance: number[];
  sentiment: SentimentAnalysis;
}

export interface RSIAnalysis {
  current: number;
  signal: 'oversold' | 'overbought' | 'neutral';
  strength: number; // 0-1
  divergence?: 'bullish' | 'bearish' | null;
}

export interface MACDAnalysis {
  macd: number;
  signal: number;
  histogram: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  crossover?: 'bullish' | 'bearish' | null;
}

export interface BollingerBandsAnalysis {
  upper: number;
  middle: number;
  lower: number;
  position: 'above_upper' | 'below_lower' | 'middle' | 'upper_half' | 'lower_half';
  squeeze: boolean;
  expansion: boolean;
}

export interface MovingAveragesAnalysis {
  sma20: number;
  sma50: number;
  sma200: number;
  ema12: number;
  ema26: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  crossovers: string[];
}

export interface StochasticAnalysis {
  k: number;
  d: number;
  signal: 'oversold' | 'overbought' | 'neutral';
  crossover?: 'bullish' | 'bearish' | null;
}

export interface ATRAnalysis {
  current: number;
  average: number;
  volatility: 'high' | 'medium' | 'low';
}

export interface TradingSignal {
  type: 'buy' | 'sell' | 'hold';
  strength: number; // 0-1
  source: string;
  description: string;
  confidence: number; // 0-1
}

export interface TrendAnalysis {
  direction: 'uptrend' | 'downtrend' | 'sideways';
  strength: number; // 0-1
  duration: number; // periods
  reliability: number; // 0-1
}

export interface SentimentAnalysis {
  score: number; // -1 to 1 (bearish to bullish)
  label: 'very_bearish' | 'bearish' | 'neutral' | 'bullish' | 'very_bullish';
  confidence: number; // 0-1
  factors: string[];
}

export class TechnicalAnalysisEngine {
  private logger: Logger;
  private ready: boolean = false;
  private pythonEngine: Platform3EngineConnection;
  private pythonInterface: PythonEngineInterface;

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
      throw new Error('Python engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const requestId = Math.random().toString(36).substr(2, 9);
      const message = JSON.stringify({
        id: requestId,
        command,
        data,
        timestamp: Date.now()
      });

      // Set up response handler
      const timeout = setTimeout(() => {
        this.pythonEngine.communicationQueue.delete(requestId);
        reject(new Error(`Python engine timeout for command: ${command}`));
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
        reject(new Error('Python process not available'));
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
    this.logger.info('🚀 Initializing Technical Analysis Engine for humanitarian trading...');
    
    try {
      // Initialize Python engine connection
      await this.initializePythonEngine();
      
      // Validate technical indicators library
      const testData = [1, 2, 3, 4, 5];
      TI.SMA.calculate({ period: 3, values: testData });
      
      // Test Python engine communication
      await this.testPythonEngineIntegration();
      
      this.ready = true;
      this.logger.info('✅ Technical Analysis Engine initialized and Python bridge established');
    } catch (error) {
      this.logger.error('❌ Technical Analysis Engine initialization failed:', error);
      throw error;
    }
  }

  private async initializePythonEngine(): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonScriptPath = path.join(__dirname, '../../ai-platform/coordination/engine/platform3_engine.py');
      
      this.logger.info(`Starting Python engine: ${pythonScriptPath}`);
      
      const pythonProcess = spawn('python', [pythonScriptPath, '--mode=technical-analysis'], {
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
            this.logger.debug('Python output:', line);
          }
        }
      });

      pythonProcess.stderr?.on('data', (data) => {
        this.logger.error('Python engine error:', data.toString());
      });

      pythonProcess.on('close', (code) => {
        this.logger.warn(`Python engine process closed with code ${code}`);
        this.pythonEngine.initialized = false;
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python engine process error:', error);
        reject(error);
      });

      // Wait for initialization confirmation
      setTimeout(() => {
        if (pythonProcess.pid) {
          this.pythonEngine.initialized = true;
          resolve();
        } else {
          reject(new Error('Python engine failed to start'));
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

  private async testPythonEngineIntegration(): Promise<void> {
    this.logger.info('🧪 Testing Python engine integration...');
    
    try {
      // Test basic communication
      const pingResult = await this.pythonInterface.sendCommand('ping', { message: 'integration_test' });
      if (pingResult.status !== 'pong') {
        throw new Error('Python engine ping test failed');
      }

      // Test indicator calculation
      const testData = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
      const rsiResult = await this.pythonInterface.sendCommand('calculate_rsi', {
        data: testData,
        period: 5
      });

      if (!rsiResult || !Array.isArray(rsiResult.values)) {
        throw new Error('Python RSI calculation test failed');
      }

      this.logger.info('✅ Python engine integration test passed');
    } catch (error) {
      this.logger.error('❌ Python engine integration test failed:', error);
      throw error;
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  async analyze(symbol: string, marketData: MarketData[]): Promise<TechnicalAnalysisResult> {
    if (!this.ready) {
      throw new Error('Technical Analysis Engine not initialized');
    }

    if (marketData.length < 50) {
      throw new Error('Insufficient data for technical analysis (minimum 50 periods required)');
    }

    this.logger.debug(`🔍 Performing humanitarian trading analysis for ${symbol} with ${marketData.length} data points`);

    const closes = marketData.map(d => d.close);
    const highs = marketData.map(d => d.high);
    const lows = marketData.map(d => d.low);
    const volumes = marketData.map(d => d.volume || 0);

    try {
      // Enhanced analysis using both TypeScript and Python engines
      const [
        rsi,
        macd,
        bollingerBands,
        movingAverages,
        stochastic,
        atr,
        pythonEnhancedAnalysis
      ] = await Promise.all([
        this.calculateRSI(closes),
        this.calculateMACD(closes),
        this.calculateBollingerBands(closes),
        this.calculateMovingAverages(closes),
        this.calculateStochastic(highs, lows, closes),
        this.calculateATR(marketData),
        this.getPythonEnhancedAnalysis(symbol, marketData)
      ]);

      // Analyze support and resistance with Python enhancement
      const supportResistance = await this.findSupportResistanceEnhanced(highs, lows);

      // Generate trading signals with AI enhancement
      const signals = await this.generateEnhancedTradingSignals(
        rsi, macd, bollingerBands, movingAverages, stochastic, pythonEnhancedAnalysis
      );

      // Analyze trend with AI validation
      const trend = await this.analyzeTrendEnhanced(closes, movingAverages);

      // Calculate sentiment with Python AI model
      const sentiment = await this.calculateEnhancedSentiment(rsi, macd, trend, signals);

      this.logger.info(`✅ Analysis completed for ${symbol} - Generated ${signals.length} signals for humanitarian trading`);

      return {
        symbol,
        timestamp: Date.now(),
        indicators: {
          rsi,
          macd,
          bollingerBands,
          movingAverages,
          stochastic,
          atr
        },
        signals,
        trend,
        support: supportResistance.support,
        resistance: supportResistance.resistance,
        sentiment
      };
    } catch (error) {
      this.logger.error(`❌ Analysis failed for ${symbol}:`, error);
      throw error;
    }
  }

  private async getPythonEnhancedAnalysis(symbol: string, marketData: MarketData[]): Promise<any> {
    try {
      return await this.pythonInterface.sendCommand('enhanced_technical_analysis', {
        symbol,
        market_data: marketData,
        indicators: ['rsi', 'macd', 'bollinger_bands', 'sma', 'ema', 'stochastic', 'atr'],
        ai_enhancement: true,
        humanitarian_mode: true
      });
    } catch (error) {
      this.logger.warn('Python enhanced analysis unavailable, using local calculations:', error);
      return null;
    }
  }

  private calculateRSI(closes: number[]): RSIAnalysis {
    const rsiValues = TI.RSI.calculate({ period: 14, values: closes });
    const current = rsiValues[rsiValues.length - 1];

    let signal: 'oversold' | 'overbought' | 'neutral' = 'neutral';
    if (current < 30) signal = 'oversold';
    else if (current > 70) signal = 'overbought';

    const strength = current > 50 ? (current - 50) / 50 : (50 - current) / 50;

    // Check for divergence (simplified)
    const divergence = this.checkRSIDivergence(rsiValues, closes);

    return {
      current,
      signal,
      strength,
      divergence
    };
  }

  private calculateMACD(closes: number[]): MACDAnalysis {
    const macdData = TI.MACD.calculate({
      values: closes,
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      SimpleMAOscillator: false,
      SimpleMASignal: false
    });

    const latest = macdData[macdData.length - 1];
    const macd = latest.MACD;
    const signal = latest.signal;
    const histogram = latest.histogram;

    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (macd > signal && histogram > 0) trend = 'bullish';
    else if (macd < signal && histogram < 0) trend = 'bearish';

    // Check for crossover
    const crossover = this.checkMACDCrossover(macdData);

    return {
      macd,
      signal,
      histogram,
      trend,
      crossover
    };
  }

  private calculateBollingerBands(closes: number[]): BollingerBandsAnalysis {
    const bbData = TI.BollingerBands.calculate({
      period: 20,
      stdDev: 2,
      values: closes
    });

    const latest = bbData[bbData.length - 1];
    const currentPrice = closes[closes.length - 1];

    let position: BollingerBandsAnalysis['position'] = 'middle';
    if (currentPrice > latest.upper) position = 'above_upper';
    else if (currentPrice < latest.lower) position = 'below_lower';
    else if (currentPrice > latest.middle) position = 'upper_half';
    else position = 'lower_half';

    // Check for squeeze and expansion
    const bandWidth = (latest.upper - latest.lower) / latest.middle;
    const avgBandWidth = mean(bbData.slice(-20).map(bb => (bb.upper - bb.lower) / bb.middle));
    
    const squeeze = bandWidth < avgBandWidth * 0.8;
    const expansion = bandWidth > avgBandWidth * 1.2;

    return {
      upper: latest.upper,
      middle: latest.middle,
      lower: latest.lower,
      position,
      squeeze,
      expansion
    };
  }

  private calculateMovingAverages(closes: number[]): MovingAveragesAnalysis {
    const sma20 = TI.SMA.calculate({ period: 20, values: closes });
    const sma50 = TI.SMA.calculate({ period: 50, values: closes });
    const sma200 = TI.SMA.calculate({ period: 200, values: closes });
    const ema12 = TI.EMA.calculate({ period: 12, values: closes });
    const ema26 = TI.EMA.calculate({ period: 26, values: closes });

    const currentSMA20 = sma20[sma20.length - 1];
    const currentSMA50 = sma50[sma50.length - 1];
    const currentSMA200 = sma200[sma200.length - 1];
    const currentEMA12 = ema12[ema12.length - 1];
    const currentEMA26 = ema26[ema26.length - 1];

    // Determine trend
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (currentSMA20 > currentSMA50 && currentSMA50 > currentSMA200) trend = 'bullish';
    else if (currentSMA20 < currentSMA50 && currentSMA50 < currentSMA200) trend = 'bearish';

    // Check for crossovers
    const crossovers = this.checkMACrossovers(sma20, sma50, ema12, ema26);

    return {
      sma20: currentSMA20,
      sma50: currentSMA50,
      sma200: currentSMA200,
      ema12: currentEMA12,
      ema26: currentEMA26,
      trend,
      crossovers
    };
  }

  private calculateStochastic(highs: number[], lows: number[], closes: number[]): StochasticAnalysis {
    const stochData = TI.Stochastic.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: 14,
      signalPeriod: 3
    });

    const latest = stochData[stochData.length - 1];
    const k = latest.k;
    const d = latest.d;

    let signal: 'oversold' | 'overbought' | 'neutral' = 'neutral';
    if (k < 20 && d < 20) signal = 'oversold';
    else if (k > 80 && d > 80) signal = 'overbought';

    // Check for crossover
    const crossover = this.checkStochasticCrossover(stochData);

    return {
      k,
      d,
      signal,
      crossover
    };
  }

  private calculateATR(marketData: MarketData[]): ATRAnalysis {
    const atrData = TI.ATR.calculate({
      high: marketData.map(d => d.high),
      low: marketData.map(d => d.low),
      close: marketData.map(d => d.close),
      period: 14
    });

    const current = atrData[atrData.length - 1];
    const average = mean(atrData.slice(-20));

    let volatility: 'high' | 'medium' | 'low' = 'medium';
    if (current > average * 1.5) volatility = 'high';
    else if (current < average * 0.5) volatility = 'low';

    return {
      current,
      average,
      volatility
    };
  }

  // Helper methods for analysis
  private checkRSIDivergence(rsiValues: number[], closes: number[]): 'bullish' | 'bearish' | null {
    // Simplified divergence detection
    if (rsiValues.length < 20) return null;

    const recentRSI = rsiValues.slice(-10);
    const recentPrices = closes.slice(-10);

    const rsiTrend = recentRSI[recentRSI.length - 1] - recentRSI[0];
    const priceTrend = recentPrices[recentPrices.length - 1] - recentPrices[0];

    if (rsiTrend > 0 && priceTrend < 0) return 'bullish';
    if (rsiTrend < 0 && priceTrend > 0) return 'bearish';

    return null;
  }

  private checkMACDCrossover(macdData: any[]): 'bullish' | 'bearish' | null {
    if (macdData.length < 2) return null;

    const current = macdData[macdData.length - 1];
    const previous = macdData[macdData.length - 2];

    if (previous.MACD <= previous.signal && current.MACD > current.signal) return 'bullish';
    if (previous.MACD >= previous.signal && current.MACD < current.signal) return 'bearish';

    return null;
  }

  private checkMACrossovers(sma20: number[], sma50: number[], ema12: number[], ema26: number[]): string[] {
    const crossovers: string[] = [];

    // Golden Cross / Death Cross
    if (sma20.length >= 2 && sma50.length >= 2) {
      const currentSMA20 = sma20[sma20.length - 1];
      const previousSMA20 = sma20[sma20.length - 2];
      const currentSMA50 = sma50[sma50.length - 1];
      const previousSMA50 = sma50[sma50.length - 2];

      if (previousSMA20 <= previousSMA50 && currentSMA20 > currentSMA50) {
        crossovers.push('Golden Cross (SMA20 > SMA50)');
      }
      if (previousSMA20 >= previousSMA50 && currentSMA20 < currentSMA50) {
        crossovers.push('Death Cross (SMA20 < SMA50)');
      }
    }

    return crossovers;
  }

  private checkStochasticCrossover(stochData: any[]): 'bullish' | 'bearish' | null {
    if (stochData.length < 2) return null;

    const current = stochData[stochData.length - 1];
    const previous = stochData[stochData.length - 2];

    if (previous.k <= previous.d && current.k > current.d) return 'bullish';
    if (previous.k >= previous.d && current.k < current.d) return 'bearish';

    return null;
  }

  // Enhanced methods with Python AI integration
  private async findSupportResistanceEnhanced(highs: number[], lows: number[]): Promise<{ support: number[], resistance: number[] }> {
    try {
      const pythonResult = await this.pythonInterface.sendCommand('calculate_support_resistance', {
        highs,
        lows,
        method: 'ai_enhanced'
      });

      if (pythonResult && pythonResult.support && pythonResult.resistance) {
        return pythonResult;
      }
    } catch (error) {
      this.logger.warn('Using local support/resistance calculation:', error);
    }

    // Fallback to local calculation
    return this.findSupportResistance(highs, lows);
  }

  private async generateEnhancedTradingSignals(
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    bb: BollingerBandsAnalysis,
    ma: MovingAveragesAnalysis,
    stoch: StochasticAnalysis,
    pythonAnalysis: any
  ): Promise<TradingSignal[]> {
    const localSignals = this.generateTradingSignals(rsi, macd, bb, ma, stoch);

    try {
      if (pythonAnalysis && pythonAnalysis.ai_signals) {
        // Merge local and AI-enhanced signals
        const aiSignals: TradingSignal[] = pythonAnalysis.ai_signals.map((signal: any) => ({
          type: signal.type,
          strength: signal.strength,
          source: `AI-${signal.source}`,
          description: signal.description,
          confidence: signal.confidence * 1.1 // AI signals get confidence boost
        }));

        return [...localSignals, ...aiSignals];
      }
    } catch (error) {
      this.logger.warn('AI signal generation failed, using local signals:', error);
    }

    return localSignals;
  }

  private async analyzeTrendEnhanced(closes: number[], ma: MovingAveragesAnalysis): Promise<TrendAnalysis> {
    try {
      const pythonTrend = await this.pythonInterface.sendCommand('analyze_trend_ai', {
        closes,
        moving_averages: ma,
        humanitarian_mode: true
      });

      if (pythonTrend && pythonTrend.direction) {
        return pythonTrend;
      }
    } catch (error) {
      this.logger.warn('Using local trend analysis:', error);
    }

    return this.analyzeTrend(closes, ma);
  }

  private async calculateEnhancedSentiment(
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    trend: TrendAnalysis,
    signals: TradingSignal[]
  ): Promise<SentimentAnalysis> {
    try {
      const pythonSentiment = await this.pythonInterface.sendCommand('calculate_ai_sentiment', {
        rsi,
        macd,
        trend,
        signals,
        humanitarian_focus: true
      });

      if (pythonSentiment && typeof pythonSentiment.score === 'number') {
        return pythonSentiment;
      }
    } catch (error) {
      this.logger.warn('Using local sentiment analysis:', error);
    }

    return this.calculateSentiment(rsi, macd, trend, signals);
  }

  // Integration testing methods for humanitarian mission validation
  async runIntegrationTests(): Promise<boolean> {
    this.logger.info('🧪 Running Technical Analysis Engine integration tests...');

    try {
      // Test 1: Python engine connectivity
      const pingTest = await this.pythonInterface.sendCommand('ping', { test: 'integration' });
      if (pingTest.status !== 'pong') {
        throw new Error('Python engine ping test failed');
      }

      // Test 2: Technical analysis with sample data
      const sampleData: MarketData[] = this.generateSampleMarketData();
      const analysisResult = await this.analyze('TEST_SYMBOL', sampleData);
      
      if (!analysisResult || !analysisResult.indicators || !analysisResult.signals) {
        throw new Error('Technical analysis test failed');
      }

      // Test 3: AI enhancement validation
      if (analysisResult.signals.some(s => s.source.includes('AI-'))) {
        this.logger.info('✅ AI enhancement confirmed');
      }

      // Test 4: Humanitarian mode validation
      const humanitarianTest = await this.pythonInterface.sendCommand('validate_humanitarian_mode', {});
      if (!humanitarianTest.enabled) {
        throw new Error('Humanitarian mode not enabled in Python engine');
      }

      this.logger.info('✅ All Technical Analysis Engine integration tests passed');
      return true;
    } catch (error) {
      this.logger.error('❌ Integration tests failed:', error);
      return false;
    }
  }

  private generateSampleMarketData(): MarketData[] {
    const data: MarketData[] = [];
    let price = 1.1000;
    
    for (let i = 0; i < 100; i++) {
      const change = (Math.random() - 0.5) * 0.002;
      price += change;
      
      data.push({
        timestamp: Date.now() - (100 - i) * 60000,
        open: price - change,
        high: price + Math.random() * 0.001,
        low: price - Math.random() * 0.001,
        close: price,
        volume: Math.floor(Math.random() * 1000) + 500
      });
    }
    
    return data;
  }

  private findSupportResistance(highs: number[], lows: number[]): { support: number[], resistance: number[] } {
    // Simplified support/resistance detection using pivot points
    const support: number[] = [];
    const resistance: number[] = [];

    const lookback = 10;
    
    for (let i = lookback; i < lows.length - lookback; i++) {
      const isSupport = lows.slice(i - lookback, i).every(low => low >= lows[i]) &&
                       lows.slice(i + 1, i + lookback + 1).every(low => low >= lows[i]);
      
      if (isSupport) {
        support.push(lows[i]);
      }
    }

    for (let i = lookback; i < highs.length - lookback; i++) {
      const isResistance = highs.slice(i - lookback, i).every(high => high <= highs[i]) &&
                          highs.slice(i + 1, i + lookback + 1).every(high => high <= highs[i]);
      
      if (isResistance) {
        resistance.push(highs[i]);
      }
    }

    return {
      support: support.slice(-3), // Last 3 support levels
      resistance: resistance.slice(-3) // Last 3 resistance levels
    };
  }

  private generateTradingSignals(
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    bb: BollingerBandsAnalysis,
    ma: MovingAveragesAnalysis,
    stoch: StochasticAnalysis
  ): TradingSignal[] {
    const signals: TradingSignal[] = [];

    // RSI signals
    if (rsi.signal === 'oversold') {
      signals.push({
        type: 'buy',
        strength: rsi.strength,
        source: 'RSI',
        description: 'RSI indicates oversold condition',
        confidence: 0.7
      });
    } else if (rsi.signal === 'overbought') {
      signals.push({
        type: 'sell',
        strength: rsi.strength,
        source: 'RSI',
        description: 'RSI indicates overbought condition',
        confidence: 0.7
      });
    }

    // MACD signals
    if (macd.crossover === 'bullish') {
      signals.push({
        type: 'buy',
        strength: 0.8,
        source: 'MACD',
        description: 'MACD bullish crossover detected',
        confidence: 0.8
      });
    } else if (macd.crossover === 'bearish') {
      signals.push({
        type: 'sell',
        strength: 0.8,
        source: 'MACD',
        description: 'MACD bearish crossover detected',
        confidence: 0.8
      });
    }

    // Bollinger Bands signals
    if (bb.position === 'below_lower') {
      signals.push({
        type: 'buy',
        strength: 0.6,
        source: 'Bollinger Bands',
        description: 'Price below lower Bollinger Band',
        confidence: 0.6
      });
    } else if (bb.position === 'above_upper') {
      signals.push({
        type: 'sell',
        strength: 0.6,
        source: 'Bollinger Bands',
        description: 'Price above upper Bollinger Band',
        confidence: 0.6
      });
    }

    // Moving Average signals
    if (ma.trend === 'bullish') {
      signals.push({
        type: 'buy',
        strength: 0.7,
        source: 'Moving Averages',
        description: 'Bullish moving average alignment',
        confidence: 0.7
      });
    } else if (ma.trend === 'bearish') {
      signals.push({
        type: 'sell',
        strength: 0.7,
        source: 'Moving Averages',
        description: 'Bearish moving average alignment',
        confidence: 0.7
      });
    }

    return signals;
  }

  private analyzeTrend(closes: number[], ma: MovingAveragesAnalysis): TrendAnalysis {
    const recentCloses = closes.slice(-20);
    const priceChange = recentCloses[recentCloses.length - 1] - recentCloses[0];
    
    let direction: 'uptrend' | 'downtrend' | 'sideways' = 'sideways';
    if (priceChange > 0 && ma.trend === 'bullish') direction = 'uptrend';
    else if (priceChange < 0 && ma.trend === 'bearish') direction = 'downtrend';

    const strength = Math.abs(priceChange) / recentCloses[0];
    const reliability = ma.trend !== 'neutral' ? 0.8 : 0.4;

    return {
      direction,
      strength: Math.min(strength * 10, 1), // Normalize to 0-1
      duration: 20, // Using 20 periods for analysis
      reliability
    };
  }

  private calculateSentiment(
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    trend: TrendAnalysis,
    signals: TradingSignal[]
  ): SentimentAnalysis {
    let score = 0;
    const factors: string[] = [];

    // RSI contribution
    if (rsi.signal === 'oversold') {
      score += 0.3;
      factors.push('RSI oversold');
    } else if (rsi.signal === 'overbought') {
      score -= 0.3;
      factors.push('RSI overbought');
    }

    // MACD contribution
    if (macd.trend === 'bullish') {
      score += 0.3;
      factors.push('MACD bullish');
    } else if (macd.trend === 'bearish') {
      score -= 0.3;
      factors.push('MACD bearish');
    }

    // Trend contribution
    if (trend.direction === 'uptrend') {
      score += 0.4 * trend.strength;
      factors.push('Uptrend detected');
    } else if (trend.direction === 'downtrend') {
      score -= 0.4 * trend.strength;
      factors.push('Downtrend detected');
    }

    // Normalize score to -1 to 1
    score = Math.max(-1, Math.min(1, score));

    let label: SentimentAnalysis['label'] = 'neutral';
    if (score > 0.6) label = 'very_bullish';
    else if (score > 0.2) label = 'bullish';
    else if (score < -0.6) label = 'very_bearish';
    else if (score < -0.2) label = 'bearish';

    const confidence = Math.abs(score);

    return {
      score,
      label,
      confidence,
      factors
    };
  }

  async analyzeSentiment(symbol: string, marketData: MarketData[]): Promise<SentimentAnalysis> {
    const analysis = await this.analyze(symbol, marketData);
    return analysis.sentiment;
  }
}
