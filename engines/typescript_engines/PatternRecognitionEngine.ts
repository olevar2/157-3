// Pattern Recognition Engine - Chart pattern detection for trading signals
// Detects head & shoulders, triangles, support/resistance, and other chart patterns
// Bridge to Python AI pattern recognition for humanitarian forex trading platform

import { Logger } from 'winston';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import { EventEmitter } from 'events';
import { mean, standardDeviation } from 'simple-statistics';
import { MarketData } from './TechnicalAnalysisEngine';

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

// Python Pattern Recognition Integration Interface
export interface PythonPatternInterface {
  detectPatterns(symbol: string, data: MarketData[]): Promise<PatternRecognitionResult>;
  validatePattern(pattern: DetectedPattern): Promise<PatternValidation>;
  getPatternProbability(patternType: string, data: MarketData[]): Promise<number>;
}

export interface PatternValidation {
  valid: boolean;
  confidence: number;
  reliability: number;
  expectedMove: number;
}

export interface PatternRecognitionResult {
  symbol: string;
  timestamp: number;
  patterns: DetectedPattern[];
  supportLevels: SupportResistanceLevel[];
  resistanceLevels: SupportResistanceLevel[];
  trendLines: TrendLine[];
  signals: PatternSignal[];
}

export interface DetectedPattern {
  type: PatternType;
  name: string;
  confidence: number;
  startIndex: number;
  endIndex: number;
  keyPoints: PatternPoint[];
  signal: 'bullish' | 'bearish' | 'neutral';
  target?: number;
  stopLoss?: number;
  description: string;
}

export type PatternType = 
  | 'head_and_shoulders'
  | 'inverse_head_and_shoulders'
  | 'double_top'
  | 'double_bottom'
  | 'ascending_triangle'
  | 'descending_triangle'
  | 'symmetrical_triangle'
  | 'flag'
  | 'pennant'
  | 'wedge'
  | 'channel'
  | 'cup_and_handle';

export interface PatternPoint {
  index: number;
  price: number;
  timestamp: number;
  type: 'peak' | 'trough' | 'support' | 'resistance';
}

export interface SupportResistanceLevel {
  price: number;
  strength: number;
  touches: number;
  firstTouch: number;
  lastTouch: number;
  type: 'support' | 'resistance';
}

export interface TrendLine {
  startPoint: PatternPoint;
  endPoint: PatternPoint;
  slope: number;
  strength: number;
  type: 'support' | 'resistance' | 'trend';
  equation: string;
}

export interface PatternSignal {
  type: 'breakout' | 'breakdown' | 'reversal' | 'continuation';
  pattern: string;
  strength: number;
  confidence: number;
  direction: 'bullish' | 'bearish';
  entry?: number;
  target?: number;
  stopLoss?: number;
}

export class PatternRecognitionEngine {
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
      throw new Error('Python pattern recognition engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const requestId = Math.random().toString(36).substr(2, 9);
      const message = JSON.stringify({
        id: requestId,
        command,
        data,
        timestamp: Date.now(),
        engine_type: 'pattern_recognition'
      });

      // Set up response handler
      const timeout = setTimeout(() => {
        this.pythonEngine.communicationQueue.delete(requestId);
        reject(new Error(`Python pattern engine timeout for command: ${command}`));
      }, 35000); // 35 second timeout for pattern analysis

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
        reject(new Error('Python pattern process not available'));
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
    this.logger.info('🚀 Initializing Pattern Recognition Engine for humanitarian trading...');
    
    try {
      // Initialize Python pattern recognition engine connection
      await this.initializePythonEngine();
      
      // Test pattern detection algorithms
      const testData = this.generateTestData();
      const peaks = this.findPeaksAndTroughs(testData);
      
      if (peaks.peaks.length > 0 && peaks.troughs.length > 0) {
        // Test Python pattern engine integration
        await this.testPythonPatternEngineIntegration();
        
        this.ready = true;
        this.logger.info('✅ Pattern Recognition Engine initialized with Python bridge');
      } else {
        throw new Error('Pattern detection test failed');
      }
    } catch (error) {
      this.logger.error('❌ Pattern Recognition Engine initialization failed:', error);
      throw error;
    }
  }

  private async initializePythonEngine(): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonScriptPath = path.join(__dirname, '../../ai-platform/coordination/engine/platform3_engine.py');
      
      this.logger.info(`Starting Python pattern recognition engine: ${pythonScriptPath}`);
      
      const pythonProcess = spawn('python', [pythonScriptPath, '--mode=pattern-recognition'], {
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
            this.logger.debug('Python pattern output:', line);
          }
        }
      });

      pythonProcess.stderr?.on('data', (data) => {
        this.logger.error('Python pattern engine error:', data.toString());
      });

      pythonProcess.on('close', (code) => {
        this.logger.warn(`Python pattern engine process closed with code ${code}`);
        this.pythonEngine.initialized = false;
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python pattern engine process error:', error);
        reject(error);
      });

      // Wait for initialization confirmation
      setTimeout(() => {
        if (pythonProcess.pid) {
          this.pythonEngine.initialized = true;
          resolve();
        } else {
          reject(new Error('Python pattern engine failed to start'));
        }
      }, 4000);
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

  private async testPythonPatternEngineIntegration(): Promise<void> {
    this.logger.info('🧪 Testing Python pattern recognition engine integration...');
    
    try {
      // Test basic communication
      const pingResult = await this.pythonInterface.sendCommand('ping', { 
        message: 'pattern_integration_test',
        engine_type: 'pattern_recognition'
      });
      if (pingResult.status !== 'pong') {
        throw new Error('Python pattern engine ping test failed');
      }

      // Test pattern detection
      const testData = this.generateTestData();
      const patternResult = await this.pythonInterface.sendCommand('detect_advanced_patterns', {
        market_data: testData,
        pattern_types: ['head_and_shoulders', 'double_top', 'triangle'],
        ai_enhancement: true
      });

      if (!patternResult || !Array.isArray(patternResult.patterns)) {
        throw new Error('Python pattern detection test failed');
      }

      this.logger.info('✅ Python pattern recognition engine integration test passed');
    } catch (error) {
      this.logger.error('❌ Python pattern engine integration test failed:', error);
      throw error;
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  async detectPatterns(symbol: string, marketData: MarketData[]): Promise<PatternRecognitionResult> {
    if (!this.ready) {
      throw new Error('Pattern Recognition Engine not initialized');
    }

    if (marketData.length < 50) {
      throw new Error('Insufficient data for pattern recognition (minimum 50 periods required)');
    }

    this.logger.debug(`🔍 Detecting humanitarian trading patterns for ${symbol} with ${marketData.length} data points`);

    try {
      // Try Python AI-enhanced pattern detection first
      const pythonPatterns = await this.getPythonEnhancedPatterns(symbol, marketData);
      
      if (pythonPatterns) {
        this.logger.info(`✅ AI-enhanced pattern detection completed for ${symbol}`);
        return pythonPatterns;
      }
    } catch (error) {
      this.logger.warn('Python pattern detection failed, using local analysis:', error);
    }

    // Fallback to local pattern detection
    return this.generateLocalPatternAnalysis(symbol, marketData);
  }

  private async getPythonEnhancedPatterns(symbol: string, marketData: MarketData[]): Promise<PatternRecognitionResult | null> {
    try {
      const result = await this.pythonInterface.sendCommand('enhanced_pattern_detection', {
        symbol,
        market_data: marketData,
        pattern_types: [
          'head_and_shoulders', 'inverse_head_and_shoulders',
          'double_top', 'double_bottom',
          'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
          'flag', 'pennant', 'wedge', 'channel', 'cup_and_handle'
        ],
        ai_enhancement: true,
        humanitarian_mode: true,
        confidence_threshold: 0.6
      });

      if (result && result.patterns && Array.isArray(result.patterns)) {
        return {
          symbol,
          timestamp: Date.now(),
          patterns: result.patterns.map((p: any) => ({
            ...p,
            confidence: p.confidence * 1.1 // AI patterns get confidence boost
          })),
          supportLevels: result.support_levels || [],
          resistanceLevels: result.resistance_levels || [],
          trendLines: result.trend_lines || [],
          signals: result.signals || []
        };
      }
    } catch (error) {
      this.logger.debug('Python enhanced pattern detection unavailable:', error);
    }

    return null;
  }

  private async generateLocalPatternAnalysis(symbol: string, marketData: MarketData[]): Promise<PatternRecognitionResult> {

    // Find peaks and troughs
    const peaksAndTroughs = this.findPeaksAndTroughs(marketData);
    
    // Detect chart patterns
    const patterns = await this.detectChartPatterns(marketData, peaksAndTroughs);
    
    // Find support and resistance levels
    const supportResistance = this.findSupportResistanceLevels(marketData, peaksAndTroughs);
    
    // Draw trend lines
    const trendLines = this.findTrendLines(peaksAndTroughs);
    
    // Generate pattern-based signals
    const signals = this.generatePatternSignals(patterns, supportResistance, marketData);

    return {
      symbol,
      timestamp: Date.now(),
      patterns,
      supportLevels: supportResistance.support,
      resistanceLevels: supportResistance.resistance,
      trendLines,
      signals
    };
  }

  private findPeaksAndTroughs(marketData: MarketData[]): { peaks: PatternPoint[], troughs: PatternPoint[] } {
    const peaks: PatternPoint[] = [];
    const troughs: PatternPoint[] = [];
    const lookback = 5; // Look 5 periods back and forward

    for (let i = lookback; i < marketData.length - lookback; i++) {
      const current = marketData[i];
      const highs = marketData.slice(i - lookback, i + lookback + 1).map(d => d.high);
      const lows = marketData.slice(i - lookback, i + lookback + 1).map(d => d.low);

      // Check if current high is a peak
      if (current.high === Math.max(...highs)) {
        peaks.push({
          index: i,
          price: current.high,
          timestamp: current.timestamp,
          type: 'peak'
        });
      }

      // Check if current low is a trough
      if (current.low === Math.min(...lows)) {
        troughs.push({
          index: i,
          price: current.low,
          timestamp: current.timestamp,
          type: 'trough'
        });
      }
    }

    return { peaks, troughs };
  }

  private async detectChartPatterns(
    marketData: MarketData[], 
    peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] }
  ): Promise<DetectedPattern[]> {
    const patterns: DetectedPattern[] = [];

    // Detect Head and Shoulders
    patterns.push(...this.detectHeadAndShoulders(peaksAndTroughs.peaks, marketData));
    
    // Detect Double Tops/Bottoms
    patterns.push(...this.detectDoubleTops(peaksAndTroughs.peaks, marketData));
    patterns.push(...this.detectDoubleBottoms(peaksAndTroughs.troughs, marketData));
    
    // Detect Triangles
    patterns.push(...this.detectTriangles(peaksAndTroughs, marketData));
    
    // Detect Flags and Pennants
    patterns.push(...this.detectFlags(marketData, peaksAndTroughs));
    
    // Detect Channels
    patterns.push(...this.detectChannels(peaksAndTroughs, marketData));

    return patterns.filter(pattern => pattern.confidence > 0.6); // Only return high-confidence patterns
  }

  private detectHeadAndShoulders(peaks: PatternPoint[], marketData: MarketData[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    for (let i = 0; i < peaks.length - 2; i++) {
      const leftShoulder = peaks[i];
      const head = peaks[i + 1];
      const rightShoulder = peaks[i + 2];

      // Check head and shoulders criteria
      if (head.price > leftShoulder.price && head.price > rightShoulder.price) {
        const shoulderSymmetry = Math.abs(leftShoulder.price - rightShoulder.price) / leftShoulder.price;
        
        if (shoulderSymmetry < 0.02) { // Shoulders should be roughly equal (within 2%)
          const confidence = Math.max(0.6, 1 - shoulderSymmetry * 10);
          
          // Calculate neckline and target
          const neckline = (leftShoulder.price + rightShoulder.price) / 2;
          const target = neckline - (head.price - neckline);
          
          patterns.push({
            type: 'head_and_shoulders',
            name: 'Head and Shoulders',
            confidence,
            startIndex: leftShoulder.index,
            endIndex: rightShoulder.index,
            keyPoints: [leftShoulder, head, rightShoulder],
            signal: 'bearish',
            target,
            stopLoss: head.price,
            description: 'Bearish reversal pattern with head higher than shoulders'
          });
        }
      }
    }

    return patterns;
  }

  private detectDoubleTops(peaks: PatternPoint[], marketData: MarketData[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    for (let i = 0; i < peaks.length - 1; i++) {
      const firstPeak = peaks[i];
      const secondPeak = peaks[i + 1];

      const priceDifference = Math.abs(firstPeak.price - secondPeak.price) / firstPeak.price;
      const timeDifference = secondPeak.index - firstPeak.index;

      // Double top criteria
      if (priceDifference < 0.015 && timeDifference > 10 && timeDifference < 50) {
        const confidence = Math.max(0.6, 1 - priceDifference * 20);
        
        // Find valley between peaks
        const valleyStart = firstPeak.index;
        const valleyEnd = secondPeak.index;
        const valleyData = marketData.slice(valleyStart, valleyEnd + 1);
        const valleyLow = Math.min(...valleyData.map(d => d.low));
        
        const target = valleyLow - (firstPeak.price - valleyLow);

        patterns.push({
          type: 'double_top',
          name: 'Double Top',
          confidence,
          startIndex: firstPeak.index,
          endIndex: secondPeak.index,
          keyPoints: [firstPeak, secondPeak],
          signal: 'bearish',
          target,
          stopLoss: Math.max(firstPeak.price, secondPeak.price),
          description: 'Bearish reversal pattern with two similar peaks'
        });
      }
    }

    return patterns;
  }

  private detectDoubleBottoms(troughs: PatternPoint[], marketData: MarketData[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    for (let i = 0; i < troughs.length - 1; i++) {
      const firstTrough = troughs[i];
      const secondTrough = troughs[i + 1];

      const priceDifference = Math.abs(firstTrough.price - secondTrough.price) / firstTrough.price;
      const timeDifference = secondTrough.index - firstTrough.index;

      // Double bottom criteria
      if (priceDifference < 0.015 && timeDifference > 10 && timeDifference < 50) {
        const confidence = Math.max(0.6, 1 - priceDifference * 20);
        
        // Find peak between troughs
        const peakStart = firstTrough.index;
        const peakEnd = secondTrough.index;
        const peakData = marketData.slice(peakStart, peakEnd + 1);
        const peakHigh = Math.max(...peakData.map(d => d.high));
        
        const target = peakHigh + (peakHigh - firstTrough.price);

        patterns.push({
          type: 'double_bottom',
          name: 'Double Bottom',
          confidence,
          startIndex: firstTrough.index,
          endIndex: secondTrough.index,
          keyPoints: [firstTrough, secondTrough],
          signal: 'bullish',
          target,
          stopLoss: Math.min(firstTrough.price, secondTrough.price),
          description: 'Bullish reversal pattern with two similar troughs'
        });
      }
    }

    return patterns;
  }

  private detectTriangles(
    peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] },
    marketData: MarketData[]
  ): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    // Combine and sort peaks and troughs by index
    const allPoints = [...peaksAndTroughs.peaks, ...peaksAndTroughs.troughs]
      .sort((a, b) => a.index - b.index);

    if (allPoints.length < 4) return patterns;

    // Look for triangle patterns
    for (let i = 0; i < allPoints.length - 3; i++) {
      const points = allPoints.slice(i, i + 4);
      const triangle = this.analyzeTrianglePattern(points, marketData);
      
      if (triangle) {
        patterns.push(triangle);
      }
    }

    return patterns;
  }

  private analyzeTrianglePattern(points: PatternPoint[], marketData: MarketData[]): DetectedPattern | null {
    if (points.length < 4) return null;

    const highs = points.filter(p => p.type === 'peak');
    const lows = points.filter(p => p.type === 'trough');

    if (highs.length < 2 || lows.length < 2) return null;

    // Calculate trend lines
    const highTrend = this.calculateTrendLine(highs);
    const lowTrend = this.calculateTrendLine(lows);

    // Determine triangle type
    let triangleType: PatternType;
    let signal: 'bullish' | 'bearish' | 'neutral' = 'neutral';

    if (highTrend.slope < -0.001 && Math.abs(lowTrend.slope) < 0.001) {
      triangleType = 'descending_triangle';
      signal = 'bearish';
    } else if (Math.abs(highTrend.slope) < 0.001 && lowTrend.slope > 0.001) {
      triangleType = 'ascending_triangle';
      signal = 'bullish';
    } else if (highTrend.slope < -0.001 && lowTrend.slope > 0.001) {
      triangleType = 'symmetrical_triangle';
      signal = 'neutral';
    } else {
      return null;
    }

    const confidence = this.calculateTriangleConfidence(points, highTrend, lowTrend);

    return {
      type: triangleType,
      name: triangleType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      confidence,
      startIndex: points[0].index,
      endIndex: points[points.length - 1].index,
      keyPoints: points,
      signal,
      description: `${signal} triangle pattern indicating potential ${signal === 'neutral' ? 'breakout' : signal} move`
    };
  }

  private calculateTrendLine(points: PatternPoint[]): { slope: number, intercept: number } {
    if (points.length < 2) return { slope: 0, intercept: 0 };

    const n = points.length;
    const sumX = points.reduce((sum, p) => sum + p.index, 0);
    const sumY = points.reduce((sum, p) => sum + p.price, 0);
    const sumXY = points.reduce((sum, p) => sum + p.index * p.price, 0);
    const sumX2 = points.reduce((sum, p) => sum + p.index * p.index, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return { slope, intercept };
  }

  private calculateTriangleConfidence(
    points: PatternPoint[], 
    highTrend: { slope: number, intercept: number },
    lowTrend: { slope: number, intercept: number }
  ): number {
    // Calculate how well points fit the trend lines
    const highs = points.filter(p => p.type === 'peak');
    const lows = points.filter(p => p.type === 'trough');

    let highFit = 0;
    let lowFit = 0;

    for (const point of highs) {
      const expectedPrice = highTrend.slope * point.index + highTrend.intercept;
      const error = Math.abs(point.price - expectedPrice) / point.price;
      highFit += 1 - error;
    }

    for (const point of lows) {
      const expectedPrice = lowTrend.slope * point.index + lowTrend.intercept;
      const error = Math.abs(point.price - expectedPrice) / point.price;
      lowFit += 1 - error;
    }

    const avgFit = (highFit / highs.length + lowFit / lows.length) / 2;
    return Math.max(0.5, Math.min(1, avgFit));
  }

  private detectFlags(marketData: MarketData[], peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] }): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    // Simplified flag detection - look for consolidation after strong moves
    for (let i = 20; i < marketData.length - 10; i++) {
      const preTrend = this.calculateTrendStrength(marketData, i - 20, i);
      const consolidation = this.detectConsolidation(marketData, i, i + 10);
      
      if (Math.abs(preTrend) > 0.02 && consolidation.isConsolidating) {
        const signal = preTrend > 0 ? 'bullish' : 'bearish';
        
        patterns.push({
          type: 'flag',
          name: 'Flag Pattern',
          confidence: consolidation.confidence,
          startIndex: i,
          endIndex: i + 10,
          keyPoints: [],
          signal,
          description: `${signal} flag pattern indicating trend continuation`
        });
      }
    }

    return patterns;
  }

  private detectChannels(peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] }, marketData: MarketData[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    if (peaksAndTroughs.peaks.length < 2 || peaksAndTroughs.troughs.length < 2) {
      return patterns;
    }

    // Find parallel trend lines
    const upperTrendLine = this.calculateTrendLine(peaksAndTroughs.peaks);
    const lowerTrendLine = this.calculateTrendLine(peaksAndTroughs.troughs);

    // Check if trend lines are roughly parallel
    const slopeDifference = Math.abs(upperTrendLine.slope - lowerTrendLine.slope);
    
    if (slopeDifference < 0.001) { // Parallel lines
      const confidence = this.calculateChannelConfidence(peaksAndTroughs, upperTrendLine, lowerTrendLine);
      
      if (confidence > 0.6) {
        patterns.push({
          type: 'channel',
          name: 'Price Channel',
          confidence,
          startIndex: Math.min(...peaksAndTroughs.peaks.map(p => p.index), ...peaksAndTroughs.troughs.map(p => p.index)),
          endIndex: Math.max(...peaksAndTroughs.peaks.map(p => p.index), ...peaksAndTroughs.troughs.map(p => p.index)),
          keyPoints: [...peaksAndTroughs.peaks, ...peaksAndTroughs.troughs],
          signal: 'neutral',
          description: 'Price channel with parallel support and resistance lines'
        });
      }
    }

    return patterns;
  }

  private calculateTrendStrength(marketData: MarketData[], startIndex: number, endIndex: number): number {
    const startPrice = marketData[startIndex].close;
    const endPrice = marketData[endIndex].close;
    return (endPrice - startPrice) / startPrice;
  }

  private detectConsolidation(marketData: MarketData[], startIndex: number, endIndex: number): { isConsolidating: boolean, confidence: number } {
    const prices = marketData.slice(startIndex, endIndex + 1).map(d => d.close);
    const volatility = standardDeviation(prices) / mean(prices);
    
    const isConsolidating = volatility < 0.01; // Low volatility indicates consolidation
    const confidence = Math.max(0.5, 1 - volatility * 50);
    
    return { isConsolidating, confidence };
  }

  private calculateChannelConfidence(
    peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] },
    upperTrend: { slope: number, intercept: number },
    lowerTrend: { slope: number, intercept: number }
  ): number {
    // Similar to triangle confidence calculation
    let totalFit = 0;
    let pointCount = 0;

    for (const point of peaksAndTroughs.peaks) {
      const expectedPrice = upperTrend.slope * point.index + upperTrend.intercept;
      const error = Math.abs(point.price - expectedPrice) / point.price;
      totalFit += 1 - error;
      pointCount++;
    }

    for (const point of peaksAndTroughs.troughs) {
      const expectedPrice = lowerTrend.slope * point.index + lowerTrend.intercept;
      const error = Math.abs(point.price - expectedPrice) / point.price;
      totalFit += 1 - error;
      pointCount++;
    }

    return Math.max(0.5, Math.min(1, totalFit / pointCount));
  }

  private findSupportResistanceLevels(
    marketData: MarketData[], 
    peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] }
  ): { support: SupportResistanceLevel[], resistance: SupportResistanceLevel[] } {
    const support: SupportResistanceLevel[] = [];
    const resistance: SupportResistanceLevel[] = [];

    // Group similar price levels
    const tolerance = 0.002; // 0.2% tolerance

    // Process troughs for support levels
    const supportGroups = this.groupSimilarLevels(peaksAndTroughs.troughs, tolerance);
    for (const group of supportGroups) {
      if (group.length >= 2) { // At least 2 touches
        const avgPrice = mean(group.map(p => p.price));
        const strength = Math.min(1, group.length / 5); // Max strength at 5 touches
        
        support.push({
          price: avgPrice,
          strength,
          touches: group.length,
          firstTouch: Math.min(...group.map(p => p.timestamp)),
          lastTouch: Math.max(...group.map(p => p.timestamp)),
          type: 'support'
        });
      }
    }

    // Process peaks for resistance levels
    const resistanceGroups = this.groupSimilarLevels(peaksAndTroughs.peaks, tolerance);
    for (const group of resistanceGroups) {
      if (group.length >= 2) { // At least 2 touches
        const avgPrice = mean(group.map(p => p.price));
        const strength = Math.min(1, group.length / 5); // Max strength at 5 touches
        
        resistance.push({
          price: avgPrice,
          strength,
          touches: group.length,
          firstTouch: Math.min(...group.map(p => p.timestamp)),
          lastTouch: Math.max(...group.map(p => p.timestamp)),
          type: 'resistance'
        });
      }
    }

    return { 
      support: support.sort((a, b) => b.strength - a.strength).slice(0, 5),
      resistance: resistance.sort((a, b) => b.strength - a.strength).slice(0, 5)
    };
  }

  private groupSimilarLevels(points: PatternPoint[], tolerance: number): PatternPoint[][] {
    const groups: PatternPoint[][] = [];
    const used = new Set<number>();

    for (let i = 0; i < points.length; i++) {
      if (used.has(i)) continue;

      const group = [points[i]];
      used.add(i);

      for (let j = i + 1; j < points.length; j++) {
        if (used.has(j)) continue;

        const priceDiff = Math.abs(points[i].price - points[j].price) / points[i].price;
        if (priceDiff <= tolerance) {
          group.push(points[j]);
          used.add(j);
        }
      }

      groups.push(group);
    }

    return groups;
  }

  private findTrendLines(peaksAndTroughs: { peaks: PatternPoint[], troughs: PatternPoint[] }): TrendLine[] {
    const trendLines: TrendLine[] = [];

    // Create trend lines from peaks (resistance trend lines)
    if (peaksAndTroughs.peaks.length >= 2) {
      for (let i = 0; i < peaksAndTroughs.peaks.length - 1; i++) {
        const start = peaksAndTroughs.peaks[i];
        const end = peaksAndTroughs.peaks[i + 1];
        const slope = (end.price - start.price) / (end.index - start.index);
        
        trendLines.push({
          startPoint: start,
          endPoint: end,
          slope,
          strength: 0.7,
          type: 'resistance',
          equation: `y = ${slope.toFixed(6)}x + ${(start.price - slope * start.index).toFixed(6)}`
        });
      }
    }

    // Create trend lines from troughs (support trend lines)
    if (peaksAndTroughs.troughs.length >= 2) {
      for (let i = 0; i < peaksAndTroughs.troughs.length - 1; i++) {
        const start = peaksAndTroughs.troughs[i];
        const end = peaksAndTroughs.troughs[i + 1];
        const slope = (end.price - start.price) / (end.index - start.index);
        
        trendLines.push({
          startPoint: start,
          endPoint: end,
          slope,
          strength: 0.7,
          type: 'support',
          equation: `y = ${slope.toFixed(6)}x + ${(start.price - slope * start.index).toFixed(6)}`
        });
      }
    }

    return trendLines;
  }

  private generatePatternSignals(
    patterns: DetectedPattern[], 
    supportResistance: { support: SupportResistanceLevel[], resistance: SupportResistanceLevel[] },
    marketData: MarketData[]
  ): PatternSignal[] {
    const signals: PatternSignal[] = [];
    const currentPrice = marketData[marketData.length - 1].close;

    // Generate signals from patterns
    for (const pattern of patterns) {
      if (pattern.confidence > 0.7) {
        signals.push({
          type: pattern.signal === 'bullish' ? 'breakout' : pattern.signal === 'bearish' ? 'breakdown' : 'reversal',
          pattern: pattern.name,
          strength: pattern.confidence,
          confidence: pattern.confidence,
          direction: pattern.signal,
          entry: currentPrice,
          target: pattern.target,
          stopLoss: pattern.stopLoss
        });
      }
    }

    // Generate signals from support/resistance breaks
    for (const support of supportResistance.support) {
      if (currentPrice < support.price * 0.999) { // Price broke below support
        signals.push({
          type: 'breakdown',
          pattern: 'Support Break',
          strength: support.strength,
          confidence: support.strength,
          direction: 'bearish',
          entry: currentPrice,
          target: support.price - (support.price * 0.01),
          stopLoss: support.price * 1.005
        });
      }
    }

    for (const resistance of supportResistance.resistance) {
      if (currentPrice > resistance.price * 1.001) { // Price broke above resistance
        signals.push({
          type: 'breakout',
          pattern: 'Resistance Break',
          strength: resistance.strength,
          confidence: resistance.strength,
          direction: 'bullish',
          entry: currentPrice,
          target: resistance.price + (resistance.price * 0.01),
          stopLoss: resistance.price * 0.995
        });
      }
    }

    return signals.sort((a, b) => b.confidence - a.confidence).slice(0, 5); // Top 5 signals
  }

  private generateTestData(): MarketData[] {
    const data: MarketData[] = [];
    let price = 1.0;
    
    for (let i = 0; i < 100; i++) {
      price += (Math.random() - 0.5) * 0.01;
      data.push({
        timestamp: Date.now() + i * 60000,
        open: price,
        high: price + Math.random() * 0.005,
        low: price - Math.random() * 0.005,
        close: price,
        volume: 1000 + Math.random() * 5000
      });
    }
    
    return data;
  }

  // Integration testing methods for humanitarian mission validation
  async runIntegrationTests(): Promise<boolean> {
    this.logger.info('🧪 Running Pattern Recognition Engine integration tests...');

    try {
      // Test 1: Python pattern engine connectivity
      const pingTest = await this.pythonInterface.sendCommand('ping', { 
        test: 'pattern_integration',
        engine_type: 'pattern_recognition'
      });
      if (pingTest.status !== 'pong') {
        throw new Error('Python pattern engine ping test failed');
      }

      // Test 2: Pattern detection with sample data
      const sampleData: MarketData[] = this.generateSamplePatternData();
      const patternResult = await this.detectPatterns('TEST_SYMBOL', sampleData);
      
      if (!patternResult || !patternResult.patterns || patternResult.patterns.length === 0) {
        throw new Error('Pattern detection test failed');
      }

      // Test 3: Support/resistance detection test
      if (patternResult.supportLevels.length === 0 || patternResult.resistanceLevels.length === 0) {
        throw new Error('Support/resistance detection test failed');
      }

      // Test 4: AI enhancement validation
      const aiTest = await this.pythonInterface.sendCommand('validate_ai_enhancement', {
        module: 'pattern_recognition',
        test_type: 'advanced_patterns'
      });
      
      if (!aiTest.enabled) {
        throw new Error('AI enhancement not enabled in Python pattern engine');
      }

      // Test 5: Humanitarian mode validation
      const humanitarianTest = await this.pythonInterface.sendCommand('validate_humanitarian_pattern_mode', {});
      if (!humanitarianTest.enabled) {
        throw new Error('Humanitarian pattern mode not enabled in Python engine');
      }

      this.logger.info('✅ All Pattern Recognition Engine integration tests passed');
      return true;
    } catch (error) {
      this.logger.error('❌ Pattern integration tests failed:', error);
      return false;
    }
  }

  private generateSamplePatternData(): MarketData[] {
    const data: MarketData[] = [];
    let price = 1.2000;
    
    // Generate data with a potential double bottom pattern
    for (let i = 0; i < 50; i++) {
      if (i < 20) {
        price -= 0.001 + (Math.random() * 0.001);
      } else if (i < 30) {
        price += 0.0015 + (Math.random() * 0.001);
      } else if (i < 40) {
        price -= 0.001 + (Math.random() * 0.001);
      } else {
        price += 0.002 + (Math.random() * 0.001);
      }

      data.push({
        timestamp: Date.now() - (50 - i) * 60000,
        open: price - 0.0005,
        high: price + 0.001,
        low: price - 0.001,
        close: price,
        volume: 1000 + Math.random() * 5000
      });
    }
    
    return data;
  }
}
