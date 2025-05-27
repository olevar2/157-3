/**
 * Comprehensive Indicator Engine for Platform3
 * Integrates the 67-indicator adapter with the analytics service
 */

import { spawn } from 'child_process';
import path from 'path';
import winston from 'winston';

export interface MarketDataInput {
  timestamps: number[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface IndicatorResult {
  indicator_name: string;
  category: string;
  values: any;
  signals: any;
  metadata: any;
  calculation_time: number;
  success: boolean;
  error_message?: string;
}

export interface ComprehensiveAnalysisResult {
  symbol: string;
  timeframe: string;
  timestamp: string;
  total_indicators: number;
  successful_indicators: number;
  failed_indicators: number;
  success_rate: number;
  total_calculation_time: number;
  categories: {
    [category: string]: {
      indicators: IndicatorResult[];
      success_count: number;
      total_count: number;
      success_rate: number;
    };
  };
  summary: {
    momentum: IndicatorResult[];
    trend: IndicatorResult[];
    volatility: IndicatorResult[];
    volume: IndicatorResult[];
    cycle: IndicatorResult[];
    advanced: IndicatorResult[];
    gann: IndicatorResult[];
    scalping: IndicatorResult[];
    daytrading: IndicatorResult[];
    swingtrading: IndicatorResult[];
    signals: IndicatorResult[];
  };
}

export class ComprehensiveIndicatorEngine {
  private logger: winston.Logger;
  private pythonScriptPath: string;
  private isInitialized: boolean = false;

  constructor(logger: winston.Logger) {
    this.logger = logger;
    this.pythonScriptPath = path.resolve(__dirname, '../scripts/indicator_bridge.py');
  }

  async initialize(): Promise<void> {
    try {
      this.logger.info('🚀 Initializing Comprehensive Indicator Engine...');

      // Verify Python script exists
      const fs = require('fs');
      if (!fs.existsSync(this.pythonScriptPath)) {
        throw new Error(`Python adapter not found at: ${this.pythonScriptPath}`);
      }

      // Test the adapter
      const testResult = await this.testAdapter();
      if (!testResult) {
        throw new Error('Adapter test failed');
      }

      this.isInitialized = true;
      this.logger.info('✅ Comprehensive Indicator Engine initialized - 67 indicators ready');
    } catch (error) {
      this.logger.error('❌ Failed to initialize Comprehensive Indicator Engine:', error);
      throw error;
    }
  }
  private async testAdapter(): Promise<boolean> {
    return new Promise((resolve) => {
      this.logger.info(`Spawning Python test from working directory: ${process.cwd()}`);
      this.logger.info(`Python script path: ${this.pythonScriptPath}`);
        const testProcess = spawn('python', [
        this.pythonScriptPath,
        '--test'
      ], {
        cwd: path.dirname(this.pythonScriptPath), // Set working directory to script location
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' } // Set UTF-8 encoding
      });

      let output = '';
      let errorOutput = '';

      testProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      testProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      testProcess.on('close', (code) => {
        this.logger.info(`Test process output: ${output}`);
        this.logger.info(`Test process error output: ${errorOutput}`);
        this.logger.info(`Test process exit code: ${code}`);
        if (code === 0 && output.includes('TEST_SUCCESS')) {
          this.logger.info('✅ Adapter test passed');
          resolve(true);
        } else {
          this.logger.error('❌ Adapter test failed');
          this.logger.error(`Expected TEST_SUCCESS in output, got: ${output}`);
          resolve(false);
        }
      });

      testProcess.on('error', (error) => {
        this.logger.error('❌ Python process error:', error);
        resolve(false);
      });
    });
  }

  async calculateIndicator(
    indicatorName: string,
    marketData: MarketDataInput,
    symbol: string,
    timeframe: string = '1h'
  ): Promise<IndicatorResult> {
    if (!this.isInitialized) {
      throw new Error('Engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const inputData = {
        action: 'calculate_single',
        indicator_name: indicatorName,
        market_data: marketData,
        symbol,
        timeframe
      };

      const pythonProcess = spawn('python', [this.pythonScriptPath]);

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`Python process failed with code ${code}: ${errorOutput}`);
          reject(new Error(`Indicator calculation failed: ${errorOutput}`));
          return;
        }

        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (error) {
          this.logger.error('Failed to parse indicator result:', error);
          reject(new Error('Invalid JSON response from indicator calculation'));
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python process error:', error);
        reject(error);
      });

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();
    });
  }

  async calculateAllIndicators(
    marketData: MarketDataInput,
    symbol: string,
    timeframe: string = '1h'
  ): Promise<ComprehensiveAnalysisResult> {
    if (!this.isInitialized) {
      throw new Error('Engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const inputData = {
        action: 'calculate_all',
        market_data: marketData,
        symbol,
        timeframe
      };

      const pythonProcess = spawn('python', [this.pythonScriptPath]);

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`Python process failed with code ${code}: ${errorOutput}`);
          reject(new Error(`Comprehensive analysis failed: ${errorOutput}`));
          return;
        }

        try {
          const result = JSON.parse(output);
          this.logger.info(`✅ Comprehensive analysis completed: ${result.successful_indicators}/${result.total_indicators} indicators (${result.success_rate.toFixed(1)}%)`);
          resolve(result);
        } catch (error) {
          this.logger.error('Failed to parse comprehensive analysis result:', error);
          reject(new Error('Invalid JSON response from comprehensive analysis'));
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python process error:', error);
        reject(error);
      });

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();
    });
  }

  async batchCalculateIndicators(
    indicatorNames: string[],
    marketData: MarketDataInput,
    symbol: string,
    timeframe: string = '1h'
  ): Promise<{ [indicatorName: string]: IndicatorResult }> {
    if (!this.isInitialized) {
      throw new Error('Engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const inputData = {
        action: 'calculate_batch',
        indicator_names: indicatorNames,
        market_data: marketData,
        symbol,
        timeframe
      };

      const pythonProcess = spawn('python', [this.pythonScriptPath]);

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`Python process failed with code ${code}: ${errorOutput}`);
          reject(new Error(`Batch calculation failed: ${errorOutput}`));
          return;
        }

        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (error) {
          this.logger.error('Failed to parse batch calculation result:', error);
          reject(new Error('Invalid JSON response from batch calculation'));
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python process error:', error);
        reject(error);
      });

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();
    });
  }

  async getAvailableIndicators(): Promise<{ [category: string]: string[] }> {
    if (!this.isInitialized) {
      throw new Error('Engine not initialized');
    }

    return new Promise((resolve, reject) => {
      const inputData = {
        action: 'list_indicators'
      };

      const pythonProcess = spawn('python', [this.pythonScriptPath]);

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`Python process failed with code ${code}: ${errorOutput}`);
          reject(new Error(`List indicators failed: ${errorOutput}`));
          return;
        }

        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (error) {
          this.logger.error('Failed to parse indicator list:', error);
          reject(new Error('Invalid JSON response from indicator list'));
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python process error:', error);
        reject(error);
      });

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();
    });
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  getPerformanceStats(): any {
    return {
      initialized: this.isInitialized,
      script_path: this.pythonScriptPath,
      total_indicators: 67
    };
  }
}
