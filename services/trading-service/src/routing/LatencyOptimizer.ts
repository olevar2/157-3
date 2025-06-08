import axios from 'axios';
import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';
import { ExecutionVenue } from './SpeedOptimizedExecution';

/**
 * Latency Optimizer
 * Optimizes order routing to minimize execution latency
 */

export interface LatencyMetrics {
  venueId: string;
  avgLatency: number;
  minLatency: number;
  maxLatency: number;
  samples: number;
  lastUpdated: number;
}

export interface RouteCandidate {
  venue: string;
  expectedLatency: number;
  confidence: number;       // 0-1 confidence in latency estimate
  cost: number;            // Execution cost estimate
  liquidity: number;       // Available liquidity
  score: number;           // Overall routing score
}

export interface LatencyMeasurement {
  venue: string;
  latency: number;
  timestamp: Date;
  orderType: string;
  success: boolean;
}

export class LatencyOptimizer extends EventEmitter {
  private venueMetrics: Map<string, LatencyMetrics> = new Map();
  private latencyHistory: Map<string, LatencyMeasurement[]> = new Map();
  private pythonEngineUrl: string;
  private measurementWindow = 1000; // Keep last 1000 measurements per venue
  private updateInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
    
    // Initialize with default venues and baseline metrics
    this.initializeDefaultVenues();
    
    // Start periodic metrics update
    this.startPeriodicUpdates();
    
    console.log('LatencyOptimizer initialized');
  }

  /**
   * Initialize default trading venues with baseline latency estimates
   */
  private initializeDefaultVenues(): void {
    const defaultVenues = [
      { id: 'binance', name: 'Binance', baseLatency: 5 },
      { id: 'coinbase', name: 'Coinbase', baseLatency: 10 },
      { id: 'kraken', name: 'Kraken', baseLatency: 15 }
    ];

    defaultVenues.forEach(venue => {
      this.venueMetrics.set(venue.id, {
        venueId: venue.id,
        avgLatency: venue.baseLatency,
        minLatency: venue.baseLatency,
        maxLatency: venue.baseLatency,
        samples: 1,
        lastUpdated: Date.now()
      });
    });
  }

  /**
   * Record a latency measurement for a venue
   */
  public recordLatencyMeasurement(measurement: LatencyMeasurement): void {
    // Add to history
    if (!this.latencyHistory.has(measurement.venue)) {
      this.latencyHistory.set(measurement.venue, []);
    }

    const history = this.latencyHistory.get(measurement.venue)!;
    history.push(measurement);

    // Keep only recent measurements
    if (history.length > this.measurementWindow) {
      history.splice(0, history.length - this.measurementWindow);
    }

    // Update metrics for this venue
    this.updateVenueMetrics(measurement.venue);

    console.log(`Recorded latency for ${measurement.venue}: ${measurement.latency}ms`);
  }

  /**
   * Update metrics for a specific venue based on historical measurements
   */
  private updateVenueMetrics(venue: string): void {
    const history = this.latencyHistory.get(venue);
    if (!history || history.length === 0) return;

    // Only consider successful measurements
    const successfulMeasurements = history.filter(m => m.success).map(m => m.latency);
    if (successfulMeasurements.length === 0) return;

    // Calculate metrics
    const sortedLatencies = successfulMeasurements.sort((a, b) => a - b);
    const averageLatency = sortedLatencies.reduce((sum, lat) => sum + lat, 0) / sortedLatencies.length;
    const p95Index = Math.floor(sortedLatencies.length * 0.95);
    const p99Index = Math.floor(sortedLatencies.length * 0.99);

    const updatedMetrics: LatencyMetrics = {
      venueId: venue,
      avgLatency: averageLatency,
      minLatency: sortedLatencies[0] || averageLatency,
      maxLatency: sortedLatencies[sortedLatencies.length - 1] || averageLatency,
      samples: successfulMeasurements.length,
      lastUpdated: Date.now()
    };

    this.venueMetrics.set(venue, updatedMetrics);

    this.emit('metricsUpdated', { venue, metrics: updatedMetrics });
  }

  /**
   * Find the optimal venue for routing based on latency and other factors
   */
  public async findOptimalRoute(
    symbol: string,
    orderSize: number,
    orderType: 'MARKET' | 'LIMIT',
    urgency: 'LOW' | 'MEDIUM' | 'HIGH' = 'MEDIUM'
  ): Promise<RouteCandidate[]> {
    const candidates: RouteCandidate[] = [];

    // Get current market data and liquidity from Python engines
    const marketData = await this.getMarketData(symbol);

    for (const [venue, metrics] of this.venueMetrics) {
      // Calculate expected latency based on order type and urgency
      let expectedLatency = metrics.avgLatency;
      
      if (orderType === 'MARKET') {
        expectedLatency = metrics.avgLatency; // Market orders typically faster
      } else {
        expectedLatency = metrics.avgLatency * 1.2; // Limit orders may take longer
      }

      if (urgency === 'HIGH') {
        expectedLatency = metrics.maxLatency; // Use more conservative estimate for urgent orders
      } else if (urgency === 'LOW') {
        expectedLatency = metrics.avgLatency * 0.8; // Optimistic for non-urgent orders
      }

      // Estimate execution cost (simplified)
      const estimatedCost = this.estimateExecutionCost(venue, symbol, orderSize, orderType);

      // Estimate available liquidity (simplified)
      const estimatedLiquidity = this.estimateLiquidity(venue, symbol, marketData);

      // Calculate confidence based on measurement count and recency
      const confidence = Math.min(1.0, metrics.samples / 100) * 
                        Math.max(0.5, 1 - (Date.now() - metrics.lastUpdated) / (24 * 60 * 60 * 1000));

      // Calculate overall score (lower is better for latency, higher for liquidity)
      const latencyScore = 1 / (expectedLatency + 1); // Inverse of latency
      const liquidityScore = Math.min(1, estimatedLiquidity / orderSize);
      const costScore = 1 / (estimatedCost + 1); // Inverse of cost
      
      const score = (latencyScore * 0.4 + liquidityScore * 0.3 + costScore * 0.3) * confidence;

      candidates.push({
        venue,
        expectedLatency,
        confidence,
        cost: estimatedCost,
        liquidity: estimatedLiquidity,
        score
      });
    }

    // Sort by score (highest first)
    candidates.sort((a, b) => b.score - a.score);

    console.log(`Optimal routes for ${symbol} (${orderType}):`, 
                candidates.slice(0, 3).map(c => `${c.venue}: ${c.expectedLatency.toFixed(1)}ms (score: ${c.score.toFixed(3)})`));

    return candidates;
  }

  /**
   * Get market data from Python engines
   */
  private async getMarketData(symbol: string): Promise<any> {
    try {
      const response = await axios.get(`${this.pythonEngineUrl}/api/market-data/${symbol}`);
      return response.data;
    } catch (error) {
      console.warn(`Failed to get market data for ${symbol}, using defaults`);
      return { spread: 0.0002, volume: 1000000 }; // Default values
    }
  }

  /**
   * Estimate execution cost for a venue
   */
  private estimateExecutionCost(venue: string, symbol: string, orderSize: number, orderType: string): number {
    // Simplified cost model - in reality this would be much more sophisticated
    const baseCost = 0.00015; // 1.5 pips base cost
    
    // Different venues have different cost structures
    const venueCostMultipliers: { [key: string]: number } = {
      'FXPro': 1.0,
      'ICMarkets': 0.8,
      'Pepperstone': 0.9,
      'FXCM': 1.2,
      'Oanda': 1.1,
      'Interactive_Brokers': 0.7
    };

    const venueMultiplier = venueCostMultipliers[venue] || 1.0;
    const sizeMultiplier = Math.log(orderSize / 100000 + 1) / 10; // Larger orders cost more
    const typeMultiplier = orderType === 'MARKET' ? 1.0 : 0.8; // Limit orders typically cheaper

    return baseCost * venueMultiplier * (1 + sizeMultiplier) * typeMultiplier;
  }

  /**
   * Estimate available liquidity for a venue
   */
  private estimateLiquidity(venue: string, symbol: string, marketData: any): number {
    // Simplified liquidity model
    const baseLiquidity = marketData?.volume || 1000000;
    
    // Different venues have different liquidity
    const venueLiquidityMultipliers: { [key: string]: number } = {
      'FXPro': 1.0,
      'ICMarkets': 1.2,
      'Pepperstone': 0.9,
      'FXCM': 1.1,
      'Oanda': 0.8,
      'Interactive_Brokers': 1.3
    };

    const venueMultiplier = venueLiquidityMultipliers[venue] || 1.0;
    return baseLiquidity * venueMultiplier;
  }

  /**
   * Get current metrics for a venue
   */
  public getVenueMetrics(venue: string): LatencyMetrics | undefined {
    return this.venueMetrics.get(venue);
  }

  /**
   * Get all venue metrics
   */
  public getAllVenueMetrics(): Map<string, LatencyMetrics> {
    return new Map(this.venueMetrics);
  }

  /**
   * Get recent latency measurements for a venue
   */
  public getRecentMeasurements(venue: string, count: number = 10): LatencyMeasurement[] {
    const history = this.latencyHistory.get(venue) || [];
    return history.slice(-count);
  }

  /**
   * Start periodic metric updates
   */
  private startPeriodicUpdates(): void {
    this.updateInterval = setInterval(() => {
      this.updateLatencyMetrics();
    }, 5000); // Update every 5 seconds
  }

  /**
   * Stop periodic updates
   */
  public dispose(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    console.log('LatencyOptimizer disposed');
  }

  /**
   * Benchmark latency for all venues
   */
  public async benchmarkAllVenues(): Promise<void> {
    console.log('Starting latency benchmark for all venues...');
    
    const venues = Array.from(this.venueMetrics.keys());
    
    for (const venue of venues) {
      try {
        // Simulate latency measurement
        const startTime = performance.now();
        await this.simulateVenueLatency(venue);
        const latency = performance.now() - startTime;
        
        this.recordLatencyMeasurement({
          venue,
          latency,
          timestamp: new Date(),
          orderType: 'MARKET',
          success: true
        });
        
        console.log(`Benchmarked ${venue}: ${latency.toFixed(2)}ms`);
      } catch (error) {
        console.error(`Failed to benchmark ${venue}:`, error);
      }
    }
  }

  /**
   * Simulate latency for a venue (for testing/benchmarking)
   */
  private async simulateVenueLatency(venue: string): Promise<void> {
    const baseLatency = this.venueMetrics.get(venue)?.avgLatency || 20;
    const jitter = (Math.random() - 0.5) * 10; // ±5ms jitter
    const delay = Math.max(5, baseLatency + jitter);
    
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  private async updateLatencyMetrics(): Promise<void> {
    for (const [venueId, metrics] of this.venueMetrics.entries()) {
      const latency = await this.measureVenueLatency(venueId);
      this.updateVenueMetrics(venueId, latency);
    }
  }

  private async measureVenueLatency(venueId: string): Promise<number> {
    const start = performance.now();
    
    try {
      // Simulate latency measurement (replace with actual venue ping)
      await new Promise(resolve => setTimeout(resolve, Math.random() * 20));
      
      const latency = performance.now() - start;
      return latency;
    } catch (error) {
      console.error(`Failed to measure latency for ${venueId}:`, error);
      return 999; // High penalty for failed measurements
    }
  }

  private updateVenueMetrics(venueId: string, newLatency: number): void {
    const metrics = this.venueMetrics.get(venueId);
    if (!metrics) return;

    // Update with exponential moving average
    const alpha = 0.3; // Smoothing factor
    metrics.avgLatency = alpha * newLatency + (1 - alpha) * metrics.avgLatency;
    metrics.minLatency = Math.min(metrics.minLatency, newLatency);
    metrics.maxLatency = Math.max(metrics.maxLatency, newLatency);
    metrics.samples++;
    metrics.lastUpdated = Date.now();

    this.emit('metricsUpdated', { venueId, metrics });
  }

  public getOptimalVenue(venues: ExecutionVenue[]): ExecutionVenue | null {
    if (venues.length === 0) return null;

    let optimalVenue = venues[0];
    let lowestLatency = Infinity;

    for (const venue of venues) {
      const metrics = this.venueMetrics.get(venue.id);
      if (metrics && metrics.avgLatency < lowestLatency) {
        lowestLatency = metrics.avgLatency;
        optimalVenue = venue;
      }
    }

    return optimalVenue;
  }

  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}
  }

  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}
