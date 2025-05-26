/**
 * Day Trading Position Sizer
 * 
 * Advanced position sizing system for day trading strategies with dynamic
 * risk-based sizing, volatility adjustments, and session-based optimization.
 * 
 * Key Features:
 * - Kelly Criterion-based position sizing
 * - Volatility-adjusted sizing
 * - Session-based risk adjustments
 * - Portfolio heat management
 * - Dynamic stop-loss positioning
 * - Risk-reward optimization
 * 
 * Author: Platform3 Trading Team
 * Version: 1.0.0
 */

import { EventEmitter } from 'events';

export enum SizingMethod {
    FIXED = 'FIXED',
    KELLY = 'KELLY',
    VOLATILITY_ADJUSTED = 'VOLATILITY_ADJUSTED',
    RISK_PARITY = 'RISK_PARITY',
    OPTIMAL_F = 'OPTIMAL_F'
}

export enum SessionType {
    LONDON_OPEN = 'LONDON_OPEN',
    NY_OPEN = 'NY_OPEN',
    OVERLAP = 'OVERLAP',
    ASIAN = 'ASIAN',
    OFF_HOURS = 'OFF_HOURS'
}

export interface SizingParameters {
    basePositionSize: number;
    maxPositionSize: number;
    riskPerTrade: number; // Percentage of account
    maxPortfolioHeat: number; // Maximum total risk exposure
    kellyFraction: number; // Kelly multiplier (0.25 = quarter Kelly)
    volatilityLookback: number; // Days for volatility calculation
    sessionMultipliers: Record<SessionType, number>;
    correlationThreshold: number;
    maxDrawdownThreshold: number;
}

export interface MarketData {
    symbol: string;
    price: number;
    volatility: number;
    atr: number; // Average True Range
    volume: number;
    spread: number;
    timestamp: number;
}

export interface StrategyStats {
    symbol: string;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    profitFactor: number;
    sharpeRatio: number;
    maxDrawdown: number;
    totalTrades: number;
    recentPerformance: number[]; // Last 20 trades
}

export interface PositionSizeRequest {
    symbol: string;
    strategy: string;
    direction: 'LONG' | 'SHORT';
    entryPrice: number;
    stopLoss: number;
    takeProfit?: number;
    confidence: number; // 0-1 signal confidence
    sessionType: SessionType;
    accountBalance: number;
    currentPositions: Position[];
}

export interface Position {
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
    riskAmount: number;
}

export interface SizingResult {
    recommendedSize: number;
    maxAllowedSize: number;
    riskAmount: number;
    riskRewardRatio: number;
    sizingMethod: SizingMethod;
    adjustmentFactors: {
        volatility: number;
        session: number;
        correlation: number;
        heat: number;
        performance: number;
        kelly: number;
    };
    warnings: string[];
    confidence: number;
}

export class DayTradingPositionSizer extends EventEmitter {
    private sizingParams: SizingParameters;
    private marketData: Map<string, MarketData> = new Map();
    private strategyStats: Map<string, StrategyStats> = new Map();
    private correlationMatrix: Map<string, Map<string, number>> = new Map();
    
    // Performance tracking
    private sizingHistory: SizingResult[] = [];
    private performanceMetrics = {
        totalSizings: 0,
        averageSize: 0,
        riskUtilization: 0
    };

    constructor(sizingParams: SizingParameters) {
        super();
        this.sizingParams = { ...sizingParams };
        
        console.log('✅ DayTradingPositionSizer initialized');
    }

    /**
     * Calculate optimal position size
     */
    public calculatePositionSize(request: PositionSizeRequest): SizingResult {
        try {
            const marketData = this.marketData.get(request.symbol);
            const strategyStats = this.strategyStats.get(request.strategy);
            
            if (!marketData) {
                throw new Error(`No market data available for ${request.symbol}`);
            }

            // Calculate base risk amount
            const riskAmount = request.accountBalance * (this.sizingParams.riskPerTrade / 100);
            
            // Calculate stop distance in price terms
            const stopDistance = Math.abs(request.entryPrice - request.stopLoss);
            
            // Base position size (risk-based)
            const baseSize = riskAmount / stopDistance;
            
            // Apply sizing method
            let adjustedSize = this.applySizingMethod(
                baseSize, request, marketData, strategyStats
            );
            
            // Calculate adjustment factors
            const adjustmentFactors = this.calculateAdjustmentFactors(
                request, marketData, strategyStats
            );
            
            // Apply adjustments
            adjustedSize = this.applyAdjustments(adjustedSize, adjustmentFactors);
            
            // Apply limits and constraints
            const finalSize = this.applyConstraints(adjustedSize, request);
            
            // Calculate risk-reward ratio
            const riskRewardRatio = request.takeProfit 
                ? Math.abs(request.takeProfit - request.entryPrice) / stopDistance
                : 0;
            
            // Generate warnings
            const warnings = this.generateWarnings(request, finalSize, adjustmentFactors);
            
            const result: SizingResult = {
                recommendedSize: Math.floor(finalSize),
                maxAllowedSize: this.sizingParams.maxPositionSize,
                riskAmount: finalSize * stopDistance,
                riskRewardRatio,
                sizingMethod: this.determineSizingMethod(strategyStats),
                adjustmentFactors,
                warnings,
                confidence: this.calculateConfidence(request, adjustmentFactors)
            };
            
            // Track performance
            this.trackSizingResult(result);
            
            this.emit('positionSized', result);
            
            return result;
            
        } catch (error) {
            console.error('Error calculating position size:', error);
            return this.getDefaultSizing(request);
        }
    }

    /**
     * Apply sizing method
     */
    private applySizingMethod(
        baseSize: number,
        request: PositionSizeRequest,
        marketData: MarketData,
        strategyStats?: StrategyStats
    ): number {
        const method = this.determineSizingMethod(strategyStats);
        
        switch (method) {
            case SizingMethod.KELLY:
                return this.applyKellySizing(baseSize, strategyStats);
                
            case SizingMethod.VOLATILITY_ADJUSTED:
                return this.applyVolatilitySizing(baseSize, marketData);
                
            case SizingMethod.RISK_PARITY:
                return this.applyRiskParitySizing(baseSize, request);
                
            case SizingMethod.OPTIMAL_F:
                return this.applyOptimalFSizing(baseSize, strategyStats);
                
            default:
                return baseSize;
        }
    }

    /**
     * Apply Kelly Criterion sizing
     */
    private applyKellySizing(baseSize: number, strategyStats?: StrategyStats): number {
        if (!strategyStats || strategyStats.totalTrades < 20) {
            return baseSize; // Not enough data for Kelly
        }
        
        const winRate = strategyStats.winRate;
        const avgWin = strategyStats.avgWin;
        const avgLoss = Math.abs(strategyStats.avgLoss);
        
        if (avgLoss === 0) return baseSize;
        
        // Kelly formula: f = (bp - q) / b
        // where b = avgWin/avgLoss, p = winRate, q = 1-winRate
        const b = avgWin / avgLoss;
        const p = winRate;
        const q = 1 - winRate;
        
        const kellyFraction = (b * p - q) / b;
        
        // Apply Kelly fraction multiplier for safety
        const adjustedKelly = kellyFraction * this.sizingParams.kellyFraction;
        
        return baseSize * Math.max(0.1, Math.min(2.0, adjustedKelly));
    }

    /**
     * Apply volatility-based sizing
     */
    private applyVolatilitySizing(baseSize: number, marketData: MarketData): number {
        // Inverse volatility sizing - higher volatility = smaller size
        const volatilityAdjustment = 1 / (1 + marketData.volatility * 2);
        return baseSize * volatilityAdjustment;
    }

    /**
     * Apply risk parity sizing
     */
    private applyRiskParitySizing(baseSize: number, request: PositionSizeRequest): number {
        // Adjust based on current portfolio concentration
        const symbolExposure = this.calculateSymbolExposure(request.symbol, request.currentPositions);
        const totalExposure = this.calculateTotalExposure(request.currentPositions);
        
        if (totalExposure === 0) return baseSize;
        
        const concentrationRatio = symbolExposure / totalExposure;
        const maxConcentration = 0.3; // 30% max per symbol
        
        if (concentrationRatio > maxConcentration) {
            return baseSize * (maxConcentration / concentrationRatio);
        }
        
        return baseSize;
    }

    /**
     * Apply Optimal F sizing
     */
    private applyOptimalFSizing(baseSize: number, strategyStats?: StrategyStats): number {
        if (!strategyStats || strategyStats.recentPerformance.length < 10) {
            return baseSize;
        }
        
        // Simplified Optimal F calculation
        const returns = strategyStats.recentPerformance;
        const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
        
        if (variance === 0) return baseSize;
        
        const optimalF = avgReturn / variance;
        return baseSize * Math.max(0.1, Math.min(2.0, optimalF));
    }

    /**
     * Calculate adjustment factors
     */
    private calculateAdjustmentFactors(
        request: PositionSizeRequest,
        marketData: MarketData,
        strategyStats?: StrategyStats
    ): SizingResult['adjustmentFactors'] {
        return {
            volatility: this.calculateVolatilityAdjustment(marketData),
            session: this.calculateSessionAdjustment(request.sessionType),
            correlation: this.calculateCorrelationAdjustment(request),
            heat: this.calculateHeatAdjustment(request),
            performance: this.calculatePerformanceAdjustment(strategyStats),
            kelly: strategyStats ? this.calculateKellyAdjustment(strategyStats) : 1.0
        };
    }

    /**
     * Calculate volatility adjustment
     */
    private calculateVolatilityAdjustment(marketData: MarketData): number {
        // Normalize volatility (assuming 2% is normal daily volatility)
        const normalVolatility = 0.02;
        const volatilityRatio = marketData.volatility / normalVolatility;
        
        // Inverse relationship: higher volatility = lower size
        return 1 / Math.sqrt(volatilityRatio);
    }

    /**
     * Calculate session adjustment
     */
    private calculateSessionAdjustment(sessionType: SessionType): number {
        return this.sizingParams.sessionMultipliers[sessionType] || 1.0;
    }

    /**
     * Calculate correlation adjustment
     */
    private calculateCorrelationAdjustment(request: PositionSizeRequest): number {
        const correlations = this.correlationMatrix.get(request.symbol);
        if (!correlations) return 1.0;
        
        let maxCorrelation = 0;
        for (const position of request.currentPositions) {
            const correlation = correlations.get(position.symbol) || 0;
            maxCorrelation = Math.max(maxCorrelation, Math.abs(correlation));
        }
        
        // Reduce size if high correlation with existing positions
        if (maxCorrelation > this.sizingParams.correlationThreshold) {
            return 1 - (maxCorrelation - this.sizingParams.correlationThreshold);
        }
        
        return 1.0;
    }

    /**
     * Calculate heat adjustment
     */
    private calculateHeatAdjustment(request: PositionSizeRequest): number {
        const currentHeat = this.calculatePortfolioHeat(request.currentPositions);
        const maxHeat = this.sizingParams.maxPortfolioHeat;
        
        if (currentHeat >= maxHeat) {
            return 0; // No new positions allowed
        }
        
        const remainingHeat = maxHeat - currentHeat;
        return Math.min(1.0, remainingHeat / (maxHeat * 0.2)); // Scale down as heat increases
    }

    /**
     * Calculate performance adjustment
     */
    private calculatePerformanceAdjustment(strategyStats?: StrategyStats): number {
        if (!strategyStats) return 1.0;
        
        // Adjust based on recent performance and drawdown
        const recentPerformance = strategyStats.recentPerformance.slice(-10);
        const avgRecentReturn = recentPerformance.reduce((sum, ret) => sum + ret, 0) / recentPerformance.length;
        
        // Reduce size if in drawdown
        if (strategyStats.maxDrawdown > this.sizingParams.maxDrawdownThreshold) {
            return 0.5; // Half size during significant drawdown
        }
        
        // Adjust based on recent performance
        if (avgRecentReturn < 0) {
            return 0.8; // Reduce size during poor performance
        }
        
        return 1.0;
    }

    /**
     * Calculate Kelly adjustment
     */
    private calculateKellyAdjustment(strategyStats: StrategyStats): number {
        if (strategyStats.totalTrades < 20) return 1.0;
        
        const winRate = strategyStats.winRate;
        const profitFactor = strategyStats.profitFactor;
        
        // Kelly-based confidence adjustment
        if (winRate > 0.6 && profitFactor > 1.5) {
            return 1.2; // Increase size for high-confidence strategies
        } else if (winRate < 0.4 || profitFactor < 1.1) {
            return 0.7; // Reduce size for low-confidence strategies
        }
        
        return 1.0;
    }

    /**
     * Apply adjustments to position size
     */
    private applyAdjustments(
        baseSize: number,
        adjustments: SizingResult['adjustmentFactors']
    ): number {
        let adjustedSize = baseSize;
        
        // Apply each adjustment factor
        adjustedSize *= adjustments.volatility;
        adjustedSize *= adjustments.session;
        adjustedSize *= adjustments.correlation;
        adjustedSize *= adjustments.heat;
        adjustedSize *= adjustments.performance;
        adjustedSize *= adjustments.kelly;
        
        return adjustedSize;
    }

    /**
     * Apply constraints and limits
     */
    private applyConstraints(size: number, request: PositionSizeRequest): number {
        // Apply maximum position size limit
        let constrainedSize = Math.min(size, this.sizingParams.maxPositionSize);
        
        // Apply minimum size (avoid dust trades)
        const minSize = this.sizingParams.basePositionSize * 0.1;
        constrainedSize = Math.max(constrainedSize, minSize);
        
        // Ensure we don't exceed account balance
        const maxAffordableSize = request.accountBalance / request.entryPrice;
        constrainedSize = Math.min(constrainedSize, maxAffordableSize);
        
        return constrainedSize;
    }

    /**
     * Generate warnings
     */
    private generateWarnings(
        request: PositionSizeRequest,
        finalSize: number,
        adjustments: SizingResult['adjustmentFactors']
    ): string[] {
        const warnings: string[] = [];
        
        if (adjustments.volatility < 0.5) {
            warnings.push('High volatility detected - position size reduced');
        }
        
        if (adjustments.correlation < 0.8) {
            warnings.push('High correlation with existing positions');
        }
        
        if (adjustments.heat < 0.5) {
            warnings.push('Portfolio heat limit approaching');
        }
        
        if (adjustments.performance < 0.8) {
            warnings.push('Strategy underperforming - reduced size');
        }
        
        if (finalSize < this.sizingParams.basePositionSize * 0.5) {
            warnings.push('Position size significantly reduced due to risk factors');
        }
        
        const riskAmount = finalSize * Math.abs(request.entryPrice - request.stopLoss);
        const riskPercent = (riskAmount / request.accountBalance) * 100;
        
        if (riskPercent > this.sizingParams.riskPerTrade * 1.5) {
            warnings.push('Risk per trade exceeds recommended threshold');
        }
        
        return warnings;
    }

    /**
     * Calculate confidence score
     */
    private calculateConfidence(
        request: PositionSizeRequest,
        adjustments: SizingResult['adjustmentFactors']
    ): number {
        // Base confidence on signal confidence and adjustments
        let confidence = request.confidence;
        
        // Reduce confidence if many negative adjustments
        const adjustmentValues = Object.values(adjustments);
        const avgAdjustment = adjustmentValues.reduce((sum, val) => sum + val, 0) / adjustmentValues.length;
        
        confidence *= avgAdjustment;
        
        return Math.max(0.1, Math.min(1.0, confidence));
    }

    /**
     * Determine sizing method based on strategy stats
     */
    private determineSizingMethod(strategyStats?: StrategyStats): SizingMethod {
        if (!strategyStats || strategyStats.totalTrades < 20) {
            return SizingMethod.VOLATILITY_ADJUSTED;
        }
        
        if (strategyStats.winRate > 0.55 && strategyStats.profitFactor > 1.3) {
            return SizingMethod.KELLY;
        }
        
        return SizingMethod.VOLATILITY_ADJUSTED;
    }

    /**
     * Calculate symbol exposure
     */
    private calculateSymbolExposure(symbol: string, positions: Position[]): number {
        return positions
            .filter(p => p.symbol === symbol)
            .reduce((sum, p) => sum + Math.abs(p.size * p.currentPrice), 0);
    }

    /**
     * Calculate total exposure
     */
    private calculateTotalExposure(positions: Position[]): number {
        return positions.reduce((sum, p) => sum + Math.abs(p.size * p.currentPrice), 0);
    }

    /**
     * Calculate portfolio heat
     */
    private calculatePortfolioHeat(positions: Position[]): number {
        return positions.reduce((sum, p) => sum + p.riskAmount, 0);
    }

    /**
     * Track sizing result
     */
    private trackSizingResult(result: SizingResult): void {
        this.sizingHistory.push(result);
        
        // Limit history size
        if (this.sizingHistory.length > 1000) {
            this.sizingHistory = this.sizingHistory.slice(-500);
        }
        
        // Update performance metrics
        this.performanceMetrics.totalSizings++;
        this.performanceMetrics.averageSize = 
            (this.performanceMetrics.averageSize * (this.performanceMetrics.totalSizings - 1) + result.recommendedSize) 
            / this.performanceMetrics.totalSizings;
    }

    /**
     * Get default sizing for errors
     */
    private getDefaultSizing(request: PositionSizeRequest): SizingResult {
        const riskAmount = request.accountBalance * (this.sizingParams.riskPerTrade / 100);
        const stopDistance = Math.abs(request.entryPrice - request.stopLoss);
        const defaultSize = Math.min(riskAmount / stopDistance, this.sizingParams.basePositionSize);
        
        return {
            recommendedSize: Math.floor(defaultSize),
            maxAllowedSize: this.sizingParams.maxPositionSize,
            riskAmount: defaultSize * stopDistance,
            riskRewardRatio: 0,
            sizingMethod: SizingMethod.FIXED,
            adjustmentFactors: {
                volatility: 1, session: 1, correlation: 1,
                heat: 1, performance: 1, kelly: 1
            },
            warnings: ['Error in sizing calculation - using default'],
            confidence: 0.5
        };
    }

    /**
     * Update market data
     */
    public updateMarketData(data: MarketData): void {
        this.marketData.set(data.symbol, data);
    }

    /**
     * Update strategy statistics
     */
    public updateStrategyStats(strategy: string, stats: StrategyStats): void {
        this.strategyStats.set(strategy, stats);
    }

    /**
     * Update correlation matrix
     */
    public updateCorrelationMatrix(correlations: Map<string, Map<string, number>>): void {
        this.correlationMatrix = correlations;
    }

    /**
     * Get sizing performance
     */
    public getSizingPerformance(): any {
        return {
            ...this.performanceMetrics,
            recentSizings: this.sizingHistory.slice(-10),
            averageAdjustments: this.calculateAverageAdjustments()
        };
    }

    /**
     * Calculate average adjustments
     */
    private calculateAverageAdjustments(): any {
        if (this.sizingHistory.length === 0) return {};
        
        const recent = this.sizingHistory.slice(-50);
        const avgAdjustments = {
            volatility: 0, session: 0, correlation: 0,
            heat: 0, performance: 0, kelly: 0
        };
        
        for (const result of recent) {
            Object.keys(avgAdjustments).forEach(key => {
                avgAdjustments[key] += result.adjustmentFactors[key];
            });
        }
        
        Object.keys(avgAdjustments).forEach(key => {
            avgAdjustments[key] /= recent.length;
        });
        
        return avgAdjustments;
    }
}

export default DayTradingPositionSizer;
