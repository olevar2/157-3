from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
/**
 * Scalping Risk Management Engine
 * 
 * Ultra-fast risk management system specifically designed for scalping strategies.
 * Provides real-time risk assessment, position sizing, and automated risk controls
 * with sub-millisecond response times.
 * 
 * Key Features:
 * - Real-time position risk monitoring
 * - Dynamic position sizing based on volatility
 * - Rapid drawdown protection
 * - Session-based risk limits
 * - Automated stop-loss and take-profit management
 * - Correlation-based exposure limits
 * 
 * Author: Platform3 Trading Team
 * Version: 1.0.0
 */

import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';

export enum RiskLevel {
    LOW = 'LOW',
    MEDIUM = 'MEDIUM',
    HIGH = 'HIGH',
    CRITICAL = 'CRITICAL',
    EMERGENCY = 'EMERGENCY'
}

export enum RiskAction {
    ALLOW = 'ALLOW',
    REDUCE_SIZE = 'REDUCE_SIZE',
    REJECT = 'REJECT',
    CLOSE_POSITION = 'CLOSE_POSITION',
    CLOSE_ALL = 'CLOSE_ALL',
    EMERGENCY_STOP = 'EMERGENCY_STOP'
}

export enum SessionType {
    LONDON_OPEN = 'LONDON_OPEN',
    NY_OPEN = 'NY_OPEN',
    OVERLAP = 'OVERLAP',
    ASIAN = 'ASIAN',
    OFF_HOURS = 'OFF_HOURS'
}

export interface RiskParameters {
    maxPositionSize: number;
    maxDailyLoss: number;
    maxDrawdown: number;
    maxCorrelatedExposure: number;
    stopLossPercent: number;
    takeProfitPercent: number;
    maxPositionsPerSymbol: number;
    maxTotalPositions: number;
    volatilityMultiplier: number;
    sessionRiskMultipliers: Record<SessionType, number>;
}

export interface Position {
    id: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
    realizedPnL: number;
    stopLoss?: number;
    takeProfit?: number;
    timestamp: number;
    session: SessionType;
}

export interface RiskAssessment {
    orderId: string;
    symbol: string;
    requestedSize: number;
    assessedSize: number;
    riskLevel: RiskLevel;
    action: RiskAction;
    reasons: string[];
    metrics: RiskMetrics;
    timestamp: number;
    processingTimeMs: number;
}

export interface RiskMetrics {
    currentDrawdown: number;
    dailyPnL: number;
    totalExposure: number;
    correlatedExposure: number;
    positionCount: number;
    volatilityScore: number;
    sessionRiskScore: number;
    marginUtilization: number;
}

export interface MarketData {
    symbol: string;
    bid: number;
    ask: number;
    volatility: number;
    correlation: Record<string, number>;
    timestamp: number;
}

export class ScalpingRiskEngine extends EventEmitter {
    private riskParams: RiskParameters;
    private positions: Map<string, Position> = new Map();
    private marketData: Map<string, MarketData> = new Map();
    private dailyPnL: number = 0;
    private maxDrawdown: number = 0;
    private currentDrawdown: number = 0;
    private sessionStartTime: number = Date.now();
    private currentSession: SessionType = SessionType.OFF_HOURS;
    
    // Performance tracking
    private assessmentCount: number = 0;
    private totalProcessingTime: number = 0;
    
    // Risk monitoring
    private isEmergencyStop: boolean = false;
    private riskViolations: string[] = [];
    
    constructor(riskParams: RiskParameters) {
        super();
        this.riskParams = { ...riskParams };
        this.initializeRiskEngine();
        
        console.log('✅ ScalpingRiskEngine initialized');
    }

    /**
     * Initialize risk engine
     */
    private initializeRiskEngine(): void {
        // Start session monitoring
        this.updateCurrentSession();
        setInterval(() => this.updateCurrentSession(), 60000); // Check every minute
        
        // Start risk monitoring
        setInterval(() => this.performRiskMonitoring(), 1000); // Check every second
        
        // Reset daily metrics at session start
        this.resetDailyMetrics();
    }

    /**
     * Assess risk for new order
     */
    public async assessOrderRisk(
        orderId: string,
        symbol: string,
        side: 'LONG' | 'SHORT',
        requestedSize: number,
        price: number
    ): Promise<RiskAssessment> {
        const startTime = performance.now();
        
        try {
            // Emergency stop check
            if (this.isEmergencyStop) {
                return this.createRiskAssessment(
                    orderId, symbol, requestedSize, 0,
                    RiskLevel.EMERGENCY, RiskAction.EMERGENCY_STOP,
                    ['Emergency stop activated'], startTime
                );
            }

            // Get current market data
            const marketData = this.marketData.get(symbol);
            if (!marketData) {
                return this.createRiskAssessment(
                    orderId, symbol, requestedSize, 0,
                    RiskLevel.HIGH, RiskAction.REJECT,
                    ['No market data available'], startTime
                );
            }

            // Calculate risk metrics
            const metrics = this.calculateRiskMetrics(symbol, side, requestedSize, price);
            
            // Assess position size
            const assessedSize = this.calculateOptimalPositionSize(
                symbol, side, requestedSize, price, metrics
            );
            
            // Determine risk level and action
            const { riskLevel, action, reasons } = this.determineRiskAction(
                symbol, side, requestedSize, assessedSize, metrics
            );

            const assessment = this.createRiskAssessment(
                orderId, symbol, requestedSize, assessedSize,
                riskLevel, action, reasons, startTime, metrics
            );

            // Update tracking
            this.assessmentCount++;
            this.totalProcessingTime += assessment.processingTimeMs;

            // Emit risk event
            this.emit('riskAssessment', assessment);

            return assessment;

        } catch (error) {
            console.error('Error in risk assessment:', error);
            return this.createRiskAssessment(
                orderId, symbol, requestedSize, 0,
                RiskLevel.CRITICAL, RiskAction.REJECT,
                [`Risk assessment error: ${error.message}`], startTime
            );
        }
    }

    /**
     * Calculate comprehensive risk metrics
     */
    private calculateRiskMetrics(
        symbol: string,
        side: 'LONG' | 'SHORT',
        size: number,
        price: number
    ): RiskMetrics {
        const marketData = this.marketData.get(symbol);
        const notionalValue = size * price;
        
        // Current drawdown
        const currentDrawdown = Math.max(0, this.maxDrawdown - this.dailyPnL);
        
        // Total exposure
        let totalExposure = notionalValue;
        for (const position of this.positions.values()) {
            totalExposure += Math.abs(position.size * position.currentPrice);
        }
        
        // Correlated exposure
        const correlatedExposure = this.calculateCorrelatedExposure(symbol, size, price);
        
        // Position count
        const positionCount = this.positions.size;
        
        // Volatility score
        const volatilityScore = marketData ? marketData.volatility : 0;
        
        // Session risk score
        const sessionMultiplier = this.riskParams.sessionRiskMultipliers[this.currentSession] || 1.0;
        const sessionRiskScore = volatilityScore * sessionMultiplier;
        
        // Margin utilization (simplified)
        const marginUtilization = totalExposure / (this.riskParams.maxPositionSize * 10); // Assume 10x max as total margin
        
        return {
            currentDrawdown,
            dailyPnL: this.dailyPnL,
            totalExposure,
            correlatedExposure,
            positionCount,
            volatilityScore,
            sessionRiskScore,
            marginUtilization
        };
    }

    /**
     * Calculate correlated exposure
     */
    private calculateCorrelatedExposure(symbol: string, size: number, price: number): number {
        const marketData = this.marketData.get(symbol);
        if (!marketData || !marketData.correlation) return 0;
        
        let correlatedExposure = 0;
        
        for (const position of this.positions.values()) {
            const correlation = marketData.correlation[position.symbol] || 0;
            if (Math.abs(correlation) > 0.5) { // Significant correlation
                const positionValue = Math.abs(position.size * position.currentPrice);
                correlatedExposure += positionValue * Math.abs(correlation);
            }
        }
        
        // Add current order exposure
        correlatedExposure += size * price;
        
        return correlatedExposure;
    }

    /**
     * Calculate optimal position size
     */
    private calculateOptimalPositionSize(
        symbol: string,
        side: 'LONG' | 'SHORT',
        requestedSize: number,
        price: number,
        metrics: RiskMetrics
    ): number {
        let optimalSize = requestedSize;
        
        // Volatility-based sizing
        const marketData = this.marketData.get(symbol);
        if (marketData) {
            const volatilityAdjustment = 1 / (1 + marketData.volatility * this.riskParams.volatilityMultiplier);
            optimalSize *= volatilityAdjustment;
        }
        
        // Session-based sizing
        const sessionMultiplier = this.riskParams.sessionRiskMultipliers[this.currentSession] || 1.0;
        optimalSize *= sessionMultiplier;
        
        // Drawdown-based sizing
        if (metrics.currentDrawdown > 0) {
            const drawdownAdjustment = 1 - (metrics.currentDrawdown / this.riskParams.maxDrawdown);
            optimalSize *= Math.max(0.1, drawdownAdjustment); // Minimum 10% of original size
        }
        
        // Correlation-based sizing
        if (metrics.correlatedExposure > this.riskParams.maxCorrelatedExposure) {
            const correlationAdjustment = this.riskParams.maxCorrelatedExposure / metrics.correlatedExposure;
            optimalSize *= correlationAdjustment;
        }
        
        // Ensure within absolute limits
        optimalSize = Math.min(optimalSize, this.riskParams.maxPositionSize);
        
        return Math.max(0, Math.floor(optimalSize));
    }

    /**
     * Determine risk action
     */
    private determineRiskAction(
        symbol: string,
        side: 'LONG' | 'SHORT',
        requestedSize: number,
        assessedSize: number,
        metrics: RiskMetrics
    ): { riskLevel: RiskLevel; action: RiskAction; reasons: string[] } {
        const reasons: string[] = [];
        let riskLevel = RiskLevel.LOW;
        let action = RiskAction.ALLOW;
        
        // Check daily loss limit
        if (metrics.dailyPnL < -this.riskParams.maxDailyLoss) {
            reasons.push('Daily loss limit exceeded');
            riskLevel = RiskLevel.CRITICAL;
            action = RiskAction.REJECT;
        }
        
        // Check drawdown limit
        if (metrics.currentDrawdown > this.riskParams.maxDrawdown) {
            reasons.push('Maximum drawdown exceeded');
            riskLevel = RiskLevel.CRITICAL;
            action = RiskAction.REJECT;
        }
        
        // Check position limits
        if (metrics.positionCount >= this.riskParams.maxTotalPositions) {
            reasons.push('Maximum total positions reached');
            riskLevel = RiskLevel.HIGH;
            action = RiskAction.REJECT;
        }
        
        // Check symbol-specific position limit
        const symbolPositions = Array.from(this.positions.values())
            .filter(p => p.symbol === symbol).length;
        if (symbolPositions >= this.riskParams.maxPositionsPerSymbol) {
            reasons.push('Maximum positions per symbol reached');
            riskLevel = RiskLevel.HIGH;
            action = RiskAction.REJECT;
        }
        
        // Check correlated exposure
        if (metrics.correlatedExposure > this.riskParams.maxCorrelatedExposure) {
            reasons.push('Correlated exposure limit exceeded');
            riskLevel = RiskLevel.MEDIUM;
            if (assessedSize < requestedSize) {
                action = RiskAction.REDUCE_SIZE;
            } else {
                action = RiskAction.REJECT;
            }
        }
        
        // Check volatility
        if (metrics.volatilityScore > 0.05) { // 5% volatility threshold
            reasons.push('High market volatility detected');
            riskLevel = Math.max(riskLevel as any, RiskLevel.MEDIUM as any);
            if (assessedSize < requestedSize) {
                action = RiskAction.REDUCE_SIZE;
            }
        }
        
        // Check margin utilization
        if (metrics.marginUtilization > 0.8) { // 80% margin utilization
            reasons.push('High margin utilization');
            riskLevel = Math.max(riskLevel as any, RiskLevel.HIGH as any);
            action = RiskAction.REJECT;
        }
        
        // Size reduction check
        if (assessedSize < requestedSize && action === RiskAction.ALLOW) {
            reasons.push('Position size reduced due to risk factors');
            riskLevel = RiskLevel.MEDIUM;
            action = RiskAction.REDUCE_SIZE;
        }
        
        // Emergency conditions
        if (metrics.dailyPnL < -this.riskParams.maxDailyLoss * 1.5) {
            reasons.push('Emergency loss threshold reached');
            riskLevel = RiskLevel.EMERGENCY;
            action = RiskAction.EMERGENCY_STOP;
            this.triggerEmergencyStop();
        }
        
        return { riskLevel, action, reasons };
    }

    /**
     * Create risk assessment result
     */
    private createRiskAssessment(
        orderId: string,
        symbol: string,
        requestedSize: number,
        assessedSize: number,
        riskLevel: RiskLevel,
        action: RiskAction,
        reasons: string[],
        startTime: number,
        metrics?: RiskMetrics
    ): RiskAssessment {
        const processingTime = performance.now() - startTime;
        
        return {
            orderId,
            symbol,
            requestedSize,
            assessedSize,
            riskLevel,
            action,
            reasons,
            metrics: metrics || this.calculateRiskMetrics(symbol, 'LONG', 0, 0),
            timestamp: Date.now(),
            processingTimeMs: processingTime
        };
    }

    /**
     * Update position
     */
    public updatePosition(position: Position): void {
        this.positions.set(position.id, position);
        
        // Update daily P&L
        this.updateDailyPnL();
        
        // Check for automatic stop-loss/take-profit
        this.checkAutomaticExits(position);
        
        this.emit('positionUpdated', position);
    }

    /**
     * Close position
     */
    public closePosition(positionId: string, price: number): boolean {
        const position = this.positions.get(positionId);
        if (!position) return false;
        
        // Calculate final P&L
        const pnl = position.side === 'LONG' 
            ? (price - position.entryPrice) * position.size
            : (position.entryPrice - price) * position.size;
        
        position.realizedPnL += pnl;
        position.unrealizedPnL = 0;
        
        // Update daily P&L
        this.dailyPnL += pnl;
        
        // Remove position
        this.positions.delete(positionId);
        
        this.emit('positionClosed', { position, finalPnL: pnl });
        
        return true;
    }

    /**
     * Update market data
     */
    public updateMarketData(data: MarketData): void {
        this.marketData.set(data.symbol, data);
        
        // Update position current prices
        for (const position of this.positions.values()) {
            if (position.symbol === data.symbol) {
                position.currentPrice = (data.bid + data.ask) / 2;
                
                // Update unrealized P&L
                position.unrealizedPnL = position.side === 'LONG'
                    ? (position.currentPrice - position.entryPrice) * position.size
                    : (position.entryPrice - position.currentPrice) * position.size;
            }
        }
        
        this.updateDailyPnL();
    }

    /**
     * Update daily P&L
     */
    private updateDailyPnL(): void {
        let totalUnrealizedPnL = 0;
        let totalRealizedPnL = 0;
        
        for (const position of this.positions.values()) {
            totalUnrealizedPnL += position.unrealizedPnL;
            totalRealizedPnL += position.realizedPnL;
        }
        
        const currentTotalPnL = totalRealizedPnL + totalUnrealizedPnL;
        
        // Update drawdown
        if (currentTotalPnL > this.maxDrawdown) {
            this.maxDrawdown = currentTotalPnL;
        }
        
        this.currentDrawdown = this.maxDrawdown - currentTotalPnL;
        this.dailyPnL = currentTotalPnL;
    }

    /**
     * Check automatic exits
     */
    private checkAutomaticExits(position: Position): void {
        if (!position.stopLoss && !position.takeProfit) return;
        
        const currentPrice = position.currentPrice;
        let shouldClose = false;
        let reason = '';
        
        if (position.side === 'LONG') {
            if (position.stopLoss && currentPrice <= position.stopLoss) {
                shouldClose = true;
                reason = 'Stop loss triggered';
            } else if (position.takeProfit && currentPrice >= position.takeProfit) {
                shouldClose = true;
                reason = 'Take profit triggered';
            }
        } else { // SHORT
            if (position.stopLoss && currentPrice >= position.stopLoss) {
                shouldClose = true;
                reason = 'Stop loss triggered';
            } else if (position.takeProfit && currentPrice <= position.takeProfit) {
                shouldClose = true;
                reason = 'Take profit triggered';
            }
        }
        
        if (shouldClose) {
            this.emit('automaticExit', { position, reason, price: currentPrice });
        }
    }

    /**
     * Perform risk monitoring
     */
    private performRiskMonitoring(): void {
        const metrics = this.calculateRiskMetrics('', 'LONG', 0, 0);
        
        // Check for risk violations
        const violations: string[] = [];
        
        if (metrics.dailyPnL < -this.riskParams.maxDailyLoss) {
            violations.push('Daily loss limit exceeded');
        }
        
        if (metrics.currentDrawdown > this.riskParams.maxDrawdown) {
            violations.push('Maximum drawdown exceeded');
        }
        
        if (metrics.marginUtilization > 0.9) {
            violations.push('Critical margin utilization');
        }
        
        if (violations.length > 0) {
            this.riskViolations = violations;
            this.emit('riskViolation', { violations, metrics });
        }
        
        // Emergency stop conditions
        if (metrics.dailyPnL < -this.riskParams.maxDailyLoss * 1.5) {
            this.triggerEmergencyStop();
        }
    }

    /**
     * Trigger emergency stop
     */
    private triggerEmergencyStop(): void {
        if (this.isEmergencyStop) return;
        
        this.isEmergencyStop = true;
        
        console.error('🚨 EMERGENCY STOP TRIGGERED');
        
        this.emit('emergencyStop', {
            reason: 'Critical risk threshold exceeded',
            timestamp: Date.now(),
            positions: Array.from(this.positions.values()),
            dailyPnL: this.dailyPnL
        });
    }

    /**
     * Reset emergency stop
     */
    public resetEmergencyStop(): void {
        this.isEmergencyStop = false;
        this.riskViolations = [];
        console.log('✅ Emergency stop reset');
    }

    /**
     * Update current session
     */
    private updateCurrentSession(): void {
        const now = new Date();
        const utcHour = now.getUTCHours();
        
        if (utcHour >= 8 && utcHour < 12) {
            this.currentSession = SessionType.LONDON_OPEN;
        } else if (utcHour >= 12 && utcHour < 16) {
            this.currentSession = SessionType.OVERLAP;
        } else if (utcHour >= 16 && utcHour < 24) {
            this.currentSession = SessionType.NY_OPEN;
        } else if (utcHour >= 0 && utcHour < 8) {
            this.currentSession = SessionType.ASIAN;
        } else {
            this.currentSession = SessionType.OFF_HOURS;
        }
    }

    /**
     * Reset daily metrics
     */
    private resetDailyMetrics(): void {
        this.dailyPnL = 0;
        this.maxDrawdown = 0;
        this.currentDrawdown = 0;
        this.sessionStartTime = Date.now();
        
        console.log('📊 Daily risk metrics reset');
    }

    /**
     * Get risk status
     */
    public getRiskStatus(): any {
        const metrics = this.calculateRiskMetrics('', 'LONG', 0, 0);
        
        return {
            isEmergencyStop: this.isEmergencyStop,
            currentSession: this.currentSession,
            metrics,
            violations: this.riskViolations,
            performance: {
                assessmentCount: this.assessmentCount,
                averageProcessingTimeMs: this.assessmentCount > 0 
                    ? this.totalProcessingTime / this.assessmentCount 
                    : 0
            },
            positions: {
                count: this.positions.size,
                symbols: Array.from(new Set(Array.from(this.positions.values()).map(p => p.symbol)))
            }
        };
    }

    /**
     * Update risk parameters
     */
    public updateRiskParameters(newParams: Partial<RiskParameters>): void {
        this.riskParams = { ...this.riskParams, ...newParams };
        this.emit('riskParametersUpdated', this.riskParams);
        console.log('✅ Risk parameters updated');
    }
}

export default ScalpingRiskEngine;
