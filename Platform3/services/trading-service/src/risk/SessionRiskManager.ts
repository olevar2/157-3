/**
 * Session Risk Manager
 *
 * Advanced session-based risk management system for forex trading.
 * Provides dynamic risk controls based on trading sessions (Asian/London/NY/Overlap)
 * with real-time session transition handling and time-zone aware risk adjustments.
 *
 * Key Features:
 * - Session-based risk multipliers and position limits
 * - Real-time session transition detection and handling
 * - Dynamic risk adjustments based on session volatility profiles
 * - Time-zone aware risk controls for global trading
 * - Session-specific exposure limits and monitoring
 * - Performance tracking with sub-millisecond response times
 *
 * Author: Platform3 Trading Team
 * Version: 1.0.0
 */

import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';
import winston from 'winston';
import {
    SessionType,
    RiskParameters,
    Position,
    RiskMetrics,
    RiskLevel,
    RiskAction
} from './ScalpingRiskEngine';

export interface SessionRiskParameters {
    sessionLimits: Record<SessionType, SessionLimits>;
    transitionBufferMs: number; // Buffer time during session transitions
    sessionVolatilityProfiles: Record<SessionType, VolatilityProfile>;
    emergencySessionLimits: SessionLimits;
    maxSessionTransitionsPerHour: number;
}

export interface SessionLimits {
    maxPositions: number;
    maxExposure: number;
    maxDailyLoss: number;
    riskMultiplier: number;
    positionSizeMultiplier: number;
    stopLossMultiplier: number;
}

export interface VolatilityProfile {
    expectedVolatility: number;
    liquidityLevel: 'HIGH' | 'MEDIUM' | 'LOW';
    spreadMultiplier: number;
    slippageExpectation: number;
    recommendedTimeframes: string[];
}

export interface SessionTransition {
    fromSession: SessionType;
    toSession: SessionType;
    transitionTime: number;
    riskAdjustments: RiskAdjustment[];
    positionsAffected: string[];
}

export interface RiskAdjustment {
    type: 'POSITION_SIZE' | 'STOP_LOSS' | 'EXPOSURE_LIMIT' | 'RISK_MULTIPLIER';
    oldValue: number;
    newValue: number;
    reason: string;
    appliedAt: number;
}

export interface SessionRiskAssessment {
    sessionType: SessionType;
    riskLevel: RiskLevel;
    action: RiskAction;
    sessionLimitsStatus: SessionLimitsStatus;
    recommendations: string[];
    nextSessionTransition: number;
    processingTimeMs: number;
}

export interface SessionLimitsStatus {
    positionCount: number;
    maxPositions: number;
    currentExposure: number;
    maxExposure: number;
    sessionPnL: number;
    maxSessionLoss: number;
    utilizationPercentage: number;
}

export class SessionRiskManager extends EventEmitter {
    private sessionParams: SessionRiskParameters;
    private currentSession: SessionType = SessionType.OFF_HOURS;
    private previousSession: SessionType = SessionType.OFF_HOURS;
    private sessionStartTime: number = Date.now();
    private sessionTransitionCount: number = 0;
    private lastTransitionTime: number = 0;

    // Session tracking
    private sessionPositions: Map<SessionType, Set<string>> = new Map();
    private sessionPnL: Map<SessionType, number> = new Map();
    private sessionExposure: Map<SessionType, number> = new Map();
    private sessionRiskAdjustments: RiskAdjustment[] = [];

    // Performance tracking
    private assessmentCount: number = 0;
    private totalProcessingTime: number = 0;
    private sessionTransitions: SessionTransition[] = [];

    // Risk monitoring
    private isSessionTransition: boolean = false;
    private sessionViolations: string[] = [];

    private logger: winston.Logger;

    constructor(sessionParams: SessionRiskParameters, logger?: winston.Logger) {
        super();
        this.sessionParams = { ...sessionParams };
        this.logger = logger || this.createDefaultLogger();
        this.initializeSessionManager();

        this.logger.info('✅ SessionRiskManager initialized', {
            component: 'SessionRiskManager',
            sessionParams: this.sessionParams
        });
    }

    /**
     * Create default logger if none provided
     */
    private createDefaultLogger(): winston.Logger {
        return winston.createLogger({
            level: process.env.LOG_LEVEL || 'info',
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.errors({ stack: true }),
                winston.format.json()
            ),
            transports: [
                new winston.transports.Console({
                    format: winston.format.combine(
                        winston.format.colorize(),
                        winston.format.simple()
                    )
                })
            ]
        });
    }

    /**
     * Initialize session manager
     */
    private initializeSessionManager(): void {
        // Initialize session tracking maps
        Object.values(SessionType).forEach(session => {
            this.sessionPositions.set(session, new Set());
            this.sessionPnL.set(session, 0);
            this.sessionExposure.set(session, 0);
        });

        // Start session monitoring
        this.updateCurrentSession();
        setInterval(() => this.updateCurrentSession(), 30000); // Check every 30 seconds

        // Start session risk monitoring
        setInterval(() => this.performSessionRiskMonitoring(), 5000); // Check every 5 seconds

        // Reset session metrics at session start
        this.resetSessionMetrics();
    }

    /**
     * Assess session-based risk for new order
     */
    public async assessSessionRisk(
        symbol: string,
        side: 'LONG' | 'SHORT',
        requestedSize: number,
        price: number,
        positions: Position[]
    ): Promise<SessionRiskAssessment> {
        const startTime = performance.now();

        try {
            // Check if in session transition
            if (this.isSessionTransition) {
                return this.createSessionRiskAssessment(
                    RiskLevel.HIGH, RiskAction.REJECT,
                    ['Session transition in progress'], startTime
                );
            }

            // Get current session limits
            const sessionLimits = this.sessionParams.sessionLimits[this.currentSession];
            if (!sessionLimits) {
                return this.createSessionRiskAssessment(
                    RiskLevel.CRITICAL, RiskAction.REJECT,
                    ['No session limits configured'], startTime
                );
            }

            // Calculate session metrics
            const sessionStatus = this.calculateSessionLimitsStatus(positions);

            // Check session-specific limits
            const { riskLevel, action, recommendations } = this.evaluateSessionLimits(
                sessionStatus, requestedSize, price
            );

            const assessment = this.createSessionRiskAssessment(
                riskLevel, action, recommendations, startTime, sessionStatus
            );

            // Update tracking
            this.assessmentCount++;
            this.totalProcessingTime += assessment.processingTimeMs;

            // Emit session risk event
            this.emit('sessionRiskAssessment', assessment);

            return assessment;

        } catch (error) {
            this.logger.error('Error in session risk assessment:', error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            return this.createSessionRiskAssessment(
                RiskLevel.CRITICAL, RiskAction.REJECT,
                [`Session risk assessment error: ${errorMessage}`], startTime
            );
        }
    }

    /**
     * Update current session based on UTC time
     */
    private updateCurrentSession(): void {
        const now = new Date();
        const utcHour = now.getUTCHours();

        let newSession: SessionType;

        // Session times in UTC
        if (utcHour >= 8 && utcHour < 12) {
            newSession = SessionType.LONDON_OPEN;
        } else if (utcHour >= 12 && utcHour < 16) {
            newSession = SessionType.OVERLAP;
        } else if (utcHour >= 16 && utcHour < 24) {
            newSession = SessionType.NY_OPEN;
        } else if (utcHour >= 0 && utcHour < 8) {
            newSession = SessionType.ASIAN;
        } else {
            newSession = SessionType.OFF_HOURS;
        }

        // Handle session transition
        if (newSession !== this.currentSession) {
            this.handleSessionTransition(this.currentSession, newSession);
        }
    }

    /**
     * Handle session transition
     */
    private handleSessionTransition(fromSession: SessionType, toSession: SessionType): void {
        const transitionTime = Date.now();

        this.logger.info('🔄 Session transition detected', {
            component: 'SessionRiskManager',
            fromSession,
            toSession,
            transitionTime
        });

        // Set transition flag
        this.isSessionTransition = true;
        this.previousSession = fromSession;
        this.currentSession = toSession;
        this.sessionStartTime = transitionTime;
        this.sessionTransitionCount++;
        this.lastTransitionTime = transitionTime;

        // Calculate risk adjustments
        const riskAdjustments = this.calculateSessionRiskAdjustments(fromSession, toSession);

        // Create transition record
        const transition: SessionTransition = {
            fromSession,
            toSession,
            transitionTime,
            riskAdjustments,
            positionsAffected: Array.from(this.sessionPositions.get(fromSession) || [])
        };

        this.sessionTransitions.push(transition);

        // Apply transition buffer
        setTimeout(() => {
            this.isSessionTransition = false;
            this.logger.info('✅ Session transition completed', {
                component: 'SessionRiskManager',
                session: toSession,
                bufferMs: this.sessionParams.transitionBufferMs
            });
        }, this.sessionParams.transitionBufferMs);

        // Emit session change event
        this.emit('sessionChange', transition);

        // Reset session metrics for new session
        this.resetSessionMetrics();
    }

    /**
     * Calculate session risk adjustments for transition
     */
    private calculateSessionRiskAdjustments(fromSession: SessionType, toSession: SessionType): RiskAdjustment[] {
        const adjustments: RiskAdjustment[] = [];
        const fromLimits = this.sessionParams.sessionLimits[fromSession];
        const toLimits = this.sessionParams.sessionLimits[toSession];

        if (!fromLimits || !toLimits) return adjustments;

        const now = Date.now();

        // Risk multiplier adjustment
        if (fromLimits.riskMultiplier !== toLimits.riskMultiplier) {
            adjustments.push({
                type: 'RISK_MULTIPLIER',
                oldValue: fromLimits.riskMultiplier,
                newValue: toLimits.riskMultiplier,
                reason: `Session transition from ${fromSession} to ${toSession}`,
                appliedAt: now
            });
        }

        // Position size multiplier adjustment
        if (fromLimits.positionSizeMultiplier !== toLimits.positionSizeMultiplier) {
            adjustments.push({
                type: 'POSITION_SIZE',
                oldValue: fromLimits.positionSizeMultiplier,
                newValue: toLimits.positionSizeMultiplier,
                reason: `Session volatility profile change`,
                appliedAt: now
            });
        }

        // Stop loss multiplier adjustment
        if (fromLimits.stopLossMultiplier !== toLimits.stopLossMultiplier) {
            adjustments.push({
                type: 'STOP_LOSS',
                oldValue: fromLimits.stopLossMultiplier,
                newValue: toLimits.stopLossMultiplier,
                reason: `Session liquidity profile change`,
                appliedAt: now
            });
        }

        // Exposure limit adjustment
        if (fromLimits.maxExposure !== toLimits.maxExposure) {
            adjustments.push({
                type: 'EXPOSURE_LIMIT',
                oldValue: fromLimits.maxExposure,
                newValue: toLimits.maxExposure,
                reason: `Session exposure limit change`,
                appliedAt: now
            });
        }

        this.sessionRiskAdjustments.push(...adjustments);
        return adjustments;
    }

    /**
     * Calculate session limits status
     */
    private calculateSessionLimitsStatus(positions: Position[]): SessionLimitsStatus {
        const sessionLimits = this.sessionParams.sessionLimits[this.currentSession];
        if (!sessionLimits) {
            return {
                positionCount: 0,
                maxPositions: 0,
                currentExposure: 0,
                maxExposure: 0,
                sessionPnL: 0,
                maxSessionLoss: 0,
                utilizationPercentage: 0
            };
        }

        // Filter positions for current session
        const sessionPositions = positions.filter(p => p.session === this.currentSession);

        // Calculate current exposure
        const currentExposure = sessionPositions.reduce((total, position) => {
            return total + Math.abs(position.size * position.currentPrice);
        }, 0);

        // Calculate session P&L
        const sessionPnL = sessionPositions.reduce((total, position) => {
            return total + position.unrealizedPnL + position.realizedPnL;
        }, 0);

        // Calculate utilization percentage
        const utilizationPercentage = sessionLimits.maxExposure > 0
            ? (currentExposure / sessionLimits.maxExposure) * 100
            : 0;

        return {
            positionCount: sessionPositions.length,
            maxPositions: sessionLimits.maxPositions,
            currentExposure,
            maxExposure: sessionLimits.maxExposure,
            sessionPnL,
            maxSessionLoss: sessionLimits.maxDailyLoss,
            utilizationPercentage
        };
    }

    /**
     * Get higher risk level between two risk levels
     */
    private getHigherRiskLevel(current: RiskLevel, candidate: RiskLevel): RiskLevel {
        const riskOrder = {
            [RiskLevel.LOW]: 0,
            [RiskLevel.MEDIUM]: 1,
            [RiskLevel.HIGH]: 2,
            [RiskLevel.CRITICAL]: 3,
            [RiskLevel.EMERGENCY]: 4
        };

        return riskOrder[current] >= riskOrder[candidate] ? current : candidate;
    }

    /**
     * Evaluate session limits
     */
    private evaluateSessionLimits(
        sessionStatus: SessionLimitsStatus,
        requestedSize: number,
        price: number
    ): { riskLevel: RiskLevel; action: RiskAction; recommendations: string[] } {
        const recommendations: string[] = [];
        let riskLevel = RiskLevel.LOW;
        let action = RiskAction.ALLOW;

        const requestedExposure = requestedSize * price;

        // Check position count limit
        if (sessionStatus.positionCount >= sessionStatus.maxPositions) {
            recommendations.push('Maximum session positions reached');
            riskLevel = RiskLevel.HIGH;
            action = RiskAction.REJECT;
        }

        // Check exposure limit
        if (sessionStatus.currentExposure + requestedExposure > sessionStatus.maxExposure) {
            recommendations.push('Session exposure limit would be exceeded');
            riskLevel = this.getHigherRiskLevel(riskLevel, RiskLevel.MEDIUM);
            if (action === RiskAction.ALLOW) {
                action = RiskAction.REDUCE_SIZE;
            }
        }

        // Check session P&L limit
        if (sessionStatus.sessionPnL < -sessionStatus.maxSessionLoss) {
            recommendations.push('Session loss limit exceeded');
            riskLevel = RiskLevel.CRITICAL;
            action = RiskAction.REJECT;
        }

        // Check utilization percentage
        if (sessionStatus.utilizationPercentage > 80) {
            recommendations.push('High session utilization detected');
            riskLevel = this.getHigherRiskLevel(riskLevel, RiskLevel.MEDIUM);
        }

        // Session-specific recommendations
        const volatilityProfile = this.sessionParams.sessionVolatilityProfiles[this.currentSession];
        if (volatilityProfile) {
            if (volatilityProfile.liquidityLevel === 'LOW') {
                recommendations.push('Low liquidity session - consider reduced position sizes');
                riskLevel = this.getHigherRiskLevel(riskLevel, RiskLevel.MEDIUM);
            }

            if (volatilityProfile.expectedVolatility > 0.03) {
                recommendations.push('High volatility session - increased risk monitoring');
                riskLevel = this.getHigherRiskLevel(riskLevel, RiskLevel.MEDIUM);
            }
        }

        return { riskLevel, action, recommendations };
    }

    /**
     * Create session risk assessment result
     */
    private createSessionRiskAssessment(
        riskLevel: RiskLevel,
        action: RiskAction,
        recommendations: string[],
        startTime: number,
        sessionStatus?: SessionLimitsStatus
    ): SessionRiskAssessment {
        const processingTime = performance.now() - startTime;

        // Calculate next session transition time
        const nextTransition = this.calculateNextSessionTransition();

        return {
            sessionType: this.currentSession,
            riskLevel,
            action,
            sessionLimitsStatus: sessionStatus || {
                positionCount: 0,
                maxPositions: 0,
                currentExposure: 0,
                maxExposure: 0,
                sessionPnL: 0,
                maxSessionLoss: 0,
                utilizationPercentage: 0
            },
            recommendations,
            nextSessionTransition: nextTransition,
            processingTimeMs: processingTime
        };
    }

    /**
     * Calculate next session transition time
     */
    private calculateNextSessionTransition(): number {
        const now = new Date();
        const utcHour = now.getUTCHours();

        let nextTransitionHour: number;

        if (utcHour < 8) {
            nextTransitionHour = 8; // London open
        } else if (utcHour < 12) {
            nextTransitionHour = 12; // Overlap start
        } else if (utcHour < 16) {
            nextTransitionHour = 16; // NY open
        } else {
            nextTransitionHour = 24; // Asian session (next day)
        }

        const nextTransition = new Date(now);
        nextTransition.setUTCHours(nextTransitionHour, 0, 0, 0);

        // If next transition is tomorrow
        if (nextTransitionHour === 24) {
            nextTransition.setUTCDate(nextTransition.getUTCDate() + 1);
            nextTransition.setUTCHours(0, 0, 0, 0);
        }

        return nextTransition.getTime();
    }

    /**
     * Perform session risk monitoring
     */
    private performSessionRiskMonitoring(): void {
        try {
            // Check session transition frequency
            const now = Date.now();
            const hourAgo = now - (60 * 60 * 1000);
            const recentTransitions = this.sessionTransitions.filter(t => t.transitionTime > hourAgo);

            if (recentTransitions.length > this.sessionParams.maxSessionTransitionsPerHour) {
                this.sessionViolations.push('Excessive session transitions detected');
                this.emit('sessionRiskViolation', {
                    type: 'EXCESSIVE_TRANSITIONS',
                    count: recentTransitions.length,
                    limit: this.sessionParams.maxSessionTransitionsPerHour
                });
            }

            // Monitor session exposure
            const currentExposure = this.sessionExposure.get(this.currentSession) || 0;
            const sessionLimits = this.sessionParams.sessionLimits[this.currentSession];

            if (sessionLimits && currentExposure > sessionLimits.maxExposure * 0.9) {
                this.emit('sessionLimitExceeded', {
                    session: this.currentSession,
                    currentExposure,
                    limit: sessionLimits.maxExposure,
                    utilizationPercentage: (currentExposure / sessionLimits.maxExposure) * 100
                });
            }

        } catch (error) {
            this.logger.error('Error in session risk monitoring:', error);
        }
    }

    /**
     * Reset session metrics
     */
    private resetSessionMetrics(): void {
        // Reset session-specific tracking
        this.sessionPnL.set(this.currentSession, 0);
        this.sessionExposure.set(this.currentSession, 0);
        this.sessionPositions.set(this.currentSession, new Set());

        // Clear old violations
        this.sessionViolations = [];

        this.logger.info('📊 Session metrics reset', {
            component: 'SessionRiskManager',
            session: this.currentSession
        });
    }

    /**
     * Update position tracking for session
     */
    public updatePositionForSession(position: Position): void {
        const sessionPositions = this.sessionPositions.get(position.session) || new Set();
        sessionPositions.add(position.id);
        this.sessionPositions.set(position.session, sessionPositions);

        // Update session exposure
        const currentExposure = this.sessionExposure.get(position.session) || 0;
        const positionExposure = Math.abs(position.size * position.currentPrice);
        this.sessionExposure.set(position.session, currentExposure + positionExposure);

        // Update session P&L
        const currentPnL = this.sessionPnL.get(position.session) || 0;
        this.sessionPnL.set(position.session, currentPnL + position.unrealizedPnL + position.realizedPnL);

        this.emit('sessionRiskUpdate', {
            session: position.session,
            positionId: position.id,
            exposure: positionExposure,
            pnl: position.unrealizedPnL + position.realizedPnL
        });
    }

    /**
     * Remove position from session tracking
     */
    public removePositionFromSession(positionId: string, session: SessionType): void {
        const sessionPositions = this.sessionPositions.get(session);
        if (sessionPositions) {
            sessionPositions.delete(positionId);
        }

        this.emit('sessionRiskUpdate', {
            session,
            positionId,
            action: 'REMOVED'
        });
    }

    /**
     * Get current session information
     */
    public getCurrentSessionInfo(): {
        session: SessionType;
        sessionStartTime: number;
        isTransition: boolean;
        nextTransition: number;
        volatilityProfile: VolatilityProfile | undefined;
    } {
        return {
            session: this.currentSession,
            sessionStartTime: this.sessionStartTime,
            isTransition: this.isSessionTransition,
            nextTransition: this.calculateNextSessionTransition(),
            volatilityProfile: this.sessionParams.sessionVolatilityProfiles[this.currentSession]
        };
    }

    /**
     * Get session risk status
     */
    public getSessionRiskStatus(): {
        currentSession: SessionType;
        sessionLimits: SessionLimits | undefined;
        sessionMetrics: {
            positionCount: number;
            exposure: number;
            pnl: number;
        };
        recentTransitions: SessionTransition[];
        violations: string[];
        performance: {
            assessmentCount: number;
            averageProcessingTimeMs: number;
        };
    } {
        const sessionLimits = this.sessionParams.sessionLimits[this.currentSession];
        const positionCount = (this.sessionPositions.get(this.currentSession) || new Set()).size;
        const exposure = this.sessionExposure.get(this.currentSession) || 0;
        const pnl = this.sessionPnL.get(this.currentSession) || 0;

        return {
            currentSession: this.currentSession,
            sessionLimits,
            sessionMetrics: {
                positionCount,
                exposure,
                pnl
            },
            recentTransitions: this.sessionTransitions.slice(-10), // Last 10 transitions
            violations: this.sessionViolations,
            performance: {
                assessmentCount: this.assessmentCount,
                averageProcessingTimeMs: this.assessmentCount > 0
                    ? this.totalProcessingTime / this.assessmentCount
                    : 0
            }
        };
    }

    /**
     * Update session risk parameters
     */
    public updateSessionRiskParameters(newParams: Partial<SessionRiskParameters>): void {
        this.sessionParams = { ...this.sessionParams, ...newParams };
        this.emit('sessionParametersUpdated', this.sessionParams);

        this.logger.info('✅ Session risk parameters updated', {
            component: 'SessionRiskManager',
            newParams
        });
    }
}

export default SessionRiskManager;
