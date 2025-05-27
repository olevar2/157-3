"""
Time-Weighted Volatility Analysis Module

This module provides advanced time-weighted volatility calculations for forex trading,
incorporating session-based weighting, intraday patterns, and market regime detection.
Optimized for scalping (M1-M5), day trading (M15-H1), and swing trading (H4) strategies.

Features:
- Session-weighted volatility (Asian/London/NY/Overlap)
- Intraday volatility patterns and clustering
- Market regime detection (Low/Normal/High/Extreme volatility)
- Time-decay weighted calculations
- Real-time volatility forecasting
- Risk-adjusted volatility metrics

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, time
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class TradingSession(Enum):
    """Trading session classification"""
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"

@dataclass
class VolatilityMetrics:
    """Container for volatility analysis results"""
    current_volatility: float
    weighted_volatility: float
    session_volatility: float
    regime: VolatilityRegime
    session: TradingSession
    forecast: float
    confidence: float
    risk_adjustment: float
    percentile_rank: float
    z_score: float

class TimeWeightedVolatility:
    """
    Advanced Time-Weighted Volatility Analysis
    
    Provides sophisticated volatility calculations with session weighting,
    time decay, and market regime detection for forex trading strategies.
    """
    
    def __init__(self, 
                 lookback_periods: int = 20,
                 decay_factor: float = 0.94,
                 session_weights: Optional[Dict[str, float]] = None,
                 regime_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize Time-Weighted Volatility analyzer
        
        Args:
            lookback_periods: Number of periods for volatility calculation
            decay_factor: Exponential decay factor for time weighting (0-1)
            session_weights: Custom session weights for volatility calculation
            regime_thresholds: Custom thresholds for volatility regime classification
        """
        self.lookback_periods = lookback_periods
        self.decay_factor = decay_factor
        
        # Default session weights (higher weight = more volatile session)
        self.session_weights = session_weights or {
            TradingSession.ASIAN.value: 0.7,
            TradingSession.LONDON.value: 1.2,
            TradingSession.NY.value: 1.1,
            TradingSession.OVERLAP_LONDON_NY.value: 1.5,
            TradingSession.OVERLAP_ASIAN_LONDON.value: 0.9
        }
        
        # Default volatility regime thresholds (percentiles)
        self.regime_thresholds = regime_thresholds or {
            VolatilityRegime.LOW.value: 25.0,
            VolatilityRegime.NORMAL.value: 75.0,
            VolatilityRegime.HIGH.value: 90.0
        }
        
        # Internal state
        self.volatility_history: List[float] = []
        self.session_history: List[TradingSession] = []
        self.timestamp_history: List[datetime] = []
        
        logger.info(f"TimeWeightedVolatility initialized with {lookback_periods} periods, decay={decay_factor}")
    
    def identify_session(self, timestamp: datetime) -> TradingSession:
        """
        Identify trading session based on UTC timestamp
        
        Args:
            timestamp: UTC timestamp
            
        Returns:
            TradingSession enum value
        """
        utc_time = timestamp.time()
        
        # Session times in UTC
        asian_start = time(22, 0)  # 22:00 UTC (previous day)
        asian_end = time(8, 0)     # 08:00 UTC
        london_start = time(8, 0)   # 08:00 UTC
        london_end = time(16, 0)    # 16:00 UTC
        ny_start = time(13, 0)      # 13:00 UTC
        ny_end = time(22, 0)        # 22:00 UTC
        
        # Check for overlaps first
        if time(13, 0) <= utc_time <= time(16, 0):  # London-NY overlap
            return TradingSession.OVERLAP_LONDON_NY
        elif time(8, 0) <= utc_time <= time(9, 0):  # Asian-London overlap
            return TradingSession.OVERLAP_ASIAN_LONDON
        elif london_start <= utc_time < time(13, 0):  # London only
            return TradingSession.LONDON
        elif time(16, 0) < utc_time <= ny_end:  # NY only
            return TradingSession.NY
        else:  # Asian session (includes overnight)
            return TradingSession.ASIAN
    
    def calculate_returns_volatility(self, prices: np.ndarray) -> float:
        """
        Calculate returns-based volatility
        
        Args:
            prices: Array of price values
            
        Returns:
            Volatility value (annualized)
        """
        if len(prices) < 2:
            return 0.0
        
        # Calculate log returns
        returns = np.diff(np.log(prices))
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(returns) * np.sqrt(252 * 24)  # Annualized for forex (24/7)
        
        return volatility
    
    def apply_time_weights(self, values: np.ndarray) -> np.ndarray:
        """
        Apply exponential time decay weights to values
        
        Args:
            values: Array of values to weight
            
        Returns:
            Array of time-weighted values
        """
        if len(values) == 0:
            return values
        
        # Create exponential decay weights (most recent = highest weight)
        weights = np.array([self.decay_factor ** i for i in range(len(values))])
        weights = weights[::-1]  # Reverse so most recent has highest weight
        weights = weights / np.sum(weights)  # Normalize
        
        return values * weights
    
    def calculate_session_weighted_volatility(self, 
                                            volatilities: List[float],
                                            sessions: List[TradingSession]) -> float:
        """
        Calculate session-weighted volatility
        
        Args:
            volatilities: List of volatility values
            sessions: List of corresponding trading sessions
            
        Returns:
            Session-weighted volatility
        """
        if not volatilities or not sessions:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for vol, session in zip(volatilities, sessions):
            weight = self.session_weights.get(session.value, 1.0)
            weighted_sum += vol * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def classify_volatility_regime(self, current_vol: float) -> VolatilityRegime:
        """
        Classify current volatility regime based on historical percentiles
        
        Args:
            current_vol: Current volatility value
            
        Returns:
            VolatilityRegime classification
        """
        if len(self.volatility_history) < 10:
            return VolatilityRegime.NORMAL
        
        # Calculate percentile rank of current volatility
        percentile = (np.sum(np.array(self.volatility_history) <= current_vol) / 
                     len(self.volatility_history)) * 100
        
        if percentile <= self.regime_thresholds[VolatilityRegime.LOW.value]:
            return VolatilityRegime.LOW
        elif percentile <= self.regime_thresholds[VolatilityRegime.NORMAL.value]:
            return VolatilityRegime.NORMAL
        elif percentile <= self.regime_thresholds[VolatilityRegime.HIGH.value]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def forecast_volatility(self, current_vol: float, session: TradingSession) -> Tuple[float, float]:
        """
        Forecast next period volatility with confidence
        
        Args:
            current_vol: Current volatility value
            session: Current trading session
            
        Returns:
            Tuple of (forecasted_volatility, confidence)
        """
        if len(self.volatility_history) < 5:
            return current_vol, 0.5
        
        # Simple EWMA forecast
        recent_vols = np.array(self.volatility_history[-10:])
        weights = np.array([self.decay_factor ** i for i in range(len(recent_vols))])
        weights = weights[::-1] / np.sum(weights[::-1])
        
        forecast = np.sum(recent_vols * weights)
        
        # Adjust for session characteristics
        session_multiplier = self.session_weights.get(session.value, 1.0)
        forecast *= session_multiplier
        
        # Calculate confidence based on recent volatility stability
        recent_std = np.std(recent_vols[-5:]) if len(recent_vols) >= 5 else 0.0
        confidence = max(0.1, 1.0 - (recent_std / np.mean(recent_vols)) if np.mean(recent_vols) > 0 else 0.5)
        
        return forecast, min(0.95, confidence)
    
    def calculate_risk_adjustment(self, volatility: float, regime: VolatilityRegime) -> float:
        """
        Calculate risk adjustment factor based on volatility regime
        
        Args:
            volatility: Current volatility value
            regime: Volatility regime classification
            
        Returns:
            Risk adjustment factor (1.0 = normal, >1.0 = higher risk, <1.0 = lower risk)
        """
        base_adjustments = {
            VolatilityRegime.LOW: 0.8,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.3,
            VolatilityRegime.EXTREME: 1.8
        }
        
        return base_adjustments.get(regime, 1.0)
    
    def analyze(self, 
                prices: Union[List[float], np.ndarray],
                timestamps: Optional[List[datetime]] = None) -> VolatilityMetrics:
        """
        Perform comprehensive time-weighted volatility analysis
        
        Args:
            prices: Price data for analysis
            timestamps: Corresponding timestamps (optional, uses current time if None)
            
        Returns:
            VolatilityMetrics object with analysis results
        """
        try:
            prices = np.array(prices)
            
            if len(prices) < 2:
                logger.warning("Insufficient price data for volatility analysis")
                return VolatilityMetrics(
                    current_volatility=0.0,
                    weighted_volatility=0.0,
                    session_volatility=0.0,
                    regime=VolatilityRegime.NORMAL,
                    session=TradingSession.ASIAN,
                    forecast=0.0,
                    confidence=0.0,
                    risk_adjustment=1.0,
                    percentile_rank=50.0,
                    z_score=0.0
                )
            
            # Use current time if timestamps not provided
            if timestamps is None:
                timestamps = [datetime.utcnow()] * len(prices)
            
            current_timestamp = timestamps[-1] if timestamps else datetime.utcnow()
            current_session = self.identify_session(current_timestamp)
            
            # Calculate current volatility
            current_volatility = self.calculate_returns_volatility(prices[-self.lookback_periods:])
            
            # Update history
            self.volatility_history.append(current_volatility)
            self.session_history.append(current_session)
            self.timestamp_history.append(current_timestamp)
            
            # Maintain history size
            max_history = self.lookback_periods * 3
            if len(self.volatility_history) > max_history:
                self.volatility_history = self.volatility_history[-max_history:]
                self.session_history = self.session_history[-max_history:]
                self.timestamp_history = self.timestamp_history[-max_history:]
            
            # Calculate time-weighted volatility
            recent_vols = np.array(self.volatility_history[-self.lookback_periods:])
            weighted_vols = self.apply_time_weights(recent_vols)
            weighted_volatility = np.sum(weighted_vols)
            
            # Calculate session-weighted volatility
            recent_sessions = self.session_history[-self.lookback_periods:]
            session_volatility = self.calculate_session_weighted_volatility(
                self.volatility_history[-self.lookback_periods:], recent_sessions
            )
            
            # Classify volatility regime
            regime = self.classify_volatility_regime(current_volatility)
            
            # Forecast volatility
            forecast, confidence = self.forecast_volatility(current_volatility, current_session)
            
            # Calculate risk adjustment
            risk_adjustment = self.calculate_risk_adjustment(current_volatility, regime)
            
            # Calculate percentile rank and z-score
            if len(self.volatility_history) > 1:
                percentile_rank = (np.sum(np.array(self.volatility_history) <= current_volatility) / 
                                 len(self.volatility_history)) * 100
                mean_vol = np.mean(self.volatility_history)
                std_vol = np.std(self.volatility_history)
                z_score = (current_volatility - mean_vol) / std_vol if std_vol > 0 else 0.0
            else:
                percentile_rank = 50.0
                z_score = 0.0
            
            result = VolatilityMetrics(
                current_volatility=current_volatility,
                weighted_volatility=weighted_volatility,
                session_volatility=session_volatility,
                regime=regime,
                session=current_session,
                forecast=forecast,
                confidence=confidence,
                risk_adjustment=risk_adjustment,
                percentile_rank=percentile_rank,
                z_score=z_score
            )
            
            logger.info(f"Volatility analysis complete: {regime.value} regime, "
                       f"current={current_volatility:.4f}, forecast={forecast:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            raise
    
    def get_trading_recommendations(self, metrics: VolatilityMetrics) -> Dict[str, Union[str, float]]:
        """
        Generate trading recommendations based on volatility analysis
        
        Args:
            metrics: VolatilityMetrics from analysis
            
        Returns:
            Dictionary with trading recommendations
        """
        recommendations = {
            "position_size_multiplier": 1.0 / metrics.risk_adjustment,
            "stop_loss_multiplier": metrics.risk_adjustment,
            "take_profit_multiplier": metrics.risk_adjustment * 0.8,
            "session_preference": metrics.session.value,
            "regime_status": metrics.regime.value,
            "confidence_level": metrics.confidence
        }
        
        # Session-specific recommendations
        if metrics.session in [TradingSession.OVERLAP_LONDON_NY, TradingSession.OVERLAP_ASIAN_LONDON]:
            recommendations["strategy_preference"] = "scalping"
            recommendations["timeframe_preference"] = "M1-M5"
        elif metrics.session == TradingSession.LONDON:
            recommendations["strategy_preference"] = "day_trading"
            recommendations["timeframe_preference"] = "M15-H1"
        elif metrics.session == TradingSession.NY:
            recommendations["strategy_preference"] = "day_trading"
            recommendations["timeframe_preference"] = "M15-H1"
        else:  # Asian session
            recommendations["strategy_preference"] = "swing_trading"
            recommendations["timeframe_preference"] = "H1-H4"
        
        # Regime-specific recommendations
        if metrics.regime == VolatilityRegime.EXTREME:
            recommendations["trading_advice"] = "reduce_exposure"
            recommendations["risk_level"] = "high"
        elif metrics.regime == VolatilityRegime.HIGH:
            recommendations["trading_advice"] = "cautious_trading"
            recommendations["risk_level"] = "elevated"
        elif metrics.regime == VolatilityRegime.LOW:
            recommendations["trading_advice"] = "range_trading"
            recommendations["risk_level"] = "low"
        else:
            recommendations["trading_advice"] = "normal_trading"
            recommendations["risk_level"] = "normal"
        
        return recommendations
