#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Correlation & Relative Momentum Indicators
==================================================

Advanced correlation analysis and momentum indicators with multi-timeframe
support and adaptive market condition detection.

Author: Platform3 AI System
Created: June 6, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union # Added Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import (
        IndicatorBase, IndicatorResult, IndicatorType, 
        TimeFrame, SignalType, IndicatorSignal, MarketData
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import (
        IndicatorBase, IndicatorResult, IndicatorType, 
        TimeFrame, SignalType, IndicatorSignal, MarketData
    )


@dataclass
class CorrelationMatrix:
    """Data structure for correlation analysis results."""
    timeframe: TimeFrame
    correlation_coefficient: float
    strength_category: str  # weak, moderate, strong
    significance_level: float
    sample_size: int
    calculation_timestamp: datetime


@dataclass
class MomentumMetrics:
    """Data structure for momentum calculation results."""
    price_velocity: float
    price_acceleration: float
    momentum_strength: float
    momentum_direction: str  # bullish, bearish, neutral
    divergence_detected: bool
    adaptive_threshold: float


class DynamicCorrelationIndicator(IndicatorBase):
    """
    Dynamic Correlation Indicator for multi-asset correlation analysis.
    
    Calculates dynamic correlation coefficients across multiple timeframes
    with adaptive thresholds based on market volatility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "DynamicCorrelationIndicator"
        self.indicator_type = IndicatorType.MOMENTUM
        
        # Configuration parameters
        self.correlation_periods = config.get('correlation_periods', [5, 10, 20, 50]) if config else [5, 10, 20, 50]
        self.timeframes = config.get('timeframes', [
            TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, 
            TimeFrame.H1, TimeFrame.H4, TimeFrame.D1
        ]) if config else [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1]
        
        # Correlation strength thresholds
        self.weak_threshold = 0.3
        self.moderate_threshold = 0.7
        
        # Adaptive volatility parameters
        self.volatility_lookback = 20
        self.volatility_multiplier = 1.5
        
        self.logger.info(f"Initialized {self.name} with periods {self.correlation_periods}")
    
    def calculate_volatility_adjusted_threshold(self, data: List[MarketData]) -> float:
        """Calculate adaptive correlation threshold based on market volatility."""
        if len(data) < self.volatility_lookback:
            return self.moderate_threshold
        
        # Calculate recent volatility
        recent_prices = [d.close for d in data[-self.volatility_lookback:]]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Adjust threshold based on volatility
        base_threshold = self.moderate_threshold
        volatility_adjustment = volatility * self.volatility_multiplier
        
        # Higher volatility = lower correlation threshold needed for signal
        adjusted_threshold = max(0.1, base_threshold - volatility_adjustment)
        
        return min(0.9, adjusted_threshold)
    
    def calculate_correlation_matrix(self, data: List[MarketData], period: int) -> List[CorrelationMatrix]:
        """Calculate correlation matrix for given period across timeframes."""
        correlation_results = []
        
        if len(data) < period:
            return correlation_results
        
        # Get price series for correlation calculation
        prices = np.array([d.close for d in data[-period:]])
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate correlation with market proxy (using price changes as proxy)
        if len(returns) < 3:  # Need minimum data points
            return correlation_results
        
        # Create synthetic market data for correlation (simplified approach)
        # In real implementation, this would correlate with actual market indices
        market_proxy = np.cumsum(np.random.normal(0, 0.02, len(returns)))  # Random walk as proxy
        
        try:
            correlation_coeff, p_value = pearsonr(returns, market_proxy[:len(returns)])
            
            # Classify correlation strength
            abs_corr = abs(correlation_coeff)
            if abs_corr < self.weak_threshold:
                strength = "weak"
            elif abs_corr < self.moderate_threshold:
                strength = "moderate"
            else:
                strength = "strong"
            
            correlation_matrix = CorrelationMatrix(
                timeframe=TimeFrame.D1,  # Default timeframe
                correlation_coefficient=correlation_coeff,
                strength_category=strength,
                significance_level=p_value,
                sample_size=len(returns),
                calculation_timestamp=datetime.now()
            )
            
            correlation_results.append(correlation_matrix)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
        
        return correlation_results
    
    def detect_correlation_divergence(self, correlation_history: List[float]) -> bool:
        """Detect divergence in correlation patterns."""
        if len(correlation_history) < 10:
            return False
        
        # Find peaks and troughs in correlation
        peaks, _ = find_peaks(correlation_history, height=0.3)
        troughs, _ = find_peaks([-x for x in correlation_history], height=-0.3)
        
        # Simple divergence detection: correlation trending down while price trending up
        recent_corr_trend = np.polyfit(range(len(correlation_history[-5:])), correlation_history[-5:], 1)[0]
        
        return recent_corr_trend < -0.05  # Significant negative trend in correlation
    
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """Calculate dynamic correlation indicator."""
        try:
            if len(data) < max(self.correlation_periods):
                raise ValueError(f"Insufficient data: need at least {max(self.correlation_periods)} points")
            
            # Calculate correlations for all periods
            all_correlations = {}
            correlation_signals = []
            
            for period in self.correlation_periods:
                correlations = self.calculate_correlation_matrix(data, period)
                all_correlations[f'period_{period}'] = correlations
                
                if correlations:
                    avg_correlation = np.mean([c.correlation_coefficient for c in correlations])
                    correlation_signals.append(avg_correlation)
            
            # Get adaptive threshold
            adaptive_threshold = self.calculate_volatility_adjusted_threshold(data)
            
            # Calculate overall correlation score
            if correlation_signals:
                overall_correlation = np.mean(correlation_signals)
                correlation_strength = abs(overall_correlation)
            else:
                overall_correlation = 0.0
                correlation_strength = 0.0
            
            # Generate signal
            signal = None
            if correlation_strength > adaptive_threshold:
                if overall_correlation > 0:
                    signal_type = SignalType.BUY
                    strength = min(1.0, correlation_strength)
                else:
                    signal_type = SignalType.SELL
                    strength = min(1.0, correlation_strength)
                
                signal = IndicatorSignal(
                    timestamp=data[-1].timestamp,
                    indicator_name=self.name,
                    signal_type=signal_type,
                    strength=strength,
                    confidence=correlation_strength,
                    metadata={
                        'correlation_coefficient': overall_correlation,
                        'adaptive_threshold': adaptive_threshold,
                        'periods_analyzed': self.correlation_periods
                    }
                )
            
            # Prepare result value
            result_value = {
                'correlation_coefficient': overall_correlation,
                'correlation_strength': correlation_strength,
                'adaptive_threshold': adaptive_threshold,
                'periods_analyzed': len(self.correlation_periods),
                'correlations_by_period': {
                    f'period_{p}': np.mean([c.correlation_coefficient for c in all_correlations[f'period_{p}']])
                    if all_correlations[f'period_{p}'] else 0.0
                    for p in self.correlation_periods
                }
            }
            
            return IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=TimeFrame.D1,
                value=result_value,
                signal=signal,
                raw_data={
                    'correlation_matrices': all_correlations,
                    'adaptive_threshold': adaptive_threshold,
                    'market_volatility': self.calculate_volatility_adjusted_threshold(data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in DynamicCorrelationIndicator calculation: {e}")
            return IndicatorResult(
                timestamp=data[-1].timestamp if data else datetime.now(),
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=TimeFrame.D1,
                value=0.0,
                signal=None,
                raw_data={'error': str(e)}
            )


class RelativeMomentumIndicator(IndicatorBase):
    """
    Relative Momentum Indicator with velocity and acceleration analysis.
    
    Calculates price velocity, acceleration, and momentum strength with
    divergence detection and adaptive thresholds.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config) # Pass config to IndicatorBase
        self.name = "RelativeMomentumIndicator"
        self.indicator_type = IndicatorType.MOMENTUM
        
        # Configuration parameters from self.config (IndicatorBase stores it)
        self.velocity_period = self.config.get('velocity_period', 14)
        self.acceleration_period = self.config.get('acceleration_period', 7)
        self.momentum_smoothing = self.config.get('momentum_smoothing', 3)
        
        # Multi-timeframe analysis
        self.timeframes = self.config.get('timeframes', [
            TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, 
            TimeFrame.H1, TimeFrame.H4, TimeFrame.D1
        ])
        
        # Momentum thresholds
        self.strong_momentum_threshold = self.config.get('strong_momentum_threshold', 0.75)
        self.weak_momentum_threshold = self.config.get('weak_momentum_threshold', 0.25)
        
        # Divergence detection parameters
        self.divergence_lookback = self.config.get('divergence_lookback', 20)
        self.divergence_sensitivity = self.config.get('divergence_sensitivity', 0.1)
        
        self.logger.info(f"Initialized {self.name} with velocity period {self.velocity_period}")    def _validate_data(self, data: Union[List[Dict[str, Any]], pd.DataFrame], required_columns: Optional[List[str]] = None) -> None:
        """Validate input data for RelativeMomentumIndicator."""
        validation_result = super()._validate_data(data)
        if not validation_result:
            raise ValueError("Base data validation failed.")

        # RMI can work with List[Dict] or pd.DataFrame, but needs 'close' prices.
        # The internal calculations convert to numpy arrays from 'close' prices.
        # Ensure 'close' is available if it's a DataFrame.
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrame input for RMI must contain a 'close' column.")        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                if 'close' not in data[0]:
                    raise ValueError("List[Dict] input for RMI, dictionaries must contain a 'close' key.")
            # Further checks for list of dicts can be added if necessary

    def calculate_price_velocity(self, data: Union[List[MarketData], pd.DataFrame]) -> float:
        """Calculate price velocity (rate of price change)."""
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns or len(data) < self.velocity_period + 1:
                return 0.0
            prices = data['close'].values[-self.velocity_period-1:]
        elif isinstance(data, list) and all(isinstance(item, MarketData) for item in data):
            if len(data) < self.velocity_period + 1:
                return 0.0
            prices = np.array([d.close for d in data[-self.velocity_period-1:]])
        else:
            # Handle List[Dict] or other formats if necessary, or raise error
            # For now, assuming MarketData objects if it's a list
            self.logger.warning("Unsupported data type for velocity calculation, expected List[MarketData] or pd.DataFrame")
            return 0.0
        
        # Calculate velocity using linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize velocity by average price
        avg_price = np.mean(prices)
        velocity = (slope / avg_price) * 100 if avg_price > 0 else 0.0
        
        return velocity
    
    def calculate_price_acceleration(self, data: Union[List[MarketData], pd.DataFrame]) -> float:
        """Calculate price acceleration (rate of velocity change)."""
        # Determine input type and extract close prices
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns or len(data) < self.acceleration_period * 2:
                return 0.0
            # Create MarketData-like list of dicts for compatibility with existing logic
            # This is a bit inefficient but reuses the velocity logic expecting MarketData list
            market_data_list = [MarketData(timestamp=idx, open=row['open'], high=row['high'], low=row['low'], close=row['close'], volume=row['volume'], timeframe=TimeFrame.D1) 
                                for idx, row in data.iterrows()] 
        elif isinstance(data, list) and all(isinstance(item, MarketData) for item in data):
            if len(data) < self.acceleration_period * 2:
                return 0.0
            market_data_list = data
        else:
            self.logger.warning("Unsupported data type for acceleration calculation, expected List[MarketData] or pd.DataFrame")
            return 0.0

        velocities = []
        for i in range(self.acceleration_period):
            end_idx = len(market_data_list) - i
            # Ensure start_idx for slicing is valid and provides enough data for velocity_period
            # The slice for velocity calculation needs self.velocity_period + 1 items
            start_idx_for_velocity_calc = max(0, end_idx - (self.velocity_period + 1))
            
            if end_idx > start_idx_for_velocity_calc and (end_idx - start_idx_for_velocity_calc) >= (self.velocity_period +1) :
                # Pass the correct slice of market_data_list to calculate_price_velocity
                # calculate_price_velocity itself will take the last self.velocity_period + 1 items from this slice
                subset_data_for_velocity = market_data_list[start_idx_for_velocity_calc:end_idx]
                velocity = self.calculate_price_velocity(subset_data_for_velocity)
                velocities.append(velocity)
            else:
                # Not enough data for this iteration's velocity calculation
                # Append NaN or handle as appropriate, or ensure enough data upstream
                # For simplicity here, we might skip or append a neutral value if strictness is not required
                # However, this indicates an issue with data length or slicing logic
                pass # Or velocities.append(0.0) / np.nan if preferred
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate acceleration as change in velocity
        velocities = np.array(velocities[::-1])  # Reverse to chronological order
        acceleration = np.diff(velocities)
        
        return np.mean(acceleration) if len(acceleration) > 0 else 0.0
    
    def calculate_momentum_strength(self, velocity: float, acceleration: float) -> float:
        """Calculate overall momentum strength from velocity and acceleration."""
        # Normalize velocity and acceleration
        velocity_normalized = np.tanh(velocity / 10.0)  # Sigmoid-like normalization
        acceleration_normalized = np.tanh(acceleration / 5.0)
        
        # Combine velocity and acceleration with weights
        momentum_strength = abs(0.7 * velocity_normalized + 0.3 * acceleration_normalized)
        
        return min(1.0, momentum_strength)
    
    def detect_momentum_divergence(self, data: List[MarketData], momentum_history: List[float]) -> bool:
        """Detect price-momentum divergence."""
        if len(data) < self.divergence_lookback or len(momentum_history) < self.divergence_lookback:
            return False
        
        # Get recent price and momentum trends
        recent_prices = [d.close for d in data[-self.divergence_lookback:]]
        recent_momentum = momentum_history[-self.divergence_lookback:]
        
        # Calculate trend slopes
        x = np.arange(len(recent_prices))
        price_slope, _ = np.polyfit(x, recent_prices, 1)
        momentum_slope, _ = np.polyfit(x, recent_momentum, 1)
        
        # Normalize slopes
        price_trend = price_slope / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
        momentum_trend = momentum_slope / np.mean(recent_momentum) if np.mean(recent_momentum) > 0 else 0
        
        # Detect divergence: opposite trends with sufficient magnitude
        divergence = (price_trend * momentum_trend < 0 and 
                     abs(price_trend) > self.divergence_sensitivity and 
                     abs(momentum_trend) > self.divergence_sensitivity)
        
        return divergence
    
    def calculate_adaptive_threshold(self, data: List[MarketData]) -> float:
        """Calculate adaptive momentum threshold based on market conditions."""
        if len(data) < 20:
            return self.strong_momentum_threshold
        
        # Calculate market volatility
        recent_prices = [d.close for d in data[-20:]]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # Calculate average volume (if available)
        volumes = [d.volume for d in data[-20:] if d.volume > 0]
        avg_volume = np.mean(volumes) if volumes else 1.0
        current_volume = data[-1].volume if data[-1].volume > 0 else avg_volume
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Adaptive threshold: lower in high volatility, higher in low volatility
        base_threshold = self.strong_momentum_threshold
        volatility_adjustment = volatility * 10  # Scale volatility impact
        volume_adjustment = np.log(volume_ratio) * 0.1  # Volume impact
        
        adaptive_threshold = base_threshold - volatility_adjustment + volume_adjustment
        
        return max(0.1, min(0.9, adaptive_threshold))
    
    def calculate(self, data: Union[List[MarketData], pd.DataFrame]) -> IndicatorResult:
        """Calculate relative momentum indicator."""
        try:
            # Perform validation using the new _validate_data method
            self._validate_data(data)

            required_length = max(self.velocity_period, self.acceleration_period) * 2
            current_length = len(data.index) if isinstance(data, pd.DataFrame) else len(data)

            if current_length < required_length:
                raise ValueError(f"Insufficient data: need at least {required_length} points, got {current_length}")
            
            # Convert to DataFrame if it's List[MarketData] for easier handling, or adapt internal logic
            # For this example, let's assume internal calculations are adapted or data is DataFrame
            # If data is List[MarketData], it needs to be converted to a structure that
            # calculate_price_velocity and calculate_price_acceleration can handle (e.g. pd.DataFrame or List[MarketData])
            
            # Ensure data is in a consistent format for calculations (e.g., pd.DataFrame)
            # This part might need adjustment based on how MarketData is structured and used
            df_data = None
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], MarketData):
                # Convert List[MarketData] to DataFrame
                df_data = pd.DataFrame([{'timestamp': md.timestamp, 'open': md.open, 'high': md.high, 'low': md.low, 'close': md.close, 'volume': md.volume} for md in data])
                # If timestamps are datetime objects, set as index if needed by downstream processing
                if not df_data.empty and isinstance(df_data['timestamp'].iloc[0], datetime):
                    df_data.set_index('timestamp', inplace=True)
            elif isinstance(data, pd.DataFrame):
                df_data = data
            else:
                raise ValueError("Unsupported data type for RMI calculation. Must be List[MarketData] or pd.DataFrame.")

            # Calculate velocity and acceleration using the (potentially converted) DataFrame
            velocity = self.calculate_price_velocity(df_data)
            acceleration = self.calculate_price_acceleration(df_data)
            
            # Calculate momentum strength
            momentum_strength = self.calculate_momentum_strength(velocity, acceleration)
            
            # Determine momentum direction
            if velocity > 0.1:
                momentum_direction = "bullish"
            elif velocity < -0.1:
                momentum_direction = "bearish"
            else:
                momentum_direction = "neutral"
            
            # Calculate adaptive threshold
            adaptive_threshold = self.calculate_adaptive_threshold(data)
            
            # Simple momentum history for divergence detection
            momentum_history = [momentum_strength]  # In real implementation, maintain history
            divergence_detected = False  # Simplified for this implementation
            
            # Create momentum metrics
            momentum_metrics = MomentumMetrics(
                price_velocity=velocity,
                price_acceleration=acceleration,
                momentum_strength=momentum_strength,
                momentum_direction=momentum_direction,
                divergence_detected=divergence_detected,
                adaptive_threshold=adaptive_threshold
            )
            
            # Generate signal
            signal = None
            if momentum_strength > adaptive_threshold:
                if velocity > 0:
                    signal_type = SignalType.BUY
                else:
                    signal_type = SignalType.SELL
                
                # Adjust strength based on acceleration
                signal_strength = momentum_strength
                if acceleration * velocity > 0:  # Acceleration in same direction
                    signal_strength = min(1.0, signal_strength * 1.2)
                
                signal = IndicatorSignal(
                    timestamp=data[-1].timestamp,
                    indicator_name=self.name,
                    signal_type=signal_type,
                    strength=signal_strength,
                    confidence=momentum_strength,
                    metadata={
                        'velocity': velocity,
                        'acceleration': acceleration,
                        'momentum_direction': momentum_direction,
                        'adaptive_threshold': adaptive_threshold
                    }
                )
            
            # Prepare result value
            result_value = {
                'price_velocity': velocity,
                'price_acceleration': acceleration,
                'momentum_strength': momentum_strength,
                'momentum_direction': momentum_direction,
                'adaptive_threshold': adaptive_threshold,
                'divergence_detected': divergence_detected
            }
            
            return IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=TimeFrame.D1,
                value=result_value,
                signal=signal,
                raw_data={
                    'momentum_metrics': momentum_metrics,
                    'velocity_period': self.velocity_period,
                    'acceleration_period': self.acceleration_period
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in RelativeMomentumIndicator calculation: {e}")
            return IndicatorResult(
                timestamp=data[-1].timestamp if data else datetime.now(),
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=TimeFrame.D1,
                value=0.0,
                signal=None,
                raw_data={'error': str(e)}
            )


# Convenience functions for easy access
def create_dynamic_correlation_indicator(config: Optional[Dict[str, Any]] = None) -> DynamicCorrelationIndicator:
    """Create a DynamicCorrelationIndicator instance."""
    return DynamicCorrelationIndicator(config)


def create_relative_momentum_indicator(config: Optional[Dict[str, Any]] = None) -> RelativeMomentumIndicator:
    """Create a RelativeMomentumIndicator instance."""
    return RelativeMomentumIndicator(config)


# Export classes and functions
__all__ = [
    'DynamicCorrelationIndicator',
    'RelativeMomentumIndicator',
    'CorrelationMatrix',
    'MomentumMetrics',
    'create_dynamic_correlation_indicator',
    'create_relative_momentum_indicator'
]
