#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear Regression Indicator - High-Quality Implementation
Platform3 Phase 3 - Enhanced Trading Engine for Charitable Profits
Helping sick and poor children through advanced trading algorithms
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorSignal, SignalType, IndicatorType, TimeFrame
from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError


class LinearRegressionIndicator(IndicatorBase):
    """
    Advanced Linear Regression Indicator with Multi-Timeframe Analysis
    
    Features:
    - Linear regression trend analysis with confidence intervals
    - Support and resistance level detection
    - Trend strength and direction measurement
    - R-squared correlation analysis
    - Slope-based signal generation
    - Statistical significance testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Linear Regression Indicator"""
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Configuration parameters
        self.period = config.get('period', 20) if config else 20
        self.confidence_level = config.get('confidence_level', 0.95) if config else 0.95
        self.min_r_squared = config.get('min_r_squared', 0.7) if config else 0.7
        self.slope_threshold = config.get('slope_threshold', 0.001) if config else 0.001
        
        self.logger.info(f"LinearRegressionIndicator initialized with period={self.period}")
    
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform high-precision linear regression calculation
        
        Returns comprehensive regression analysis including:
        - Regression line values
        - Upper and lower bounds
        - R-squared value
        - Slope and intercept
        - Statistical significance
        """
        try:
            if len(data) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data)}")
            
            # Extract price data
            prices = np.array([float(item['close']) for item in data[-self.period:]])
            timestamps = np.array(range(len(prices)))
            
            # Perform linear regression
            X = timestamps.reshape(-1, 1)
            y = prices
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate predictions
            y_pred = model.predict(X)
            
            # Calculate statistical metrics
            r_squared = r2_score(y, y_pred)
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Calculate standard error and confidence intervals
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(y) - 2)
            margin_error = t_critical * std_error
            
            upper_bound = y_pred + margin_error
            lower_bound = y_pred - margin_error
            
            # Calculate trend strength
            trend_strength = min(abs(slope) * 1000, 1.0)  # Normalize to 0-1
            
            # Determine trend direction
            if slope > self.slope_threshold:
                trend_direction = "upward"
            elif slope < -self.slope_threshold:
                trend_direction = "downward"
            else:
                trend_direction = "sideways"
            
            # Calculate support and resistance levels
            support_level = np.min(lower_bound)
            resistance_level = np.max(upper_bound)
            
            # Current price position relative to regression line
            current_price = prices[-1]
            regression_value = y_pred[-1]
            price_deviation = (current_price - regression_value) / regression_value * 100
            
            return {
                'regression_line': y_pred.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist(),
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_squared),
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'support_level': float(support_level),
                'resistance_level': float(resistance_level),
                'current_deviation': float(price_deviation),
                'standard_error': float(std_error),
                'statistical_significance': r_squared >= self.min_r_squared,
                'timestamps': timestamps.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Linear regression calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on linear regression analysis
        
        Signal Logic:
        - BUY: Price near lower bound of upward trend with high R²
        - SELL: Price near upper bound of downward trend with high R²
        - Strong signals require high statistical significance
        """
        try:
            result = self._perform_calculation(data)
            
            if not result['statistical_significance']:
                return None
            
            current_price = float(data[-1]['close'])
            regression_value = result['regression_line'][-1]
            upper_bound = result['upper_bound'][-1]
            lower_bound = result['lower_bound'][-1]
            
            slope = result['slope']
            r_squared = result['r_squared']
            trend_strength = result['trend_strength']
            
            # Calculate signal strength based on multiple factors
            confidence = min(r_squared * trend_strength * 1.2, 1.0)
            
            # Price position relative to bounds
            price_position = (current_price - lower_bound) / (upper_bound - lower_bound)
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            
            # Strong upward trend signals
            if slope > self.slope_threshold and r_squared > 0.8:
                if price_position < 0.3:  # Near lower bound
                    signal_type = SignalType.BUY
                    strength = confidence * (1 - price_position)
                elif price_position > 0.7:  # Near upper bound
                    signal_type = SignalType.SELL
                    strength = confidence * price_position * 0.6
            
            # Strong downward trend signals
            elif slope < -self.slope_threshold and r_squared > 0.8:
                if price_position > 0.7:  # Near upper bound
                    signal_type = SignalType.SELL
                    strength = confidence * price_position
                elif price_position < 0.3:  # Near lower bound
                    signal_type = SignalType.BUY
                    strength = confidence * (1 - price_position) * 0.6
            
            # Sideways market - mean reversion signals
            elif abs(slope) <= self.slope_threshold and r_squared > 0.6:
                if price_position < 0.2:
                    signal_type = SignalType.BUY
                    strength = confidence * 0.7
                elif price_position > 0.8:
                    signal_type = SignalType.SELL
                    strength = confidence * 0.7
            
            if signal_type != SignalType.NEUTRAL and strength > 0.3:
                # Calculate targets
                price_range = upper_bound - lower_bound
                
                if signal_type == SignalType.BUY:
                    take_profit = current_price + (price_range * 0.6)
                    stop_loss = lower_bound * 0.995
                else:
                    take_profit = current_price - (price_range * 0.6)
                    stop_loss = upper_bound * 1.005
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='LinearRegression',
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'slope': slope,
                        'r_squared': r_squared,
                        'trend_direction': result['trend_direction'],
                        'price_position': price_position,
                        'regression_value': regression_value
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None
    
    def get_support_resistance(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get dynamic support and resistance levels"""
        try:
            result = self._perform_calculation(data)
            return {
                'support': result['support_level'],
                'resistance': result['resistance_level'],
                'current_regression': result['regression_line'][-1]
            }
        except Exception as e:
            self.logger.error(f"Support/resistance calculation failed: {e}")
            return {}
    
    def get_trend_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive trend analysis"""
        try:
            result = self._perform_calculation(data)
            return {
                'direction': result['trend_direction'],
                'strength': result['trend_strength'],
                'slope': result['slope'],
                'r_squared': result['r_squared'],
                'statistical_significance': result['statistical_significance']
            }
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = []
    base_price = 100.0
    for i in range(30):
        price = base_price + (i * 0.5) + np.random.normal(0, 0.2)
        sample_data.append({
            'timestamp': datetime.now().isoformat(),
            'open': price - 0.1,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1000
        })
    
    # Initialize indicator
    config = {'period': 20, 'confidence_level': 0.95}
    indicator = LinearRegressionIndicator(config)
    
    # Test calculation
    result = indicator.calculate(sample_data)
    print("Linear Regression Result:", result)
    
    # Test signal generation
    signal = indicator.generate_signal(sample_data)
    if signal:
        print(f"Signal: {signal.signal_type.value}, Strength: {signal.strength:.3f}")