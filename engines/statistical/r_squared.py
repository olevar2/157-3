#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-Squared Indicator - High-Quality Implementation
Platform3 Phase 3 - Enhanced Trading Engine for Charitable Profits  
Helping sick and poor children through advanced trading algorithms
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

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


class RSquaredIndicator(IndicatorBase):
    """
    Advanced R-Squared Indicator with Trend Strength Analysis
    
    Features:
    - R-squared calculation for trend strength
    - Multiple regression models (linear, polynomial)
    - Rolling R-squared analysis
    - Trend reliability measurement
    - Forecast confidence assessment
    - Model selection optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize R-Squared Indicator"""
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Configuration parameters
        self.period = config.get('period', 20) if config else 20
        self.polynomial_degree = config.get('polynomial_degree', 2) if config else 2
        self.confidence_threshold = config.get('confidence_threshold', 0.7) if config else 0.7
        self.trend_threshold = config.get('trend_threshold', 0.5) if config else 0.5
        
        self.logger.info(f"RSquaredIndicator initialized with period={self.period}")
    
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform high-precision R-squared calculation
        
        Returns comprehensive trend strength analysis including:
        - Linear R-squared values
        - Polynomial R-squared values
        - Rolling R-squared
        - Trend reliability metrics
        - Model comparison
        """
        try:
            if len(data) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data)}")
            
            # Extract price data
            prices = np.array([float(item['close']) for item in data])
            
            # Calculate rolling R-squared values
            linear_r_squared = []
            polynomial_r_squared = []
            trend_strength = []
            model_reliability = []
            
            for i in range(len(prices)):
                if i < self.period - 1:
                    linear_r_squared.append(np.nan)
                    polynomial_r_squared.append(np.nan)
                    trend_strength.append(np.nan)
                    model_reliability.append(np.nan)
                else:
                    # Get period data
                    period_prices = prices[i - self.period + 1:i + 1]
                    X = np.arange(len(period_prices)).reshape(-1, 1)
                    y = period_prices
                    
                    try:
                        # Linear regression R-squared
                        linear_model = LinearRegression()
                        linear_model.fit(X, y)
                        y_pred_linear = linear_model.predict(X)
                        r2_linear = r2_score(y, y_pred_linear)
                        
                        # Polynomial regression R-squared
                        X_poly = np.column_stack([X.flatten() ** j for j in range(1, self.polynomial_degree + 1)])
                        poly_model = LinearRegression()
                        poly_model.fit(X_poly, y)
                        y_pred_poly = poly_model.predict(X_poly)
                        r2_poly = r2_score(y, y_pred_poly)
                        
                        # Ensure R-squared values are valid
                        r2_linear = max(0, min(1, r2_linear)) if not np.isnan(r2_linear) else 0
                        r2_poly = max(0, min(1, r2_poly)) if not np.isnan(r2_poly) else 0
                        
                        linear_r_squared.append(r2_linear)
                        polynomial_r_squared.append(r2_poly)
                        
                        # Calculate trend strength based on R-squared
                        strength = max(r2_linear, r2_poly)
                        trend_strength.append(strength)
                        
                        # Model reliability (consistency over time)
                        if len(linear_r_squared) >= 5:
                            recent_r2 = [r for r in linear_r_squared[-5:] if not np.isnan(r)]
                            if recent_r2:
                                reliability = 1 - np.std(recent_r2) / (np.mean(recent_r2) + 0.001)
                                reliability = max(0, min(1, reliability))
                            else:
                                reliability = 0
                        else:
                            reliability = 0
                        
                        model_reliability.append(reliability)
                        
                    except Exception as e:
                        self.logger.warning(f"R-squared calculation failed for period {i}: {e}")
                        linear_r_squared.append(0)
                        polynomial_r_squared.append(0)
                        trend_strength.append(0)
                        model_reliability.append(0)
            
            # Current values
            current_linear_r2 = linear_r_squared[-1] if not np.isnan(linear_r_squared[-1]) else 0
            current_poly_r2 = polynomial_r_squared[-1] if not np.isnan(polynomial_r_squared[-1]) else 0
            current_trend_strength = trend_strength[-1] if not np.isnan(trend_strength[-1]) else 0
            current_reliability = model_reliability[-1] if not np.isnan(model_reliability[-1]) else 0
            
            # Trend classification based on R-squared
            if current_trend_strength >= 0.8:
                trend_classification = "very_strong"
            elif current_trend_strength >= 0.6:
                trend_classification = "strong"
            elif current_trend_strength >= 0.4:
                trend_classification = "moderate"
            elif current_trend_strength >= 0.2:
                trend_classification = "weak"
            else:
                trend_classification = "no_trend"
            
            # Model selection (linear vs polynomial)
            if current_poly_r2 > current_linear_r2 + 0.1:
                best_model = "polynomial"
                best_r_squared = current_poly_r2
            else:
                best_model = "linear"
                best_r_squared = current_linear_r2
            
            # Calculate forecast confidence
            if best_r_squared > 0.7 and current_reliability > 0.6:
                forecast_confidence = "high"
            elif best_r_squared > 0.5 and current_reliability > 0.4:
                forecast_confidence = "medium"
            elif best_r_squared > 0.3:
                forecast_confidence = "low"
            else:
                forecast_confidence = "very_low"
            
            # Trend direction analysis
            if len(prices) >= 2:
                recent_slope = (prices[-1] - prices[-2]) / prices[-2]
                if abs(recent_slope) > 0.001:
                    trend_direction = "upward" if recent_slope > 0 else "downward"
                else:
                    trend_direction = "sideways"
            else:
                trend_direction = "unknown"
            
            # Calculate R-squared momentum (rate of change)
            valid_linear_r2 = [r for r in linear_r_squared if not np.isnan(r)]
            if len(valid_linear_r2) >= 3:
                r2_momentum = valid_linear_r2[-1] - valid_linear_r2[-3]
            else:
                r2_momentum = 0
            
            return {
                'linear_r_squared': [r if not np.isnan(r) else None for r in linear_r_squared],
                'polynomial_r_squared': [r if not np.isnan(r) else None for r in polynomial_r_squared],
                'trend_strength': [t if not np.isnan(t) else None for t in trend_strength],
                'model_reliability': [m if not np.isnan(m) else None for m in model_reliability],
                'current_linear_r2': float(current_linear_r2),
                'current_poly_r2': float(current_poly_r2),
                'current_trend_strength': float(current_trend_strength),
                'current_reliability': float(current_reliability),
                'trend_classification': trend_classification,
                'best_model': best_model,
                'best_r_squared': float(best_r_squared),
                'forecast_confidence': forecast_confidence,
                'trend_direction': trend_direction,
                'r2_momentum': float(r2_momentum)
            }
            
        except Exception as e:
            self.logger.error(f"R-squared calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on R-squared analysis
        
        Signal Logic:
        - High R-squared + strong trend = follow trend
        - Low R-squared = avoid trend following
        - R-squared momentum changes = trend change signals
        """
        try:
            result = self._perform_calculation(data)
            
            current_trend_strength = result['current_trend_strength']
            current_reliability = result['current_reliability']
            trend_direction = result['trend_direction']
            forecast_confidence = result['forecast_confidence']
            r2_momentum = result['r2_momentum']
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = current_reliability
            
            # Strong trend signals
            if current_trend_strength > self.confidence_threshold and current_reliability > 0.5:
                if trend_direction == "upward":
                    signal_type = SignalType.BUY
                    strength = current_trend_strength * current_reliability
                elif trend_direction == "downward":
                    signal_type = SignalType.SELL
                    strength = current_trend_strength * current_reliability
            
            # R-squared momentum signals (trend change detection)
            elif abs(r2_momentum) > 0.2 and current_trend_strength > 0.4:
                if r2_momentum > 0.2:  # R-squared increasing
                    if trend_direction == "upward":
                        signal_type = SignalType.BUY
                    elif trend_direction == "downward":
                        signal_type = SignalType.SELL
                    strength = min(r2_momentum * 3, 1.0)
                elif r2_momentum < -0.2:  # R-squared decreasing (trend weakening)
                    signal_type = SignalType.WARNING
                    strength = min(abs(r2_momentum) * 2, 1.0)
            
            # Mean reversion signals (low R-squared)
            elif current_trend_strength < 0.3 and current_reliability > 0.3:
                # In low trend strength, look for mean reversion opportunities
                current_price = float(data[-1]['close'])
                if len(data) >= self.period:
                    mean_price = np.mean([float(item['close']) for item in data[-self.period:]])
                    price_deviation = (current_price - mean_price) / mean_price
                    
                    if price_deviation > 0.02:  # 2% above mean
                        signal_type = SignalType.SELL
                        strength = min(abs(price_deviation) * 10, 1.0) * (1 - current_trend_strength)
                    elif price_deviation < -0.02:  # 2% below mean
                        signal_type = SignalType.BUY
                        strength = min(abs(price_deviation) * 10, 1.0) * (1 - current_trend_strength)
            
            # Minimum signal threshold
            if signal_type != SignalType.NEUTRAL and strength > 0.3 and confidence > 0.3:
                current_price = float(data[-1]['close'])
                
                # Calculate targets based on trend strength
                price_factor = current_trend_strength * 0.03  # Up to 3% based on trend strength
                
                if signal_type == SignalType.BUY:
                    take_profit = current_price * (1 + price_factor)
                    stop_loss = current_price * (1 - price_factor * 0.6)
                elif signal_type == SignalType.SELL:
                    take_profit = current_price * (1 - price_factor)
                    stop_loss = current_price * (1 + price_factor * 0.6)
                else:  # WARNING
                    take_profit = None
                    stop_loss = current_price * (1 - price_factor * 0.8)
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='RSquared',
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'trend_strength': current_trend_strength,
                        'reliability': current_reliability,
                        'trend_direction': trend_direction,
                        'forecast_confidence': forecast_confidence,
                        'r2_momentum': r2_momentum,
                        'best_model': result['best_model']
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None