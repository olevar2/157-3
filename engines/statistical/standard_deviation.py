#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standard Deviation Indicator - High-Quality Implementation
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

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorSignal, SignalType, IndicatorType, TimeFrame
from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError


class StandardDeviationIndicator(IndicatorBase):
    """
    Advanced Standard Deviation Indicator with Volatility Analysis
    
    Features:
    - Rolling standard deviation calculation
    - Bollinger Bands integration
    - Volatility clustering detection
    - Adaptive period adjustment
    - Multi-timeframe volatility analysis
    - Risk-adjusted signal generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Standard Deviation Indicator"""
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Configuration parameters
        self.period = config.get('period', 20) if config else 20
        self.std_multiplier = config.get('std_multiplier', 2.0) if config else 2.0
        self.adaptive_period = config.get('adaptive_period', True) if config else True
        self.min_period = config.get('min_period', 10) if config else 10
        self.max_period = config.get('max_period', 50) if config else 50
        
        self.logger.info(f"StandardDeviationIndicator initialized with period={self.period}")
    
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform high-precision standard deviation calculation
        
        Returns comprehensive volatility analysis including:
        - Rolling standard deviation
        - Bollinger Bands
        - Volatility percentile
        - Adaptive period recommendations
        """
        try:
            if len(data) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data)}")
            
            # Extract price data
            prices = np.array([float(item['close']) for item in data])
            high_prices = np.array([float(item['high']) for item in data])
            low_prices = np.array([float(item['low']) for item in data])
            
            # Calculate returns for volatility analysis
            returns = np.diff(np.log(prices))
            
            # Calculate rolling standard deviation
            std_values = []
            mean_values = []
            upper_bands = []
            lower_bands = []
            
            for i in range(len(prices)):
                if i < self.period - 1:
                    std_values.append(np.nan)
                    mean_values.append(np.nan)
                    upper_bands.append(np.nan)
                    lower_bands.append(np.nan)
                else:
                    # Get period data
                    period_prices = prices[i - self.period + 1:i + 1]
                    
                    # Calculate statistics
                    mean_price = np.mean(period_prices)
                    std_price = np.std(period_prices, ddof=1)
                    
                    std_values.append(std_price)
                    mean_values.append(mean_price)
                    upper_bands.append(mean_price + (self.std_multiplier * std_price))
                    lower_bands.append(mean_price - (self.std_multiplier * std_price))
            
            # Calculate additional volatility metrics
            current_std = std_values[-1] if not np.isnan(std_values[-1]) else 0
            
            # Historical volatility percentile
            valid_stds = [s for s in std_values if not np.isnan(s)]
            if valid_stds:
                volatility_percentile = stats.percentileofscore(valid_stds, current_std) / 100
            else:
                volatility_percentile = 0.5
            
            # True Range for alternative volatility measure
            tr_values = []
            for i in range(1, len(prices)):
                tr = max(
                    high_prices[i] - low_prices[i],
                    abs(high_prices[i] - prices[i-1]),
                    abs(low_prices[i] - prices[i-1])
                )
                tr_values.append(tr)
            
            # Average True Range
            if len(tr_values) >= self.period:
                atr = np.mean(tr_values[-self.period:])
            else:
                atr = np.mean(tr_values) if tr_values else 0
            
            # Volatility clustering detection
            if len(returns) >= self.period:
                recent_vol = np.std(returns[-self.period//2:]) if len(returns) >= self.period//2 else 0
                historical_vol = np.std(returns[-self.period:])
                volatility_clustering = recent_vol / historical_vol if historical_vol > 0 else 1
            else:
                volatility_clustering = 1
            
            # Adaptive period suggestion
            if self.adaptive_period and current_std > 0:
                # Adjust period based on volatility
                if volatility_percentile > 0.8:  # High volatility
                    suggested_period = max(self.min_period, self.period // 2)
                elif volatility_percentile < 0.2:  # Low volatility
                    suggested_period = min(self.max_period, self.period * 2)
                else:
                    suggested_period = self.period
            else:
                suggested_period = self.period
            
            # Current price position relative to bands
            current_price = prices[-1]
            current_upper = upper_bands[-1] if not np.isnan(upper_bands[-1]) else current_price
            current_lower = lower_bands[-1] if not np.isnan(lower_bands[-1]) else current_price
            
            if current_upper != current_lower:
                band_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                band_position = 0.5
            
            # Band width (volatility measure)
            if not np.isnan(current_upper) and not np.isnan(current_lower):
                band_width = (current_upper - current_lower) / mean_values[-1] * 100
            else:
                band_width = 0
            
            return {
                'std_values': [v if not np.isnan(v) else None for v in std_values],
                'mean_values': [v if not np.isnan(v) else None for v in mean_values],
                'upper_bands': [v if not np.isnan(v) else None for v in upper_bands],
                'lower_bands': [v if not np.isnan(v) else None for v in lower_bands],
                'current_std': float(current_std),
                'volatility_percentile': float(volatility_percentile),
                'atr': float(atr),
                'volatility_clustering': float(volatility_clustering),
                'suggested_period': int(suggested_period),
                'band_position': float(band_position),
                'band_width': float(band_width),
                'returns': returns.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Standard deviation calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on standard deviation analysis
        
        Signal Logic:
        - BUY: Price near lower band with low volatility percentile
        - SELL: Price near upper band with low volatility percentile
        - Avoid signals during high volatility clustering
        """
        try:
            result = self._perform_calculation(data)
            
            current_price = float(data[-1]['close'])
            band_position = result['band_position']
            volatility_percentile = result['volatility_percentile']
            volatility_clustering = result['volatility_clustering']
            band_width = result['band_width']
            
            # Skip signals during extreme volatility
            if volatility_clustering > 2.0 or volatility_percentile > 0.9:
                return None
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = 0.5
            
            # Calculate base confidence from volatility stability
            volatility_stability = 1 - abs(volatility_clustering - 1)
            confidence = min(volatility_stability * (1 - volatility_percentile * 0.5), 1.0)
            
            # Signal generation based on band position
            if band_position <= 0.1:  # Near lower band
                signal_type = SignalType.BUY
                strength = (0.1 - band_position) * 10  # Stronger near lower bound
            elif band_position >= 0.9:  # Near upper band
                signal_type = SignalType.SELL
                strength = (band_position - 0.9) * 10  # Stronger near upper bound
            elif band_position <= 0.2 and volatility_percentile < 0.3:  # Moderate buy zone
                signal_type = SignalType.BUY
                strength = (0.2 - band_position) * 5
            elif band_position >= 0.8 and volatility_percentile < 0.3:  # Moderate sell zone
                signal_type = SignalType.SELL
                strength = (band_position - 0.8) * 5
            
            # Adjust strength based on band width (volatility)
            if band_width < 2:  # Low volatility enhances signals
                strength *= 1.5
            elif band_width > 10:  # High volatility reduces signals
                strength *= 0.5
            
            # Minimum signal threshold
            if signal_type != SignalType.NEUTRAL and strength > 0.3 and confidence > 0.4:
                # Calculate targets based on standard deviation
                std_value = result['current_std']
                
                if signal_type == SignalType.BUY:
                    take_profit = current_price + (std_value * 1.5)
                    stop_loss = current_price - (std_value * 0.8)
                else:
                    take_profit = current_price - (std_value * 1.5)
                    stop_loss = current_price + (std_value * 0.8)
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='StandardDeviation',
                    signal_type=signal_type,
                    strength=min(strength, 1.0),
                    confidence=confidence,
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'band_position': band_position,
                        'volatility_percentile': volatility_percentile,
                        'band_width': band_width,
                        'std_value': std_value,
                        'volatility_clustering': volatility_clustering
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None