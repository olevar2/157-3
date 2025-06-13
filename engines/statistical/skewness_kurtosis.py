#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skewness Indicator - High-Quality Implementation
Platform3 Phase 3 - Enhanced Trading Engine for Charitable Profits
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy import stats

from engines.indicator_base import IndicatorBase, IndicatorSignal, SignalType
from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import ServiceError


class SkewnessIndicator(IndicatorBase):
    """Advanced Skewness Indicator for Distribution Analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        self.period = config.get('period', 20) if config else 20
        self.threshold = config.get('threshold', 0.5) if config else 0.5
        
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate rolling skewness"""
        try:
            if len(data) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data)}")
            
            prices = np.array([float(item['close']) for item in data])
            returns = np.diff(np.log(prices))
            
            skewness_values = []
            for i in range(len(returns)):
                if i < self.period - 1:
                    skewness_values.append(np.nan)
                else:
                    period_returns = returns[i - self.period + 1:i + 1]
                    skew = stats.skew(period_returns)
                    skewness_values.append(skew if not np.isnan(skew) else 0)
            
            current_skewness = skewness_values[-1] if skewness_values else 0
            
            # Classify skewness
            if current_skewness > self.threshold:
                skew_class = "positive_skew"
                interpretation = "right_tail_risk"
            elif current_skewness < -self.threshold:
                skew_class = "negative_skew"  
                interpretation = "left_tail_risk"
            else:
                skew_class = "normal"
                interpretation = "symmetric_distribution"
            
            return {
                'skewness_values': [s if not np.isnan(s) else None for s in skewness_values],
                'current_skewness': float(current_skewness),
                'skew_class': skew_class,
                'interpretation': interpretation
            }
            
        except Exception as e:
            self.logger.error(f"Skewness calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """Generate signals based on skewness analysis"""
        try:
            result = self._perform_calculation(data)
            current_skewness = result['current_skewness']
            skew_class = result['skew_class']
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            
            # Extreme skewness signals
            if abs(current_skewness) > 1.0:
                if current_skewness > 1.0:  # Positive skew - downside risk
                    signal_type = SignalType.SELL
                    strength = min(current_skewness / 2, 1.0)
                elif current_skewness < -1.0:  # Negative skew - upside potential
                    signal_type = SignalType.BUY
                    strength = min(abs(current_skewness) / 2, 1.0)
            
            if signal_type != SignalType.NEUTRAL and strength > 0.3:
                current_price = float(data[-1]['close'])
                price_factor = 0.015 * strength
                
                if signal_type == SignalType.BUY:
                    take_profit = current_price * (1 + price_factor)
                    stop_loss = current_price * (1 - price_factor * 0.7)
                else:
                    take_profit = current_price * (1 - price_factor)
                    stop_loss = current_price * (1 + price_factor * 0.7)
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='Skewness',
                    signal_type=signal_type,
                    strength=strength,
                    confidence=min(abs(current_skewness) / 2, 1.0),
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'current_skewness': current_skewness,
                        'skew_class': skew_class,
                        'interpretation': result['interpretation']
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None


class KurtosisIndicator(IndicatorBase):
    """Advanced Kurtosis Indicator for Tail Risk Analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        self.period = config.get('period', 20) if config else 20
        self.threshold = config.get('threshold', 3.0) if config else 3.0
        
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate rolling kurtosis"""
        try:
            if len(data) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data)}")
            
            prices = np.array([float(item['close']) for item in data])
            returns = np.diff(np.log(prices))
            
            kurtosis_values = []
            for i in range(len(returns)):
                if i < self.period - 1:
                    kurtosis_values.append(np.nan)
                else:
                    period_returns = returns[i - self.period + 1:i + 1]
                    kurt = stats.kurtosis(period_returns, fisher=True)  # Excess kurtosis
                    kurtosis_values.append(kurt if not np.isnan(kurt) else 0)
            
            current_kurtosis = kurtosis_values[-1] if kurtosis_values else 0
            
            # Classify kurtosis
            if current_kurtosis > 1.0:
                kurt_class = "leptokurtic"
                interpretation = "fat_tails_high_risk"
            elif current_kurtosis < -1.0:
                kurt_class = "platykurtic"
                interpretation = "thin_tails_low_risk"
            else:
                kurt_class = "mesokurtic"
                interpretation = "normal_tail_risk"
            
            return {
                'kurtosis_values': [k if not np.isnan(k) else None for k in kurtosis_values],
                'current_kurtosis': float(current_kurtosis),
                'kurt_class': kurt_class,
                'interpretation': interpretation
            }
            
        except Exception as e:
            self.logger.error(f"Kurtosis calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """Generate signals based on kurtosis analysis"""
        try:
            result = self._perform_calculation(data)
            current_kurtosis = result['current_kurtosis']
            kurt_class = result['kurt_class']
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            
            # Extreme kurtosis warnings
            if current_kurtosis > 3.0:  # Very fat tails
                signal_type = SignalType.WARNING
                strength = min(current_kurtosis / 5, 1.0)
            elif current_kurtosis > 1.5:  # Moderately fat tails - reduce position
                signal_type = SignalType.SELL
                strength = min(current_kurtosis / 4, 1.0)
            
            if signal_type != SignalType.NEUTRAL and strength > 0.2:
                current_price = float(data[-1]['close'])
                price_factor = 0.01 * strength
                
                if signal_type == SignalType.SELL:
                    take_profit = current_price * (1 - price_factor)
                    stop_loss = current_price * (1 + price_factor * 1.5)
                else:  # WARNING
                    take_profit = None
                    stop_loss = current_price * (1 - price_factor * 2)
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='Kurtosis',
                    signal_type=signal_type,
                    strength=strength,
                    confidence=min(current_kurtosis / 5, 1.0),
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'current_kurtosis': current_kurtosis,
                        'kurt_class': kurt_class,
                        'interpretation': result['interpretation']
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None


# Export for use
__all__ = ['SkewnessIndicator', 'KurtosisIndicator']