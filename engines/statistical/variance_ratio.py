#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Variance Ratio Indicator - High-Quality Implementation
Platform3 Phase 3 - Enhanced Trading Engine for Charitable Profits
Helping sick and poor children through advanced trading algorithms
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


class VarianceRatioIndicator(IndicatorBase):
    """
    Advanced Variance Ratio Indicator for Market Efficiency Testing
    
    Features:
    - Lo-MacKinlay variance ratio test
    - Random walk hypothesis testing
    - Multiple time horizons analysis
    - Mean reversion detection
    - Market microstructure analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        
        self.period = config.get('period', 20) if config else 20
        self.test_lags = config.get('test_lags', [2, 4, 8, 16]) if config else [2, 4, 8, 16]
        self.confidence_level = config.get('confidence_level', 0.95) if config else 0.95
        
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate variance ratios for multiple lags"""
        try:
            if len(data) < max(self.test_lags) * 2:
                raise ServiceError(f"Insufficient data for variance ratio calculation")
            
            prices = np.array([float(item['close']) for item in data])
            log_prices = np.log(prices)
            returns = np.diff(log_prices)
            
            variance_ratios = {}
            z_statistics = {}
            p_values = {}
            
            # Calculate variance ratio for each lag
            for lag in self.test_lags:
                if len(returns) >= lag * 10:  # Minimum data requirement
                    # Calculate overlapping returns
                    overlapping_returns = []
                    for i in range(len(returns) - lag + 1):
                        overlapping_returns.append(np.sum(returns[i:i+lag]))
                    
                    overlapping_returns = np.array(overlapping_returns)
                    
                    # Variance ratio calculation
                    var_1 = np.var(returns, ddof=1)
                    var_lag = np.var(overlapping_returns, ddof=1) / lag
                    
                    if var_1 > 0:
                        vr = var_lag / var_1
                        
                        # Lo-MacKinlay test statistic
                        n = len(returns)
                        z_stat = (vr - 1) * np.sqrt(n * lag / (2 * (lag - 1)))
                        
                        # P-value (two-tailed test)
                        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        
                        variance_ratios[f'lag_{lag}'] = float(vr)
                        z_statistics[f'lag_{lag}'] = float(z_stat)
                        p_values[f'lag_{lag}'] = float(p_val)
            
            # Overall market efficiency assessment
            significant_departures = sum(1 for p in p_values.values() if p < (1 - self.confidence_level))
            efficiency_score = 1 - (significant_departures / len(p_values)) if p_values else 0.5
            
            # Mean reversion tendency
            mean_vr = np.mean(list(variance_ratios.values())) if variance_ratios else 1.0
            if mean_vr < 0.8:
                market_behavior = "mean_reverting"
            elif mean_vr > 1.2:
                market_behavior = "momentum"
            else:
                market_behavior = "random_walk"
            
            return {
                'variance_ratios': variance_ratios,
                'z_statistics': z_statistics,
                'p_values': p_values,
                'efficiency_score': float(efficiency_score),
                'market_behavior': market_behavior,
                'mean_variance_ratio': float(mean_vr)
            }
            
        except Exception as e:
            self.logger.error(f"Variance ratio calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """Generate signals based on variance ratio analysis"""
        try:
            result = self._perform_calculation(data)
            
            market_behavior = result['market_behavior']
            efficiency_score = result['efficiency_score']
            mean_vr = result['mean_variance_ratio']
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = efficiency_score
            
            # Mean reversion signals
            if market_behavior == "mean_reverting" and mean_vr < 0.7:
                current_price = float(data[-1]['close'])
                recent_prices = [float(item['close']) for item in data[-10:]]
                mean_price = np.mean(recent_prices)
                
                if current_price > mean_price * 1.02:
                    signal_type = SignalType.SELL
                    strength = min((1 - mean_vr) * 2, 1.0)
                elif current_price < mean_price * 0.98:
                    signal_type = SignalType.BUY
                    strength = min((1 - mean_vr) * 2, 1.0)
            
            # Momentum signals
            elif market_behavior == "momentum" and mean_vr > 1.3:
                if len(data) >= 3:
                    recent_trend = (float(data[-1]['close']) - float(data[-3]['close'])) / float(data[-3]['close'])
                    if abs(recent_trend) > 0.01:
                        signal_type = SignalType.BUY if recent_trend > 0 else SignalType.SELL
                        strength = min((mean_vr - 1) * 1.5, 1.0)
            
            if signal_type != SignalType.NEUTRAL and strength > 0.3:
                current_price = float(data[-1]['close'])
                price_factor = 0.02 * strength  # Up to 2% based on signal strength
                
                if signal_type == SignalType.BUY:
                    take_profit = current_price * (1 + price_factor)
                    stop_loss = current_price * (1 - price_factor * 0.6)
                else:
                    take_profit = current_price * (1 - price_factor)
                    stop_loss = current_price * (1 + price_factor * 0.6)
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='VarianceRatio',
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'market_behavior': market_behavior,
                        'efficiency_score': efficiency_score,
                        'mean_variance_ratio': mean_vr
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None


# Export for use
__all__ = ['VarianceRatioIndicator']