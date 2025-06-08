#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation Coefficient Indicator - High-Quality Implementation
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


class CorrelationCoefficientIndicator(IndicatorBase):
    """
    Advanced Correlation Coefficient Indicator with Multi-Asset Analysis
    
    Features:
    - Pearson correlation coefficient calculation
    - Spearman rank correlation
    - Rolling correlation analysis
    - Cross-asset correlation detection
    - Correlation stability measurement
    - Divergence signal generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Correlation Coefficient Indicator"""
        super().__init__(config)
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Configuration parameters
        self.period = config.get('period', 20) if config else 20
        self.reference_data = config.get('reference_data', None) if config else None
        self.correlation_threshold = config.get('correlation_threshold', 0.7) if config else 0.7
        self.stability_period = config.get('stability_period', 5) if config else 5
        
        self.logger.info(f"CorrelationCoefficientIndicator initialized with period={self.period}")
    
    def _perform_calculation(self, data: List[Dict[str, Any]], reference_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform high-precision correlation coefficient calculation
        
        Args:
            data: Primary asset data
            reference_data: Secondary asset data for correlation analysis
        
        Returns comprehensive correlation analysis including:
        - Pearson correlation coefficients
        - Spearman rank correlations  
        - Rolling correlations
        - Correlation stability metrics
        """
        try:
            if len(data) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data)}")
            
            # Extract price data
            prices = np.array([float(item['close']) for item in data])
            returns = np.diff(np.log(prices))
            
            # If no reference data provided, use price vs time correlation
            if reference_data is None or len(reference_data) < len(data):
                reference_series = np.arange(len(prices))
                reference_returns = np.diff(reference_series)
                analysis_type = "time_correlation"
            else:
                ref_prices = np.array([float(item['close']) for item in reference_data[:len(data)]])
                reference_series = ref_prices
                reference_returns = np.diff(np.log(ref_prices))
                analysis_type = "asset_correlation"
            
            # Calculate rolling correlations
            pearson_correlations = []
            spearman_correlations = []
            
            for i in range(len(prices)):
                if i < self.period - 1:
                    pearson_correlations.append(np.nan)
                    spearman_correlations.append(np.nan)
                else:
                    # Get period data
                    if analysis_type == "time_correlation":
                        period_prices = prices[i - self.period + 1:i + 1]
                        period_ref = reference_series[i - self.period + 1:i + 1]
                    else:
                        period_returns = returns[max(0, i - self.period):i]
                        period_ref_returns = reference_returns[max(0, i - self.period):i]
                        
                        if len(period_returns) == len(period_ref_returns) and len(period_returns) > 1:
                            # Pearson correlation
                            try:
                                pearson_corr, _ = stats.pearsonr(period_returns, period_ref_returns)
                                pearson_correlations.append(pearson_corr if not np.isnan(pearson_corr) else 0)
                            except:
                                pearson_correlations.append(0)
                            
                            # Spearman correlation
                            try:
                                spearman_corr, _ = stats.spearmanr(period_returns, period_ref_returns)
                                spearman_correlations.append(spearman_corr if not np.isnan(spearman_corr) else 0)
                            except:
                                spearman_correlations.append(0)
                        else:
                            pearson_correlations.append(0)
                            spearman_correlations.append(0)
                        continue
                    
                    # Time correlation calculation
                    try:
                        pearson_corr, _ = stats.pearsonr(period_prices, period_ref)
                        pearson_correlations.append(pearson_corr if not np.isnan(pearson_corr) else 0)
                        
                        spearman_corr, _ = stats.spearmanr(period_prices, period_ref)
                        spearman_correlations.append(spearman_corr if not np.isnan(spearman_corr) else 0)
                    except:
                        pearson_correlations.append(0)
                        spearman_correlations.append(0)
            
            # Calculate correlation stability
            valid_pearson = [c for c in pearson_correlations if not np.isnan(c)]
            valid_spearman = [c for c in spearman_correlations if not np.isnan(c)]
            
            if len(valid_pearson) >= self.stability_period:
                recent_pearson = valid_pearson[-self.stability_period:]
                pearson_stability = 1 - np.std(recent_pearson)
            else:
                pearson_stability = 0
            
            if len(valid_spearman) >= self.stability_period:
                recent_spearman = valid_spearman[-self.stability_period:]
                spearman_stability = 1 - np.std(recent_spearman)
            else:
                spearman_stability = 0
            
            # Current correlation values
            current_pearson = valid_pearson[-1] if valid_pearson else 0
            current_spearman = valid_spearman[-1] if valid_spearman else 0
            
            # Correlation strength classification
            abs_pearson = abs(current_pearson)
            if abs_pearson >= 0.8:
                correlation_strength = "very_strong"
            elif abs_pearson >= 0.6:
                correlation_strength = "strong"
            elif abs_pearson >= 0.4:
                correlation_strength = "moderate"
            elif abs_pearson >= 0.2:
                correlation_strength = "weak"
            else:
                correlation_strength = "very_weak"
            
            # Correlation direction
            if current_pearson > 0.1:
                correlation_direction = "positive"
            elif current_pearson < -0.1:
                correlation_direction = "negative"
            else:
                correlation_direction = "neutral"
            
            # Calculate correlation breakdown/divergence signals
            correlation_changes = np.diff(valid_pearson) if len(valid_pearson) > 1 else []
            recent_change = correlation_changes[-1] if correlation_changes else 0
            
            # Divergence detection
            if len(correlation_changes) >= 3:
                trend_change = np.mean(correlation_changes[-3:])
                if abs(trend_change) > 0.1:
                    divergence_signal = "breaking" if trend_change < 0 else "strengthening"
                else:
                    divergence_signal = "stable"
            else:
                divergence_signal = "insufficient_data"
            
            return {
                'pearson_correlations': [c if not np.isnan(c) else None for c in pearson_correlations],
                'spearman_correlations': [c if not np.isnan(c) else None for c in spearman_correlations],
                'current_pearson': float(current_pearson),
                'current_spearman': float(current_spearman),
                'pearson_stability': float(pearson_stability),
                'spearman_stability': float(spearman_stability),
                'correlation_strength': correlation_strength,
                'correlation_direction': correlation_direction,
                'recent_change': float(recent_change),
                'divergence_signal': divergence_signal,
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            raise ServiceError(f"Calculation error: {str(e)}")
    
    def generate_signal(self, data: List[Dict[str, Any]], reference_data: Optional[List[Dict[str, Any]]] = None) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on correlation analysis
        
        Signal Logic:
        - Divergence signals when correlation breaks down
        - Strength signals when correlation is very strong
        - Direction change signals
        """
        try:
            result = self._perform_calculation(data, reference_data)
            
            current_pearson = result['current_pearson']
            pearson_stability = result['pearson_stability']
            divergence_signal = result['divergence_signal']
            recent_change = result['recent_change']
            correlation_strength = result['correlation_strength']
            
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = pearson_stability
            
            # Divergence signals (correlation breakdown)
            if divergence_signal == "breaking" and abs(current_pearson) > 0.3:
                if current_pearson > 0:  # Positive correlation breaking down
                    signal_type = SignalType.SELL if recent_change < -0.2 else SignalType.WARNING
                else:  # Negative correlation breaking down
                    signal_type = SignalType.BUY if recent_change > 0.2 else SignalType.WARNING
                strength = min(abs(recent_change) * 3, 1.0)
            
            # Strong correlation signals
            elif correlation_strength in ["strong", "very_strong"] and pearson_stability > 0.7:
                if current_pearson > 0.7:  # Strong positive correlation
                    signal_type = SignalType.BUY
                    strength = current_pearson * pearson_stability
                elif current_pearson < -0.7:  # Strong negative correlation
                    signal_type = SignalType.SELL
                    strength = abs(current_pearson) * pearson_stability
            
            # Correlation reversal signals
            elif abs(recent_change) > 0.3 and pearson_stability > 0.5:
                if recent_change > 0.3:  # Correlation strengthening positively
                    signal_type = SignalType.BUY
                elif recent_change < -0.3:  # Correlation weakening or turning negative
                    signal_type = SignalType.SELL
                strength = min(abs(recent_change) * 2, 1.0)
            
            # Minimum signal threshold
            if signal_type != SignalType.NEUTRAL and strength > 0.3 and confidence > 0.4:
                current_price = float(data[-1]['close'])
                price_change_factor = abs(current_pearson) * 0.02  # 2% max based on correlation
                
                if signal_type in [SignalType.BUY]:
                    take_profit = current_price * (1 + price_change_factor)
                    stop_loss = current_price * (1 - price_change_factor * 0.5)
                else:
                    take_profit = current_price * (1 - price_change_factor)
                    stop_loss = current_price * (1 + price_change_factor * 0.5)
                
                return IndicatorSignal(
                    timestamp=datetime.fromisoformat(data[-1]['timestamp']),
                    indicator_name='CorrelationCoefficient',
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    price_target=take_profit,
                    stop_loss=stop_loss,
                    metadata={
                        'current_pearson': current_pearson,
                        'correlation_strength': correlation_strength,
                        'divergence_signal': divergence_signal,
                        'recent_change': recent_change,
                        'stability': pearson_stability
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None