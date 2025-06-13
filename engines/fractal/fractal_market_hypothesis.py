"""
Fractal Market Hypothesis Indicator
===================================

The Fractal Market Hypothesis (FMH) indicator analyzes market efficiency 
and liquidity across different time horizons using fractal analysis.
Based on Edgar Peters' Fractal Market Analysis theory.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
        from indicator_base import IndicatorBase


class FractalMarketHypothesis(IndicatorBase):
    """
    Fractal Market Hypothesis (FMH) indicator.
    
    Analyzes market structure based on:
    - Liquidity across time horizons
    - Information flow efficiency
    - Market stability vs chaos
    - Investment horizons diversity
    """
    
    def __init__(self, 
                 short_period: int = 5,
                 medium_period: int = 20,
                 long_period: int = 60,
                 hurst_window: int = 50):
        """
        Initialize Fractal Market Hypothesis indicator.
        
        Args:
            short_period: Short-term analysis period
            medium_period: Medium-term analysis period  
            long_period: Long-term analysis period
            hurst_window: Window for Hurst exponent calculation
        """
        super().__init__()
        
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.hurst_window = hurst_window
        
        # Validation
        if any(p <= 0 for p in [short_period, medium_period, long_period, hurst_window]):
            raise ValueError("All periods must be positive")
        if short_period >= medium_period or medium_period >= long_period:
            raise ValueError("Periods must be in ascending order")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Fractal Market Hypothesis metrics.
        
        Args:
            data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            Dictionary containing FMH analysis and signals
        """
        try:
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_columns)
            
            if len(data) < self.long_period:
                raise ValueError(f"Insufficient data: need at least {self.long_period} periods")
            
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate FMH components
            fmh_metrics = self._calculate_fmh_metrics(close, volume, high, low)
            
            # Generate signals
            signals = self._generate_signals(fmh_metrics)
            
            # Calculate additional metrics
            metrics = self._calculate_additional_metrics(fmh_metrics)
            
            return {
                'fmh_metrics': fmh_metrics,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(fmh_metrics, signals[-1] if len(signals) > 0 else 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Market Hypothesis: {e}")
            raise
    
    def _calculate_fmh_metrics(self, close: np.ndarray, volume: np.ndarray, 
                              high: np.ndarray, low: np.ndarray) -> Dict:
        """Calculate core Fractal Market Hypothesis metrics."""
        
        # 1. Liquidity Concentration Index
        liquidity_concentration = self._calculate_liquidity_concentration(volume)
        
        # 2. Information Efficiency Ratio
        info_efficiency = self._calculate_information_efficiency(close)
        
        # 3. Market Stability Index
        market_stability = self._calculate_market_stability(close, high, low)
        
        # 4. Horizon Diversity Index
        horizon_diversity = self._calculate_horizon_diversity(close)
        
        # 5. Fractal Dimension across time horizons
        fractal_dimensions = self._calculate_multi_horizon_fractal_dim(close)
        
        # 6. Hurst Exponent Evolution
        hurst_evolution = self._calculate_hurst_evolution(close)
        
        return {
            'liquidity_concentration': liquidity_concentration,
            'information_efficiency': info_efficiency,
            'market_stability': market_stability,
            'horizon_diversity': horizon_diversity,
            'fractal_dimensions': fractal_dimensions,
            'hurst_evolution': hurst_evolution
        }
    
    def _calculate_liquidity_concentration(self, volume: np.ndarray) -> np.ndarray:
        """Calculate liquidity concentration across time horizons."""
        concentration = np.full(len(volume), np.nan)
        
        for i in range(self.long_period - 1, len(volume)):
            # Volume distributions across different periods
            short_vol = volume[i - self.short_period + 1:i + 1]
            medium_vol = volume[i - self.medium_period + 1:i + 1]
            long_vol = volume[i - self.long_period + 1:i + 1]
            
            # Calculate concentration ratios
            short_concentration = np.std(short_vol) / (np.mean(short_vol) + 1e-8)
            medium_concentration = np.std(medium_vol) / (np.mean(medium_vol) + 1e-8)
            long_concentration = np.std(long_vol) / (np.mean(long_vol) + 1e-8)
            
            # Weighted concentration index
            concentration[i] = (
                0.5 * short_concentration + 
                0.3 * medium_concentration + 
                0.2 * long_concentration
            )
        
        return concentration
    
    def _calculate_information_efficiency(self, close: np.ndarray) -> np.ndarray:
        """Calculate information flow efficiency."""
        efficiency = np.full(len(close), np.nan)
        
        for i in range(self.medium_period - 1, len(close)):
            # Price returns at different horizons
            prices = close[i - self.medium_period + 1:i + 1]
            returns = np.diff(np.log(prices))
            
            # Serial correlation test for efficiency
            if len(returns) > 1:
                # Lag-1 autocorrelation
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0
                
                # Information efficiency (lower autocorr = higher efficiency)
                efficiency[i] = 1 - abs(autocorr)
            
        return efficiency
    
    def _calculate_market_stability(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate market stability index."""
        stability = np.full(len(close), np.nan)
        
        for i in range(self.medium_period - 1, len(close)):
            # Price volatility
            prices = close[i - self.medium_period + 1:i + 1]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns)
            
            # Range volatility
            highs = high[i - self.medium_period + 1:i + 1]
            lows = low[i - self.medium_period + 1:i + 1]
            closes = close[i - self.medium_period + 1:i + 1]
            
            true_ranges = np.maximum(
                highs - lows,
                np.maximum(
                    np.abs(highs - np.roll(closes, 1)),
                    np.abs(lows - np.roll(closes, 1))
                )
            )[1:]  # Remove first element due to roll
            
            avg_true_range = np.mean(true_ranges)
            range_volatility = avg_true_range / (np.mean(closes) + 1e-8)
            
            # Stability index (inverse of combined volatility)
            combined_volatility = (volatility + range_volatility) / 2
            stability[i] = 1 / (1 + combined_volatility)
        
        return stability
    
    def _calculate_horizon_diversity(self, close: np.ndarray) -> np.ndarray:
        """Calculate investment horizon diversity."""
        diversity = np.full(len(close), np.nan)
        
        for i in range(self.long_period - 1, len(close)):
            # Returns at different horizons
            short_returns = np.diff(np.log(close[i - self.short_period:i + 1]))
            medium_returns = np.diff(np.log(close[i - self.medium_period:i + 1:self.short_period]))
            long_returns = np.diff(np.log(close[i - self.long_period:i + 1:self.medium_period]))
            
            # Correlation between different horizon returns
            correlations = []
            
            if len(short_returns) > 1 and len(medium_returns) > 1:
                # Align arrays for correlation
                min_len = min(len(short_returns), len(medium_returns))
                if min_len > 1:
                    corr = np.corrcoef(short_returns[-min_len:], medium_returns[-min_len:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if len(medium_returns) > 1 and len(long_returns) > 1:
                min_len = min(len(medium_returns), len(long_returns))
                if min_len > 1:
                    corr = np.corrcoef(medium_returns[-min_len:], long_returns[-min_len:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            # Diversity index (lower correlation = higher diversity)
            if correlations:
                diversity[i] = 1 - np.mean(correlations)
            
        return diversity
    
    def _calculate_multi_horizon_fractal_dim(self, close: np.ndarray) -> Dict:
        """Calculate fractal dimensions across multiple time horizons."""
        dims = {
            'short': np.full(len(close), np.nan),
            'medium': np.full(len(close), np.nan),
            'long': np.full(len(close), np.nan)
        }
        
        for i in range(self.long_period - 1, len(close)):
            # Short horizon fractal dimension
            short_prices = close[i - self.short_period + 1:i + 1]
            dims['short'][i] = self._calculate_fractal_dimension(short_prices)
            
            # Medium horizon fractal dimension
            medium_prices = close[i - self.medium_period + 1:i + 1]
            dims['medium'][i] = self._calculate_fractal_dimension(medium_prices)
            
            # Long horizon fractal dimension
            long_prices = close[i - self.long_period + 1:i + 1]
            dims['long'][i] = self._calculate_fractal_dimension(long_prices)
        
        return dims
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using Higuchi method."""
        if len(prices) < 4:
            return 1.0
        
        # Higuchi fractal dimension
        k_max = min(8, len(prices) // 4)
        lk = []
        
        for k in range(1, k_max + 1):
            l_mk = 0
            for m in range(k):
                ll = 0
                n = (len(prices) - m - 1) // k
                if n > 0:
                    for i in range(1, n + 1):
                        ll += abs(prices[m + i * k] - prices[m + (i - 1) * k])
                    ll = ll * (len(prices) - 1) / (n * k)
                    l_mk += ll
            
            if k > 0:
                lk.append(l_mk / k)
            else:
                lk.append(0)
        
        # Calculate fractal dimension
        if len(lk) > 1:
            k_values = np.arange(1, len(lk) + 1)
            lk = np.array(lk)
            lk = lk[lk > 0]  # Remove zeros
            k_values = k_values[:len(lk)]
            
            if len(lk) > 1:
                # Linear regression in log-log space
                log_k = np.log(k_values)
                log_lk = np.log(lk)
                slope = np.polyfit(log_k, log_lk, 1)[0]
                return 2 - slope
        
        return 1.5  # Default fractal dimension
    
    def _calculate_hurst_evolution(self, close: np.ndarray) -> np.ndarray:
        """Calculate evolving Hurst exponent."""
        hurst = np.full(len(close), np.nan)
        
        for i in range(self.hurst_window - 1, len(close)):
            prices = close[i - self.hurst_window + 1:i + 1]
            hurst[i] = self._calculate_hurst_exponent(prices)
        
        return hurst
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        if len(prices) < 4:
            return 0.5
        
        # Calculate log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        if len(returns) < 2:
            return 0.5
        
        # R/S analysis
        n = len(returns)
        mean_return = np.mean(returns)
        
        # Cumulative deviations from mean
        cum_dev = np.cumsum(returns - mean_return)
        
        # Range of cumulative deviations
        R = np.max(cum_dev) - np.min(cum_dev)
        
        # Standard deviation
        S = np.std(returns)
        
        if S > 0:
            rs = R / S
            if rs > 0:
                # Hurst exponent
                return np.log(rs) / np.log(n)
        
        return 0.5
    
    def _generate_signals(self, fmh_metrics: Dict) -> np.ndarray:
        """Generate trading signals based on FMH analysis."""
        # Use liquidity concentration as primary signal source
        liquidity = fmh_metrics['liquidity_concentration']
        efficiency = fmh_metrics['information_efficiency']
        stability = fmh_metrics['market_stability']
        
        signals = np.zeros(len(liquidity))
        
        for i in range(1, len(signals)):
            if np.isnan(liquidity[i]) or np.isnan(efficiency[i]) or np.isnan(stability[i]):
                continue
            
            # High efficiency + high stability + moderate liquidity = Buy
            if (efficiency[i] > 0.7 and stability[i] > 0.6 and 
                0.3 < liquidity[i] < 0.8):
                signals[i] = 1
            
            # Low efficiency + low stability + high liquidity concentration = Sell
            elif (efficiency[i] < 0.4 and stability[i] < 0.4 and 
                  liquidity[i] > 0.8):
                signals[i] = -1
            
            # Moderate conditions = weak signals
            elif efficiency[i] > 0.6 and stability[i] > 0.5:
                signals[i] = 0.5
            elif efficiency[i] < 0.5 and stability[i] < 0.5:
                signals[i] = -0.5
        
        return signals
    
    def _calculate_additional_metrics(self, fmh_metrics: Dict) -> Dict:
        """Calculate additional FMH metrics."""
        liquidity = fmh_metrics['liquidity_concentration']
        efficiency = fmh_metrics['information_efficiency']
        stability = fmh_metrics['market_stability']
        diversity = fmh_metrics['horizon_diversity']
        
        valid_data = ~(np.isnan(liquidity) | np.isnan(efficiency) | 
                      np.isnan(stability) | np.isnan(diversity))
        
        if not np.any(valid_data):
            return {}
        
        valid_liquidity = liquidity[valid_data]
        valid_efficiency = efficiency[valid_data]
        valid_stability = stability[valid_data]
        valid_diversity = diversity[valid_data]
        
        return {
            'avg_liquidity_concentration': np.mean(valid_liquidity),
            'avg_information_efficiency': np.mean(valid_efficiency),
            'avg_market_stability': np.mean(valid_stability),
            'avg_horizon_diversity': np.mean(valid_diversity),
            'market_regime': self._classify_market_regime(
                valid_liquidity[-1] if len(valid_liquidity) > 0 else 0.5,
                valid_efficiency[-1] if len(valid_efficiency) > 0 else 0.5,
                valid_stability[-1] if len(valid_stability) > 0 else 0.5
            ),
            'fractal_efficiency': np.mean(valid_efficiency) * np.mean(valid_stability),
            'liquidity_volatility': np.std(valid_liquidity),
            'efficiency_trend': np.mean(np.diff(valid_efficiency[-10:])) if len(valid_efficiency) > 10 else 0
        }
    
    def _classify_market_regime(self, liquidity: float, efficiency: float, stability: float) -> str:
        """Classify current market regime based on FMH metrics."""
        if efficiency > 0.7 and stability > 0.6:
            return "EFFICIENT_STABLE"
        elif efficiency > 0.6 and stability < 0.4:
            return "EFFICIENT_VOLATILE"
        elif efficiency < 0.4 and stability > 0.6:
            return "INEFFICIENT_STABLE"
        elif efficiency < 0.4 and stability < 0.4:
            return "CHAOTIC"
        elif liquidity > 0.8:
            return "ILLIQUID"
        else:
            return "TRANSITIONAL"
    
    def _interpret_signals(self, fmh_metrics: Dict, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        liquidity = fmh_metrics['liquidity_concentration']
        efficiency = fmh_metrics['information_efficiency']
        stability = fmh_metrics['market_stability']
        
        if np.isnan(liquidity[-1]) or np.isnan(efficiency[-1]) or np.isnan(stability[-1]):
            return "Insufficient data for FMH analysis"
        
        current_liquidity = liquidity[-1]
        current_efficiency = efficiency[-1]
        current_stability = stability[-1]
        
        regime = self._classify_market_regime(current_liquidity, current_efficiency, current_stability)
        
        signal_desc = {
            1: "BUY signal (Favorable FMH conditions)",
            0.5: "Weak BUY signal (Moderate FMH conditions)",
            -0.5: "Weak SELL signal (Deteriorating FMH conditions)",
            -1: "SELL signal (Unfavorable FMH conditions)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return (f"FMH Analysis - Regime: {regime} | "
                f"Efficiency: {current_efficiency:.3f} | "
                f"Stability: {current_stability:.3f} | "
                f"Liquidity: {current_liquidity:.3f} - {signal_desc}")


def create_fractal_market_hypothesis(short_period: int = 5, **kwargs) -> FractalMarketHypothesis:
    """Factory function to create Fractal Market Hypothesis indicator."""
    return FractalMarketHypothesis(short_period=short_period, **kwargs)
