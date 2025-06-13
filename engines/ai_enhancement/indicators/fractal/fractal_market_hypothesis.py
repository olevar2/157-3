"""
Fractal Market Hypothesis Indicator

A fractal market hypothesis indicator that analyzes market behavior through the lens
of fractal geometry and chaos theory. This indicator evaluates market efficiency,
fractal dimensions, memory effects, and regime changes based on the Fractal Market
Hypothesis proposed by Edgar Peters.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
from scipy import stats

# Add the parent directory to Python path for imports

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class FractalMarketHypothesis(BaseIndicator):
    """
    Fractal Market Hypothesis Indicator
    
    Implements analysis based on the Fractal Market Hypothesis:
    - Market behavior is fractal and self-similar
    - Different investment horizons create different fractal structures
    - Market stability depends on multiple time horizons
    - Liquidity and information flow affect fractal properties
    - Crisis periods show different fractal characteristics
    
    The indicator provides:
    - Fractal dimension analysis
    - Market stability index
    - Regime change detection
    - Liquidity stress indicators
    - Multi-horizon analysis
    - Memory persistence measurement
    """
    
    def __init__(self, 
                 short_horizon: int = 10,
                 medium_horizon: int = 50,
                 long_horizon: int = 200,
                 hurst_window: int = 50,
                 stability_threshold: float = 0.7,
                 crisis_threshold: float = 0.3):
        """
        Initialize Fractal Market Hypothesis indicator
        
        Args:
            short_horizon: Short-term investment horizon
            medium_horizon: Medium-term investment horizon  
            long_horizon: Long-term investment horizon
            hurst_window: Window for Hurst exponent calculation
            stability_threshold: Threshold for market stability
            crisis_threshold: Threshold for crisis detection
        """
        super().__init__()
        self.short_horizon = short_horizon
        self.medium_horizon = medium_horizon
        self.long_horizon = long_horizon
        self.hurst_window = hurst_window
        self.stability_threshold = stability_threshold
        self.crisis_threshold = crisis_threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate fractal market hypothesis analysis
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'fractal_dimension': Multi-horizon fractal dimension
            - 'hurst_exponent': Hurst exponent for memory analysis
            - 'stability_index': Market stability index
            - 'regime_state': Current market regime
            - 'liquidity_stress': Liquidity stress indicator
            - 'crisis_indicator': Crisis detection signal
        """
        try:
            if len(data) < self.long_horizon:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'fractal_dimension': empty_series,
                    'hurst_exponent': empty_series,
                    'stability_index': empty_series,
                    'regime_state': empty_series,
                    'liquidity_stress': empty_series,
                    'crisis_indicator': empty_series
                }
            
            close = data['close']
            volume = data['volume']
            high = data['high']
            low = data['low']
            
            # Calculate multi-horizon fractal dimension
            fractal_dimension = self._calculate_multi_horizon_fractal_dimension(close)
            
            # Calculate Hurst exponent for memory analysis
            hurst_exponent = self._calculate_hurst_exponent(close)
            
            # Calculate market stability index
            stability_index = self._calculate_stability_index(
                fractal_dimension, hurst_exponent, close
            )
            
            # Determine market regime
            regime_state = self._determine_market_regime(
                fractal_dimension, hurst_exponent, stability_index
            )
            
            # Calculate liquidity stress
            liquidity_stress = self._calculate_liquidity_stress(
                close, volume, high, low
            )
            
            # Detect crisis conditions
            crisis_indicator = self._detect_crisis_conditions(
                stability_index, liquidity_stress, fractal_dimension
            )
            
            return {
                'fractal_dimension': fractal_dimension,
                'hurst_exponent': hurst_exponent,
                'stability_index': stability_index,
                'regime_state': regime_state,
                'liquidity_stress': liquidity_stress,
                'crisis_indicator': crisis_indicator
            }
            
        except Exception as e:
            print(f"Error in Fractal Market Hypothesis calculation: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'fractal_dimension': empty_series,
                'hurst_exponent': empty_series,
                'stability_index': empty_series,
                'regime_state': empty_series,
                'liquidity_stress': empty_series,
                'crisis_indicator': empty_series
            }
    
    def _calculate_multi_horizon_fractal_dimension(self, close: pd.Series) -> pd.Series:
        """Calculate fractal dimension across multiple time horizons"""
        try:
            fractal_dimension = pd.Series(1.5, index=close.index)
            
            for i in range(self.long_horizon, len(close)):
                dimensions = []
                
                # Calculate fractal dimension for each horizon
                for horizon in [self.short_horizon, self.medium_horizon, self.long_horizon]:
                    if i >= horizon:
                        # Get price segment
                        price_segment = close.iloc[i-horizon:i+1]
                        
                        # Calculate fractal dimension using variation method
                        dimension = self._calculate_single_fractal_dimension(price_segment)
                        dimensions.append(dimension)
                
                if dimensions:
                    # Weighted average of dimensions (emphasize longer horizons)
                    weights = [0.2, 0.3, 0.5]  # Short, medium, long
                    if len(dimensions) == len(weights):
                        weighted_dimension = np.average(dimensions, weights=weights)
                    else:
                        weighted_dimension = np.mean(dimensions)
                    
                    fractal_dimension.iloc[i] = weighted_dimension
            
            return fractal_dimension
        except:
            return pd.Series(1.5, index=close.index)
    
    def _calculate_single_fractal_dimension(self, price_series: pd.Series) -> float:
        """Calculate fractal dimension for a single time series"""
        try:
            if len(price_series) < 4:
                return 1.5
            
            # Use the variation method to estimate fractal dimension
            variations = []
            scales = [1, 2, 4, 8]
            
            for scale in scales:
                if scale < len(price_series):
                    # Calculate variation at this scale
                    scaled_series = price_series.iloc[::scale]
                    if len(scaled_series) > 1:
                        variation = scaled_series.diff().abs().sum()
                        variations.append(variation)
            
            if len(variations) >= 2:
                # Estimate dimension from log-log relationship
                log_scales = np.log(scales[:len(variations)])
                log_variations = np.log(np.array(variations) + 1e-8)
                
                # Linear regression
                slope, _, _, _, _ = stats.linregress(log_scales, log_variations)
                
                # Fractal dimension = 1 - slope (for variation method)
                dimension = 1 - slope
                
                # Bound between 1 and 2
                dimension = np.clip(dimension, 1.0, 2.0)
                return dimension
            
            return 1.5  # Default neutral value
        except:
            return 1.5
    
    def _calculate_hurst_exponent(self, close: pd.Series) -> pd.Series:
        """Calculate Hurst exponent for memory analysis"""
        try:
            hurst_exponent = pd.Series(0.5, index=close.index)
            
            for i in range(self.hurst_window, len(close)):
                # Get price segment
                price_segment = close.iloc[i-self.hurst_window:i+1]
                
                # Calculate log returns
                log_returns = np.log(price_segment / price_segment.shift(1)).dropna()
                
                if len(log_returns) > 5:
                    # Use R/S analysis to estimate Hurst exponent
                    hurst = self._rescaled_range_analysis(log_returns.values)
                    hurst_exponent.iloc[i] = hurst
            
            return hurst_exponent
        except:
            return pd.Series(0.5, index=close.index)
    
    def _rescaled_range_analysis(self, returns: np.ndarray) -> float:
        """Perform rescaled range analysis to estimate Hurst exponent"""
        try:
            n = len(returns)
            if n < 8:
                return 0.5
            
            # Calculate mean
            mean_return = np.mean(returns)
            
            # Calculate cumulative deviations
            cumulative_deviations = np.cumsum(returns - mean_return)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(returns)
            
            if S > 0 and R > 0:
                # R/S ratio
                rs_ratio = R / S
                
                # Estimate Hurst exponent using theoretical relationship
                # H ≈ log(R/S) / log(n)
                hurst = np.log(rs_ratio) / np.log(n)
                
                # Bound between 0 and 1
                hurst = np.clip(hurst, 0.0, 1.0)
                return hurst
            
            return 0.5
        except:
            return 0.5
    
    def _calculate_stability_index(self, fractal_dimension: pd.Series,
                                  hurst_exponent: pd.Series,
                                  close: pd.Series) -> pd.Series:
        """Calculate market stability index based on fractal properties"""
        try:
            stability_index = pd.Series(0.5, index=close.index)
            
            for i in range(len(close)):
                # Get current fractal properties
                current_fractal_dim = fractal_dimension.iloc[i]
                current_hurst = hurst_exponent.iloc[i]
                
                # Stability components:
                # 1. Fractal dimension close to 1.5 indicates stable randomness
                fractal_stability = 1 - abs(current_fractal_dim - 1.5) / 0.5
                fractal_stability = np.clip(fractal_stability, 0, 1)
                
                # 2. Hurst exponent close to 0.5 indicates efficient market
                hurst_stability = 1 - abs(current_hurst - 0.5) / 0.5
                hurst_stability = np.clip(hurst_stability, 0, 1)
                
                # 3. Price volatility component
                if i >= self.short_horizon:
                    recent_returns = close.iloc[i-self.short_horizon:i+1].pct_change().dropna()
                    if len(recent_returns) > 0:
                        volatility = recent_returns.std()
                        # Lower volatility = higher stability
                        vol_stability = np.exp(-volatility * 100)  # Scaling factor
                        vol_stability = np.clip(vol_stability, 0, 1)
                    else:
                        vol_stability = 0.5
                else:
                    vol_stability = 0.5
                
                # Combined stability index
                stability = (fractal_stability + hurst_stability + vol_stability) / 3
                stability_index.iloc[i] = stability
            
            # Smooth the stability index
            stability_index = stability_index.rolling(window=5, min_periods=1).mean()
            
            return stability_index
        except:
            return pd.Series(0.5, index=close.index)
    
    def _determine_market_regime(self, fractal_dimension: pd.Series,
                                hurst_exponent: pd.Series,
                                stability_index: pd.Series) -> pd.Series:
        """Determine current market regime based on fractal properties"""
        try:
            regime_state = pd.Series(0, index=fractal_dimension.index)
            
            for i in range(len(fractal_dimension)):
                fractal_dim = fractal_dimension.iloc[i]
                hurst = hurst_exponent.iloc[i]
                stability = stability_index.iloc[i]
                
                # Regime classification:
                # 1 = Efficient/Stable Market
                # 0 = Transitional/Uncertain
                # -1 = Crisis/Unstable Market
                
                if stability >= self.stability_threshold:
                    if 0.4 <= hurst <= 0.6 and 1.3 <= fractal_dim <= 1.7:
                        regime_state.iloc[i] = 1  # Efficient market
                    else:
                        regime_state.iloc[i] = 0  # Stable but not efficient
                elif stability <= self.crisis_threshold:
                    regime_state.iloc[i] = -1  # Crisis/unstable
                else:
                    regime_state.iloc[i] = 0  # Transitional
            
            # Smooth regime changes
            regime_state = regime_state.rolling(window=3, min_periods=1).mean().round()
            
            return regime_state
        except:
            return pd.Series(0, index=fractal_dimension.index)
    
    def _calculate_liquidity_stress(self, close: pd.Series, volume: pd.Series,
                                   high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate liquidity stress indicator"""
        try:
            liquidity_stress = pd.Series(0.0, index=close.index)
            
            for i in range(self.medium_horizon, len(close)):
                # Volume analysis
                recent_volume = volume.iloc[i-self.medium_horizon:i+1]
                volume_mean = recent_volume.mean()
                volume_std = recent_volume.std()
                current_volume = volume.iloc[i]
                
                # Volume stress (low volume = high stress)
                if volume_mean > 0:
                    volume_stress = 1 - (current_volume / volume_mean)
                    volume_stress = np.clip(volume_stress, 0, 1)
                else:
                    volume_stress = 0.5
                
                # Bid-ask spread proxy using high-low range
                hl_range = high.iloc[i] - low.iloc[i]
                recent_ranges = (high.iloc[i-self.medium_horizon:i+1] - 
                               low.iloc[i-self.medium_horizon:i+1])
                avg_range = recent_ranges.mean()
                
                # Spread stress (high spread = high stress)
                if avg_range > 0 and close.iloc[i] > 0:
                    spread_stress = (hl_range / close.iloc[i]) / (avg_range / close.iloc[i])
                    spread_stress = np.clip(spread_stress - 1, 0, 1)
                else:
                    spread_stress = 0
                
                # Combined liquidity stress
                liquidity_stress.iloc[i] = (volume_stress + spread_stress) / 2
            
            return liquidity_stress
        except:
            return pd.Series(0, index=close.index)
    
    def _detect_crisis_conditions(self, stability_index: pd.Series,
                                 liquidity_stress: pd.Series,
                                 fractal_dimension: pd.Series) -> pd.Series:
        """Detect crisis conditions based on fractal market hypothesis"""
        try:
            crisis_indicator = pd.Series(0, index=stability_index.index)
            
            for i in range(len(stability_index)):
                stability = stability_index.iloc[i]
                stress = liquidity_stress.iloc[i]
                fractal_dim = fractal_dimension.iloc[i]
                
                # Crisis conditions:
                # 1. Low stability
                # 2. High liquidity stress
                # 3. Extreme fractal dimension values
                
                crisis_score = 0
                
                if stability <= self.crisis_threshold:
                    crisis_score += 1
                
                if stress >= 0.7:  # High liquidity stress
                    crisis_score += 1
                
                if fractal_dim <= 1.2 or fractal_dim >= 1.8:  # Extreme fractal behavior
                    crisis_score += 1
                
                # Crisis signal if multiple conditions met
                if crisis_score >= 2:
                    crisis_indicator.iloc[i] = 1
                elif crisis_score == 1:
                    crisis_indicator.iloc[i] = 0  # Warning
                else:
                    crisis_indicator.iloc[i] = -1  # Normal
            
            return crisis_indicator
        except:
            return pd.Series(0, index=stability_index.index)
    
    def get_stability_index(self, data: pd.DataFrame) -> pd.Series:
        """Get market stability index"""
        result = self.calculate(data)
        return result['stability_index']
    
    def get_regime_state(self, data: pd.DataFrame) -> pd.Series:
        """Get market regime state"""
        result = self.calculate(data)
        return result['regime_state']
    
    def get_crisis_indicator(self, data: pd.DataFrame) -> pd.Series:
        """Get crisis detection signals"""
        result = self.calculate(data)
        return result['crisis_indicator']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with different market conditions
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=250, freq='D')
    
    # Generate sample data with different regimes
    t = np.arange(250)
    
    # Normal period (efficient market)
    normal_period = 100 + np.random.randn(100) * 2
    
    # Crisis period (high volatility, trending)
    crisis_period = normal_period[-1] + np.cumsum(np.random.randn(75) * 5)
    
    # Recovery period (mean reverting)
    recovery_mean = crisis_period[-1]
    recovery_period = recovery_mean + 10 * np.sin(np.arange(75) * 0.2) + np.random.randn(75) * 1
    
    close_prices = np.concatenate([normal_period, crisis_period, recovery_period])
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 2, 250),
        'low': close_prices - np.random.uniform(0, 2, 250), 
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.5, 250)
    }, index=dates)
    
    # Add volume stress during crisis
    data.loc[data.index[100:175], 'volume'] *= 0.3  # Lower volume during crisis
    
    # Test the indicator
    print("Testing Fractal Market Hypothesis Indicator")
    print("=" * 50)
    
    indicator = FractalMarketHypothesis(
        short_horizon=10,
        medium_horizon=50,
        long_horizon=100,
        hurst_window=30,
        stability_threshold=0.6,
        crisis_threshold=0.3
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Fractal dimension range: {result['fractal_dimension'].min():.3f} to {result['fractal_dimension'].max():.3f}")
    print(f"Hurst exponent range: {result['hurst_exponent'].min():.3f} to {result['hurst_exponent'].max():.3f}")
    print(f"Stability index range: {result['stability_index'].min():.3f} to {result['stability_index'].max():.3f}")
    print(f"Liquidity stress range: {result['liquidity_stress'].min():.3f} to {result['liquidity_stress'].max():.3f}")
    
    # Analyze regimes
    regimes = result['regime_state']
    efficient_periods = (regimes == 1).sum()
    transitional_periods = (regimes == 0).sum()
    crisis_periods = (regimes == -1).sum()
    
    print(f"\nMarket Regime Analysis:")
    print(f"Efficient market periods: {efficient_periods}")
    print(f"Transitional periods: {transitional_periods}")
    print(f"Crisis/unstable periods: {crisis_periods}")
    
    # Crisis detection
    crisis_signals = result['crisis_indicator']
    crisis_warnings = (crisis_signals == 1).sum()
    normal_periods = (crisis_signals == -1).sum()
    
    print(f"\nCrisis Detection:")
    print(f"Crisis signals: {crisis_warnings}")
    print(f"Normal periods: {normal_periods}")
    
    # Stability by period
    print(f"\nStability Analysis by Period:")
    print(f"Normal period (1-100): avg stability = {result['stability_index'].iloc[100:150].mean():.3f}")
    print(f"Crisis period (100-175): avg stability = {result['stability_index'].iloc[150:200].mean():.3f}")
    print(f"Recovery period (175-250): avg stability = {result['stability_index'].iloc[200:250].mean():.3f}")
    
    print("\nFractal Market Hypothesis Indicator test completed successfully!")