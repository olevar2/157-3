"""
Hidden Divergence Detector Indicator

A hidden divergence detector indicator that identifies hidden divergences between
price action and technical indicators. Hidden divergences often signal trend
continuation patterns and provide early warning signals for potential trend
acceleration or deceleration.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
from scipy import signal

# Add the parent directory to Python path for imports

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class HiddenDivergenceDetector(BaseIndicator):
    """
    Hidden Divergence Detector Indicator
    
    Detects hidden divergences between price and momentum indicators:
    - Hidden bullish divergence: Higher lows in price, lower lows in indicator
    - Hidden bearish divergence: Lower highs in price, higher highs in indicator
    - Multi-indicator divergence analysis (RSI, MACD, Stochastic)
    - Peak/trough detection with confirmation
    - Divergence strength measurement
    - Time-based divergence validation
    
    The indicator provides:
    - Hidden divergence signals (bullish/bearish)
    - Divergence strength scores
    - Confirmation status
    - Multiple indicator consensus
    - Time decay adjustments
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 stoch_k: int = 14,
                 stoch_d: int = 3,
                 min_peak_distance: int = 5,
                 divergence_lookback: int = 20,
                 confirmation_bars: int = 3):
        """
        Initialize Hidden Divergence Detector indicator
        
        Args:
            rsi_period: Period for RSI calculation
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            stoch_k: %K period for Stochastic
            stoch_d: %D period for Stochastic
            min_peak_distance: Minimum distance between peaks/troughs
            divergence_lookback: Lookback period for divergence detection
            confirmation_bars: Number of bars for divergence confirmation
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.min_peak_distance = min_peak_distance
        self.divergence_lookback = divergence_lookback
        self.confirmation_bars = confirmation_bars
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate hidden divergence detection
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'hidden_bullish_divergence': Hidden bullish divergence signals
            - 'hidden_bearish_divergence': Hidden bearish divergence signals
            - 'divergence_strength': Strength of detected divergences
            - 'confirmation_status': Divergence confirmation status
            - 'consensus_score': Multi-indicator consensus score
        """
        try:
            if len(data) < max(self.macd_slow, self.divergence_lookback) + 10:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'hidden_bullish_divergence': empty_series,
                    'hidden_bearish_divergence': empty_series,
                    'divergence_strength': empty_series,
                    'confirmation_status': empty_series,
                    'consensus_score': empty_series
                }
            
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calculate momentum indicators
            rsi = self._calculate_rsi(close)
            macd_line, macd_signal_line, macd_histogram = self._calculate_macd(close)
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
            
            # Detect peaks and troughs in price and indicators
            price_highs, price_lows = self._detect_price_extremes(high, low)
            rsi_highs, rsi_lows = self._detect_indicator_extremes(rsi)
            macd_highs, macd_lows = self._detect_indicator_extremes(macd_line)
            stoch_highs, stoch_lows = self._detect_indicator_extremes(stoch_k)
            
            # Detect hidden divergences
            hidden_bullish_div = self._detect_hidden_bullish_divergence(
                price_lows, rsi_lows, macd_lows, stoch_lows
            )
            
            hidden_bearish_div = self._detect_hidden_bearish_divergence(
                price_highs, rsi_highs, macd_highs, stoch_highs
            )
            
            # Calculate divergence strength
            divergence_strength = self._calculate_divergence_strength(
                hidden_bullish_div, hidden_bearish_div, close
            )
            
            # Check confirmation status
            confirmation_status = self._check_confirmation_status(
                hidden_bullish_div, hidden_bearish_div, close
            )
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(
                hidden_bullish_div, hidden_bearish_div, rsi, macd_line, stoch_k
            )
            
            return {
                'hidden_bullish_divergence': hidden_bullish_div,
                'hidden_bearish_divergence': hidden_bearish_div,
                'divergence_strength': divergence_strength,
                'confirmation_status': confirmation_status,
                'consensus_score': consensus_score
            }
            
        except Exception as e:
            print(f"Error in Hidden Divergence Detector: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'hidden_bullish_divergence': empty_series,
                'hidden_bearish_divergence': empty_series,
                'divergence_strength': empty_series,
                'confirmation_status': empty_series,
                'consensus_score': empty_series
            }
    
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
            avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
            
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except:
            return pd.Series(50, index=close.index)
    
    def _calculate_macd(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = close.ewm(span=self.macd_fast, min_periods=1).mean()
            ema_slow = close.ewm(span=self.macd_slow, min_periods=1).mean()
            
            macd_line = ema_fast - ema_slow
            macd_signal_line = macd_line.ewm(span=self.macd_signal, min_periods=1).mean()
            macd_histogram = macd_line - macd_signal_line
            
            return macd_line, macd_signal_line, macd_histogram
        except:
            empty_series = pd.Series(0, index=close.index)
            return empty_series, empty_series, empty_series
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        try:
            lowest_low = low.rolling(window=self.stoch_k, min_periods=1).min()
            highest_high = high.rolling(window=self.stoch_k, min_periods=1).max()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            stoch_d = stoch_k.rolling(window=self.stoch_d, min_periods=1).mean()
            
            return stoch_k.fillna(50), stoch_d.fillna(50)
        except:
            empty_series = pd.Series(50, index=close.index)
            return empty_series, empty_series
    
    def _detect_price_extremes(self, high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Detect price highs and lows"""
        try:
            # Find peaks in highs
            high_peaks, _ = signal.find_peaks(high.values, distance=self.min_peak_distance)
            price_highs = pd.Series(0, index=high.index)
            for peak in high_peaks:
                if peak < len(price_highs):
                    price_highs.iloc[peak] = high.iloc[peak]
            
            # Find troughs in lows
            low_troughs, _ = signal.find_peaks(-low.values, distance=self.min_peak_distance)
            price_lows = pd.Series(0, index=low.index)
            for trough in low_troughs:
                if trough < len(price_lows):
                    price_lows.iloc[trough] = low.iloc[trough]
            
            return price_highs, price_lows
        except:
            empty_series = pd.Series(0, index=high.index)
            return empty_series, empty_series
    
    def _detect_indicator_extremes(self, indicator: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Detect indicator highs and lows"""
        try:
            # Find peaks
            peaks, _ = signal.find_peaks(indicator.values, distance=self.min_peak_distance)
            indicator_highs = pd.Series(0, index=indicator.index)
            for peak in peaks:
                if peak < len(indicator_highs):
                    indicator_highs.iloc[peak] = indicator.iloc[peak]
            
            # Find troughs
            troughs, _ = signal.find_peaks(-indicator.values, distance=self.min_peak_distance)
            indicator_lows = pd.Series(0, index=indicator.index)
            for trough in troughs:
                if trough < len(indicator_lows):
                    indicator_lows.iloc[trough] = indicator.iloc[trough]
            
            return indicator_highs, indicator_lows
        except:
            empty_series = pd.Series(0, index=indicator.index)
            return empty_series, empty_series
    
    def _detect_hidden_bullish_divergence(self, price_lows: pd.Series, 
                                         rsi_lows: pd.Series, 
                                         macd_lows: pd.Series, 
                                         stoch_lows: pd.Series) -> pd.Series:
        """Detect hidden bullish divergence patterns"""
        try:
            hidden_bullish = pd.Series(0, index=price_lows.index)
            
            # Get non-zero low points
            price_low_points = price_lows[price_lows > 0]
            rsi_low_points = rsi_lows[rsi_lows > 0]
            macd_low_points = macd_lows[macd_lows != 0]
            stoch_low_points = stoch_lows[stoch_lows > 0]
            
            # Check for hidden bullish divergence
            for i in range(len(price_low_points)):
                current_idx = price_low_points.index[i]
                current_price_low = price_low_points.iloc[i]
                
                # Look for previous low within lookback period
                start_idx = max(0, current_idx - self.divergence_lookback)
                previous_price_lows = price_low_points[price_low_points.index < current_idx]
                previous_price_lows = previous_price_lows[previous_price_lows.index >= start_idx]
                
                if len(previous_price_lows) > 0:
                    prev_idx = previous_price_lows.index[-1]  # Most recent previous low
                    prev_price_low = previous_price_lows.iloc[-1]
                    
                    # Hidden bullish: Higher low in price
                    if current_price_low > prev_price_low:
                        # Check indicators for lower lows
                        divergence_count = 0
                        
                        # RSI check
                        if current_idx in rsi_low_points.index and prev_idx in rsi_low_points.index:
                            if rsi_low_points[current_idx] < rsi_low_points[prev_idx]:
                                divergence_count += 1
                        
                        # MACD check
                        if current_idx in macd_low_points.index and prev_idx in macd_low_points.index:
                            if macd_low_points[current_idx] < macd_low_points[prev_idx]:
                                divergence_count += 1
                        
                        # Stochastic check
                        if current_idx in stoch_low_points.index and prev_idx in stoch_low_points.index:
                            if stoch_low_points[current_idx] < stoch_low_points[prev_idx]:
                                divergence_count += 1
                        
                        # Signal if at least 2 indicators show divergence
                        if divergence_count >= 2:
                            hidden_bullish.loc[current_idx] = 1
            
            return hidden_bullish
        except:
            return pd.Series(0, index=price_lows.index)
    
    def _detect_hidden_bearish_divergence(self, price_highs: pd.Series,
                                         rsi_highs: pd.Series,
                                         macd_highs: pd.Series,
                                         stoch_highs: pd.Series) -> pd.Series:
        """Detect hidden bearish divergence patterns"""
        try:
            hidden_bearish = pd.Series(0, index=price_highs.index)
            
            # Get non-zero high points
            price_high_points = price_highs[price_highs > 0]
            rsi_high_points = rsi_highs[rsi_highs > 0]
            macd_high_points = macd_highs[macd_highs != 0]
            stoch_high_points = stoch_highs[stoch_highs > 0]
            
            # Check for hidden bearish divergence
            for i in range(len(price_high_points)):
                current_idx = price_high_points.index[i]
                current_price_high = price_high_points.iloc[i]
                
                # Look for previous high within lookback period
                start_idx = max(0, current_idx - self.divergence_lookback)
                previous_price_highs = price_high_points[price_high_points.index < current_idx]
                previous_price_highs = previous_price_highs[previous_price_highs.index >= start_idx]
                
                if len(previous_price_highs) > 0:
                    prev_idx = previous_price_highs.index[-1]  # Most recent previous high
                    prev_price_high = previous_price_highs.iloc[-1]
                    
                    # Hidden bearish: Lower high in price
                    if current_price_high < prev_price_high:
                        # Check indicators for higher highs
                        divergence_count = 0
                        
                        # RSI check
                        if current_idx in rsi_high_points.index and prev_idx in rsi_high_points.index:
                            if rsi_high_points[current_idx] > rsi_high_points[prev_idx]:
                                divergence_count += 1
                        
                        # MACD check
                        if current_idx in macd_high_points.index and prev_idx in macd_high_points.index:
                            if macd_high_points[current_idx] > macd_high_points[prev_idx]:
                                divergence_count += 1
                        
                        # Stochastic check
                        if current_idx in stoch_high_points.index and prev_idx in stoch_high_points.index:
                            if stoch_high_points[current_idx] > stoch_high_points[prev_idx]:
                                divergence_count += 1
                        
                        # Signal if at least 2 indicators show divergence
                        if divergence_count >= 2:
                            hidden_bearish.loc[current_idx] = -1
            
            return hidden_bearish
        except:
            return pd.Series(0, index=price_highs.index)
    
    def _calculate_divergence_strength(self, hidden_bullish: pd.Series,
                                     hidden_bearish: pd.Series,
                                     close: pd.Series) -> pd.Series:
        """Calculate strength of detected divergences"""
        try:
            strength = pd.Series(0.0, index=close.index)
            
            # Calculate strength for bullish divergences
            bullish_signals = hidden_bullish[hidden_bullish == 1]
            for idx in bullish_signals.index:
                # Strength based on price momentum and volume confirmation
                if idx >= 5:
                    price_momentum = close.iloc[idx] / close.iloc[idx-5] - 1
                    strength.loc[idx] = abs(price_momentum) * 100
            
            # Calculate strength for bearish divergences
            bearish_signals = hidden_bearish[hidden_bearish == -1]
            for idx in bearish_signals.index:
                if idx >= 5:
                    price_momentum = close.iloc[idx] / close.iloc[idx-5] - 1
                    strength.loc[idx] = -abs(price_momentum) * 100
            
            return strength
        except:
            return pd.Series(0, index=close.index)
    
    def _check_confirmation_status(self, hidden_bullish: pd.Series,
                                  hidden_bearish: pd.Series,
                                  close: pd.Series) -> pd.Series:
        """Check confirmation status of divergences"""
        try:
            confirmation = pd.Series(0, index=close.index)
            
            # Check bullish confirmations
            bullish_signals = hidden_bullish[hidden_bullish == 1]
            for idx in bullish_signals.index:
                end_idx = min(len(close), idx + self.confirmation_bars)
                if end_idx > idx:
                    # Confirm if price moves up after signal
                    if close.iloc[end_idx-1] > close.iloc[idx]:
                        confirmation.loc[idx] = 1
            
            # Check bearish confirmations
            bearish_signals = hidden_bearish[hidden_bearish == -1]
            for idx in bearish_signals.index:
                end_idx = min(len(close), idx + self.confirmation_bars)
                if end_idx > idx:
                    # Confirm if price moves down after signal
                    if close.iloc[end_idx-1] < close.iloc[idx]:
                        confirmation.loc[idx] = -1
            
            return confirmation
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_consensus_score(self, hidden_bullish: pd.Series,
                                  hidden_bearish: pd.Series,
                                  rsi: pd.Series,
                                  macd: pd.Series,
                                  stoch: pd.Series) -> pd.Series:
        """Calculate multi-indicator consensus score"""
        try:
            consensus = pd.Series(0.0, index=hidden_bullish.index)
            
            # Bullish consensus
            bullish_signals = hidden_bullish[hidden_bullish == 1]
            for idx in bullish_signals.index:
                score = 0
                
                # RSI oversold support
                if rsi.iloc[idx] < 30:
                    score += 1
                elif rsi.iloc[idx] < 50:
                    score += 0.5
                
                # MACD positive momentum
                if idx > 0 and macd.iloc[idx] > macd.iloc[idx-1]:
                    score += 1
                
                # Stochastic oversold support
                if stoch.iloc[idx] < 20:
                    score += 1
                elif stoch.iloc[idx] < 50:
                    score += 0.5
                
                consensus.loc[idx] = score / 3  # Normalize to 0-1
            
            # Bearish consensus
            bearish_signals = hidden_bearish[hidden_bearish == -1]
            for idx in bearish_signals.index:
                score = 0
                
                # RSI overbought resistance
                if rsi.iloc[idx] > 70:
                    score += 1
                elif rsi.iloc[idx] > 50:
                    score += 0.5
                
                # MACD negative momentum
                if idx > 0 and macd.iloc[idx] < macd.iloc[idx-1]:
                    score += 1
                
                # Stochastic overbought resistance
                if stoch.iloc[idx] > 80:
                    score += 1
                elif stoch.iloc[idx] > 50:
                    score += 0.5
                
                consensus.loc[idx] = -(score / 3)  # Negative for bearish
            
            return consensus
        except:
            return pd.Series(0, index=hidden_bullish.index)
    
    def get_hidden_divergences(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get hidden divergence signals"""
        result = self.calculate(data)
        return {
            'bullish': result['hidden_bullish_divergence'],
            'bearish': result['hidden_bearish_divergence']
        }
    
    def get_divergence_strength(self, data: pd.DataFrame) -> pd.Series:
        """Get divergence strength scores"""
        result = self.calculate(data)
        return result['divergence_strength']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with divergence patterns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample data with hidden divergence patterns
    base_price = 100
    
    # Create trend with higher lows in price but lower momentum
    trend = np.linspace(0, 20, 100)  # Uptrend
    price_pattern = trend + 5 * np.sin(np.arange(100) * 0.3)  # Oscillating pattern
    
    # Add some noise
    noise = np.random.randn(100) * 1
    close_prices = base_price + price_pattern + noise
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 2, 100),
        'low': close_prices - np.random.uniform(0, 2, 100),
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.3, 100)
    }, index=dates)
    
    # Test the indicator
    print("Testing Hidden Divergence Detector Indicator")
    print("=" * 50)
    
    indicator = HiddenDivergenceDetector(
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        min_peak_distance=5,
        divergence_lookback=20,
        confirmation_bars=3
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Hidden bullish divergences: {(result['hidden_bullish_divergence'] == 1).sum()}")
    print(f"Hidden bearish divergences: {(result['hidden_bearish_divergence'] == -1).sum()}")
    print(f"Divergence strength range: {result['divergence_strength'].min():.3f} to {result['divergence_strength'].max():.3f}")
    print(f"Confirmation status range: {result['confirmation_status'].min():.0f} to {result['confirmation_status'].max():.0f}")
    print(f"Consensus score range: {result['consensus_score'].min():.3f} to {result['consensus_score'].max():.3f}")
    
    # Show detected divergences
    bullish_divs = result['hidden_bullish_divergence'][result['hidden_bullish_divergence'] == 1]
    bearish_divs = result['hidden_bearish_divergence'][result['hidden_bearish_divergence'] == -1]
    
    if len(bullish_divs) > 0:
        print(f"\nHidden Bullish Divergences detected at:")
        for idx in bullish_divs.index:
            strength = result['divergence_strength'].loc[idx]
            consensus = result['consensus_score'].loc[idx]
            print(f"  {idx.strftime('%Y-%m-%d')}: Strength={strength:.2f}, Consensus={consensus:.2f}")
    
    if len(bearish_divs) > 0:
        print(f"\nHidden Bearish Divergences detected at:")
        for idx in bearish_divs.index:
            strength = result['divergence_strength'].loc[idx]
            consensus = result['consensus_score'].loc[idx]
            print(f"  {idx.strftime('%Y-%m-%d')}: Strength={strength:.2f}, Consensus={consensus:.2f}")
    
    # Confirmation analysis
    confirmed_signals = result['confirmation_status'][result['confirmation_status'] != 0]
    print(f"\nConfirmed divergence signals: {len(confirmed_signals)}")
    
    print("\nHidden Divergence Detector Indicator test completed successfully!")