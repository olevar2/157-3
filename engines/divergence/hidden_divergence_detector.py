"""
Platform3 Hidden Divergence Detector
Advanced detection of hidden (continuation) divergences for trend confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import ta

class HiddenDivergenceDetector:
    """
    Advanced hidden divergence detector for trend continuation signals.
    Hidden divergences indicate trend strength rather than reversal.
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 stoch_k: int = 14,
                 stoch_d: int = 3,
                 momentum_period: int = 10,
                 lookback_period: int = 80,
                 min_bars_between_points: int = 8,
                 peak_prominence: float = 0.4,
                 trend_strength_threshold: float = 0.6,
                 divergence_threshold: float = 0.65,
                 trend_lookback: int = 50):
        """
        Initialize Hidden Divergence Detector
        
        Parameters:
        -----------
        rsi_period : int
            RSI calculation period
        macd_fast : int
            MACD fast EMA period
        macd_slow : int
            MACD slow EMA period
        macd_signal : int
            MACD signal line period
        stoch_k : int
            Stochastic %K period
        stoch_d : int
            Stochastic %D period
        momentum_period : int
            Momentum period
        lookback_period : int
            Lookback period for divergence detection
        min_bars_between_points : int
            Minimum bars between divergence points
        peak_prominence : float
            Minimum prominence for peak detection
        trend_strength_threshold : float
            Minimum trend strength for hidden divergence validity
        divergence_threshold : float
            Correlation threshold for divergence confirmation
        trend_lookback : int
            Period for trend strength calculation
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.momentum_period = momentum_period
        self.lookback_period = lookback_period
        self.min_bars_between_points = min_bars_between_points
        self.peak_prominence = peak_prominence
        self.trend_strength_threshold = trend_strength_threshold
        self.divergence_threshold = divergence_threshold
        self.trend_lookback = trend_lookback
        
        # Oscillator configurations for hidden divergence analysis
        self.oscillator_configs = {
            'rsi': {'overbought': 70, 'oversold': 30, 'weight': 0.3},
            'macd': {'weight': 0.35},
            'stochastic': {'overbought': 80, 'oversold': 20, 'weight': 0.2},
            'momentum': {'weight': 0.15}
        }
    
    def calculate_oscillators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum oscillators for hidden divergence analysis"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        oscillators = {}
        
        # RSI
        oscillators['rsi'] = ta.momentum.RSIIndicator(
            close=close, window=self.rsi_period
        ).rsi()
        
        # MACD
        macd_indicator = ta.trend.MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        oscillators['macd'] = macd_indicator.macd()
        oscillators['macd_histogram'] = macd_indicator.macd_diff()
        
        # Stochastic
        stoch_indicator = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close,
            window=self.stoch_k,
            smooth_window=self.stoch_d
        )
        oscillators['stochastic'] = stoch_indicator.stoch()
        
        # Momentum
        oscillators['momentum'] = ta.momentum.ROCIndicator(
            close=close, window=self.momentum_period
        ).roc()
        
        return oscillators
    
    def detect_trend_direction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current trend direction and strength
        Hidden divergences only valid in trending markets
        """
        close = data['close'].tail(self.trend_lookback)
        
        # Calculate multiple trend indicators
        ema_short = close.ewm(span=20).mean()
        ema_long = close.ewm(span=50).mean()
        
        # Linear regression trend
        x = np.arange(len(close))
        slope, intercept, r_value, _, _ = stats.linregress(x, close)
        
        # ADX for trend strength
        try:
            adx = ta.trend.ADXIndicator(
                high=data['high'].tail(self.trend_lookback),
                low=data['low'].tail(self.trend_lookback),
                close=close,
                window=14
            ).adx().iloc[-1]
        except:
            adx = 25  # Default moderate trend strength
        
        # Determine trend direction
        ema_trend = 1 if ema_short.iloc[-1] > ema_long.iloc[-1] else -1
        slope_trend = 1 if slope > 0 else -1
        
        # Calculate trend strength (0-1)
        trend_strength = min(adx / 50, 1.0)  # Normalize ADX
        r_squared_strength = abs(r_value)  # R-squared from linear regression
        
        overall_strength = (trend_strength + r_squared_strength) / 2
        
        # Consensus trend direction
        trend_consensus = ema_trend * slope_trend
        trend_direction = 'uptrend' if trend_consensus > 0 else 'downtrend' if trend_consensus < 0 else 'sideways'
        
        return {
            'direction': trend_direction,
            'strength': overall_strength,
            'adx': adx,
            'slope': slope,
            'r_squared': r_value ** 2,
            'ema_trend': ema_trend,
            'slope_trend': slope_trend,
            'is_trending': overall_strength >= self.trend_strength_threshold
        }
    
    def detect_peaks_troughs(self, series: pd.Series, inverted: bool = False) -> np.ndarray:
        """Detect peaks or troughs in a series"""
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return np.array([])
        
        # Smooth the series
        try:
            window_length = min(11, len(clean_series) // 3 * 2 + 1)
            if window_length < 3:
                window_length = 3
            smoothed = savgol_filter(clean_series, window_length=window_length, polyorder=2)
        except:
            smoothed = clean_series.values
        
        if inverted:
            smoothed = -smoothed
        
        # Find peaks
        peaks, _ = find_peaks(smoothed, 
                             distance=self.min_bars_between_points,
                             prominence=self.peak_prominence * np.std(smoothed))
        
        return peaks
    
    def detect_hidden_bullish_divergence(self, 
                                       price_series: pd.Series, 
                                       oscillator_series: pd.Series,
                                       trend_info: Dict) -> Dict[str, Any]:
        """
        Detect hidden bullish divergence (trend continuation in uptrend)
        Price: higher low, Oscillator: lower low
        """
        if trend_info['direction'] != 'uptrend' or not trend_info['is_trending']:
            return {'divergences': [], 'count': 0, 'max_strength': 0.0}
        
        # Find price troughs and oscillator troughs
        price_troughs = self.detect_peaks_troughs(price_series, inverted=True)
        osc_troughs = self.detect_peaks_troughs(oscillator_series, inverted=True)
        
        divergences = []
        
        if len(price_troughs) >= 2 and len(osc_troughs) >= 2:
            # Focus on recent troughs
            recent_price_troughs = price_troughs[-4:] if len(price_troughs) >= 4 else price_troughs
            recent_osc_troughs = osc_troughs[-4:] if len(osc_troughs) >= 4 else osc_troughs
            
            for i in range(1, len(recent_price_troughs)):
                price_idx1, price_idx2 = recent_price_troughs[i-1], recent_price_troughs[i]
                
                # Find corresponding oscillator troughs within time window
                for j in range(len(recent_osc_troughs)-1):
                    osc_idx1, osc_idx2 = recent_osc_troughs[j], recent_osc_troughs[j+1]
                    
                    # Check time alignment (allow some flexibility)
                    if (abs(price_idx1 - osc_idx1) <= 6 and abs(price_idx2 - osc_idx2) <= 6 and
                        price_idx2 > price_idx1 and osc_idx2 > osc_idx1):
                        
                        price_val1, price_val2 = price_series.iloc[price_idx1], price_series.iloc[price_idx2]
                        osc_val1, osc_val2 = oscillator_series.iloc[osc_idx1], oscillator_series.iloc[osc_idx2]
                        
                        # Hidden bullish divergence: higher price low, lower oscillator low
                        if price_val2 > price_val1 and osc_val2 < osc_val1:
                            # Calculate divergence strength
                            price_change = (price_val2 - price_val1) / price_val1
                            osc_change = (osc_val2 - osc_val1) / abs(osc_val1 + 1e-8)
                            
                            # Strength based on magnitude of opposing movements
                            strength = min(abs(price_change) + abs(osc_change), 1.0)
                            
                            # Additional confirmation: time between points
                            time_factor = min((price_idx2 - price_idx1) / self.min_bars_between_points, 2.0) / 2.0
                            strength *= time_factor
                            
                            if strength >= self.divergence_threshold:
                                divergences.append({
                                    'type': 'hidden_bullish',
                                    'price_points': (price_idx1, price_idx2),
                                    'oscillator_points': (osc_idx1, osc_idx2),
                                    'strength': strength,
                                    'price_values': (price_val1, price_val2),
                                    'oscillator_values': (osc_val1, osc_val2),
                                    'price_change': price_change,
                                    'oscillator_change': osc_change
                                })
        
        return {
            'divergences': divergences,
            'count': len(divergences),
            'max_strength': max([d['strength'] for d in divergences]) if divergences else 0.0
        }
    
    def detect_hidden_bearish_divergence(self, 
                                       price_series: pd.Series, 
                                       oscillator_series: pd.Series,
                                       trend_info: Dict) -> Dict[str, Any]:
        """
        Detect hidden bearish divergence (trend continuation in downtrend)
        Price: lower high, Oscillator: higher high
        """
        if trend_info['direction'] != 'downtrend' or not trend_info['is_trending']:
            return {'divergences': [], 'count': 0, 'max_strength': 0.0}
        
        # Find price peaks and oscillator peaks
        price_peaks = self.detect_peaks_troughs(price_series, inverted=False)
        osc_peaks = self.detect_peaks_troughs(oscillator_series, inverted=False)
        
        divergences = []
        
        if len(price_peaks) >= 2 and len(osc_peaks) >= 2:
            # Focus on recent peaks
            recent_price_peaks = price_peaks[-4:] if len(price_peaks) >= 4 else price_peaks
            recent_osc_peaks = osc_peaks[-4:] if len(osc_peaks) >= 4 else osc_peaks
            
            for i in range(1, len(recent_price_peaks)):
                price_idx1, price_idx2 = recent_price_peaks[i-1], recent_price_peaks[i]
                
                # Find corresponding oscillator peaks within time window
                for j in range(len(recent_osc_peaks)-1):
                    osc_idx1, osc_idx2 = recent_osc_peaks[j], recent_osc_peaks[j+1]
                    
                    # Check time alignment
                    if (abs(price_idx1 - osc_idx1) <= 6 and abs(price_idx2 - osc_idx2) <= 6 and
                        price_idx2 > price_idx1 and osc_idx2 > osc_idx1):
                        
                        price_val1, price_val2 = price_series.iloc[price_idx1], price_series.iloc[price_idx2]
                        osc_val1, osc_val2 = oscillator_series.iloc[osc_idx1], oscillator_series.iloc[osc_idx2]
                        
                        # Hidden bearish divergence: lower price high, higher oscillator high
                        if price_val2 < price_val1 and osc_val2 > osc_val1:
                            # Calculate divergence strength
                            price_change = (price_val1 - price_val2) / price_val1
                            osc_change = (osc_val2 - osc_val1) / abs(osc_val1 + 1e-8)
                            
                            # Strength based on magnitude of opposing movements
                            strength = min(abs(price_change) + abs(osc_change), 1.0)
                            
                            # Additional confirmation: time between points
                            time_factor = min((price_idx2 - price_idx1) / self.min_bars_between_points, 2.0) / 2.0
                            strength *= time_factor
                            
                            if strength >= self.divergence_threshold:
                                divergences.append({
                                    'type': 'hidden_bearish',
                                    'price_points': (price_idx1, price_idx2),
                                    'oscillator_points': (osc_idx1, osc_idx2),
                                    'strength': strength,
                                    'price_values': (price_val1, price_val2),
                                    'oscillator_values': (osc_val1, osc_val2),
                                    'price_change': price_change,
                                    'oscillator_change': osc_change
                                })
        
        return {
            'divergences': divergences,
            'count': len(divergences),
            'max_strength': max([d['strength'] for d in divergences]) if divergences else 0.0
        }
    
    def calculate_consensus_score(self, 
                                oscillator_results: Dict[str, Dict],
                                trend_info: Dict) -> Dict[str, float]:
        """Calculate consensus hidden divergence score"""
        hidden_bullish_score = 0.0
        hidden_bearish_score = 0.0
        
        for oscillator, config in self.oscillator_configs.items():
            if oscillator in oscillator_results:
                weight = config['weight']
                
                bullish_strength = oscillator_results[oscillator]['hidden_bullish']['max_strength']
                bearish_strength = oscillator_results[oscillator]['hidden_bearish']['max_strength']
                
                # Apply trend bias - hidden divergences should align with trend
                trend_bias = trend_info['strength']
                
                if trend_info['direction'] == 'uptrend':
                    hidden_bullish_score += bullish_strength * weight * trend_bias
                elif trend_info['direction'] == 'downtrend':
                    hidden_bearish_score += bearish_strength * weight * trend_bias
        
        return {
            'hidden_bullish_consensus': hidden_bullish_score,
            'hidden_bearish_consensus': hidden_bearish_score,
            'net_consensus': hidden_bullish_score - hidden_bearish_score,
            'trend_alignment_score': trend_info['strength']
        }
    
    def detect_hidden_divergences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive hidden divergence detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            
        Returns:
        --------
        Dict containing hidden divergence analysis results
        """
        if len(data) < max(self.lookback_period, 60):
            return {
                'error': 'Insufficient data',
                'required_length': max(self.lookback_period, 60),
                'actual_length': len(data)
            }
        
        # Use recent data for analysis
        recent_data = data.tail(self.lookback_period)
        
        # Detect trend first (essential for hidden divergences)
        trend_info = self.detect_trend_direction(recent_data)
        
        # Calculate oscillators
        oscillators = self.calculate_oscillators(recent_data)
        
        # Price series for comparison
        price_series = recent_data['close']
        
        # Analyze hidden divergences for each oscillator
        oscillator_results = {}
        
        for osc_name, osc_series in oscillators.items():
            if osc_name in ['macd_histogram']:  # Skip derivative indicators
                continue
            
            if osc_series.dropna().empty:
                continue
            
            hidden_bullish = self.detect_hidden_bullish_divergence(
                price_series, osc_series, trend_info
            )
            hidden_bearish = self.detect_hidden_bearish_divergence(
                price_series, osc_series, trend_info
            )
            
            oscillator_results[osc_name] = {
                'hidden_bullish': hidden_bullish,
                'hidden_bearish': hidden_bearish,
                'total_strength': hidden_bullish['max_strength'] + hidden_bearish['max_strength']
            }
        
        # Calculate consensus scores
        consensus = self.calculate_consensus_score(oscillator_results, trend_info)
        
        # Determine overall signal
        signal_strength = abs(consensus['net_consensus'])
        
        if consensus['net_consensus'] > 0.3 and trend_info['direction'] == 'uptrend':
            overall_signal = 'HIDDEN_BULLISH_DIVERGENCE'
        elif consensus['net_consensus'] < -0.3 and trend_info['direction'] == 'downtrend':
            overall_signal = 'HIDDEN_BEARISH_DIVERGENCE'
        elif not trend_info['is_trending']:
            overall_signal = 'NO_TREND'
        else:
            overall_signal = 'NO_HIDDEN_DIVERGENCE'
        
        # Calculate confidence (higher when trend is strong and divergences align)
        confidence = min(signal_strength * trend_info['strength'] * 100, 100)
        
        return {
            'timestamp': data.index[-1],
            'overall_signal': overall_signal,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'trend_info': trend_info,
            'consensus_scores': consensus,
            'oscillator_results': oscillator_results,
            'oscillator_values': {
                name: series.iloc[-1] if not series.dropna().empty else None 
                for name, series in oscillators.items()
            },
            'summary': {
                'total_hidden_bullish': sum(
                    result['hidden_bullish']['count'] for result in oscillator_results.values()
                ),
                'total_hidden_bearish': sum(
                    result['hidden_bearish']['count'] for result in oscillator_results.values()
                ),
                'strongest_oscillator': max(
                    oscillator_results.keys(),
                    key=lambda x: oscillator_results[x]['total_strength']
                ) if oscillator_results else None,
                'trend_alignment': trend_info['is_trending'] and signal_strength > 0.3
            }
        }
    
    def get_hidden_divergence_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate hidden divergence signals as a DataFrame
        
        Returns:
        --------
        pd.DataFrame with hidden divergence signals and trend information
        """
        results = []
        
        # Calculate signals for each row where possible
        min_required = max(self.lookback_period // 2, 40)
        
        for i in range(min_required, len(data)):
            subset_data = data.iloc[:i+1]
            detection_result = self.detect_hidden_divergences(subset_data)
            
            if 'error' not in detection_result:
                results.append({
                    'timestamp': subset_data.index[-1],
                    'signal': detection_result['overall_signal'],
                    'strength': detection_result['signal_strength'],
                    'confidence': detection_result['confidence'],
                    'trend_direction': detection_result['trend_info']['direction'],
                    'trend_strength': detection_result['trend_info']['strength'],
                    'hidden_bullish_consensus': detection_result['consensus_scores']['hidden_bullish_consensus'],
                    'hidden_bearish_consensus': detection_result['consensus_scores']['hidden_bearish_consensus'],
                    'trend_alignment_score': detection_result['consensus_scores']['trend_alignment_score']
                })
            else:
                results.append({
                    'timestamp': subset_data.index[-1],
                    'signal': 'INSUFFICIENT_DATA',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'trend_direction': 'unknown',
                    'trend_strength': 0.0,
                    'hidden_bullish_consensus': 0.0,
                    'hidden_bearish_consensus': 0.0,
                    'trend_alignment_score': 0.0
                })
        
        return pd.DataFrame(results).set_index('timestamp')

# Example usage and testing
if __name__ == "__main__":
    # Create sample data with clear trend
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    
    # Generate trending price data
    trend = np.cumsum(np.random.randn(150) * 0.01) + 2  # Upward trend
    noise = np.random.randn(150) * 0.3
    price_base = 100 + trend + noise
    
    sample_data = pd.DataFrame({
        'open': price_base * 0.999,
        'high': price_base * 1.002,
        'low': price_base * 0.998,
        'close': price_base,
        'volume': np.random.randint(1000, 10000, 150)
    }, index=dates)
    
    # Test the hidden divergence detector
    detector = HiddenDivergenceDetector()
    
    print("Testing Hidden Divergence Detector...")
    print("=" * 50)
    
    # Test detection
    result = detector.detect_hidden_divergences(sample_data)
    print(f"Overall Signal: {result['overall_signal']}")
    print(f"Signal Strength: {result['signal_strength']:.3f}")
    print(f"Confidence: {result['confidence']:.1f}%")
    
    trend_info = result['trend_info']
    print(f"\nTrend Analysis:")
    print(f"Direction: {trend_info['direction']}")
    print(f"Strength: {trend_info['strength']:.3f}")
    print(f"Is Trending: {trend_info['is_trending']}")
    print(f"ADX: {trend_info['adx']:.1f}")
    
    consensus = result['consensus_scores']
    print(f"\nConsensus Scores:")
    print(f"Hidden Bullish: {consensus['hidden_bullish_consensus']:.3f}")
    print(f"Hidden Bearish: {consensus['hidden_bearish_consensus']:.3f}")
    print(f"Net Consensus: {consensus['net_consensus']:.3f}")
    print(f"Trend Alignment: {consensus['trend_alignment_score']:.3f}")
    
    summary = result['summary']
    print(f"\nSummary:")
    print(f"Hidden Bullish Divergences: {summary['total_hidden_bullish']}")
    print(f"Hidden Bearish Divergences: {summary['total_hidden_bearish']}")
    print(f"Strongest Oscillator: {summary['strongest_oscillator']}")
    print(f"Trend Alignment: {summary['trend_alignment']}")
    
    # Test signal generation
    signals_df = detector.get_hidden_divergence_signals(sample_data)
    print(f"\nGenerated {len(signals_df)} hidden divergence signals")
    print(f"Recent signals:\n{signals_df.tail()}")
