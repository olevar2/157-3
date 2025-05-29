"""
Z-Score Calculations - Statistical Deviation Measurement
Essential for statistical arbitrage and mean reversion strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from ..indicator_base import IndicatorBase

class ZScoreCalculator(IndicatorBase):
    """
    Z-Score Statistical Deviation Measurement
    
    The Z-Score measures how many standard deviations a data point is
    from the mean. Essential for:
    - Statistical arbitrage strategies
    - Mean reversion identification
    - Outlier detection
    - Risk-adjusted trading signals
    
    Z-Score = (X - μ) / σ
    Where:
    - X = Current value
    - μ = Mean of the data
    - σ = Standard deviation
    """
    
    def __init__(self, 
                 period: int = 20,
                 smoothing_period: int = 5,
                 z_threshold: float = 2.0):
        """
        Initialize Z-Score Calculator
        
        Args:
            period: Lookback period for mean and std calculation
            smoothing_period: Period for smoothing the Z-score
            z_threshold: Threshold for extreme Z-score alerts
        """
        super().__init__()
        self.period = period
        self.smoothing_period = smoothing_period
        self.z_threshold = z_threshold
        
        # State tracking
        self.price_window = []
        self.z_scores = []
        self.signals = []
        
    def calculate(self, 
                 data: Union[pd.DataFrame, Dict],
                 price_column: str = 'close') -> Dict:
        """
        Calculate Z-Score with comprehensive analysis
        
        Args:
            data: Price data (DataFrame or dict)
            price_column: Column name for price data
            
        Returns:
            Dict containing Z-score analysis
        """
        try:
            # Extract price data
            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
                timestamps = data.index if hasattr(data, 'index') else range(len(prices))
            else:
                prices = data.get(price_column, [])
                timestamps = data.get('timestamp', range(len(prices)))
            
            if len(prices) < self.period:
                return self._empty_result()
            
            # Calculate rolling Z-scores
            z_scores = []
            z_score_signals = []
            price_means = []
            price_stds = []
            
            for i in range(len(prices)):
                if i >= self.period - 1:
                    # Get window data
                    window_prices = prices[max(0, i - self.period + 1):i + 1]
                    
                    # Calculate mean and standard deviation
                    mean_price = np.mean(window_prices)
                    std_price = np.std(window_prices, ddof=1)
                    
                    # Calculate Z-score
                    if std_price > 0:
                        z_score = (prices[i] - mean_price) / std_price
                    else:
                        z_score = 0.0
                    
                    z_scores.append(z_score)
                    price_means.append(mean_price)
                    price_stds.append(std_price)
                    
                    # Generate signals
                    signal = self._generate_z_score_signal(z_score)
                    z_score_signals.append(signal)
                else:
                    z_scores.append(0.0)
                    price_means.append(prices[i])
                    price_stds.append(0.0)
                    z_score_signals.append('NEUTRAL')
            
            # Smooth Z-scores
            smoothed_z_scores = self._smooth_z_scores(z_scores)
            
            # Calculate statistical metrics
            stats = self._calculate_statistics(z_scores, prices)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(smoothed_z_scores, z_scores)
            
            return {
                'z_scores': z_scores,
                'smoothed_z_scores': smoothed_z_scores,
                'price_means': price_means,
                'price_stds': price_stds,
                'signals': z_score_signals,
                'trading_signals': trading_signals,
                'statistics': stats,
                'threshold': self.z_threshold,
                'extreme_readings': self._find_extreme_readings(z_scores),
                'mean_reversion_signals': self._detect_mean_reversion(z_scores, smoothed_z_scores),
                'timestamp': timestamps[-1] if timestamps else None,
                'period': self.period,
                'indicator_name': 'Z-Score Calculator'
            }
            
        except Exception as e:
            return {'error': f"Z-Score calculation failed: {str(e)}"}
    
    def _generate_z_score_signal(self, z_score: float) -> str:
        """Generate signal based on Z-score value"""
        if z_score > self.z_threshold:
            return 'EXTREMELY_OVERBOUGHT'
        elif z_score > 1.0:
            return 'OVERBOUGHT'
        elif z_score < -self.z_threshold:
            return 'EXTREMELY_OVERSOLD'
        elif z_score < -1.0:
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'
    
    def _smooth_z_scores(self, z_scores: List[float]) -> List[float]:
        """Apply smoothing to Z-scores"""
        if len(z_scores) < self.smoothing_period:
            return z_scores.copy()
        
        smoothed = []
        for i in range(len(z_scores)):
            if i >= self.smoothing_period - 1:
                window = z_scores[max(0, i - self.smoothing_period + 1):i + 1]
                smoothed.append(np.mean(window))
            else:
                smoothed.append(z_scores[i])
        
        return smoothed
    
    def _calculate_statistics(self, z_scores: List[float], prices: List[float]) -> Dict:
        """Calculate comprehensive statistics"""
        valid_z_scores = [z for z in z_scores if z != 0.0]
        
        if not valid_z_scores:
            return {}
        
        return {
            'mean_z_score': np.mean(valid_z_scores),
            'std_z_score': np.std(valid_z_scores),
            'max_z_score': max(valid_z_scores),
            'min_z_score': min(valid_z_scores),
            'current_z_score': z_scores[-1] if z_scores else 0.0,
            'extreme_count': len([z for z in valid_z_scores if abs(z) > self.z_threshold]),
            'overbought_count': len([z for z in valid_z_scores if z > 1.0]),
            'oversold_count': len([z for z in valid_z_scores if z < -1.0]),
            'normality_test': self._test_normality(valid_z_scores),
            'current_percentile': self._calculate_percentile(z_scores[-1], valid_z_scores) if z_scores else 0
        }
    
    def _generate_trading_signals(self, smoothed_z_scores: List[float], raw_z_scores: List[float]) -> Dict:
        """Generate comprehensive trading signals"""
        if len(smoothed_z_scores) < 2:
            return {'action': 'HOLD', 'strength': 0, 'confidence': 0}
        
        current_smooth = smoothed_z_scores[-1]
        current_raw = raw_z_scores[-1]
        prev_smooth = smoothed_z_scores[-2]
        
        # Signal generation logic
        action = 'HOLD'
        strength = 0
        confidence = 0
        
        # Mean reversion signals
        if current_raw > self.z_threshold and current_smooth < prev_smooth:
            action = 'SELL'
            strength = min(100, int(abs(current_raw) * 30))
            confidence = min(95, int(abs(current_raw - self.z_threshold) * 40))
        elif current_raw < -self.z_threshold and current_smooth > prev_smooth:
            action = 'BUY'
            strength = min(100, int(abs(current_raw) * 30))
            confidence = min(95, int(abs(current_raw + self.z_threshold) * 40))
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'signal_type': 'MEAN_REVERSION',
            'z_score_level': abs(current_raw),
            'smoothed_direction': 'UP' if current_smooth > prev_smooth else 'DOWN'
        }
    
    def _find_extreme_readings(self, z_scores: List[float]) -> Dict:
        """Find extreme Z-score readings"""
        extreme_readings = []
        
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > self.z_threshold:
                extreme_readings.append({
                    'index': i,
                    'z_score': z_score,
                    'severity': abs(z_score) / self.z_threshold,
                    'type': 'OVERBOUGHT' if z_score > 0 else 'OVERSOLD'
                })
        
        return {
            'count': len(extreme_readings),
            'readings': extreme_readings[-10:],  # Last 10 extreme readings
            'recent_extremes': len([r for r in extreme_readings if len(z_scores) - r['index'] <= 10])
        }
    
    def _detect_mean_reversion(self, raw_z_scores: List[float], smoothed_z_scores: List[float]) -> Dict:
        """Detect mean reversion opportunities"""
        if len(smoothed_z_scores) < 3:
            return {}
        
        # Look for divergence between raw and smoothed Z-scores
        current_raw = raw_z_scores[-1]
        current_smooth = smoothed_z_scores[-1]
        prev_smooth = smoothed_z_scores[-2]
        
        reversion_signals = []
        
        # High Z-score with decreasing smoothed Z-score = potential reversion
        if current_raw > 1.5 and current_smooth < prev_smooth:
            reversion_signals.append({
                'type': 'BEARISH_REVERSION',
                'probability': min(90, int((current_raw - 1.5) * 50)),
                'z_score': current_raw
            })
        
        # Low Z-score with increasing smoothed Z-score = potential reversion
        elif current_raw < -1.5 and current_smooth > prev_smooth:
            reversion_signals.append({
                'type': 'BULLISH_REVERSION',
                'probability': min(90, int((abs(current_raw) - 1.5) * 50)),
                'z_score': current_raw
            })
        
        return {
            'signals': reversion_signals,
            'reversion_probability': reversion_signals[0]['probability'] if reversion_signals else 0
        }
    
    def _test_normality(self, z_scores: List[float]) -> Dict:
        """Test if Z-scores follow normal distribution"""
        if len(z_scores) < 10:
            return {'sufficient_data': False}
        
        # Simple normality tests
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
        
        # Check if mean is close to 0 and std is close to 1
        normal_mean = abs(mean_z) < 0.2
        normal_std = 0.8 < std_z < 1.2
        
        return {
            'sufficient_data': True,
            'normal_mean': normal_mean,
            'normal_std': normal_std,
            'likely_normal': normal_mean and normal_std,
            'actual_mean': mean_z,
            'actual_std': std_z
        }
    
    def _calculate_percentile(self, current_z: float, all_z_scores: List[float]) -> float:
        """Calculate percentile rank of current Z-score"""
        if not all_z_scores:
            return 50.0
        
        return (sum(1 for z in all_z_scores if z <= current_z) / len(all_z_scores)) * 100
    
    def _empty_result(self) -> Dict:
        """Return empty result when insufficient data"""
        return {
            'z_scores': [],
            'smoothed_z_scores': [],
            'signals': [],
            'statistics': {},
            'indicator_name': 'Z-Score Calculator',
            'error': 'Insufficient data for calculation'
        }

def calculate_z_score(data: Union[pd.DataFrame, Dict], 
                     period: int = 20,
                     smoothing_period: int = 5,
                     z_threshold: float = 2.0,
                     price_column: str = 'close') -> Dict:
    """
    Convenience function for Z-Score calculation
    
    Args:
        data: Price data
        period: Lookback period for statistics
        smoothing_period: Smoothing period for Z-scores
        z_threshold: Threshold for extreme readings
        price_column: Price column name
        
    Returns:
        Z-Score analysis results
    """
    calculator = ZScoreCalculator(period, smoothing_period, z_threshold)
    return calculator.calculate(data, price_column)
