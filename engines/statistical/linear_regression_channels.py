"""
Linear Regression Channels - Trend-based Statistical Channels
Advanced trend strength validation using regression analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from ..indicator_base import IndicatorBase

class LinearRegressionChannels(IndicatorBase):
    """
    Linear Regression Channels - Trend-based Statistical Channels
    
    Creates price channels based on linear regression with statistical deviation bands.
    Provides superior trend analysis by combining:
    - Linear regression centerline for trend direction
    - Standard error bands for price channel boundaries
    - R-squared analysis for trend strength validation
    - Slope analysis for trend momentum
    
    Components:
    - Regression Line: Best-fit line through price data
    - Upper/Lower Channels: Regression line ± (Standard Error * multiplier)
    - Trend Strength: R-squared coefficient
    - Trend Direction: Regression slope
    """
    
    def __init__(self, 
                 period: int = 20,
                 std_error_multipliers: List[float] = [1.0, 2.0],
                 min_r_squared: float = 0.5,
                 trend_sensitivity: float = 0.001):
        """
        Initialize Linear Regression Channels
        
        Args:
            period: Lookback period for regression calculation
            std_error_multipliers: Multipliers for standard error bands
            min_r_squared: Minimum R-squared for strong trend classification
            trend_sensitivity: Minimum slope for trend detection
        """
        super().__init__()
        self.period = period
        self.std_error_multipliers = sorted(std_error_multipliers)
        self.min_r_squared = min_r_squared
        self.trend_sensitivity = trend_sensitivity
        
        # State tracking
        self.price_history = []
        self.regression_lines = []
        self.r_squared_values = []
        self.slopes = []
        
    def calculate(self, 
                 data: Union[pd.DataFrame, Dict],
                 price_column: str = 'close') -> Dict:
        """
        Calculate Linear Regression Channels with comprehensive analysis
        
        Args:
            data: Price data (DataFrame or dict)
            price_column: Column name for price data
            
        Returns:
            Dict containing regression channel analysis
        """
        try:
            # Extract price data
            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
                timestamps = data.index if hasattr(data, 'index') else range(len(prices))
                highs = data.get('high', prices).values if 'high' in data.columns else prices
                lows = data.get('low', prices).values if 'low' in data.columns else prices
            else:
                prices = data.get(price_column, [])
                timestamps = data.get('timestamp', range(len(prices)))
                highs = data.get('high', prices)
                lows = data.get('low', prices)
            
            if len(prices) < self.period:
                return self._empty_result()
            
            # Calculate rolling regression analysis
            regression_data = self._calculate_rolling_regression(prices)
            
            # Calculate regression channels
            channels = self._calculate_regression_channels(regression_data)
            
            # Analyze trend strength and quality
            trend_analysis = self._analyze_trend_strength(regression_data)
            
            # Generate trading signals
            signals = self._generate_signals(prices, highs, lows, channels, regression_data)
            
            # Detect breakouts and reversals
            breakout_analysis = self._analyze_breakouts(prices, highs, lows, channels)
            
            # Calculate performance statistics
            statistics = self._calculate_statistics(prices, channels, regression_data)
            
            return {
                'regression_lines': [rd['regression_line'] for rd in regression_data],
                'slopes': [rd['slope'] for rd in regression_data],
                'r_squared': [rd['r_squared'] for rd in regression_data],
                'standard_errors': [rd['standard_error'] for rd in regression_data],
                'channels': channels,
                'trend_analysis': trend_analysis,
                'signals': signals,
                'breakouts': breakout_analysis,
                'statistics': statistics,
                'current_position': self._calculate_position(prices[-1], channels[-1]) if channels else {},
                'forecast': self._generate_forecast(regression_data[-1]) if regression_data else {},
                'timestamp': timestamps[-1] if timestamps else None,
                'period': self.period,
                'multipliers': self.std_error_multipliers,
                'indicator_name': 'Linear Regression Channels'
            }
            
        except Exception as e:
            return {'error': f"Linear Regression Channels calculation failed: {str(e)}"}
    
    def _calculate_rolling_regression(self, prices: List[float]) -> List[Dict]:
        """Calculate rolling linear regression analysis"""
        regression_data = []
        
        for i in range(len(prices)):
            if i >= self.period - 1:
                # Get window data
                window_prices = prices[max(0, i - self.period + 1):i + 1]
                x = np.arange(len(window_prices))
                
                # Calculate linear regression
                reg_analysis = self._calculate_regression(x, window_prices)
                
                # Current regression line value
                reg_analysis['regression_line'] = reg_analysis['slope'] * (len(window_prices) - 1) + reg_analysis['intercept']
                
                regression_data.append(reg_analysis)
            else:
                # Insufficient data
                regression_data.append({
                    'slope': 0.0,
                    'intercept': prices[i],
                    'r_squared': 0.0,
                    'standard_error': 0.0,
                    'regression_line': prices[i],
                    'trend_strength': 'INSUFFICIENT_DATA'
                })
        
        return regression_data
    
    def _calculate_regression(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Calculate linear regression with comprehensive statistics"""
        try:
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predicted values
            y_pred = slope * x + intercept
            
            # R-squared calculation
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Standard error of regression
            n = len(y)
            if n > 2:
                standard_error = np.sqrt(ss_res / (n - 2))
            else:
                standard_error = 0
            
            # Trend strength classification
            trend_strength = self._classify_trend_strength(r_squared, abs(slope))
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': max(0, min(1, r_squared)),  # Clamp between 0 and 1
                'standard_error': standard_error,
                'trend_strength': trend_strength,
                'predicted_values': y_pred.tolist()
            }
            
        except Exception as e:
            return {
                'slope': 0.0,
                'intercept': np.mean(y) if len(y) > 0 else 0.0,
                'r_squared': 0.0,
                'standard_error': 0.0,
                'trend_strength': 'ERROR',
                'predicted_values': y.tolist() if hasattr(y, 'tolist') else list(y)
            }
    
    def _classify_trend_strength(self, r_squared: float, slope_magnitude: float) -> str:
        """Classify trend strength based on R-squared and slope"""
        if r_squared < 0.3:
            return 'WEAK'
        elif r_squared < self.min_r_squared:
            return 'MODERATE'
        elif r_squared >= self.min_r_squared and slope_magnitude >= self.trend_sensitivity:
            return 'STRONG'
        else:
            return 'SIDEWAYS'
    
    def _calculate_regression_channels(self, regression_data: List[Dict]) -> List[Dict]:
        """Calculate regression channel boundaries"""
        channels = []
        
        for reg_data in regression_data:
            channel = {
                'regression_line': reg_data['regression_line'],
                'slope': reg_data['slope'],
                'r_squared': reg_data['r_squared']
            }
            
            # Calculate channel boundaries for each multiplier
            for multiplier in self.std_error_multipliers:
                upper_key = f'upper_{multiplier}'
                lower_key = f'lower_{multiplier}'
                
                channel[upper_key] = reg_data['regression_line'] + (reg_data['standard_error'] * multiplier)
                channel[lower_key] = reg_data['regression_line'] - (reg_data['standard_error'] * multiplier)
            
            channels.append(channel)
        
        return channels
    
    def _analyze_trend_strength(self, regression_data: List[Dict]) -> Dict:
        """Analyze trend strength and consistency"""
        if not regression_data:
            return {}
        
        current_data = regression_data[-1]
        
        # Recent trend analysis
        recent_period = min(10, len(regression_data))
        recent_data = regression_data[-recent_period:]
        
        # Average R-squared over recent period
        avg_r_squared = np.mean([rd['r_squared'] for rd in recent_data])
        
        # Slope consistency
        slopes = [rd['slope'] for rd in recent_data]
        slope_consistency = self._calculate_slope_consistency(slopes)
        
        # Trend direction analysis
        current_slope = current_data['slope']
        trend_direction = self._classify_trend_direction(current_slope)
        
        # Trend momentum (slope acceleration)
        slope_momentum = self._calculate_slope_momentum(regression_data)
        
        return {
            'current_strength': current_data['trend_strength'],
            'current_r_squared': current_data['r_squared'],
            'average_r_squared': avg_r_squared,
            'slope_consistency': slope_consistency,
            'trend_direction': trend_direction,
            'current_slope': current_slope,
            'slope_momentum': slope_momentum,
            'trend_quality': self._assess_trend_quality(avg_r_squared, slope_consistency, current_slope),
            'trend_persistence': self._calculate_trend_persistence(regression_data)
        }
    
    def _classify_trend_direction(self, slope: float) -> str:
        """Classify trend direction based on slope"""
        if slope > self.trend_sensitivity:
            return 'BULLISH'
        elif slope < -self.trend_sensitivity:
            return 'BEARISH'
        else:
            return 'SIDEWAYS'
    
    def _calculate_slope_consistency(self, slopes: List[float]) -> Dict:
        """Calculate slope consistency metrics"""
        if len(slopes) < 2:
            return {'score': 0, 'classification': 'INSUFFICIENT_DATA'}
        
        slope_std = np.std(slopes)
        slope_mean = np.mean(slopes)
        
        # Coefficient of variation (relative consistency)
        if abs(slope_mean) > 0:
            cv = slope_std / abs(slope_mean)
            consistency_score = max(0, 100 - (cv * 100))
        else:
            consistency_score = 0
        
        # Classification
        if consistency_score > 80:
            classification = 'HIGHLY_CONSISTENT'
        elif consistency_score > 60:
            classification = 'CONSISTENT'
        elif consistency_score > 40:
            classification = 'MODERATELY_CONSISTENT'
        else:
            classification = 'INCONSISTENT'
        
        return {
            'score': consistency_score,
            'classification': classification,
            'slope_std': slope_std,
            'slope_cv': cv if abs(slope_mean) > 0 else float('inf')
        }
    
    def _calculate_slope_momentum(self, regression_data: List[Dict]) -> Dict:
        """Calculate slope momentum (acceleration/deceleration)"""
        if len(regression_data) < 3:
            return {'momentum': 0, 'direction': 'NEUTRAL'}
        
        # Calculate slope changes
        recent_slopes = [rd['slope'] for rd in regression_data[-3:]]
        slope_changes = [recent_slopes[i] - recent_slopes[i-1] for i in range(1, len(recent_slopes))]
        
        avg_momentum = np.mean(slope_changes)
        
        # Classify momentum
        if avg_momentum > self.trend_sensitivity / 10:
            direction = 'ACCELERATING'
        elif avg_momentum < -self.trend_sensitivity / 10:
            direction = 'DECELERATING'
        else:
            direction = 'STABLE'
        
        return {
            'momentum': avg_momentum,
            'direction': direction,
            'magnitude': abs(avg_momentum)
        }
    
    def _assess_trend_quality(self, avg_r_squared: float, slope_consistency: Dict, current_slope: float) -> str:
        """Assess overall trend quality"""
        # Quality factors
        strong_r_squared = avg_r_squared >= self.min_r_squared
        consistent_slope = slope_consistency['score'] > 60
        significant_slope = abs(current_slope) >= self.trend_sensitivity
        
        if strong_r_squared and consistent_slope and significant_slope:
            return 'EXCELLENT'
        elif (strong_r_squared and consistent_slope) or (strong_r_squared and significant_slope):
            return 'GOOD'
        elif strong_r_squared or (consistent_slope and significant_slope):
            return 'FAIR'
        else:
            return 'POOR'
    
    def _calculate_trend_persistence(self, regression_data: List[Dict]) -> Dict:
        """Calculate trend persistence metrics"""
        if len(regression_data) < 5:
            return {'score': 0, 'classification': 'INSUFFICIENT_DATA'}
        
        # Look for trend direction changes
        recent_period = min(20, len(regression_data))
        recent_data = regression_data[-recent_period:]
        
        # Count direction changes
        direction_changes = 0
        prev_direction = None
        
        for rd in recent_data:
            current_direction = self._classify_trend_direction(rd['slope'])
            if prev_direction and current_direction != prev_direction and current_direction != 'SIDEWAYS' and prev_direction != 'SIDEWAYS':
                direction_changes += 1
            prev_direction = current_direction
        
        # Persistence score (fewer changes = higher persistence)
        persistence_score = max(0, 100 - (direction_changes * 10))
        
        return {
            'score': persistence_score,
            'direction_changes': direction_changes,
            'classification': 'HIGH' if persistence_score > 70 else 'MEDIUM' if persistence_score > 40 else 'LOW'
        }
    
    def _generate_signals(self, prices: List[float], highs: List[float], lows: List[float],
                         channels: List[Dict], regression_data: List[Dict]) -> Dict:
        """Generate trading signals based on regression channel analysis"""
        if not channels or not regression_data:
            return {'action': 'HOLD', 'strength': 0, 'confidence': 0}
        
        current_price = prices[-1]
        current_channel = channels[-1]
        current_regression = regression_data[-1]
        
        # Calculate position within channel
        position = self._calculate_position(current_price, current_channel)
        
        # Signal generation
        action = 'HOLD'
        strength = 0
        confidence = 0
        signal_type = 'TREND'
        
        primary_multiplier = self.std_error_multipliers[0]
        upper_band = current_channel[f'upper_{primary_multiplier}']
        lower_band = current_channel[f'lower_{primary_multiplier}']
        regression_line = current_channel['regression_line']
        
        # Trend-following signals
        slope = current_regression['slope']
        r_squared = current_regression['r_squared']
        
        if r_squared >= self.min_r_squared:  # Strong trend
            if slope > self.trend_sensitivity:  # Bullish trend
                if current_price > regression_line:
                    action = 'BUY'
                    strength = min(100, int(r_squared * 100))
                    confidence = min(95, int(r_squared * 80 + 15))
                    signal_type = 'TREND_FOLLOWING_BULLISH'
            elif slope < -self.trend_sensitivity:  # Bearish trend
                if current_price < regression_line:
                    action = 'SELL'
                    strength = min(100, int(r_squared * 100))
                    confidence = min(95, int(r_squared * 80 + 15))
                    signal_type = 'TREND_FOLLOWING_BEARISH'
        
        # Channel breakout signals
        if current_price > upper_band:
            if action == 'HOLD':  # No trend signal
                action = 'BUY'
                strength = min(100, int((current_price - upper_band) / upper_band * 500))
                confidence = 70
                signal_type = 'BREAKOUT_BULLISH'
        elif current_price < lower_band:
            if action == 'HOLD':  # No trend signal
                action = 'SELL'
                strength = min(100, int((lower_band - current_price) / lower_band * 500))
                confidence = 70
                signal_type = 'BREAKOUT_BEARISH'
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'signal_type': signal_type,
            'trend_strength': current_regression['trend_strength'],
            'r_squared': r_squared,
            'slope': slope,
            'distance_from_regression': (current_price - regression_line) / regression_line * 100,
            'channel_position': position
        }
    
    def _calculate_position(self, price: float, channel: Dict) -> Dict:
        """Calculate price position within regression channels"""
        position = {}
        
        regression_line = channel['regression_line']
        
        for multiplier in self.std_error_multipliers:
            upper = channel[f'upper_{multiplier}']
            lower = channel[f'lower_{multiplier}']
            
            # Calculate percentage position within this band
            if upper > lower:
                band_position = (price - lower) / (upper - lower) * 100
                band_position = max(0, min(100, band_position))
            else:
                band_position = 50
            
            position[f'band_{multiplier}_position'] = band_position
            
            # Determine if price is outside this band
            if price > upper:
                position[f'band_{multiplier}_status'] = 'ABOVE'
            elif price < lower:
                position[f'band_{multiplier}_status'] = 'BELOW'
            else:
                position[f'band_{multiplier}_status'] = 'WITHIN'
        
        # Position relative to regression line
        position['regression_position'] = ((price - regression_line) / regression_line * 100) if regression_line > 0 else 0
        
        return position
    
    def _analyze_breakouts(self, prices: List[float], highs: List[float], lows: List[float],
                          channels: List[Dict]) -> Dict:
        """Analyze channel breakouts and their validity"""
        breakouts = []
        
        if len(channels) < 2:
            return {'breakouts': [], 'recent_count': 0}
        
        primary_multiplier = self.std_error_multipliers[0]
        
        for i in range(1, len(channels)):
            upper = channels[i][f'upper_{primary_multiplier}']
            lower = channels[i][f'lower_{primary_multiplier}']
            r_squared = channels[i]['r_squared']
            
            # Check for valid breakouts (higher R-squared = more reliable)
            breakout_confidence = r_squared * 100
            
            if highs[i] > upper and highs[i-1] <= channels[i-1][f'upper_{primary_multiplier}']:
                breakouts.append({
                    'index': i,
                    'type': 'BULLISH',
                    'strength': (highs[i] - upper) / upper * 100,
                    'confidence': breakout_confidence,
                    'price': highs[i],
                    'r_squared': r_squared
                })
            elif lows[i] < lower and lows[i-1] >= channels[i-1][f'lower_{primary_multiplier}']:
                breakouts.append({
                    'index': i,
                    'type': 'BEARISH',
                    'strength': (lower - lows[i]) / lower * 100,
                    'confidence': breakout_confidence,
                    'price': lows[i],
                    'r_squared': r_squared
                })
        
        # Filter high-confidence breakouts
        high_confidence_breakouts = [b for b in breakouts if b['confidence'] > 50]
        
        return {
            'breakouts': breakouts[-5:],  # Last 5 breakouts
            'high_confidence_breakouts': high_confidence_breakouts[-3:],  # Last 3 high-confidence
            'recent_count': len([b for b in breakouts if len(channels) - b['index'] <= 20]),
            'avg_confidence': np.mean([b['confidence'] for b in breakouts]) if breakouts else 0
        }
    
    def _generate_forecast(self, regression_data: Dict) -> Dict:
        """Generate price forecast based on regression analysis"""
        if not regression_data or regression_data['r_squared'] < 0.3:
            return {'reliable': False, 'reason': 'Insufficient trend strength'}
        
        slope = regression_data['slope']
        regression_line = regression_data['regression_line']
        standard_error = regression_data['standard_error']
        
        # Forecast next few periods
        forecast_periods = [1, 2, 3, 5]
        forecasts = []
        
        for period in forecast_periods:
            forecast_price = regression_line + (slope * period)
            confidence_interval = standard_error * 1.96  # 95% confidence interval
            
            forecasts.append({
                'period': period,
                'forecast_price': forecast_price,
                'upper_bound': forecast_price + confidence_interval,
                'lower_bound': forecast_price - confidence_interval,
                'confidence': regression_data['r_squared'] * 100
            })
        
        return {
            'reliable': True,
            'forecasts': forecasts,
            'trend_direction': self._classify_trend_direction(slope),
            'base_r_squared': regression_data['r_squared']
        }
    
    def _calculate_statistics(self, prices: List[float], channels: List[Dict], 
                             regression_data: List[Dict]) -> Dict:
        """Calculate comprehensive regression channel statistics"""
        if not channels or not regression_data:
            return {}
        
        # Channel efficiency
        primary_multiplier = self.std_error_multipliers[0]
        within_channel_count = 0
        
        for i, price in enumerate(prices):
            if i < len(channels):
                upper = channels[i][f'upper_{primary_multiplier}']
                lower = channels[i][f'lower_{primary_multiplier}']
                if lower <= price <= upper:
                    within_channel_count += 1
        
        channel_efficiency = (within_channel_count / len(prices)) * 100 if prices else 0
        
        # R-squared statistics
        r_squared_values = [rd['r_squared'] for rd in regression_data]
        valid_r_squared = [r for r in r_squared_values if r > 0]
        
        # Slope statistics
        slopes = [rd['slope'] for rd in regression_data]
        
        return {
            'channel_efficiency': channel_efficiency,
            'average_r_squared': np.mean(valid_r_squared) if valid_r_squared else 0,
            'current_r_squared': regression_data[-1]['r_squared'] if regression_data else 0,
            'strong_trend_percentage': (len([r for r in valid_r_squared if r >= self.min_r_squared]) / len(valid_r_squared) * 100) if valid_r_squared else 0,
            'average_slope': np.mean(slopes) if slopes else 0,
            'slope_volatility': np.std(slopes) if slopes else 0,
            'current_slope': regression_data[-1]['slope'] if regression_data else 0,
            'total_channels': len(channels),
            'multipliers_used': self.std_error_multipliers
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result when insufficient data"""
        return {
            'regression_lines': [],
            'channels': [],
            'signals': {},
            'statistics': {},
            'indicator_name': 'Linear Regression Channels',
            'error': 'Insufficient data for calculation'
        }

def calculate_regression_channels(data: Union[pd.DataFrame, Dict],
                                 period: int = 20,
                                 std_error_multipliers: List[float] = [1.0, 2.0],
                                 min_r_squared: float = 0.5,
                                 price_column: str = 'close') -> Dict:
    """
    Convenience function for Linear Regression Channels calculation
    
    Args:
        data: Price data
        period: Lookback period
        std_error_multipliers: Standard error multipliers
        min_r_squared: Minimum R-squared for strong trend
        price_column: Price column name
        
    Returns:
        Linear Regression Channels analysis results
    """
    calculator = LinearRegressionChannels(period, std_error_multipliers, min_r_squared)
    return calculator.calculate(data, price_column)
