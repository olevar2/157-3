"""
HiddenDivergenceDetector - Hidden Divergence Detection Indicator for Platform3

This indicator identifies hidden divergences between price action and technical
indicators, which often signal trend continuation rather than reversal. Hidden
divergences are powerful signals for identifying potential trend continuation points.

Version: 1.0.0
Category: Divergence
Complexity: Advanced
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import logging

from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface


class HiddenDivergenceDetectorIndicator(StandardIndicatorInterface):
    """
    Advanced Hidden Divergence Detection Indicator
    
    Identifies hidden divergences between price and momentum indicators:
    - Bullish hidden divergence: Higher lows in price, lower lows in indicator
    - Bearish hidden divergence: Lower highs in price, higher highs in indicator
    - Multi-timeframe divergence analysis
    - Divergence strength and reliability scoring
    - Volume confirmation integration
    
    Hidden divergences typically indicate trend continuation rather than reversal,
    making them valuable for trend-following strategies.
    """
    
    # Class-level metadata
    INDICATOR_NAME = "HiddenDivergenceDetector"
    INDICATOR_VERSION = "1.0.0"
    INDICATOR_CATEGORY = "divergence"
    INDICATOR_TYPE = "advanced"
    INDICATOR_COMPLEXITY = "advanced"
    
    def __init__(self, **kwargs):
        """
        Initialize HiddenDivergenceDetector indicator
        
        Args:
            parameters: Dictionary containing indicator parameters
                - period: Analysis period (default: 14)
                - momentum_periods: List of momentum indicator periods (default: [14, 21, 34])
                - min_pivot_distance: Minimum distance between pivots (default: 5)
                - divergence_threshold: Minimum divergence strength threshold (default: 0.02)
                - volume_confirmation: Whether to require volume confirmation (default: True)
                - rsi_overbought: RSI overbought level (default: 70)
                - rsi_oversold: RSI oversold level (default: 30)
                - lookback_period: Period to look back for divergences (default: 50)
        """
        super().__init__(**kwargs)
        
        # Get parameters with defaults
        self.period = int(self.parameters.get('period', 14))
        self.momentum_periods = self.parameters.get('momentum_periods', [14, 21, 34])
        self.min_pivot_distance = int(self.parameters.get('min_pivot_distance', 5))
        self.divergence_threshold = float(self.parameters.get('divergence_threshold', 0.02))
        self.volume_confirmation = bool(self.parameters.get('volume_confirmation', True))
        self.rsi_overbought = float(self.parameters.get('rsi_overbought', 70))
        self.rsi_oversold = float(self.parameters.get('rsi_oversold', 30))
        self.lookback_period = int(self.parameters.get('lookback_period', 50))
        
        # Validation
        if self.period < 5:
            raise ValueError("Period must be at least 5")
        if self.min_pivot_distance < 2:
            raise ValueError("Minimum pivot distance must be at least 2")
        if not 0 < self.divergence_threshold < 1:
            raise ValueError("Divergence threshold must be between 0 and 1")
            
        # Initialize state
        self.price_pivots = []
        self.indicator_pivots = []
        self.divergences = []
        self.momentum_indicators = {}
        
        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for HiddenDivergenceDetector calculation"""
        try:
            required_columns = ['high', 'low', 'close']
            if self.volume_confirmation:
                required_columns.append('volume')
                
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Need: {required_columns}")
                return False
                
            if len(data) < max(self.momentum_periods) + self.lookback_period:
                self.logger.warning(f"Insufficient data length: {len(data)} < {max(self.momentum_periods) + self.lookback_period}")
                return False
                
            if data[required_columns].isnull().any().any():
                self.logger.warning("Data contains NaN values")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various momentum indicators for divergence detection"""
        try:
            indicators = {}
            
            for period in self.momentum_periods:
                # RSI
                rsi = self._calculate_rsi(data['close'], period)
                indicators[f'rsi_{period}'] = rsi
                
                # MACD
                macd, macd_signal, macd_histogram = self._calculate_macd(data['close'], period)
                indicators[f'macd_{period}'] = macd
                indicators[f'macd_histogram_{period}'] = macd_histogram
                
                # Stochastic
                stoch_k, stoch_d = self._calculate_stochastic(data, period)
                indicators[f'stoch_k_{period}'] = stoch_k
                
                # CCI
                cci = self._calculate_cci(data, period)
                indicators[f'cci_{period}'] = cci
                
                # Williams %R
                williams_r = self._calculate_williams_r(data, period)
                indicators[f'williams_r_{period}'] = williams_r
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def _calculate_macd(self, prices: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            fast_period = max(period // 2, 12)
            slow_period = period
            signal_period = max(period // 3, 9)
            
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal_period).mean()
            macd_histogram = macd - macd_signal
            
            return macd, macd_signal, macd_histogram
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        try:
            lowest_low = data['low'].rolling(window=period).min()
            highest_high = data['high'].rolling(window=period).max()
            
            k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=3).mean()
            
            return k_percent, d_percent
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return pd.Series(), pd.Series()
    
    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            sma = typical_price.rolling(window=period).mean()
            
            # Calculate mean absolute deviation
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            
            cci = (typical_price - sma) / (0.015 * mad)
            
            return cci
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {str(e)}")
            return pd.Series()
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = data['high'].rolling(window=period).max()
            lowest_low = data['low'].rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
            
            return williams_r
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series()
    
    def _find_pivots(self, series: pd.Series, pivot_type: str) -> List[Dict]:
        """Find pivot points in a price or indicator series"""
        try:
            pivots = []
            
            for i in range(self.min_pivot_distance, len(series) - self.min_pivot_distance):
                if pivot_type == 'high':
                    # Check for pivot high
                    is_pivot = True
                    for j in range(1, self.min_pivot_distance + 1):
                        if (series.iloc[i] <= series.iloc[i-j] or 
                            series.iloc[i] <= series.iloc[i+j]):
                            is_pivot = False
                            break
                    
                    if is_pivot:
                        pivots.append({
                            'index': i,
                            'value': series.iloc[i],
                            'type': 'high',
                            'timestamp': series.index[i] if hasattr(series.index, 'to_list') else i
                        })
                
                elif pivot_type == 'low':
                    # Check for pivot low
                    is_pivot = True
                    for j in range(1, self.min_pivot_distance + 1):
                        if (series.iloc[i] >= series.iloc[i-j] or 
                            series.iloc[i] >= series.iloc[i+j]):
                            is_pivot = False
                            break
                    
                    if is_pivot:
                        pivots.append({
                            'index': i,
                            'value': series.iloc[i],
                            'type': 'low',
                            'timestamp': series.index[i] if hasattr(series.index, 'to_list') else i
                        })
            
            return pivots
            
        except Exception as e:
            self.logger.error(f"Error finding pivots: {str(e)}")
            return []
    
    def _detect_hidden_divergences(self, price_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> List[Dict]:
        """Detect hidden divergences between price and indicators"""
        try:
            divergences = []
            
            # Find price pivots
            price_highs = self._find_pivots(price_data['high'], 'high')
            price_lows = self._find_pivots(price_data['low'], 'low')
            
            # Check each momentum indicator
            for indicator_name, indicator_series in indicators.items():
                if indicator_series.empty:
                    continue
                
                # Find indicator pivots
                indicator_highs = self._find_pivots(indicator_series, 'high')
                indicator_lows = self._find_pivots(indicator_series, 'low')
                
                # Detect bullish hidden divergences (higher lows in price, lower lows in indicator)
                bullish_divergences = self._find_bullish_hidden_divergences(
                    price_lows, indicator_lows, indicator_name
                )
                divergences.extend(bullish_divergences)
                
                # Detect bearish hidden divergences (lower highs in price, higher highs in indicator)
                bearish_divergences = self._find_bearish_hidden_divergences(
                    price_highs, indicator_highs, indicator_name
                )
                divergences.extend(bearish_divergences)
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden divergences: {str(e)}")
            return []
    
    def _find_bullish_hidden_divergences(self, price_lows: List[Dict], indicator_lows: List[Dict], indicator_name: str) -> List[Dict]:
        """Find bullish hidden divergences"""
        try:
            divergences = []
            
            # Need at least 2 lows for comparison
            if len(price_lows) < 2 or len(indicator_lows) < 2:
                return divergences
            
            # Check recent pivots within lookback period
            recent_price_lows = [p for p in price_lows if p['index'] >= len(price_lows) - self.lookback_period]
            recent_indicator_lows = [p for p in indicator_lows if p['index'] >= len(indicator_lows) - self.lookback_period]
            
            # Compare consecutive lows
            for i in range(len(recent_price_lows) - 1):
                price_low_1 = recent_price_lows[i]
                price_low_2 = recent_price_lows[i + 1]
                
                # Find corresponding indicator lows (within reasonable time proximity)
                indicator_low_1 = self._find_nearest_pivot(price_low_1['index'], recent_indicator_lows)
                indicator_low_2 = self._find_nearest_pivot(price_low_2['index'], recent_indicator_lows)
                
                if indicator_low_1 and indicator_low_2:
                    # Check for bullish hidden divergence
                    # Price: higher low, Indicator: lower low
                    if (price_low_2['value'] > price_low_1['value'] and 
                        indicator_low_2['value'] < indicator_low_1['value']):
                        
                        # Calculate divergence strength
                        price_change = (price_low_2['value'] - price_low_1['value']) / price_low_1['value']
                        indicator_change = (indicator_low_1['value'] - indicator_low_2['value']) / abs(indicator_low_1['value'])
                        
                        if price_change > self.divergence_threshold and indicator_change > self.divergence_threshold:
                            divergence_strength = (price_change + indicator_change) / 2
                            
                            divergences.append({
                                'type': 'bullish_hidden',
                                'indicator': indicator_name,
                                'price_point_1': price_low_1,
                                'price_point_2': price_low_2,
                                'indicator_point_1': indicator_low_1,
                                'indicator_point_2': indicator_low_2,
                                'strength': divergence_strength,
                                'confidence': self._calculate_divergence_confidence(
                                    price_low_1, price_low_2, indicator_low_1, indicator_low_2, 'bullish_hidden'
                                )
                            })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error finding bullish hidden divergences: {str(e)}")
            return []
    
    def _find_bearish_hidden_divergences(self, price_highs: List[Dict], indicator_highs: List[Dict], indicator_name: str) -> List[Dict]:
        """Find bearish hidden divergences"""
        try:
            divergences = []
            
            # Need at least 2 highs for comparison
            if len(price_highs) < 2 or len(indicator_highs) < 2:
                return divergences
            
            # Check recent pivots within lookback period
            recent_price_highs = [p for p in price_highs if p['index'] >= len(price_highs) - self.lookback_period]
            recent_indicator_highs = [p for p in indicator_highs if p['index'] >= len(indicator_highs) - self.lookback_period]
            
            # Compare consecutive highs
            for i in range(len(recent_price_highs) - 1):
                price_high_1 = recent_price_highs[i]
                price_high_2 = recent_price_highs[i + 1]
                
                # Find corresponding indicator highs
                indicator_high_1 = self._find_nearest_pivot(price_high_1['index'], recent_indicator_highs)
                indicator_high_2 = self._find_nearest_pivot(price_high_2['index'], recent_indicator_highs)
                
                if indicator_high_1 and indicator_high_2:
                    # Check for bearish hidden divergence
                    # Price: lower high, Indicator: higher high
                    if (price_high_2['value'] < price_high_1['value'] and 
                        indicator_high_2['value'] > indicator_high_1['value']):
                        
                        # Calculate divergence strength
                        price_change = (price_high_1['value'] - price_high_2['value']) / price_high_1['value']
                        indicator_change = (indicator_high_2['value'] - indicator_high_1['value']) / abs(indicator_high_1['value'])
                        
                        if price_change > self.divergence_threshold and indicator_change > self.divergence_threshold:
                            divergence_strength = (price_change + indicator_change) / 2
                            
                            divergences.append({
                                'type': 'bearish_hidden',
                                'indicator': indicator_name,
                                'price_point_1': price_high_1,
                                'price_point_2': price_high_2,
                                'indicator_point_1': indicator_high_1,
                                'indicator_point_2': indicator_high_2,
                                'strength': divergence_strength,
                                'confidence': self._calculate_divergence_confidence(
                                    price_high_1, price_high_2, indicator_high_1, indicator_high_2, 'bearish_hidden'
                                )
                            })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error finding bearish hidden divergences: {str(e)}")
            return []
    
    def _find_nearest_pivot(self, target_index: int, pivots: List[Dict], max_distance: int = 10) -> Optional[Dict]:
        """Find the nearest pivot to a target index"""
        try:
            if not pivots:
                return None
            
            nearest_pivot = None
            min_distance = float('inf')
            
            for pivot in pivots:
                distance = abs(pivot['index'] - target_index)
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_pivot = pivot
            
            return nearest_pivot
            
        except Exception as e:
            self.logger.error(f"Error finding nearest pivot: {str(e)}")
            return None
    
    def _calculate_divergence_confidence(self, price_point_1: Dict, price_point_2: Dict, 
                                       indicator_point_1: Dict, indicator_point_2: Dict, 
                                       divergence_type: str) -> float:
        """Calculate confidence score for a divergence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Time proximity factor (closer points in time = higher confidence)
            time_diff_price = abs(price_point_2['index'] - price_point_1['index'])
            time_diff_indicator = abs(indicator_point_2['index'] - indicator_point_1['index'])
            time_proximity = 1 / (1 + abs(time_diff_price - time_diff_indicator) / 10)
            confidence += time_proximity * 0.2
            
            # Strength factor (stronger divergences = higher confidence)
            if divergence_type == 'bullish_hidden':
                price_strength = (price_point_2['value'] - price_point_1['value']) / price_point_1['value']
                indicator_strength = (indicator_point_1['value'] - indicator_point_2['value']) / abs(indicator_point_1['value'])
            else:  # bearish_hidden
                price_strength = (price_point_1['value'] - price_point_2['value']) / price_point_1['value']
                indicator_strength = (indicator_point_2['value'] - indicator_point_1['value']) / abs(indicator_point_1['value'])
            
            avg_strength = (price_strength + indicator_strength) / 2
            confidence += min(avg_strength, 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating divergence confidence: {str(e)}")
            return 0.5
    
    def _apply_volume_confirmation(self, divergences: List[Dict], volume_data: pd.Series) -> List[Dict]:
        """Apply volume confirmation to divergences"""
        try:
            if not self.volume_confirmation or volume_data.empty:
                return divergences
            
            confirmed_divergences = []
            
            for divergence in divergences:
                # Get volume at divergence points
                vol_1 = volume_data.iloc[divergence['price_point_1']['index']]
                vol_2 = volume_data.iloc[divergence['price_point_2']['index']]
                
                # Volume confirmation logic
                if divergence['type'] == 'bullish_hidden':
                    # For bullish hidden divergence, we want increasing volume on the higher low
                    if vol_2 > vol_1 * 1.1:  # 10% increase
                        divergence['volume_confirmed'] = True
                        divergence['confidence'] = min(divergence['confidence'] + 0.1, 1.0)
                    else:
                        divergence['volume_confirmed'] = False
                        
                elif divergence['type'] == 'bearish_hidden':
                    # For bearish hidden divergence, we want increasing volume on the lower high
                    if vol_2 > vol_1 * 1.1:  # 10% increase
                        divergence['volume_confirmed'] = True
                        divergence['confidence'] = min(divergence['confidence'] + 0.1, 1.0)
                    else:
                        divergence['volume_confirmed'] = False
                
                confirmed_divergences.append(divergence)
            
            return confirmed_divergences
            
        except Exception as e:
            self.logger.error(f"Error applying volume confirmation: {str(e)}")
            return divergences
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate HiddenDivergenceDetector indicator
        
        Args:
            data: DataFrame with OHLC and optionally volume data
            
        Returns:
            Dictionary containing hidden divergence analysis
        """
        try:
            if not self.validate_data(data):
                return {}
            
            # Calculate momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(data)
            self.momentum_indicators = momentum_indicators
            
            if not momentum_indicators:
                return {
                    'divergences': [],
                    'bullish_hidden_count': 0,
                    'bearish_hidden_count': 0,
                    'analysis_complete': False
                }
            
            # Detect hidden divergences
            divergences = self._detect_hidden_divergences(data, momentum_indicators)
            
            # Apply volume confirmation if enabled
            if self.volume_confirmation and 'volume' in data.columns:
                divergences = self._apply_volume_confirmation(divergences, data['volume'])
            
            # Categorize divergences
            bullish_hidden = [d for d in divergences if d['type'] == 'bullish_hidden']
            bearish_hidden = [d for d in divergences if d['type'] == 'bearish_hidden']
            
            # Calculate overall signal strength
            current_signal = self._determine_current_signal(bullish_hidden, bearish_hidden)
            
            # Store state
            self.divergences = divergences
            
            result = {
                'divergences': divergences,
                'bullish_hidden_count': len(bullish_hidden),
                'bearish_hidden_count': len(bearish_hidden),
                'total_divergences': len(divergences),
                'current_signal': current_signal['signal'],
                'signal_strength': current_signal['strength'],
                'signal_confidence': current_signal['confidence'],
                'analysis_complete': True,
                'recent_divergences': divergences[-5:] if len(divergences) > 5 else divergences,
                'strongest_divergence': max(divergences, key=lambda x: x['strength']) if divergences else None
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating HiddenDivergenceDetector: {str(e)}")
            return {}
    
    def _determine_current_signal(self, bullish_hidden: List[Dict], bearish_hidden: List[Dict]) -> Dict:
        """Determine current signal based on recent divergences"""
        try:
            recent_lookback = 20  # Look at last 20 periods
            
            # Filter recent divergences
            recent_bullish = [d for d in bullish_hidden 
                            if d['price_point_2']['index'] >= len(bullish_hidden) - recent_lookback]
            recent_bearish = [d for d in bearish_hidden 
                            if d['price_point_2']['index'] >= len(bearish_hidden) - recent_lookback]
            
            if not recent_bullish and not recent_bearish:
                return {'signal': 'neutral', 'strength': 0.0, 'confidence': 0.0}
            
            # Calculate signal strength
            bullish_strength = sum(d['strength'] for d in recent_bullish) / max(len(recent_bullish), 1)
            bearish_strength = sum(d['strength'] for d in recent_bearish) / max(len(recent_bearish), 1)
            
            # Calculate confidence
            bullish_confidence = sum(d['confidence'] for d in recent_bullish) / max(len(recent_bullish), 1)
            bearish_confidence = sum(d['confidence'] for d in recent_bearish) / max(len(recent_bearish), 1)
            
            # Determine signal
            if bullish_strength > bearish_strength:
                return {
                    'signal': 'bullish_continuation',
                    'strength': bullish_strength,
                    'confidence': bullish_confidence
                }
            elif bearish_strength > bullish_strength:
                return {
                    'signal': 'bearish_continuation',
                    'strength': bearish_strength,
                    'confidence': bearish_confidence
                }
            else:
                return {
                    'signal': 'neutral',
                    'strength': (bullish_strength + bearish_strength) / 2,
                    'confidence': (bullish_confidence + bearish_confidence) / 2
                }
                
        except Exception as e:
            self.logger.error(f"Error determining current signal: {str(e)}")
            return {'signal': 'neutral', 'strength': 0.0, 'confidence': 0.0}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata"""
        return {
            'name': self.INDICATOR_NAME,
            'version': self.INDICATOR_VERSION,
            'category': self.INDICATOR_CATEGORY,
            'type': self.INDICATOR_TYPE,
            'complexity': self.INDICATOR_COMPLEXITY,
            'parameters': {
                'period': self.period,
                'momentum_periods': self.momentum_periods,
                'min_pivot_distance': self.min_pivot_distance,
                'divergence_threshold': self.divergence_threshold,
                'volume_confirmation': self.volume_confirmation,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'lookback_period': self.lookback_period
            },
            'data_requirements': ['high', 'low', 'close', 'volume (optional)'],
            'output_format': 'hidden_divergence_analysis'
        }
    def validate_parameters(self) -> bool:
        """Validate parameters"""
        # Add specific validation logic as needed
        return True



def export() -> Dict[str, Any]:
    """
    Export function for the HiddenDivergenceDetector indicator.
    
    This function is used by the indicator registry to discover and load the indicator.
    
    Returns:
        Dictionary containing indicator information for registry
    """
    return {
        'class': HiddenDivergenceDetectorIndicator,
        'name': 'HiddenDivergenceDetector',
        'category': 'divergence',
        'version': '1.0.0',
        'description': 'Advanced hidden divergence detection for trend continuation signals',
        'complexity': 'advanced',
        'parameters': {
            'period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'momentum_periods': {'type': 'list', 'default': [14, 21, 34]},
            'min_pivot_distance': {'type': 'int', 'default': 5, 'min': 2, 'max': 15},
            'divergence_threshold': {'type': 'float', 'default': 0.02, 'min': 0.001, 'max': 0.1},
            'volume_confirmation': {'type': 'bool', 'default': True},
            'rsi_overbought': {'type': 'float', 'default': 70, 'min': 60, 'max': 90},
            'rsi_oversold': {'type': 'float', 'default': 30, 'min': 10, 'max': 40},
            'lookback_period': {'type': 'int', 'default': 50, 'min': 20, 'max': 200}
        },
        'data_requirements': ['high', 'low', 'close', 'volume'],
        'output_type': 'divergence_analysis'
    }