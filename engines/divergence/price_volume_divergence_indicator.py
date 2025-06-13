"""
PriceVolumeDivergence - Price-Volume Divergence Analysis Indicator for Platform3

This indicator analyzes divergences between price movements and volume patterns,
identifying situations where price and volume trends contradict each other,
which can signal potential trend changes or confirmations.

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


class PriceVolumeDivergenceIndicator(StandardIndicatorInterface):
    """
    Advanced Price-Volume Divergence Analysis Indicator
    
    Analyzes divergences between price and volume patterns:
    - Price-volume trend divergences
    - Volume accumulation/distribution analysis
    - On-Balance Volume (OBV) divergences
    - Volume Rate of Change (VROC) analysis
    - Price-volume momentum divergences
    - Volume confirmation strength
    
    This indicator helps identify when volume is not confirming price moves,
    which can be an early warning of potential trend changes.
    """
    
    # Class-level metadata
    INDICATOR_NAME = "PriceVolumeDivergence"
    INDICATOR_VERSION = "1.0.0"
    INDICATOR_CATEGORY = "divergence"
    INDICATOR_TYPE = "advanced"
    INDICATOR_COMPLEXITY = "advanced"
    
    def __init__(self, **kwargs):
        """
        Initialize PriceVolumeDivergence indicator
        
        Args:
            parameters: Dictionary containing indicator parameters
                - period: Main analysis period (default: 20)
                - volume_ma_period: Volume moving average period (default: 20)
                - price_ma_period: Price moving average period (default: 20)
                - divergence_threshold: Minimum divergence strength (default: 0.03)
                - volume_spike_threshold: Volume spike detection threshold (default: 2.0)
                - trend_confirmation_period: Trend confirmation period (default: 5)
                - obv_smoothing: OBV smoothing period (default: 3)
                - vroc_period: Volume Rate of Change period (default: 10)
        """
        super().__init__(**kwargs)
        
        # Get parameters with defaults
        self.period = int(self.parameters.get('period', 20))
        self.volume_ma_period = int(self.parameters.get('volume_ma_period', 20))
        self.price_ma_period = int(self.parameters.get('price_ma_period', 20))
        self.divergence_threshold = float(self.parameters.get('divergence_threshold', 0.03))
        self.volume_spike_threshold = float(self.parameters.get('volume_spike_threshold', 2.0))
        self.trend_confirmation_period = int(self.parameters.get('trend_confirmation_period', 5))
        self.obv_smoothing = int(self.parameters.get('obv_smoothing', 3))
        self.vroc_period = int(self.parameters.get('vroc_period', 10))
        
        # Validation
        if self.period < 5:
            raise ValueError("Period must be at least 5")
        if self.divergence_threshold <= 0:
            raise ValueError("Divergence threshold must be positive")
        if self.volume_spike_threshold <= 1:
            raise ValueError("Volume spike threshold must be greater than 1")
            
        # Initialize state
        self.obv_values = []
        self.volume_indicators = {}
        self.price_volume_divergences = []
        self.volume_trend = 'neutral'
        self.price_trend = 'neutral'
        
        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for PriceVolumeDivergence calculation"""
        try:
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Need: {required_columns}")
                return False
                
            if len(data) < max(self.period, self.volume_ma_period, self.price_ma_period) + 10:
                self.logger.warning(f"Insufficient data length for analysis")
                return False
                
            if data[required_columns].isnull().any().any():
                self.logger.warning("Data contains NaN values")
                return False
                
            # Check for realistic volume values
            if (data['volume'] <= 0).any():
                self.logger.warning("Data contains zero or negative volume values")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive volume-based indicators"""
        try:
            indicators = {}
            
            # On-Balance Volume (OBV)
            obv = self._calculate_obv(data)
            indicators['obv'] = obv
            indicators['obv_smoothed'] = obv.rolling(window=self.obv_smoothing).mean()
            
            # Volume Rate of Change (VROC)
            vroc = self._calculate_vroc(data['volume'])
            indicators['vroc'] = vroc
            
            # Volume Moving Averages
            indicators['volume_sma'] = data['volume'].rolling(window=self.volume_ma_period).mean()
            indicators['volume_ema'] = data['volume'].ewm(span=self.volume_ma_period).mean()
            
            # Accumulation/Distribution Line
            ad_line = self._calculate_ad_line(data)
            indicators['ad_line'] = ad_line
            
            # Chaikin Money Flow
            cmf = self._calculate_chaikin_money_flow(data)
            indicators['cmf'] = cmf
            
            # Volume Oscillator
            volume_osc = self._calculate_volume_oscillator(data['volume'])
            indicators['volume_oscillator'] = volume_osc
            
            # Price Volume Trend (PVT)
            pvt = self._calculate_pvt(data)
            indicators['pvt'] = pvt
            
            # Volume Weighted Average Price (VWAP) deviation
            vwap_dev = self._calculate_vwap_deviation(data)
            indicators['vwap_deviation'] = vwap_dev
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            return {}
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = pd.Series(index=data.index, dtype='float64')
            obv.iloc[0] = data['volume'].iloc[0]
            
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series()
    
    def _calculate_vroc(self, volume: pd.Series) -> pd.Series:
        """Calculate Volume Rate of Change"""
        try:
            vroc = ((volume - volume.shift(self.vroc_period)) / volume.shift(self.vroc_period)) * 100
            return vroc
            
        except Exception as e:
            self.logger.error(f"Error calculating VROC: {str(e)}")
            return pd.Series()
    
    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)  # Handle division by zero
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # Accumulation/Distribution Line
            ad_line = mfv.cumsum()
            
            return ad_line
            
        except Exception as e:
            self.logger.error(f"Error calculating A/D Line: {str(e)}")
            return pd.Series()
    
    def _calculate_chaikin_money_flow(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # Chaikin Money Flow
            cmf = mfv.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
            
            return cmf
            
        except Exception as e:
            self.logger.error(f"Error calculating Chaikin Money Flow: {str(e)}")
            return pd.Series()
    
    def _calculate_volume_oscillator(self, volume: pd.Series) -> pd.Series:
        """Calculate Volume Oscillator"""
        try:
            short_period = max(self.volume_ma_period // 2, 5)
            long_period = self.volume_ma_period
            
            short_ma = volume.rolling(window=short_period).mean()
            long_ma = volume.rolling(window=long_period).mean()
            
            volume_osc = ((short_ma - long_ma) / long_ma) * 100
            
            return volume_osc
            
        except Exception as e:
            self.logger.error(f"Error calculating Volume Oscillator: {str(e)}")
            return pd.Series()
    
    def _calculate_pvt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Price Volume Trend"""
        try:
            close = data['close']
            volume = data['volume']
            
            # Price change percentage
            price_change_pct = close.pct_change()
            
            # PVT calculation
            pvt = (price_change_pct * volume).cumsum()
            
            return pvt
            
        except Exception as e:
            self.logger.error(f"Error calculating PVT: {str(e)}")
            return pd.Series()
    
    def _calculate_vwap_deviation(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP deviation"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            volume = data['volume']
            
            # VWAP calculation
            vwap = (typical_price * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
            
            # Deviation from VWAP
            vwap_deviation = ((data['close'] - vwap) / vwap) * 100
            
            return vwap_deviation
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP deviation: {str(e)}")
            return pd.Series()
    
    def _calculate_price_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price-based indicators for comparison"""
        try:
            indicators = {}
            
            # Price moving averages
            indicators['price_sma'] = data['close'].rolling(window=self.price_ma_period).mean()
            indicators['price_ema'] = data['close'].ewm(span=self.price_ma_period).mean()
            
            # Price Rate of Change
            indicators['price_roc'] = ((data['close'] - data['close'].shift(self.period)) / data['close'].shift(self.period)) * 100
            
            # Price momentum
            indicators['price_momentum'] = data['close'] / data['close'].shift(self.period) * 100
            
            # Price trend strength
            indicators['price_trend_strength'] = self._calculate_price_trend_strength(data['close'])
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating price indicators: {str(e)}")
            return {}
    
    def _calculate_price_trend_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate price trend strength"""
        try:
            # Simple trend strength based on moving average slope
            ma = prices.rolling(window=self.trend_confirmation_period).mean()
            trend_strength = (ma - ma.shift(1)) / ma.shift(1) * 100
            
            return trend_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating price trend strength: {str(e)}")
            return pd.Series()
    
    def _detect_price_volume_divergences(self, price_indicators: Dict, volume_indicators: Dict) -> List[Dict]:
        """Detect divergences between price and volume indicators"""
        try:
            divergences = []
            
            # Compare price trend with volume trend
            price_trends = self._identify_trends(price_indicators['price_trend_strength'])
            volume_trends = self._identify_trends(volume_indicators['vroc'])
            
            # Check for divergences in recent periods
            lookback = min(self.period, len(price_trends), len(volume_trends))
            
            for i in range(max(0, len(price_trends) - lookback), len(price_trends)):
                if i < len(volume_trends):
                    price_trend = price_trends[i]
                    volume_trend = volume_trends[i]
                    
                    # Detect divergence
                    if self._is_divergent(price_trend, volume_trend):
                        divergence_strength = self._calculate_pv_divergence_strength(
                            price_indicators, volume_indicators, i
                        )
                        
                        if divergence_strength >= self.divergence_threshold:
                            divergences.append({
                                'index': i,
                                'type': self._classify_pv_divergence(price_trend, volume_trend),
                                'price_trend': price_trend,
                                'volume_trend': volume_trend,
                                'strength': divergence_strength,
                                'confidence': self._calculate_pv_confidence(price_indicators, volume_indicators, i),
                                'timestamp': price_indicators['price_sma'].index[i] if i < len(price_indicators['price_sma']) else i
                            })
            
            # OBV divergences
            obv_divergences = self._detect_obv_divergences(price_indicators, volume_indicators)
            divergences.extend(obv_divergences)
            
            # Volume spike divergences
            spike_divergences = self._detect_volume_spike_divergences(price_indicators, volume_indicators)
            divergences.extend(spike_divergences)
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting price-volume divergences: {str(e)}")
            return []
    
    def _identify_trends(self, series: pd.Series, threshold: float = 0.01) -> List[str]:
        """Identify trend direction in a series"""
        try:
            trends = []
            
            for i, value in enumerate(series):
                if pd.isna(value):
                    trends.append('neutral')
                elif value > threshold:
                    trends.append('up')
                elif value < -threshold:
                    trends.append('down')
                else:
                    trends.append('neutral')
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error identifying trends: {str(e)}")
            return []
    
    def _is_divergent(self, price_trend: str, volume_trend: str) -> bool:
        """Check if price and volume trends are divergent"""
        try:
            divergent_combinations = [
                ('up', 'down'),    # Price up, volume down
                ('down', 'up'),    # Price down, volume up
                ('up', 'neutral'), # Price up, volume neutral
                ('down', 'neutral') # Price down, volume neutral
            ]
            
            return (price_trend, volume_trend) in divergent_combinations
            
        except Exception as e:
            self.logger.error(f"Error checking divergence: {str(e)}")
            return False
    
    def _classify_pv_divergence(self, price_trend: str, volume_trend: str) -> str:
        """Classify the type of price-volume divergence"""
        try:
            if price_trend == 'up' and volume_trend in ['down', 'neutral']:
                return 'bearish_pv_divergence'  # Price up but volume not confirming
            elif price_trend == 'down' and volume_trend in ['up', 'neutral']:
                return 'bullish_pv_divergence'  # Price down but volume not confirming
            else:
                return 'neutral_pv_divergence'
                
        except Exception as e:
            self.logger.error(f"Error classifying PV divergence: {str(e)}")
            return 'unknown'
    
    def _calculate_pv_divergence_strength(self, price_indicators: Dict, volume_indicators: Dict, index: int) -> float:
        """Calculate strength of price-volume divergence"""
        try:
            if index >= len(price_indicators['price_trend_strength']) or index >= len(volume_indicators['vroc']):
                return 0.0
            
            price_strength = abs(price_indicators['price_trend_strength'].iloc[index])
            volume_strength = abs(volume_indicators['vroc'].iloc[index])
            
            # Normalize and combine
            max_price_strength = price_indicators['price_trend_strength'].rolling(window=self.period).max().iloc[index]
            max_volume_strength = volume_indicators['vroc'].rolling(window=self.period).max().iloc[index]
            
            if max_price_strength > 0 and max_volume_strength > 0:
                normalized_price = price_strength / max_price_strength
                normalized_volume = volume_strength / max_volume_strength
                
                # Divergence strength is higher when one is strong and the other is weak
                divergence_strength = abs(normalized_price - normalized_volume)
                
                return divergence_strength
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating PV divergence strength: {str(e)}")
            return 0.0
    
    def _calculate_pv_confidence(self, price_indicators: Dict, volume_indicators: Dict, index: int) -> float:
        """Calculate confidence in price-volume divergence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Volume confirmation factor
            if index < len(volume_indicators['volume_sma']):
                current_volume = volume_indicators['volume_sma'].iloc[index]
                avg_volume = volume_indicators['volume_sma'].rolling(window=self.period).mean().iloc[index]
                
                if current_volume > avg_volume:
                    confidence += 0.2  # Higher confidence with above-average volume
            
            # Trend persistence factor
            if index >= self.trend_confirmation_period:
                price_trend_consistency = self._calculate_trend_consistency(
                    price_indicators['price_trend_strength'][index-self.trend_confirmation_period:index+1]
                )
                confidence += price_trend_consistency * 0.3
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating PV confidence: {str(e)}")
            return 0.5
    
    def _calculate_trend_consistency(self, trend_series: pd.Series) -> float:
        """Calculate how consistent a trend is"""
        try:
            if len(trend_series) == 0:
                return 0.0
            
            # Calculate what percentage of values have the same sign
            positive_count = (trend_series > 0).sum()
            negative_count = (trend_series < 0).sum()
            total_count = len(trend_series)
            
            consistency = max(positive_count, negative_count) / total_count
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.0
    
    def _detect_obv_divergences(self, price_indicators: Dict, volume_indicators: Dict) -> List[Dict]:
        """Detect divergences between price and OBV"""
        try:
            divergences = []
            
            if 'obv_smoothed' not in volume_indicators:
                return divergences
            
            price_series = price_indicators['price_sma']
            obv_series = volume_indicators['obv_smoothed']
            
            # Find recent highs and lows
            recent_periods = min(self.period, len(price_series))
            
            for i in range(recent_periods, len(price_series)):
                # Look back for comparison points
                lookback_start = max(0, i - recent_periods)
                
                price_trend = self._calculate_trend_direction(price_series[lookback_start:i+1])
                obv_trend = self._calculate_trend_direction(obv_series[lookback_start:i+1])
                
                # Check for divergence
                if price_trend == 'up' and obv_trend == 'down':
                    # Bearish divergence
                    strength = self._calculate_obv_divergence_strength(price_series, obv_series, i, 'bearish')
                    if strength >= self.divergence_threshold:
                        divergences.append({
                            'index': i,
                            'type': 'bearish_obv_divergence',
                            'price_trend': price_trend,
                            'obv_trend': obv_trend,
                            'strength': strength,
                            'confidence': 0.7,
                            'timestamp': price_series.index[i] if hasattr(price_series.index, 'to_list') else i
                        })
                        
                elif price_trend == 'down' and obv_trend == 'up':
                    # Bullish divergence
                    strength = self._calculate_obv_divergence_strength(price_series, obv_series, i, 'bullish')
                    if strength >= self.divergence_threshold:
                        divergences.append({
                            'index': i,
                            'type': 'bullish_obv_divergence',
                            'price_trend': price_trend,
                            'obv_trend': obv_trend,
                            'strength': strength,
                            'confidence': 0.7,
                            'timestamp': price_series.index[i] if hasattr(price_series.index, 'to_list') else i
                        })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting OBV divergences: {str(e)}")
            return []
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate overall trend direction of a series"""
        try:
            if len(series) < 2:
                return 'neutral'
            
            start_value = series.iloc[0]
            end_value = series.iloc[-1]
            
            change_pct = (end_value - start_value) / abs(start_value) if start_value != 0 else 0
            
            if change_pct > 0.02:  # 2% threshold
                return 'up'
            elif change_pct < -0.02:
                return 'down'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error calculating trend direction: {str(e)}")
            return 'neutral'
    
    def _calculate_obv_divergence_strength(self, price_series: pd.Series, obv_series: pd.Series, 
                                         index: int, divergence_type: str) -> float:
        """Calculate OBV divergence strength"""
        try:
            lookback = min(self.period, index)
            
            price_change = (price_series.iloc[index] - price_series.iloc[index-lookback]) / price_series.iloc[index-lookback]
            obv_change = (obv_series.iloc[index] - obv_series.iloc[index-lookback]) / abs(obv_series.iloc[index-lookback])
            
            # Strength is based on how opposite the trends are
            if divergence_type == 'bearish':
                strength = price_change - obv_change  # Positive when price up and OBV down
            else:  # bullish
                strength = obv_change - price_change  # Positive when OBV up and price down
            
            return max(0, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV divergence strength: {str(e)}")
            return 0.0
    
    def _detect_volume_spike_divergences(self, price_indicators: Dict, volume_indicators: Dict) -> List[Dict]:
        """Detect divergences involving volume spikes"""
        try:
            divergences = []
            
            volume_series = volume_indicators['volume_sma']
            volume_ma = volume_indicators['volume_ema']
            price_series = price_indicators['price_sma']
            
            for i in range(self.period, len(volume_series)):
                # Check for volume spike
                current_volume = volume_series.iloc[i]
                avg_volume = volume_ma.iloc[i]
                
                if current_volume > avg_volume * self.volume_spike_threshold:
                    # Volume spike detected
                    
                    # Check corresponding price movement
                    price_change = (price_series.iloc[i] - price_series.iloc[i-1]) / price_series.iloc[i-1]
                    
                    # Divergence if high volume but small price movement
                    if abs(price_change) < 0.01:  # Less than 1% price movement
                        divergences.append({
                            'index': i,
                            'type': 'volume_spike_divergence',
                            'volume_ratio': current_volume / avg_volume,
                            'price_change': price_change,
                            'strength': (current_volume / avg_volume) * (1 - abs(price_change)),
                            'confidence': 0.6,
                            'timestamp': volume_series.index[i] if hasattr(volume_series.index, 'to_list') else i
                        })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting volume spike divergences: {str(e)}")
            return []
    
    def _analyze_current_pv_state(self, price_indicators: Dict, volume_indicators: Dict) -> Dict:
        """Analyze current price-volume relationship state"""
        try:
            latest_index = -1  # Most recent data point
            
            # Current trends
            current_price_trend = self._calculate_trend_direction(
                price_indicators['price_trend_strength'][-self.trend_confirmation_period:]
            )
            current_volume_trend = self._calculate_trend_direction(
                volume_indicators['vroc'][-self.trend_confirmation_period:]
            )
            
            # Volume confirmation strength
            volume_confirmation = self._calculate_volume_confirmation_strength(volume_indicators)
            
            # Overall relationship health
            relationship_health = self._assess_pv_relationship_health(
                current_price_trend, current_volume_trend, volume_confirmation
            )
            
            return {
                'price_trend': current_price_trend,
                'volume_trend': current_volume_trend,
                'volume_confirmation': volume_confirmation,
                'relationship_health': relationship_health,
                'is_divergent': self._is_divergent(current_price_trend, current_volume_trend)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing current PV state: {str(e)}")
            return {}
    
    def _calculate_volume_confirmation_strength(self, volume_indicators: Dict) -> float:
        """Calculate how well volume is confirming price movements"""
        try:
            # Use multiple volume indicators for confirmation
            obv_trend = self._calculate_trend_direction(volume_indicators['obv_smoothed'][-self.period:])
            vroc_trend = self._calculate_trend_direction(volume_indicators['vroc'][-self.period:])
            cmf_value = volume_indicators['cmf'].iloc[-1] if not volume_indicators['cmf'].empty else 0
            
            confirmation_score = 0.0
            
            # OBV confirmation
            if obv_trend in ['up', 'down']:
                confirmation_score += 0.3
            
            # VROC confirmation
            if vroc_trend in ['up', 'down']:
                confirmation_score += 0.3
            
            # CMF confirmation
            if abs(cmf_value) > 0.1:
                confirmation_score += 0.4
            
            return confirmation_score
            
        except Exception as e:
            self.logger.error(f"Error calculating volume confirmation strength: {str(e)}")
            return 0.0
    
    def _assess_pv_relationship_health(self, price_trend: str, volume_trend: str, volume_confirmation: float) -> str:
        """Assess the health of price-volume relationship"""
        try:
            # Healthy relationships
            if ((price_trend == 'up' and volume_trend == 'up') or 
                (price_trend == 'down' and volume_trend == 'down')):
                if volume_confirmation > 0.6:
                    return 'very_healthy'
                else:
                    return 'healthy'
            
            # Neutral relationships
            elif price_trend == 'neutral' or volume_trend == 'neutral':
                return 'neutral'
            
            # Unhealthy relationships (divergent)
            else:
                if volume_confirmation < 0.3:
                    return 'very_unhealthy'
                else:
                    return 'unhealthy'
                    
        except Exception as e:
            self.logger.error(f"Error assessing PV relationship health: {str(e)}")
            return 'unknown'
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate PriceVolumeDivergence indicator
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing price-volume divergence analysis
        """
        try:
            if not self.validate_data(data):
                return {}
            
            # Calculate volume indicators
            volume_indicators = self._calculate_volume_indicators(data)
            self.volume_indicators = volume_indicators
            
            # Calculate price indicators
            price_indicators = self._calculate_price_indicators(data)
            
            if not volume_indicators or not price_indicators:
                return {
                    'divergences': [],
                    'analysis_complete': False,
                    'error': 'Failed to calculate indicators'
                }
            
            # Detect price-volume divergences
            divergences = self._detect_price_volume_divergences(price_indicators, volume_indicators)
            self.price_volume_divergences = divergences
            
            # Analyze current state
            current_state = self._analyze_current_pv_state(price_indicators, volume_indicators)
            
            # Categorize divergences
            divergence_types = {
                'bearish_pv': [d for d in divergences if d['type'] == 'bearish_pv_divergence'],
                'bullish_pv': [d for d in divergences if d['type'] == 'bullish_pv_divergence'],
                'bearish_obv': [d for d in divergences if d['type'] == 'bearish_obv_divergence'],
                'bullish_obv': [d for d in divergences if d['type'] == 'bullish_obv_divergence'],
                'volume_spike': [d for d in divergences if d['type'] == 'volume_spike_divergence']
            }
            
            # Calculate summary statistics
            total_divergences = len(divergences)
            avg_strength = np.mean([d['strength'] for d in divergences]) if divergences else 0.0
            strongest_divergence = max(divergences, key=lambda x: x['strength']) if divergences else None
            
            result = {
                'total_divergences': total_divergences,
                'divergences': divergences,
                'divergence_types': divergence_types,
                'current_state': current_state,
                'average_strength': avg_strength,
                'strongest_divergence': strongest_divergence,
                'volume_indicators': {
                    'obv_current': volume_indicators['obv'].iloc[-1] if not volume_indicators['obv'].empty else None,
                    'cmf_current': volume_indicators['cmf'].iloc[-1] if not volume_indicators['cmf'].empty else None,
                    'vroc_current': volume_indicators['vroc'].iloc[-1] if not volume_indicators['vroc'].empty else None
                },
                'analysis_complete': True,
                'recent_divergences': divergences[-5:] if len(divergences) > 5 else divergences
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating PriceVolumeDivergence: {str(e)}")
            return {}
    
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
                'volume_ma_period': self.volume_ma_period,
                'price_ma_period': self.price_ma_period,
                'divergence_threshold': self.divergence_threshold,
                'volume_spike_threshold': self.volume_spike_threshold,
                'trend_confirmation_period': self.trend_confirmation_period,
                'obv_smoothing': self.obv_smoothing,
                'vroc_period': self.vroc_period
            },
            'data_requirements': ['high', 'low', 'close', 'volume'],
            'output_format': 'price_volume_divergence_analysis'
        }
    def validate_parameters(self) -> bool:
        """Validate parameters"""
        # Add specific validation logic as needed
        return True



def export() -> Dict[str, Any]:
    """
    Export function for the PriceVolumeDivergence indicator.
    
    This function is used by the indicator registry to discover and load the indicator.
    
    Returns:
        Dictionary containing indicator information for registry
    """
    return {
        'class': PriceVolumeDivergenceIndicator,
        'name': 'PriceVolumeDivergence',
        'category': 'divergence',
        'version': '1.0.0',
        'description': 'Advanced price-volume divergence analysis with comprehensive volume indicators',
        'complexity': 'advanced',
        'parameters': {
            'period': {'type': 'int', 'default': 20, 'min': 5, 'max': 100},
            'volume_ma_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 50},
            'price_ma_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 50},
            'divergence_threshold': {'type': 'float', 'default': 0.03, 'min': 0.001, 'max': 0.1},
            'volume_spike_threshold': {'type': 'float', 'default': 2.0, 'min': 1.1, 'max': 5.0},
            'trend_confirmation_period': {'type': 'int', 'default': 5, 'min': 3, 'max': 15},
            'obv_smoothing': {'type': 'int', 'default': 3, 'min': 1, 'max': 10},
            'vroc_period': {'type': 'int', 'default': 10, 'min': 5, 'max': 30}
        },
        'data_requirements': ['high', 'low', 'close', 'volume'],
        'output_type': 'divergence_analysis'
    }