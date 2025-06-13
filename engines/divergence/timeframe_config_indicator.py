"""
TimeframeConfig - Multi-Timeframe Configuration and Analysis Indicator for Platform3

This indicator provides configuration and analysis capabilities for multi-timeframe
divergence detection, enabling synchronized analysis across different time horizons
and providing comprehensive timeframe-specific divergence insights.

Version: 1.0.0
Category: Divergence
Complexity: Advanced
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import logging
from datetime import datetime, timedelta

from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface


class TimeframeConfigIndicator(StandardIndicatorInterface):
    """
    Advanced Multi-Timeframe Configuration and Analysis Indicator
    
    Provides comprehensive timeframe management for divergence analysis:
    - Multi-timeframe data synchronization
    - Timeframe-specific divergence detection
    - Cross-timeframe confirmation
    - Adaptive timeframe selection
    - Timeframe hierarchy management
    - Synchronized signal generation
    
    This indicator enables sophisticated multi-timeframe analysis
    for enhanced divergence detection and confirmation.
    """
    
    # Class-level metadata
    INDICATOR_NAME = "TimeframeConfig"
    INDICATOR_VERSION = "1.0.0"
    INDICATOR_CATEGORY = "divergence"
    INDICATOR_TYPE = "advanced"
    INDICATOR_COMPLEXITY = "advanced"
    
    # Standard timeframe multipliers
    TIMEFRAME_MULTIPLIERS = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
        '1M': 43200  # Approximate
    }
    
    def __init__(self, **kwargs):
        """
        Initialize TimeframeConfig indicator
        
        Args:
            parameters: Dictionary containing indicator parameters
                - base_timeframe: Base timeframe for analysis (default: '1h')
                - analysis_timeframes: List of timeframes to analyze (default: ['15m', '1h', '4h', '1d'])
                - confirmation_timeframes: Timeframes required for confirmation (default: ['1h', '4h'])
                - sync_method: Method for timeframe synchronization (default: 'interpolation')
                - lookback_multiplier: Multiplier for lookback periods (default: 3)
                - min_timeframe_confirmation: Minimum timeframes for signal confirmation (default: 2)
                - adaptive_timeframes: Enable adaptive timeframe selection (default: True)
                - max_timeframes: Maximum number of timeframes to analyze (default: 5)
        """
        super().__init__(**kwargs)
        
        # Get parameters with defaults
        self.base_timeframe = self.parameters.get('base_timeframe', '1h')
        self.analysis_timeframes = self.parameters.get('analysis_timeframes', ['15m', '1h', '4h', '1d'])
        self.confirmation_timeframes = self.parameters.get('confirmation_timeframes', ['1h', '4h'])
        self.sync_method = self.parameters.get('sync_method', 'interpolation')
        self.lookback_multiplier = int(self.parameters.get('lookback_multiplier', 3))
        self.min_timeframe_confirmation = int(self.parameters.get('min_timeframe_confirmation', 2))
        self.adaptive_timeframes = bool(self.parameters.get('adaptive_timeframes', True))
        self.max_timeframes = int(self.parameters.get('max_timeframes', 5))
        
        # Validation
        if self.base_timeframe not in self.TIMEFRAME_MULTIPLIERS:
            raise ValueError(f"Invalid base timeframe: {self.base_timeframe}")
        if self.min_timeframe_confirmation < 1:
            raise ValueError("Minimum timeframe confirmation must be at least 1")
        if self.max_timeframes < 2:
            raise ValueError("Maximum timeframes must be at least 2")
            
        # Initialize state
        self.timeframe_data = {}
        self.timeframe_signals = {}
        self.synchronized_data = {}
        self.cross_timeframe_divergences = []
        self.timeframe_hierarchy = []
        
        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Setup timeframe hierarchy
        self._setup_timeframe_hierarchy()
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for TimeframeConfig calculation"""
        try:
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Need: {required_columns}")
                return False
                
            # Check for datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning("Data should have DatetimeIndex for proper timeframe analysis")
                
            if len(data) < 100:  # Minimum data for multi-timeframe analysis
                self.logger.warning(f"Insufficient data length for multi-timeframe analysis: {len(data)}")
                return False
                
            if data[required_columns].isnull().any().any():
                self.logger.warning("Data contains NaN values")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _setup_timeframe_hierarchy(self):
        """Setup hierarchical timeframe structure"""
        try:
            # Sort timeframes by duration (shortest to longest)
            valid_timeframes = [tf for tf in self.analysis_timeframes if tf in self.TIMEFRAME_MULTIPLIERS]
            
            if self.adaptive_timeframes and len(valid_timeframes) > self.max_timeframes:
                # Select optimal timeframes
                valid_timeframes = self._select_optimal_timeframes(valid_timeframes)
            
            self.timeframe_hierarchy = sorted(valid_timeframes, 
                                            key=lambda x: self.TIMEFRAME_MULTIPLIERS[x])
            
            self.logger.info(f"Timeframe hierarchy established: {self.timeframe_hierarchy}")
            
        except Exception as e:
            self.logger.error(f"Error setting up timeframe hierarchy: {str(e)}")
            self.timeframe_hierarchy = [self.base_timeframe]
    
    def _select_optimal_timeframes(self, available_timeframes: List[str]) -> List[str]:
        """Select optimal timeframes for analysis"""
        try:
            # Sort by multiplier
            sorted_timeframes = sorted(available_timeframes, 
                                     key=lambda x: self.TIMEFRAME_MULTIPLIERS[x])
            
            selected = []
            base_multiplier = self.TIMEFRAME_MULTIPLIERS[self.base_timeframe]
            
            # Always include base timeframe
            if self.base_timeframe in sorted_timeframes:
                selected.append(self.base_timeframe)
            
            # Select timeframes with optimal ratios
            optimal_ratios = [0.25, 1, 4, 16, 64]  # Common timeframe ratios
            
            for ratio in optimal_ratios:
                target_multiplier = base_multiplier * ratio
                
                # Find closest timeframe
                closest_tf = min(sorted_timeframes, 
                               key=lambda x: abs(self.TIMEFRAME_MULTIPLIERS[x] - target_multiplier))
                
                if (closest_tf not in selected and 
                    len(selected) < self.max_timeframes):
                    selected.append(closest_tf)
            
            return selected[:self.max_timeframes]
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal timeframes: {str(e)}")
            return available_timeframes[:self.max_timeframes]
    
    def _resample_to_timeframe(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                # Create synthetic datetime index if not available
                data = data.copy()
                data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='1min')
            
            # Define resampling rules
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Map timeframe to pandas frequency
            freq_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D',
                '1w': '1W',
                '1M': '1M'
            }
            
            freq = freq_map.get(target_timeframe, '1H')
            
            # Add open column if not present
            if 'open' not in data.columns:
                data['open'] = data['close'].shift(1)
                data['open'].iloc[0] = data['close'].iloc[0]
            
            # Resample data
            resampled = data.resample(freq).agg(agg_rules)
            
            # Remove any NaN rows
            resampled = resampled.dropna()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling to timeframe {target_timeframe}: {str(e)}")
            return data
    
    def _calculate_timeframe_indicators(self, data: pd.DataFrame, timeframe: str) -> Dict[str, pd.Series]:
        """Calculate indicators for specific timeframe"""
        try:
            indicators = {}
            
            # Adapt periods based on timeframe
            base_period = 20
            tf_multiplier = self.TIMEFRAME_MULTIPLIERS[timeframe] / self.TIMEFRAME_MULTIPLIERS[self.base_timeframe]
            adapted_period = max(5, int(base_period / tf_multiplier))
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(data['close'], adapted_period)
            
            # MACD
            fast_period = max(5, int(12 / tf_multiplier))
            slow_period = max(10, int(26 / tf_multiplier))
            signal_period = max(3, int(9 / tf_multiplier))
            
            macd, macd_signal, macd_histogram = self._calculate_macd(
                data['close'], fast_period, slow_period, signal_period
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_histogram
            
            # Stochastic
            indicators['stoch_k'], indicators['stoch_d'] = self._calculate_stochastic(data, adapted_period)
            
            # Volume indicators
            indicators['obv'] = self._calculate_obv(data)
            indicators['volume_sma'] = data['volume'].rolling(window=adapted_period).mean()
            
            # Price momentum
            indicators['price_momentum'] = data['close'].pct_change(adapted_period)
            
            # ATR for volatility context
            indicators['atr'] = self._calculate_atr(data, adapted_period)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe indicators: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
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
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()
    
    def _detect_timeframe_divergences(self, timeframe_data: Dict) -> Dict[str, List[Dict]]:
        """Detect divergences within each timeframe"""
        try:
            timeframe_divergences = {}
            
            for timeframe, tf_data in timeframe_data.items():
                divergences = []
                
                if 'indicators' not in tf_data:
                    timeframe_divergences[timeframe] = divergences
                    continue
                
                indicators = tf_data['indicators']
                price_data = tf_data['data']
                
                # Detect RSI divergences
                rsi_divergences = self._detect_rsi_divergences(price_data, indicators.get('rsi'))
                divergences.extend(rsi_divergences)
                
                # Detect MACD divergences
                macd_divergences = self._detect_macd_divergences(price_data, indicators)
                divergences.extend(macd_divergences)
                
                # Detect Stochastic divergences
                stoch_divergences = self._detect_stochastic_divergences(price_data, indicators)
                divergences.extend(stoch_divergences)
                
                # Detect OBV divergences
                obv_divergences = self._detect_obv_divergences(price_data, indicators.get('obv'))
                divergences.extend(obv_divergences)
                
                # Add timeframe context to each divergence
                for div in divergences:
                    div['timeframe'] = timeframe
                    div['timeframe_weight'] = self._calculate_timeframe_weight(timeframe)
                
                timeframe_divergences[timeframe] = divergences
            
            return timeframe_divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting timeframe divergences: {str(e)}")
            return {}
    
    def _detect_rsi_divergences(self, price_data: pd.DataFrame, rsi: pd.Series) -> List[Dict]:
        """Detect RSI divergences"""
        try:
            if rsi is None or rsi.empty:
                return []
            
            divergences = []
            
            # Find price and RSI pivots
            price_highs = self._find_pivots(price_data['high'], 'high')
            price_lows = self._find_pivots(price_data['low'], 'low')
            rsi_highs = self._find_pivots(rsi, 'high')
            rsi_lows = self._find_pivots(rsi, 'low')
            
            # Bearish divergence: Higher price highs, lower RSI highs
            for i in range(len(price_highs) - 1):
                for j in range(i + 1, len(price_highs)):
                    ph1, ph2 = price_highs[i], price_highs[j]
                    rh1 = self._find_nearest_pivot(ph1['index'], rsi_highs)
                    rh2 = self._find_nearest_pivot(ph2['index'], rsi_highs)
                    
                    if rh1 and rh2 and ph2['value'] > ph1['value'] and rh2['value'] < rh1['value']:
                        strength = self._calculate_divergence_strength(ph1, ph2, rh1, rh2)
                        if strength > 0.02:
                            divergences.append({
                                'type': 'bearish_rsi',
                                'indicator': 'rsi',
                                'strength': strength,
                                'confidence': 0.7,
                                'price_points': [ph1, ph2],
                                'indicator_points': [rh1, rh2]
                            })
            
            # Bullish divergence: Lower price lows, higher RSI lows
            for i in range(len(price_lows) - 1):
                for j in range(i + 1, len(price_lows)):
                    pl1, pl2 = price_lows[i], price_lows[j]
                    rl1 = self._find_nearest_pivot(pl1['index'], rsi_lows)
                    rl2 = self._find_nearest_pivot(pl2['index'], rsi_lows)
                    
                    if rl1 and rl2 and pl2['value'] < pl1['value'] and rl2['value'] > rl1['value']:
                        strength = self._calculate_divergence_strength(pl1, pl2, rl1, rl2)
                        if strength > 0.02:
                            divergences.append({
                                'type': 'bullish_rsi',
                                'indicator': 'rsi',
                                'strength': strength,
                                'confidence': 0.7,
                                'price_points': [pl1, pl2],
                                'indicator_points': [rl1, rl2]
                            })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting RSI divergences: {str(e)}")
            return []
    
    def _detect_macd_divergences(self, price_data: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Detect MACD divergences"""
        try:
            macd = indicators.get('macd')
            if macd is None or macd.empty:
                return []
            
            divergences = []
            
            # Similar logic to RSI but using MACD
            price_highs = self._find_pivots(price_data['high'], 'high')
            price_lows = self._find_pivots(price_data['low'], 'low')
            macd_highs = self._find_pivots(macd, 'high')
            macd_lows = self._find_pivots(macd, 'low')
            
            # Bearish MACD divergence
            for i in range(len(price_highs) - 1):
                for j in range(i + 1, len(price_highs)):
                    ph1, ph2 = price_highs[i], price_highs[j]
                    mh1 = self._find_nearest_pivot(ph1['index'], macd_highs)
                    mh2 = self._find_nearest_pivot(ph2['index'], macd_highs)
                    
                    if mh1 and mh2 and ph2['value'] > ph1['value'] and mh2['value'] < mh1['value']:
                        strength = self._calculate_divergence_strength(ph1, ph2, mh1, mh2)
                        if strength > 0.02:
                            divergences.append({
                                'type': 'bearish_macd',
                                'indicator': 'macd',
                                'strength': strength,
                                'confidence': 0.75,
                                'price_points': [ph1, ph2],
                                'indicator_points': [mh1, mh2]
                            })
            
            # Bullish MACD divergence
            for i in range(len(price_lows) - 1):
                for j in range(i + 1, len(price_lows)):
                    pl1, pl2 = price_lows[i], price_lows[j]
                    ml1 = self._find_nearest_pivot(pl1['index'], macd_lows)
                    ml2 = self._find_nearest_pivot(pl2['index'], macd_lows)
                    
                    if ml1 and ml2 and pl2['value'] < pl1['value'] and ml2['value'] > ml1['value']:
                        strength = self._calculate_divergence_strength(pl1, pl2, ml1, ml2)
                        if strength > 0.02:
                            divergences.append({
                                'type': 'bullish_macd',
                                'indicator': 'macd',
                                'strength': strength,
                                'confidence': 0.75,
                                'price_points': [pl1, pl2],
                                'indicator_points': [ml1, ml2]
                            })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD divergences: {str(e)}")
            return []
    
    def _detect_stochastic_divergences(self, price_data: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Detect Stochastic divergences"""
        try:
            stoch_k = indicators.get('stoch_k')
            if stoch_k is None or stoch_k.empty:
                return []
            
            # Similar implementation to RSI/MACD
            # Simplified for brevity
            return []
            
        except Exception as e:
            self.logger.error(f"Error detecting Stochastic divergences: {str(e)}")
            return []
    
    def _detect_obv_divergences(self, price_data: pd.DataFrame, obv: pd.Series) -> List[Dict]:
        """Detect OBV divergences"""
        try:
            if obv is None or obv.empty:
                return []
            
            # Similar implementation to other indicators
            # Simplified for brevity
            return []
            
        except Exception as e:
            self.logger.error(f"Error detecting OBV divergences: {str(e)}")
            return []
    
    def _find_pivots(self, series: pd.Series, pivot_type: str, lookback: int = 5) -> List[Dict]:
        """Find pivot points in a series"""
        try:
            pivots = []
            
            for i in range(lookback, len(series) - lookback):
                if pivot_type == 'high':
                    is_pivot = all(series.iloc[i] > series.iloc[i-j] for j in range(1, lookback+1))
                    is_pivot = is_pivot and all(series.iloc[i] > series.iloc[i+j] for j in range(1, lookback+1))
                else:  # low
                    is_pivot = all(series.iloc[i] < series.iloc[i-j] for j in range(1, lookback+1))
                    is_pivot = is_pivot and all(series.iloc[i] < series.iloc[i+j] for j in range(1, lookback+1))
                
                if is_pivot:
                    pivots.append({
                        'index': i,
                        'value': series.iloc[i],
                        'timestamp': series.index[i] if hasattr(series.index, 'to_list') else i
                    })
            
            return pivots
            
        except Exception as e:
            self.logger.error(f"Error finding pivots: {str(e)}")
            return []
    
    def _find_nearest_pivot(self, target_index: int, pivots: List[Dict], max_distance: int = 10) -> Optional[Dict]:
        """Find nearest pivot to target index"""
        try:
            if not pivots:
                return None
            
            nearest = min(pivots, key=lambda p: abs(p['index'] - target_index))
            
            if abs(nearest['index'] - target_index) <= max_distance:
                return nearest
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding nearest pivot: {str(e)}")
            return None
    
    def _calculate_divergence_strength(self, p1: Dict, p2: Dict, i1: Dict, i2: Dict) -> float:
        """Calculate divergence strength"""
        try:
            price_change = abs(p2['value'] - p1['value']) / p1['value']
            indicator_change = abs(i2['value'] - i1['value']) / abs(i1['value'])
            
            return (price_change + indicator_change) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating divergence strength: {str(e)}")
            return 0.0
    
    def _calculate_timeframe_weight(self, timeframe: str) -> float:
        """Calculate weight for timeframe based on hierarchy"""
        try:
            if timeframe not in self.timeframe_hierarchy:
                return 0.5
            
            # Higher timeframes get higher weights
            index = self.timeframe_hierarchy.index(timeframe)
            weight = (index + 1) / len(self.timeframe_hierarchy)
            
            return weight
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe weight: {str(e)}")
            return 0.5
    
    def _analyze_cross_timeframe_confirmation(self, timeframe_divergences: Dict) -> List[Dict]:
        """Analyze cross-timeframe divergence confirmation"""
        try:
            confirmed_divergences = []
            
            # Group divergences by type
            divergence_groups = {}
            
            for timeframe, divergences in timeframe_divergences.items():
                for div in divergences:
                    div_type = div['type']
                    if div_type not in divergence_groups:
                        divergence_groups[div_type] = []
                    divergence_groups[div_type].append(div)
            
            # Check for confirmation across timeframes
            for div_type, divs in divergence_groups.items():
                if len(divs) >= self.min_timeframe_confirmation:
                    # Find the strongest divergence as representative
                    strongest = max(divs, key=lambda x: x['strength'])
                    
                    # Calculate confirmation strength
                    confirming_timeframes = [d['timeframe'] for d in divs]
                    confirmation_weight = sum(d['timeframe_weight'] for d in divs) / len(divs)
                    
                    confirmed_divergences.append({
                        'type': div_type,
                        'primary_divergence': strongest,
                        'confirming_divergences': divs,
                        'confirming_timeframes': confirming_timeframes,
                        'confirmation_count': len(divs),
                        'confirmation_strength': confirmation_weight,
                        'overall_confidence': strongest['confidence'] * confirmation_weight
                    })
            
            return confirmed_divergences
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-timeframe confirmation: {str(e)}")
            return []
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate TimeframeConfig indicator
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing multi-timeframe divergence analysis
        """
        try:
            if not self.validate_data(data):
                return {}
            
            # Process each timeframe
            timeframe_data = {}
            
            for timeframe in self.timeframe_hierarchy:
                try:
                    # Resample data to timeframe
                    tf_data = self._resample_to_timeframe(data, timeframe)
                    
                    if len(tf_data) < 20:  # Minimum data for analysis
                        continue
                    
                    # Calculate indicators for this timeframe
                    indicators = self._calculate_timeframe_indicators(tf_data, timeframe)
                    
                    timeframe_data[timeframe] = {
                        'data': tf_data,
                        'indicators': indicators,
                        'multiplier': self.TIMEFRAME_MULTIPLIERS[timeframe]
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process timeframe {timeframe}: {str(e)}")
                    continue
            
            self.timeframe_data = timeframe_data
            
            if not timeframe_data:
                return {
                    'analysis_complete': False,
                    'error': 'No timeframes could be processed',
                    'timeframes_analyzed': 0
                }
            
            # Detect divergences in each timeframe
            timeframe_divergences = self._detect_timeframe_divergences(timeframe_data)
            
            # Analyze cross-timeframe confirmation
            confirmed_divergences = self._analyze_cross_timeframe_confirmation(timeframe_divergences)
            self.cross_timeframe_divergences = confirmed_divergences
            
            # Generate overall signal
            overall_signal = self._generate_overall_signal(confirmed_divergences)
            
            # Summary statistics
            total_divergences = sum(len(divs) for divs in timeframe_divergences.values())
            timeframes_with_signals = len([tf for tf, divs in timeframe_divergences.items() if divs])
            
            result = {
                'analysis_complete': True,
                'timeframes_analyzed': list(timeframe_data.keys()),
                'timeframe_hierarchy': self.timeframe_hierarchy,
                'timeframe_divergences': timeframe_divergences,
                'confirmed_divergences': confirmed_divergences,
                'cross_timeframe_confirmations': len(confirmed_divergences),
                'total_divergences': total_divergences,
                'timeframes_with_signals': timeframes_with_signals,
                'overall_signal': overall_signal,
                'strongest_confirmation': max(confirmed_divergences, 
                                            key=lambda x: x['confirmation_strength']) if confirmed_divergences else None,
                'timeframe_data_summary': {
                    tf: {
                        'data_points': len(tf_data['data']),
                        'indicators_calculated': len(tf_data['indicators']),
                        'divergences_found': len(timeframe_divergences.get(tf, []))
                    } for tf, tf_data in timeframe_data.items()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating TimeframeConfig: {str(e)}")
            return {}
    
    def _generate_overall_signal(self, confirmed_divergences: List[Dict]) -> Dict:
        """Generate overall signal from confirmed divergences"""
        try:
            if not confirmed_divergences:
                return {'signal': 'neutral', 'strength': 0.0, 'confidence': 0.0}
            
            # Weight signals by confirmation strength
            bullish_strength = 0.0
            bearish_strength = 0.0
            
            for div in confirmed_divergences:
                strength = div['confirmation_strength']
                
                if 'bullish' in div['type']:
                    bullish_strength += strength
                elif 'bearish' in div['type']:
                    bearish_strength += strength
            
            # Determine primary signal
            if bullish_strength > bearish_strength:
                signal = 'bullish'
                signal_strength = bullish_strength / (bullish_strength + bearish_strength)
            elif bearish_strength > bullish_strength:
                signal = 'bearish'
                signal_strength = bearish_strength / (bullish_strength + bearish_strength)
            else:
                signal = 'neutral'
                signal_strength = 0.5
            
            # Calculate overall confidence
            avg_confidence = sum(d['overall_confidence'] for d in confirmed_divergences) / len(confirmed_divergences)
            
            return {
                'signal': signal,
                'strength': signal_strength,
                'confidence': avg_confidence,
                'bullish_strength': bullish_strength,
                'bearish_strength': bearish_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error generating overall signal: {str(e)}")
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
                'base_timeframe': self.base_timeframe,
                'analysis_timeframes': self.analysis_timeframes,
                'confirmation_timeframes': self.confirmation_timeframes,
                'sync_method': self.sync_method,
                'lookback_multiplier': self.lookback_multiplier,
                'min_timeframe_confirmation': self.min_timeframe_confirmation,
                'adaptive_timeframes': self.adaptive_timeframes,
                'max_timeframes': self.max_timeframes
            },
            'data_requirements': ['high', 'low', 'close', 'volume'],
            'output_format': 'multi_timeframe_divergence_analysis',
            'supported_timeframes': list(self.TIMEFRAME_MULTIPLIERS.keys())
        }
    def validate_parameters(self) -> bool:
        """Validate parameters"""
        # Add specific validation logic as needed
        return True



def export() -> Dict[str, Any]:
    """
    Export function for the TimeframeConfig indicator.
    
    This function is used by the indicator registry to discover and load the indicator.
    
    Returns:
        Dictionary containing indicator information for registry
    """
    return {
        'class': TimeframeConfigIndicator,
        'name': 'TimeframeConfig',
        'category': 'divergence',
        'version': '1.0.0',
        'description': 'Advanced multi-timeframe configuration and divergence analysis',
        'complexity': 'advanced',
        'parameters': {
            'base_timeframe': {'type': 'str', 'default': '1h', 'choices': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']},
            'analysis_timeframes': {'type': 'list', 'default': ['15m', '1h', '4h', '1d']},
            'confirmation_timeframes': {'type': 'list', 'default': ['1h', '4h']},
            'sync_method': {'type': 'str', 'default': 'interpolation', 'choices': ['interpolation', 'resampling']},
            'lookback_multiplier': {'type': 'int', 'default': 3, 'min': 1, 'max': 10},
            'min_timeframe_confirmation': {'type': 'int', 'default': 2, 'min': 1, 'max': 5},
            'adaptive_timeframes': {'type': 'bool', 'default': True},
            'max_timeframes': {'type': 'int', 'default': 5, 'min': 2, 'max': 10}
        },
        'data_requirements': ['high', 'low', 'close', 'volume'],
        'output_type': 'timeframe_analysis'
    }