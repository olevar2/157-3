"""
MomentumDivergenceScanner - Momentum Divergence Detection Indicator for Platform3

This indicator scans for momentum divergences across multiple timeframes and
technical indicators, providing comprehensive divergence analysis for both
regular and hidden divergences with advanced filtering and confirmation.

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


class MomentumDivergenceScannerIndicator(StandardIndicatorInterface):
    """
    Advanced Momentum Divergence Scanner Indicator
    
    Scans for momentum divergences across multiple indicators and timeframes:
    - Regular divergences (price vs momentum direction mismatch)
    - Hidden divergences (trend continuation signals)
    - Multi-indicator confirmation
    - Divergence strength scoring
    - Real-time divergence monitoring
    - False signal filtering
    
    This scanner provides comprehensive momentum divergence analysis
    for advanced trading decision support.
    """
    
    # Class-level metadata
    INDICATOR_NAME = "MomentumDivergenceScanner"
    INDICATOR_VERSION = "1.0.0"
    INDICATOR_CATEGORY = "divergence"
    INDICATOR_TYPE = "advanced"
    INDICATOR_COMPLEXITY = "advanced"
    
    def __init__(self, **kwargs):
        """
        Initialize MomentumDivergenceScanner indicator
        
        Args:
            parameters: Dictionary containing indicator parameters
                - scan_period: Main scanning period (default: 21)
                - indicator_periods: List of momentum indicator periods (default: [9, 14, 21, 34])
                - min_divergence_strength: Minimum divergence strength (default: 0.025)
                - confirmation_threshold: Number of confirming indicators (default: 2)
                - pivot_lookback: Pivot detection lookback (default: 5)
                - max_divergence_age: Maximum age of divergence in periods (default: 10)
                - volume_weight: Weight given to volume confirmation (default: 0.3)
                - trend_filter: Enable trend filtering (default: True)
        """
        super().__init__(**kwargs)
        
        # Get parameters with defaults
        self.scan_period = int(self.parameters.get('scan_period', 21))
        self.indicator_periods = self.parameters.get('indicator_periods', [9, 14, 21, 34])
        self.min_divergence_strength = float(self.parameters.get('min_divergence_strength', 0.025))
        self.confirmation_threshold = int(self.parameters.get('confirmation_threshold', 2))
        self.pivot_lookback = int(self.parameters.get('pivot_lookback', 5))
        self.max_divergence_age = int(self.parameters.get('max_divergence_age', 10))
        self.volume_weight = float(self.parameters.get('volume_weight', 0.3))
        self.trend_filter = bool(self.parameters.get('trend_filter', True))
        
        # Validation
        if self.scan_period < 10:
            raise ValueError("Scan period must be at least 10")
        if self.min_divergence_strength <= 0:
            raise ValueError("Minimum divergence strength must be positive")
        if not 0 <= self.volume_weight <= 1:
            raise ValueError("Volume weight must be between 0 and 1")
            
        # Initialize state
        self.momentum_data = {}
        self.divergence_history = []
        self.current_scan_results = {}
        self.trend_state = 'neutral'
        
        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for MomentumDivergenceScanner calculation"""
        try:
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Need: {required_columns}")
                return False
                
            if len(data) < max(self.indicator_periods) + self.scan_period:
                self.logger.warning(f"Insufficient data length: {len(data)} < {max(self.indicator_periods) + self.scan_period}")
                return False
                
            if data[required_columns].isnull().any().any():
                self.logger.warning("Data contains NaN values")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _calculate_momentum_suite(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate comprehensive momentum indicator suite"""
        try:
            momentum_suite = {}
            
            for period in self.indicator_periods:
                period_data = {}
                
                # RSI with smoothed variant
                rsi = self._calculate_rsi(data['close'], period)
                rsi_smooth = rsi.rolling(window=3).mean()
                period_data['rsi'] = rsi
                period_data['rsi_smooth'] = rsi_smooth
                
                # MACD with multiple variants
                macd, macd_signal, macd_histogram = self._calculate_macd(data['close'], period)
                period_data['macd'] = macd
                period_data['macd_signal'] = macd_signal
                period_data['macd_histogram'] = macd_histogram
                
                # Stochastic with %K and %D
                stoch_k, stoch_d = self._calculate_stochastic(data, period)
                period_data['stoch_k'] = stoch_k
                period_data['stoch_d'] = stoch_d
                
                # Williams %R
                williams_r = self._calculate_williams_r(data, period)
                period_data['williams_r'] = williams_r
                
                # Rate of Change (ROC)
                roc = self._calculate_roc(data['close'], period)
                period_data['roc'] = roc
                
                # Momentum oscillator
                momentum = self._calculate_momentum(data['close'], period)
                period_data['momentum'] = momentum
                
                # Ultimate Oscillator components
                uo = self._calculate_ultimate_oscillator(data, period)
                period_data['ultimate_oscillator'] = uo
                
                # Money Flow Index
                mfi = self._calculate_money_flow_index(data, period)
                period_data['mfi'] = mfi
                
                momentum_suite[f'period_{period}'] = pd.DataFrame(period_data)
            
            return momentum_suite
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum suite: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with Wilder's smoothing"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Wilder's smoothing
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def _calculate_macd(self, prices: pd.Series, base_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD with adaptive periods"""
        try:
            fast_period = max(base_period // 2, 12)
            slow_period = base_period
            signal_period = max(base_period // 3, 9)
            
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
        """Calculate Stochastic oscillator with smoothing"""
        try:
            lowest_low = data['low'].rolling(window=period).min()
            highest_high = data['high'].rolling(window=period).max()
            
            k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
            k_smooth = k_percent.rolling(window=3).mean()
            d_percent = k_smooth.rolling(window=3).mean()
            
            return k_smooth, d_percent
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return pd.Series(), pd.Series()
    
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
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        try:
            roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
            return roc
            
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {str(e)}")
            return pd.Series()
    
    def _calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Price Momentum"""
        try:
            momentum = prices / prices.shift(period) * 100
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating Momentum: {str(e)}")
            return pd.Series()
    
    def _calculate_ultimate_oscillator(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range and Buying Pressure
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            bp = close - np.minimum(low, close.shift(1))
            
            # Calculate for three timeframes
            period1 = period // 2
            period2 = period
            period3 = period * 2
            
            avg_bp1 = bp.rolling(window=period1).sum()
            avg_tr1 = tr.rolling(window=period1).sum()
            avg_bp2 = bp.rolling(window=period2).sum()
            avg_tr2 = tr.rolling(window=period2).sum()
            avg_bp3 = bp.rolling(window=period3).sum()
            avg_tr3 = tr.rolling(window=period3).sum()
            
            uo = 100 * ((4 * avg_bp1 / avg_tr1) + (2 * avg_bp2 / avg_tr2) + (avg_bp3 / avg_tr3)) / 7
            
            return uo
            
        except Exception as e:
            self.logger.error(f"Error calculating Ultimate Oscillator: {str(e)}")
            return pd.Series()
    
    def _calculate_money_flow_index(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            
            # Positive and negative money flow
            pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            pos_mf = pos_flow.rolling(window=period).sum()
            neg_mf = neg_flow.rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
            
            return mfi
            
        except Exception as e:
            self.logger.error(f"Error calculating Money Flow Index: {str(e)}")
            return pd.Series()
    
    def _detect_price_pivots(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect price pivot points for divergence analysis"""
        try:
            highs = []
            lows = []
            
            high_series = data['high']
            low_series = data['low']
            
            for i in range(self.pivot_lookback, len(data) - self.pivot_lookback):
                # Pivot high detection
                is_high_pivot = True
                for j in range(1, self.pivot_lookback + 1):
                    if (high_series.iloc[i] <= high_series.iloc[i-j] or 
                        high_series.iloc[i] <= high_series.iloc[i+j]):
                        is_high_pivot = False
                        break
                
                if is_high_pivot:
                    highs.append({
                        'index': i,
                        'price': high_series.iloc[i],
                        'timestamp': data.index[i] if hasattr(data.index, 'to_list') else i,
                        'volume': data['volume'].iloc[i]
                    })
                
                # Pivot low detection
                is_low_pivot = True
                for j in range(1, self.pivot_lookback + 1):
                    if (low_series.iloc[i] >= low_series.iloc[i-j] or 
                        low_series.iloc[i] >= low_series.iloc[i+j]):
                        is_low_pivot = False
                        break
                
                if is_low_pivot:
                    lows.append({
                        'index': i,
                        'price': low_series.iloc[i],
                        'timestamp': data.index[i] if hasattr(data.index, 'to_list') else i,
                        'volume': data['volume'].iloc[i]
                    })
            
            return {'highs': highs, 'lows': lows}
            
        except Exception as e:
            self.logger.error(f"Error detecting price pivots: {str(e)}")
            return {'highs': [], 'lows': []}
    
    def _detect_momentum_pivots(self, momentum_data: pd.Series, series_name: str) -> Dict[str, List[Dict]]:
        """Detect momentum indicator pivot points"""
        try:
            highs = []
            lows = []
            
            for i in range(self.pivot_lookback, len(momentum_data) - self.pivot_lookback):
                if pd.isna(momentum_data.iloc[i]):
                    continue
                
                # Momentum high pivot
                is_high_pivot = True
                for j in range(1, self.pivot_lookback + 1):
                    if (momentum_data.iloc[i] <= momentum_data.iloc[i-j] or 
                        momentum_data.iloc[i] <= momentum_data.iloc[i+j]):
                        is_high_pivot = False
                        break
                
                if is_high_pivot:
                    highs.append({
                        'index': i,
                        'value': momentum_data.iloc[i],
                        'indicator': series_name,
                        'timestamp': momentum_data.index[i] if hasattr(momentum_data.index, 'to_list') else i
                    })
                
                # Momentum low pivot
                is_low_pivot = True
                for j in range(1, self.pivot_lookback + 1):
                    if (momentum_data.iloc[i] >= momentum_data.iloc[i-j] or 
                        momentum_data.iloc[i] >= momentum_data.iloc[i+j]):
                        is_low_pivot = False
                        break
                
                if is_low_pivot:
                    lows.append({
                        'index': i,
                        'value': momentum_data.iloc[i],
                        'indicator': series_name,
                        'timestamp': momentum_data.index[i] if hasattr(momentum_data.index, 'to_list') else i
                    })
            
            return {'highs': highs, 'lows': lows}
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum pivots: {str(e)}")
            return {'highs': [], 'lows': []}
    
    def _scan_for_divergences(self, price_pivots: Dict, momentum_suite: Dict) -> List[Dict]:
        """Comprehensive divergence scanning across all momentum indicators"""
        try:
            all_divergences = []
            
            # Scan each momentum indicator
            for period_key, period_data in momentum_suite.items():
                for indicator_name, indicator_series in period_data.items():
                    if indicator_series.empty:
                        continue
                    
                    # Get momentum pivots
                    momentum_pivots = self._detect_momentum_pivots(indicator_series, f"{period_key}_{indicator_name}")
                    
                    # Detect regular divergences
                    regular_divergences = self._detect_regular_divergences(
                        price_pivots, momentum_pivots, f"{period_key}_{indicator_name}"
                    )
                    all_divergences.extend(regular_divergences)
                    
                    # Detect hidden divergences
                    hidden_divergences = self._detect_hidden_divergences(
                        price_pivots, momentum_pivots, f"{period_key}_{indicator_name}"
                    )
                    all_divergences.extend(hidden_divergences)
            
            return all_divergences
            
        except Exception as e:
            self.logger.error(f"Error scanning for divergences: {str(e)}")
            return []
    
    def _detect_regular_divergences(self, price_pivots: Dict, momentum_pivots: Dict, indicator_name: str) -> List[Dict]:
        """Detect regular (classic) divergences"""
        try:
            divergences = []
            
            # Bullish regular divergence: Lower lows in price, higher lows in momentum
            price_lows = price_pivots['lows'][-self.scan_period:]
            momentum_lows = momentum_pivots['lows'][-self.scan_period:]
            
            for i in range(len(price_lows) - 1):
                for j in range(i + 1, len(price_lows)):
                    price_low_1 = price_lows[i]
                    price_low_2 = price_lows[j]
                    
                    # Find corresponding momentum lows
                    momentum_low_1 = self._find_nearest_momentum_pivot(price_low_1['index'], momentum_lows)
                    momentum_low_2 = self._find_nearest_momentum_pivot(price_low_2['index'], momentum_lows)
                    
                    if momentum_low_1 and momentum_low_2:
                        # Check for bullish regular divergence
                        if (price_low_2['price'] < price_low_1['price'] and 
                            momentum_low_2['value'] > momentum_low_1['value']):
                            
                            strength = self._calculate_divergence_strength(
                                price_low_1, price_low_2, momentum_low_1, momentum_low_2, 'bullish_regular'
                            )
                            
                            if strength >= self.min_divergence_strength:
                                divergences.append({
                                    'type': 'bullish_regular',
                                    'indicator': indicator_name,
                                    'price_point_1': price_low_1,
                                    'price_point_2': price_low_2,
                                    'momentum_point_1': momentum_low_1,
                                    'momentum_point_2': momentum_low_2,
                                    'strength': strength,
                                    'confidence': self._calculate_divergence_confidence(price_low_1, price_low_2, momentum_low_1, momentum_low_2),
                                    'age': len(price_lows) - j
                                })
            
            # Bearish regular divergence: Higher highs in price, lower highs in momentum
            price_highs = price_pivots['highs'][-self.scan_period:]
            momentum_highs = momentum_pivots['highs'][-self.scan_period:]
            
            for i in range(len(price_highs) - 1):
                for j in range(i + 1, len(price_highs)):
                    price_high_1 = price_highs[i]
                    price_high_2 = price_highs[j]
                    
                    # Find corresponding momentum highs
                    momentum_high_1 = self._find_nearest_momentum_pivot(price_high_1['index'], momentum_highs)
                    momentum_high_2 = self._find_nearest_momentum_pivot(price_high_2['index'], momentum_highs)
                    
                    if momentum_high_1 and momentum_high_2:
                        # Check for bearish regular divergence
                        if (price_high_2['price'] > price_high_1['price'] and 
                            momentum_high_2['value'] < momentum_high_1['value']):
                            
                            strength = self._calculate_divergence_strength(
                                price_high_1, price_high_2, momentum_high_1, momentum_high_2, 'bearish_regular'
                            )
                            
                            if strength >= self.min_divergence_strength:
                                divergences.append({
                                    'type': 'bearish_regular',
                                    'indicator': indicator_name,
                                    'price_point_1': price_high_1,
                                    'price_point_2': price_high_2,
                                    'momentum_point_1': momentum_high_1,
                                    'momentum_point_2': momentum_high_2,
                                    'strength': strength,
                                    'confidence': self._calculate_divergence_confidence(price_high_1, price_high_2, momentum_high_1, momentum_high_2),
                                    'age': len(price_highs) - j
                                })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting regular divergences: {str(e)}")
            return []
    
    def _detect_hidden_divergences(self, price_pivots: Dict, momentum_pivots: Dict, indicator_name: str) -> List[Dict]:
        """Detect hidden divergences"""
        try:
            divergences = []
            
            # Bullish hidden divergence: Higher lows in price, lower lows in momentum
            price_lows = price_pivots['lows'][-self.scan_period:]
            momentum_lows = momentum_pivots['lows'][-self.scan_period:]
            
            for i in range(len(price_lows) - 1):
                for j in range(i + 1, len(price_lows)):
                    price_low_1 = price_lows[i]
                    price_low_2 = price_lows[j]
                    
                    momentum_low_1 = self._find_nearest_momentum_pivot(price_low_1['index'], momentum_lows)
                    momentum_low_2 = self._find_nearest_momentum_pivot(price_low_2['index'], momentum_lows)
                    
                    if momentum_low_1 and momentum_low_2:
                        # Check for bullish hidden divergence
                        if (price_low_2['price'] > price_low_1['price'] and 
                            momentum_low_2['value'] < momentum_low_1['value']):
                            
                            strength = self._calculate_divergence_strength(
                                price_low_1, price_low_2, momentum_low_1, momentum_low_2, 'bullish_hidden'
                            )
                            
                            if strength >= self.min_divergence_strength:
                                divergences.append({
                                    'type': 'bullish_hidden',
                                    'indicator': indicator_name,
                                    'price_point_1': price_low_1,
                                    'price_point_2': price_low_2,
                                    'momentum_point_1': momentum_low_1,
                                    'momentum_point_2': momentum_low_2,
                                    'strength': strength,
                                    'confidence': self._calculate_divergence_confidence(price_low_1, price_low_2, momentum_low_1, momentum_low_2),
                                    'age': len(price_lows) - j
                                })
            
            # Bearish hidden divergence: Lower highs in price, higher highs in momentum
            price_highs = price_pivots['highs'][-self.scan_period:]
            momentum_highs = momentum_pivots['highs'][-self.scan_period:]
            
            for i in range(len(price_highs) - 1):
                for j in range(i + 1, len(price_highs)):
                    price_high_1 = price_highs[i]
                    price_high_2 = price_highs[j]
                    
                    momentum_high_1 = self._find_nearest_momentum_pivot(price_high_1['index'], momentum_highs)
                    momentum_high_2 = self._find_nearest_momentum_pivot(price_high_2['index'], momentum_highs)
                    
                    if momentum_high_1 and momentum_high_2:
                        # Check for bearish hidden divergence
                        if (price_high_2['price'] < price_high_1['price'] and 
                            momentum_high_2['value'] > momentum_high_1['value']):
                            
                            strength = self._calculate_divergence_strength(
                                price_high_1, price_high_2, momentum_high_1, momentum_high_2, 'bearish_hidden'
                            )
                            
                            if strength >= self.min_divergence_strength:
                                divergences.append({
                                    'type': 'bearish_hidden',
                                    'indicator': indicator_name,
                                    'price_point_1': price_high_1,
                                    'price_point_2': price_high_2,
                                    'momentum_point_1': momentum_high_1,
                                    'momentum_point_2': momentum_high_2,
                                    'strength': strength,
                                    'confidence': self._calculate_divergence_confidence(price_high_1, price_high_2, momentum_high_1, momentum_high_2),
                                    'age': len(price_highs) - j
                                })
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden divergences: {str(e)}")
            return []
    
    def _find_nearest_momentum_pivot(self, target_index: int, momentum_pivots: List[Dict], max_distance: int = 8) -> Optional[Dict]:
        """Find nearest momentum pivot to price pivot"""
        try:
            if not momentum_pivots:
                return None
            
            nearest_pivot = None
            min_distance = float('inf')
            
            for pivot in momentum_pivots:
                distance = abs(pivot['index'] - target_index)
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_pivot = pivot
            
            return nearest_pivot
            
        except Exception as e:
            self.logger.error(f"Error finding nearest momentum pivot: {str(e)}")
            return None
    
    def _calculate_divergence_strength(self, price_p1: Dict, price_p2: Dict, 
                                     momentum_p1: Dict, momentum_p2: Dict, 
                                     divergence_type: str) -> float:
        """Calculate divergence strength score"""
        try:
            if divergence_type in ['bullish_regular', 'bullish_hidden']:
                price_change = abs(price_p2['price'] - price_p1['price']) / price_p1['price']
                momentum_change = abs(momentum_p2['value'] - momentum_p1['value']) / abs(momentum_p1['value'])
            else:  # bearish
                price_change = abs(price_p1['price'] - price_p2['price']) / price_p1['price']
                momentum_change = abs(momentum_p1['value'] - momentum_p2['value']) / abs(momentum_p1['value'])
            
            # Combine price and momentum changes
            strength = (price_change + momentum_change) / 2
            
            # Apply volume weighting if available
            if 'volume' in price_p1 and 'volume' in price_p2:
                volume_ratio = price_p2['volume'] / max(price_p1['volume'], 1)
                volume_factor = min(volume_ratio * self.volume_weight, self.volume_weight)
                strength += volume_factor
            
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating divergence strength: {str(e)}")
            return 0.0
    
    def _calculate_divergence_confidence(self, price_p1: Dict, price_p2: Dict, 
                                       momentum_p1: Dict, momentum_p2: Dict) -> float:
        """Calculate confidence score for divergence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Time proximity factor
            time_diff = abs(price_p2['index'] - momentum_p2['index'])
            time_factor = max(0, 1 - time_diff / 10) * 0.3
            confidence += time_factor
            
            # Age factor (newer divergences are more reliable)
            age = price_p2['index']
            age_factor = max(0, 1 - age / self.max_divergence_age) * 0.2
            confidence += age_factor
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating divergence confidence: {str(e)}")
            return 0.5
    
    def _filter_and_confirm_divergences(self, divergences: List[Dict]) -> List[Dict]:
        """Filter and confirm divergences using multiple criteria"""
        try:
            if not divergences:
                return []
            
            # Remove expired divergences
            active_divergences = [d for d in divergences if d['age'] <= self.max_divergence_age]
            
            # Group by type and find confirmations
            confirmed_divergences = []
            
            # Group by divergence type
            bullish_regular = [d for d in active_divergences if d['type'] == 'bullish_regular']
            bearish_regular = [d for d in active_divergences if d['type'] == 'bearish_regular']
            bullish_hidden = [d for d in active_divergences if d['type'] == 'bullish_hidden']
            bearish_hidden = [d for d in active_divergences if d['type'] == 'bearish_hidden']
            
            # Check for confirmation across multiple indicators
            for divergence_group in [bullish_regular, bearish_regular, bullish_hidden, bearish_hidden]:
                if len(divergence_group) >= self.confirmation_threshold:
                    # Find the strongest divergences
                    sorted_divergences = sorted(divergence_group, key=lambda x: x['strength'], reverse=True)
                    
                    # Add confirmation count to each divergence
                    for div in sorted_divergences[:3]:  # Top 3 strongest
                        div['confirmation_count'] = len(divergence_group)
                        div['multi_indicator_confirmed'] = True
                        confirmed_divergences.append(div)
            
            # Also include strong single-indicator divergences
            strong_single = [d for d in active_divergences 
                           if d['strength'] > self.min_divergence_strength * 2 
                           and d not in confirmed_divergences]
            
            for div in strong_single:
                div['confirmation_count'] = 1
                div['multi_indicator_confirmed'] = False
                confirmed_divergences.append(div)
            
            return confirmed_divergences
            
        except Exception as e:
            self.logger.error(f"Error filtering and confirming divergences: {str(e)}")
            return divergences
    
    def _determine_overall_signal(self, confirmed_divergences: List[Dict]) -> Dict:
        """Determine overall momentum divergence signal"""
        try:
            if not confirmed_divergences:
                return {'signal': 'neutral', 'strength': 0.0, 'confidence': 0.0}
            
            # Count different types of divergences
            bullish_regular_count = len([d for d in confirmed_divergences if d['type'] == 'bullish_regular'])
            bearish_regular_count = len([d for d in confirmed_divergences if d['type'] == 'bearish_regular'])
            bullish_hidden_count = len([d for d in confirmed_divergences if d['type'] == 'bullish_hidden'])
            bearish_hidden_count = len([d for d in confirmed_divergences if d['type'] == 'bearish_hidden'])
            
            # Calculate weighted signals
            bullish_weight = (bullish_regular_count * 1.5) + (bullish_hidden_count * 1.0)
            bearish_weight = (bearish_regular_count * 1.5) + (bearish_hidden_count * 1.0)
            
            # Determine primary signal
            if bullish_weight > bearish_weight:
                signal_type = 'bullish'
                signal_strength = bullish_weight / (bullish_weight + bearish_weight)
            elif bearish_weight > bullish_weight:
                signal_type = 'bearish'
                signal_strength = bearish_weight / (bullish_weight + bearish_weight)
            else:
                signal_type = 'neutral'
                signal_strength = 0.5
            
            # Calculate overall confidence
            total_confirmations = sum(d['confirmation_count'] for d in confirmed_divergences)
            avg_confidence = sum(d['confidence'] for d in confirmed_divergences) / len(confirmed_divergences)
            confirmation_factor = min(total_confirmations / (self.confirmation_threshold * 3), 1.0)
            
            overall_confidence = (avg_confidence + confirmation_factor) / 2
            
            return {
                'signal': signal_type,
                'strength': signal_strength,
                'confidence': overall_confidence,
                'bullish_count': bullish_regular_count + bullish_hidden_count,
                'bearish_count': bearish_regular_count + bearish_hidden_count
            }
            
        except Exception as e:
            self.logger.error(f"Error determining overall signal: {str(e)}")
            return {'signal': 'neutral', 'strength': 0.0, 'confidence': 0.0}
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate MomentumDivergenceScanner indicator
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing momentum divergence scan results
        """
        try:
            if not self.validate_data(data):
                return {}
            
            # Calculate momentum indicator suite
            momentum_suite = self._calculate_momentum_suite(data)
            self.momentum_data = momentum_suite
            
            if not momentum_suite:
                return {
                    'divergences': [],
                    'scan_complete': False,
                    'error': 'Failed to calculate momentum indicators'
                }
            
            # Detect price pivots
            price_pivots = self._detect_price_pivots(data)
            
            # Scan for divergences
            all_divergences = self._scan_for_divergences(price_pivots, momentum_suite)
            
            # Filter and confirm divergences
            confirmed_divergences = self._filter_and_confirm_divergences(all_divergences)
            
            # Determine overall signal
            overall_signal = self._determine_overall_signal(confirmed_divergences)
            
            # Store results
            self.divergence_history = confirmed_divergences
            self.current_scan_results = overall_signal
            
            # Categorize divergences
            divergence_summary = {
                'bullish_regular': [d for d in confirmed_divergences if d['type'] == 'bullish_regular'],
                'bearish_regular': [d for d in confirmed_divergences if d['type'] == 'bearish_regular'],
                'bullish_hidden': [d for d in confirmed_divergences if d['type'] == 'bullish_hidden'],
                'bearish_hidden': [d for d in confirmed_divergences if d['type'] == 'bearish_hidden']
            }
            
            result = {
                'scan_complete': True,
                'total_divergences': len(confirmed_divergences),
                'confirmed_divergences': confirmed_divergences,
                'divergence_summary': divergence_summary,
                'overall_signal': overall_signal['signal'],
                'signal_strength': overall_signal['strength'],
                'signal_confidence': overall_signal['confidence'],
                'bullish_divergence_count': overall_signal['bullish_count'],
                'bearish_divergence_count': overall_signal['bearish_count'],
                'strongest_divergence': max(confirmed_divergences, key=lambda x: x['strength']) if confirmed_divergences else None,
                'most_recent_divergence': max(confirmed_divergences, key=lambda x: x['price_point_2']['index']) if confirmed_divergences else None
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating MomentumDivergenceScanner: {str(e)}")
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
                'scan_period': self.scan_period,
                'indicator_periods': self.indicator_periods,
                'min_divergence_strength': self.min_divergence_strength,
                'confirmation_threshold': self.confirmation_threshold,
                'pivot_lookback': self.pivot_lookback,
                'max_divergence_age': self.max_divergence_age,
                'volume_weight': self.volume_weight,
                'trend_filter': self.trend_filter
            },
            'data_requirements': ['high', 'low', 'close', 'volume'],
            'output_format': 'comprehensive_divergence_scan'
        }
    def validate_parameters(self) -> bool:
        """Validate parameters"""
        # Add specific validation logic as needed
        return True



def export() -> Dict[str, Any]:
    """
    Export function for the MomentumDivergenceScanner indicator.
    
    This function is used by the indicator registry to discover and load the indicator.
    
    Returns:
        Dictionary containing indicator information for registry
    """
    return {
        'class': MomentumDivergenceScannerIndicator,
        'name': 'MomentumDivergenceScanner',
        'category': 'divergence',
        'version': '1.0.0',
        'description': 'Comprehensive momentum divergence scanner with multi-indicator confirmation',
        'complexity': 'advanced',
        'parameters': {
            'scan_period': {'type': 'int', 'default': 21, 'min': 10, 'max': 100},
            'indicator_periods': {'type': 'list', 'default': [9, 14, 21, 34]},
            'min_divergence_strength': {'type': 'float', 'default': 0.025, 'min': 0.001, 'max': 0.1},
            'confirmation_threshold': {'type': 'int', 'default': 2, 'min': 1, 'max': 5},
            'pivot_lookback': {'type': 'int', 'default': 5, 'min': 2, 'max': 15},
            'max_divergence_age': {'type': 'int', 'default': 10, 'min': 5, 'max': 30},
            'volume_weight': {'type': 'float', 'default': 0.3, 'min': 0.0, 'max': 1.0},
            'trend_filter': {'type': 'bool', 'default': True}
        },
        'data_requirements': ['high', 'low', 'close', 'volume'],
        'output_type': 'divergence_scan'
    }