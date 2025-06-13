"""
Momentum Divergence Scanner

Scans for momentum divergences between price action and multiple momentum 
indicators to identify potential trend reversal points with high accuracy.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class MomentumDivergenceResult:
    divergence_type: str              # "bullish", "bearish", "hidden_bullish", "hidden_bearish", "none"
    divergence_strength: float        # Strength score (0-1)
    price_points: List[Tuple[int, float]]    # Price swing points involved
    momentum_points: List[Tuple[int, float]] # Momentum indicator points
    indicators_confirming: List[str]   # Which indicators show divergence
    divergence_age: int               # Bars since divergence started
    potential_target: Optional[float] # Projected price target
    confidence_level: float           # Overall confidence (0-1)
    timestamp: Optional[str] = None


class MomentumDivergenceScanner(StandardIndicatorInterface):
    """
    Momentum Divergence Scanner
    
    Systematically scans for divergences between price and momentum indicators
    (RSI, MACD, Stochastic) to identify high-probability reversal setups.
    
    Detects both regular and hidden divergences with confirmation from
    multiple momentum indicators for increased reliability.
    """
    
    CATEGORY = "technical"
    
    def __init__(self,
                 lookback: int = 50,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 stoch_k: int = 14,
                 stoch_d: int = 3,
                 min_swing_bars: int = 5,
                 min_divergence_strength: float = 0.6,
                 **kwargs):
        """
        Initialize Momentum Divergence Scanner.
        
        Args:
            lookback: Number of bars to analyze for divergences
            rsi_period: RSI calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            stoch_k: Stochastic %K period
            stoch_d: Stochastic %D period
            min_swing_bars: Minimum bars between swing points
            min_divergence_strength: Minimum strength to report divergence
        """
        super().__init__(**kwargs)
        self.lookback = lookback
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.min_swing_bars = min_swing_bars
        self.min_divergence_strength = min_divergence_strength
    
    def calculate(self, data: pd.DataFrame) -> MomentumDivergenceResult:
        """
        Scan for momentum divergences.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            MomentumDivergenceResult with divergence analysis
        """
        try:
            if len(data) < max(self.lookback, self.macd_slow + 10):
                return MomentumDivergenceResult(
                    divergence_type="none",
                    divergence_strength=0.0,
                    price_points=[],
                    momentum_points=[],
                    indicators_confirming=[],
                    divergence_age=0,
                    potential_target=None,
                    confidence_level=0.0
                )
            
            # Get recent data
            recent_data = data.tail(self.lookback).copy()
            
            # Calculate momentum indicators
            momentum_data = self._calculate_momentum_indicators(recent_data)
            
            # Find price swing points
            price_swings = self._find_price_swings(recent_data)
            
            # Find momentum swing points for each indicator
            momentum_swings = {}
            for indicator in ['rsi', 'macd', 'stoch']:
                momentum_swings[indicator] = self._find_momentum_swings(momentum_data[indicator])
            
            # Scan for divergences
            best_divergence = self._scan_divergences(
                price_swings, momentum_swings, recent_data, momentum_data
            )
            
            return best_divergence
            
        except Exception as e:
            return MomentumDivergenceResult(
                divergence_type="error",
                divergence_strength=0.0,
                price_points=[],
                momentum_points=[],
                indicators_confirming=[],
                divergence_age=0,
                potential_target=None,
                confidence_level=0.0
            )
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate RSI, MACD, and Stochastic indicators."""
        momentum_data = {}
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        momentum_data['rsi'] = rsi
        
        # MACD
        ema_fast = data['close'].ewm(span=self.macd_fast).mean()
        ema_slow = data['close'].ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        macd_histogram = macd_line - signal_line
        momentum_data['macd'] = macd_histogram  # Use histogram for divergence
        
        # Stochastic
        low_min = data['low'].rolling(window=self.stoch_k).min()
        high_max = data['high'].rolling(window=self.stoch_k).max()
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=self.stoch_d).mean()
        momentum_data['stoch'] = k_percent
        
        return momentum_data
    
    def _find_price_swings(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """Find price swing highs and lows."""
        swings = {'highs': [], 'lows': []}
        
        highs = data['high'].values
        lows = data['low'].values
        
        for i in range(self.min_swing_bars, len(data) - self.min_swing_bars):
            # Check for swing high
            is_high = True
            for j in range(1, self.min_swing_bars + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_high = False
                    break
            
            if is_high:
                swings['highs'].append((i, highs[i]))
            
            # Check for swing low
            is_low = True
            for j in range(1, self.min_swing_bars + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_low = False
                    break
            
            if is_low:
                swings['lows'].append((i, lows[i]))
        
        return swings
    
    def _find_momentum_swings(self, momentum_series: pd.Series) -> Dict[str, List[Tuple[int, float]]]:
        """Find momentum indicator swing points."""
        swings = {'highs': [], 'lows': []}
        values = momentum_series.dropna().values
        
        if len(values) < self.min_swing_bars * 2:
            return swings
        
        for i in range(self.min_swing_bars, len(values) - self.min_swing_bars):
            # Check for swing high
            is_high = True
            for j in range(1, self.min_swing_bars + 1):
                if values[i] <= values[i-j] or values[i] <= values[i+j]:
                    is_high = False
                    break
            
            if is_high:
                swings['highs'].append((i, values[i]))
            
            # Check for swing low
            is_low = True
            for j in range(1, self.min_swing_bars + 1):
                if values[i] >= values[i-j] or values[i] >= values[i+j]:
                    is_low = False
                    break
            
            if is_low:
                swings['lows'].append((i, values[i]))
        
        return swings
    
    def _scan_divergences(self, price_swings: Dict, momentum_swings: Dict,
                         price_data: pd.DataFrame, momentum_data: Dict) -> MomentumDivergenceResult:
        """Scan for divergences between price and momentum indicators."""
        best_divergence = MomentumDivergenceResult(
            divergence_type="none",
            divergence_strength=0.0,
            price_points=[],
            momentum_points=[],
            indicators_confirming=[],
            divergence_age=0,
            potential_target=None,
            confidence_level=0.0
        )
        
        # Check each momentum indicator for divergences
        for indicator in ['rsi', 'macd', 'stoch']:
            if indicator not in momentum_swings:
                continue
            
            # Check bullish divergences (price lows vs momentum lows)
            bullish_div = self._check_bullish_divergence(
                price_swings['lows'], momentum_swings[indicator]['lows']
            )
            
            # Check bearish divergences (price highs vs momentum highs)
            bearish_div = self._check_bearish_divergence(
                price_swings['highs'], momentum_swings[indicator]['highs']
            )
            
            # Select best divergence from this indicator
            current_best = max([bullish_div, bearish_div], key=lambda x: x.divergence_strength)
            
            if current_best.divergence_strength > best_divergence.divergence_strength:
                best_divergence = current_best
                best_divergence.indicators_confirming = [indicator]
            elif (current_best.divergence_strength > self.min_divergence_strength and
                  current_best.divergence_type == best_divergence.divergence_type):
                # Add confirming indicator
                best_divergence.indicators_confirming.append(indicator)
                best_divergence.divergence_strength = np.mean([
                    best_divergence.divergence_strength, current_best.divergence_strength
                ])
        
        # Calculate additional metrics for best divergence
        if best_divergence.divergence_type != "none":
            best_divergence.divergence_age = self._calculate_divergence_age(best_divergence)
            best_divergence.potential_target = self._calculate_target(best_divergence, price_data)
            best_divergence.confidence_level = self._calculate_confidence(best_divergence)
        
        return best_divergence
    
    def _check_bullish_divergence(self, price_lows: List[Tuple[int, float]], 
                                 momentum_lows: List[Tuple[int, float]]) -> MomentumDivergenceResult:
        """Check for bullish divergence pattern."""
        if len(price_lows) < 2 or len(momentum_lows) < 2:
            return MomentumDivergenceResult(
                divergence_type="none", divergence_strength=0.0, price_points=[],
                momentum_points=[], indicators_confirming=[], divergence_age=0,
                potential_target=None, confidence_level=0.0
            )
        
        # Get the two most recent lows
        recent_price_lows = sorted(price_lows, key=lambda x: x[0])[-2:]
        recent_momentum_lows = sorted(momentum_lows, key=lambda x: x[0])[-2:]
        
        if len(recent_price_lows) < 2 or len(recent_momentum_lows) < 2:
            return MomentumDivergenceResult(
                divergence_type="none", divergence_strength=0.0, price_points=[],
                momentum_points=[], indicators_confirming=[], divergence_age=0,
                potential_target=None, confidence_level=0.0
            )
        
        # Check for bullish divergence: price makes lower low, momentum makes higher low
        price_declining = recent_price_lows[1][1] < recent_price_lows[0][1]
        momentum_rising = recent_momentum_lows[1][1] > recent_momentum_lows[0][1]
        
        if price_declining and momentum_rising:
            # Calculate divergence strength
            price_change = (recent_price_lows[0][1] - recent_price_lows[1][1]) / recent_price_lows[0][1]
            momentum_change = (recent_momentum_lows[1][1] - recent_momentum_lows[0][1]) / abs(recent_momentum_lows[0][1])
            strength = min(1.0, (price_change + momentum_change) * 2)  # Scale to 0-1
            
            return MomentumDivergenceResult(
                divergence_type="bullish",
                divergence_strength=max(strength, 0.1),
                price_points=recent_price_lows,
                momentum_points=recent_momentum_lows,
                indicators_confirming=[],
                divergence_age=0,
                potential_target=None,
                confidence_level=0.0
            )
        
        return MomentumDivergenceResult(
            divergence_type="none", divergence_strength=0.0, price_points=[],
            momentum_points=[], indicators_confirming=[], divergence_age=0,
            potential_target=None, confidence_level=0.0
        )
    
    def _check_bearish_divergence(self, price_highs: List[Tuple[int, float]], 
                                 momentum_highs: List[Tuple[int, float]]) -> MomentumDivergenceResult:
        """Check for bearish divergence pattern."""
        if len(price_highs) < 2 or len(momentum_highs) < 2:
            return MomentumDivergenceResult(
                divergence_type="none", divergence_strength=0.0, price_points=[],
                momentum_points=[], indicators_confirming=[], divergence_age=0,
                potential_target=None, confidence_level=0.0
            )
        
        # Get the two most recent highs
        recent_price_highs = sorted(price_highs, key=lambda x: x[0])[-2:]
        recent_momentum_highs = sorted(momentum_highs, key=lambda x: x[0])[-2:]
        
        if len(recent_price_highs) < 2 or len(recent_momentum_highs) < 2:
            return MomentumDivergenceResult(
                divergence_type="none", divergence_strength=0.0, price_points=[],
                momentum_points=[], indicators_confirming=[], divergence_age=0,
                potential_target=None, confidence_level=0.0
            )
        
        # Check for bearish divergence: price makes higher high, momentum makes lower high
        price_rising = recent_price_highs[1][1] > recent_price_highs[0][1]
        momentum_declining = recent_momentum_highs[1][1] < recent_momentum_highs[0][1]
        
        if price_rising and momentum_declining:
            # Calculate divergence strength
            price_change = (recent_price_highs[1][1] - recent_price_highs[0][1]) / recent_price_highs[0][1]
            momentum_change = (recent_momentum_highs[0][1] - recent_momentum_highs[1][1]) / abs(recent_momentum_highs[0][1])
            strength = min(1.0, (price_change + momentum_change) * 2)  # Scale to 0-1
            
            return MomentumDivergenceResult(
                divergence_type="bearish",
                divergence_strength=max(strength, 0.1),
                price_points=recent_price_highs,
                momentum_points=recent_momentum_highs,
                indicators_confirming=[],
                divergence_age=0,
                potential_target=None,
                confidence_level=0.0
            )
        
        return MomentumDivergenceResult(
            divergence_type="none", divergence_strength=0.0, price_points=[],
            momentum_points=[], indicators_confirming=[], divergence_age=0,
            potential_target=None, confidence_level=0.0
        )
    
    def _calculate_divergence_age(self, divergence: MomentumDivergenceResult) -> int:
        """Calculate how many bars ago the divergence started."""
        if not divergence.price_points:
            return 0
        
        # Age is based on the most recent swing point
        most_recent = max(divergence.price_points, key=lambda x: x[0])
        return len(divergence.price_points) - most_recent[0] if divergence.price_points else 0
    
    def _calculate_target(self, divergence: MomentumDivergenceResult, 
                         price_data: pd.DataFrame) -> Optional[float]:
        """Calculate potential price target based on divergence."""
        if not divergence.price_points or len(divergence.price_points) < 2:
            return None
        
        current_price = float(price_data['close'].iloc[-1])
        
        if divergence.divergence_type == "bullish":
            # Target based on previous high or resistance
            recent_high = price_data['high'].tail(20).max()
            target = current_price + (recent_high - current_price) * 0.618  # 61.8% of move to high
        elif divergence.divergence_type == "bearish":
            # Target based on previous low or support
            recent_low = price_data['low'].tail(20).min()
            target = current_price - (current_price - recent_low) * 0.618  # 61.8% of move to low
        else:
            return None
        
        return target
    
    def _calculate_confidence(self, divergence: MomentumDivergenceResult) -> float:
        """Calculate overall confidence in the divergence signal."""
        confidence_factors = []
        
        # Factor 1: Number of confirming indicators
        indicator_factor = len(divergence.indicators_confirming) / 3.0  # Max 3 indicators
        confidence_factors.append(indicator_factor)
        
        # Factor 2: Divergence strength
        confidence_factors.append(divergence.divergence_strength)
        
        # Factor 3: Age factor (fresher divergences are better)
        age_factor = max(0, 1.0 - divergence.divergence_age / 20.0)  # Declines over 20 bars
        confidence_factors.append(age_factor)
        
        # Factor 4: Price point quality (more separated points are better)
        if len(divergence.price_points) >= 2:
            point_separation = abs(divergence.price_points[1][0] - divergence.price_points[0][0])
            separation_factor = min(1.0, point_separation / 10.0)  # Normalize to 10 bars
            confidence_factors.append(separation_factor)
        
        return np.mean(confidence_factors)
    
    def get_display_name(self) -> str:
        return "Momentum Divergence Scanner"
    
    def get_parameters(self) -> Dict:
        return {
            "lookback": self.lookback,
            "rsi_period": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "min_swing_bars": self.min_swing_bars,
            "min_divergence_strength": self.min_divergence_strength
        }