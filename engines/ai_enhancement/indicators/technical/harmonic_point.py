"""
Harmonic Point Indicator

Identifies harmonic pattern completion points using Fibonacci ratios and
geometric relationships to detect potential reversal zones in price action.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class HarmonicPointResult:
    pattern_type: str                    # "gartley", "butterfly", "bat", "crab", "none"
    completion_point: Optional[float]    # Price level where pattern completes
    pattern_confidence: float            # Confidence in pattern validity (0-1)
    fibonacci_ratios: Dict[str, float]   # Key Fibonacci ratios in the pattern
    reversal_zone: Tuple[float, float]   # Price range for potential reversal
    entry_signal: str                    # "bullish", "bearish", "none"
    target_levels: List[float]           # Potential target price levels
    stop_loss: Optional[float]           # Suggested stop loss level
    timestamp: Optional[str] = None


class HarmonicPoint(StandardIndicatorInterface):
    """
    Harmonic Point Pattern Recognition Indicator
    
    Detects harmonic patterns (Gartley, Butterfly, Bat, Crab) by analyzing
    price swings and Fibonacci relationships to identify potential reversal
    points with high probability setups.
    """
    
    CATEGORY = "technical"
    
    def __init__(self,
                 lookback: int = 100,
                 min_swing_size: float = 0.02,
                 fib_tolerance: float = 0.05,
                 pattern_tolerance: float = 0.1,
                 min_pattern_confidence: float = 0.7,
                 **kwargs):
        """
        Initialize Harmonic Point indicator.
        
        Args:
            lookback: Number of periods to search for patterns
            min_swing_size: Minimum swing size as fraction (e.g., 0.02 = 2%)
            fib_tolerance: Tolerance for Fibonacci ratio matching
            pattern_tolerance: General pattern matching tolerance
            min_pattern_confidence: Minimum confidence for valid patterns
        """
        super().__init__(**kwargs)
        self.lookback = lookback
        self.min_swing_size = min_swing_size
        self.fib_tolerance = fib_tolerance
        self.pattern_tolerance = pattern_tolerance
        self.min_pattern_confidence = min_pattern_confidence
        
        # Fibonacci ratios for different harmonic patterns
        self.pattern_ratios = {
            'gartley': {
                'XA_AB': 0.618, 'AB_BC': (0.382, 0.886), 'BC_CD': 1.272, 'XA_AD': 0.786
            },
            'butterfly': {
                'XA_AB': 0.786, 'AB_BC': (0.382, 0.886), 'BC_CD': (1.618, 2.618), 'XA_AD': (1.272, 1.618)
            },
            'bat': {
                'XA_AB': (0.382, 0.5), 'AB_BC': (0.382, 0.886), 'BC_CD': (1.618, 2.618), 'XA_AD': 0.886
            },
            'crab': {
                'XA_AB': (0.382, 0.618), 'AB_BC': (0.382, 0.886), 'BC_CD': (2.24, 3.618), 'XA_AD': 1.618
            }
        }
    
    def calculate(self, data: pd.DataFrame) -> HarmonicPointResult:
        """
        Calculate harmonic pattern points.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            HarmonicPointResult with harmonic pattern analysis
        """
        try:
            if len(data) < self.lookback:
                return HarmonicPointResult(
                    pattern_type="none",
                    completion_point=None,
                    pattern_confidence=0.0,
                    fibonacci_ratios={},
                    reversal_zone=(0.0, 0.0),
                    entry_signal="none",
                    target_levels=[],
                    stop_loss=None
                )
            
            # Get recent data
            recent_data = data.tail(self.lookback).copy()
            current_price = float(recent_data['close'].iloc[-1])
            
            # Find swing points
            swing_points = self._find_swing_points(recent_data)
            
            if len(swing_points) < 5:  # Need at least X, A, B, C, D points
                return HarmonicPointResult(
                    pattern_type="none",
                    completion_point=None,
                    pattern_confidence=0.0,
                    fibonacci_ratios={},
                    reversal_zone=(current_price * 0.99, current_price * 1.01),
                    entry_signal="none",
                    target_levels=[],
                    stop_loss=None
                )
            
            # Analyze potential harmonic patterns
            best_pattern = self._analyze_harmonic_patterns(swing_points, current_price)
            
            return best_pattern
            
        except Exception as e:
            current_price = float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
            return HarmonicPointResult(
                pattern_type="error",
                completion_point=None,
                pattern_confidence=0.0,
                fibonacci_ratios={},
                reversal_zone=(current_price * 0.99, current_price * 1.01),
                entry_signal="none",
                target_levels=[],
                stop_loss=None
            )
    
    def _find_swing_points(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Find significant swing highs and lows."""
        swing_points = []
        highs = data['high'].values
        lows = data['low'].values
        
        # Find swing highs and lows with minimum swing size
        for i in range(2, len(data) - 2):
            # Check for swing high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                
                # Verify minimum swing size
                if self._is_significant_swing(data, i, 'high'):
                    swing_points.append((i, highs[i], 'high'))
            
            # Check for swing low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                
                # Verify minimum swing size
                if self._is_significant_swing(data, i, 'low'):
                    swing_points.append((i, lows[i], 'low'))
        
        # Sort by time and return most recent significant swings
        swing_points.sort(key=lambda x: x[0])
        return swing_points[-10:]  # Keep last 10 swings for analysis
    
    def _is_significant_swing(self, data: pd.DataFrame, index: int, swing_type: str) -> bool:
        """Check if swing meets minimum size requirement."""
        if swing_type == 'high':
            # Compare with nearby lows
            nearby_low = min(data['low'].iloc[max(0, index-5):index+6])
            swing_size = (data['high'].iloc[index] - nearby_low) / nearby_low
        else:
            # Compare with nearby highs
            nearby_high = max(data['high'].iloc[max(0, index-5):index+6])
            swing_size = (nearby_high - data['low'].iloc[index]) / nearby_high
        
        return swing_size >= self.min_swing_size
    
    def _analyze_harmonic_patterns(self, swing_points: List[Tuple[int, float, str]], 
                                  current_price: float) -> HarmonicPointResult:
        """Analyze swing points for harmonic patterns."""
        best_pattern = HarmonicPointResult(
            pattern_type="none",
            completion_point=None,
            pattern_confidence=0.0,
            fibonacci_ratios={},
            reversal_zone=(current_price * 0.99, current_price * 1.01),
            entry_signal="none",
            target_levels=[],
            stop_loss=None
        )
        
        if len(swing_points) < 4:
            return best_pattern
        
        # Try to identify patterns with different swing combinations
        for i in range(len(swing_points) - 4):
            # Extract XABCD points (need alternating highs/lows)
            pattern_points = swing_points[i:i+5]
            
            # Verify alternating pattern
            if not self._is_valid_swing_sequence(pattern_points):
                continue
            
            # Extract price levels
            X = pattern_points[0][1]
            A = pattern_points[1][1]
            B = pattern_points[2][1]
            C = pattern_points[3][1]
            D = pattern_points[4][1] if len(pattern_points) == 5 else current_price
            
            # Test each pattern type
            for pattern_name, ratios in self.pattern_ratios.items():
                confidence, fib_ratios = self._test_pattern_match(X, A, B, C, D, ratios, pattern_name)
                
                if confidence > best_pattern.pattern_confidence and confidence >= self.min_pattern_confidence:
                    # Calculate pattern details
                    completion_point = self._calculate_completion_point(X, A, B, C, ratios)
                    reversal_zone = self._calculate_reversal_zone(completion_point, A, X)
                    entry_signal = self._determine_entry_signal(pattern_points, current_price)
                    target_levels = self._calculate_targets(X, A, B, C, completion_point)
                    stop_loss = self._calculate_stop_loss(pattern_points, completion_point)
                    
                    best_pattern = HarmonicPointResult(
                        pattern_type=pattern_name,
                        completion_point=completion_point,
                        pattern_confidence=confidence,
                        fibonacci_ratios=fib_ratios,
                        reversal_zone=reversal_zone,
                        entry_signal=entry_signal,
                        target_levels=target_levels,
                        stop_loss=stop_loss,
                        timestamp=None
                    )
        
        return best_pattern
    
    def _is_valid_swing_sequence(self, points: List[Tuple[int, float, str]]) -> bool:
        """Check if swing points form a valid alternating sequence."""
        if len(points) < 4:
            return False
        
        # Should alternate between highs and lows
        for i in range(len(points) - 1):
            if points[i][2] == points[i+1][2]:  # Same type consecutive
                return False
        
        return True
    
    def _test_pattern_match(self, X: float, A: float, B: float, C: float, D: float,
                           ratios: Dict, pattern_name: str) -> Tuple[float, Dict[str, float]]:
        """Test if XABCD points match a specific harmonic pattern."""
        calculated_ratios = {}
        confidence_scores = []
        
        # Calculate actual ratios
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        AD = abs(D - A)
        
        if XA == 0 or AB == 0 or BC == 0:
            return 0.0, {}
        
        calculated_ratios['AB/XA'] = AB / XA
        calculated_ratios['BC/AB'] = BC / AB
        calculated_ratios['CD/BC'] = CD / BC if BC > 0 else 0
        calculated_ratios['AD/XA'] = AD / XA
        
        # Compare with expected ratios
        for ratio_name, expected in ratios.items():
            if ratio_name == 'XA_AB':
                actual = calculated_ratios['AB/XA']
                target = expected
            elif ratio_name == 'AB_BC':
                actual = calculated_ratios['BC/AB']
                target = expected if isinstance(expected, (int, float)) else expected[0]
            elif ratio_name == 'BC_CD':
                actual = calculated_ratios['CD/BC']
                target = expected if isinstance(expected, (int, float)) else expected[0]
            elif ratio_name == 'XA_AD':
                actual = calculated_ratios['AD/XA']
                target = expected if isinstance(expected, (int, float)) else expected[0]
            else:
                continue
            
            # Calculate confidence for this ratio
            if isinstance(expected, tuple):
                # Range check
                if expected[0] <= actual <= expected[1]:
                    confidence_scores.append(1.0)
                else:
                    distance = min(abs(actual - expected[0]), abs(actual - expected[1]))
                    confidence_scores.append(max(0, 1.0 - distance / self.fib_tolerance))
            else:
                # Point check
                distance = abs(actual - target) / target if target > 0 else 1.0
                confidence_scores.append(max(0, 1.0 - distance / self.fib_tolerance))
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        return overall_confidence, calculated_ratios
    
    def _calculate_completion_point(self, X: float, A: float, B: float, C: float, 
                                   ratios: Dict) -> float:
        """Calculate the D point (completion point) for the pattern."""
        # Use XA_AD ratio to project D point
        XA = abs(A - X)
        expected_ad_ratio = ratios.get('XA_AD', 0.786)
        
        if isinstance(expected_ad_ratio, tuple):
            expected_ad_ratio = np.mean(expected_ad_ratio)
        
        AD = XA * expected_ad_ratio
        
        # Determine direction based on pattern structure
        if A > X:  # Bullish pattern
            D = A - AD
        else:  # Bearish pattern
            D = A + AD
        
        return D
    
    def _calculate_reversal_zone(self, completion_point: float, A: float, X: float) -> Tuple[float, float]:
        """Calculate the price range for potential reversal."""
        if completion_point is None:
            return (0.0, 0.0)
        
        # Create zone around completion point
        zone_size = abs(A - X) * 0.05  # 5% of XA move
        
        return (completion_point - zone_size, completion_point + zone_size)
    
    def _determine_entry_signal(self, pattern_points: List[Tuple[int, float, str]], 
                               current_price: float) -> str:
        """Determine entry signal based on pattern structure."""
        if len(pattern_points) < 2:
            return "none"
        
        # Check pattern direction
        X_price = pattern_points[0][1]
        A_price = pattern_points[1][1]
        
        if A_price > X_price:  # Bullish pattern structure
            return "bullish"
        else:  # Bearish pattern structure
            return "bearish"
    
    def _calculate_targets(self, X: float, A: float, B: float, C: float, 
                          completion_point: float) -> List[float]:
        """Calculate potential target levels."""
        if completion_point is None:
            return []
        
        targets = []
        
        # Target 1: 38.2% of CD
        CD = abs(completion_point - C)
        target1 = completion_point + CD * 0.382 * (1 if C > completion_point else -1)
        targets.append(target1)
        
        # Target 2: 61.8% of CD
        target2 = completion_point + CD * 0.618 * (1 if C > completion_point else -1)
        targets.append(target2)
        
        # Target 3: Point C level
        targets.append(C)
        
        return sorted(targets)
    
    def _calculate_stop_loss(self, pattern_points: List[Tuple[int, float, str]], 
                            completion_point: float) -> Optional[float]:
        """Calculate suggested stop loss level."""
        if completion_point is None or len(pattern_points) < 1:
            return None
        
        # Stop loss beyond X point with small buffer
        X_price = pattern_points[0][1]
        buffer = abs(completion_point - X_price) * 0.1  # 10% buffer
        
        if completion_point > X_price:  # Bullish pattern
            return X_price - buffer
        else:  # Bearish pattern
            return X_price + buffer
    
    def get_display_name(self) -> str:
        return "Harmonic Point Pattern"
    
    def get_parameters(self) -> Dict:
        return {
            "lookback": self.lookback,
            "min_swing_size": self.min_swing_size,
            "fib_tolerance": self.fib_tolerance,
            "pattern_tolerance": self.pattern_tolerance,
            "min_pattern_confidence": self.min_pattern_confidence
        }