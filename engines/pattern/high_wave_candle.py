from typing import List, Dict, Optional
import numpy as np

class HighWaveCandlePattern:
    def __init__(self, position: int, open_price: float, high_price: float,
                 low_price: float, close_price: float, body_size: float,
                 upper_shadow: float, lower_shadow: float,
                 upper_shadow_ratio: float, lower_shadow_ratio: float,
                 volatility_score: float, strength: float):
        self.position = position
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.body_size = body_size
        self.upper_shadow = upper_shadow
        self.lower_shadow = lower_shadow
        self.upper_shadow_ratio = upper_shadow_ratio
        self.lower_shadow_ratio = lower_shadow_ratio
        self.volatility_score = volatility_score
        self.strength = strength

class HighWaveCandleDetector:
    def __init__(self, min_shadow_body_ratio: float = 2.0,
                 min_total_range_atr_ratio: float = 1.5):
        self.min_shadow_body_ratio = min_shadow_body_ratio
        self.min_total_range_atr_ratio = min_total_range_atr_ratio
        self.patterns: List[HighWaveCandlePattern] = []

    def detect_patterns(self, opens: List[float], highs: List[float],
                        lows: List[float], closes: List[float],
                        atr: float) -> Dict:
        for idx in range(1, len(opens) - 1):
            pattern = self._detect_pattern(opens, highs, lows, closes, atr, idx)
            if pattern is not None:
                self.patterns.append(pattern)
        return self._create_results()

    def _detect_pattern(self, opens: List[float], highs: List[float],
                        lows: List[float], closes: List[float],
                        atr: float, idx: int) -> Optional[HighWaveCandlePattern]:
        """Detect high wave candle pattern"""
        open_price = opens[idx]
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        
        # Calculate components
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0 or body_size == 0:
            return None
        
        # Check if range is significant relative to ATR
        if atr > 0 and total_range / atr < self.min_total_range_atr_ratio:
            return None
        
        # Calculate shadows
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Calculate shadow ratios
        upper_shadow_ratio = upper_shadow / body_size
        lower_shadow_ratio = lower_shadow / body_size
        
        # Check if both shadows meet minimum criteria
        if upper_shadow_ratio < self.min_shadow_body_ratio:
            return None
        if lower_shadow_ratio < self.min_shadow_body_ratio:
            return None
        
        # Calculate volatility score
        volatility_score = self._calculate_volatility_score(
            total_range, atr, upper_shadow_ratio, lower_shadow_ratio
        )
        
        # Calculate strength
        strength = self._calculate_strength(
            upper_shadow_ratio, lower_shadow_ratio, volatility_score
        )
        
        return HighWaveCandlePattern(
            position=idx,
            open_price=open_price,
            high_price=high,
            low_price=low,
            close_price=close,
            body_size=body_size,
            upper_shadow=upper_shadow,
            lower_shadow=lower_shadow,
            upper_shadow_ratio=upper_shadow_ratio,
            lower_shadow_ratio=lower_shadow_ratio,
            volatility_score=volatility_score,
            strength=strength
        )
    
    def _calculate_volatility_score(self, total_range: float, atr: float,
                                  upper_shadow_ratio: float,
                                  lower_shadow_ratio: float) -> float:
        """Calculate volatility score (0-100)"""
        score = 50.0
        
        # Range relative to ATR
        if atr > 0:
            range_ratio = total_range / atr
            if range_ratio > 3:
                score += 30
            elif range_ratio > 2:
                score += 20
            else:
                score += 10
        
        # Shadow ratios
        avg_shadow_ratio = (upper_shadow_ratio + lower_shadow_ratio) / 2
        if avg_shadow_ratio > 5:
            score += 20
        elif avg_shadow_ratio > 4:
            score += 10
        
        return min(100, max(0, score))
    
    def _calculate_strength(self, upper_shadow_ratio: float,
                          lower_shadow_ratio: float,
                          volatility_score: float) -> float:
        """Calculate pattern strength"""
        strength = 60.0
        
        # Longer shadows = stronger pattern
        avg_shadow_ratio = (upper_shadow_ratio + lower_shadow_ratio) / 2
        if avg_shadow_ratio > 5:
            strength += 25
        elif avg_shadow_ratio > 4:
            strength += 15
        
        # Higher volatility = stronger pattern
        strength += volatility_score * 0.15
        
        return min(100, max(0, strength))
    
    def _create_results(self) -> Dict:
        """Create results dictionary"""
        return {
            'patterns': [
                {
                    'position': p.position,
                    'body_size': p.body_size,
                    'upper_shadow': p.upper_shadow,
                    'lower_shadow': p.lower_shadow,
                    'upper_shadow_ratio': p.upper_shadow_ratio,
                    'lower_shadow_ratio': p.lower_shadow_ratio,
                    'volatility_score': p.volatility_score,
                    'strength': p.strength,
                    'candle': {
                        'open': p.open_price,
                        'high': p.high_price,
                        'low': p.low_price,
                        'close': p.close_price
                    }
                }
                for p in self.patterns
            ],
            'summary': {
                'total_patterns': len(self.patterns),
                'avg_strength': np.mean([p.strength for p in self.patterns]) if self.patterns else 0,
                'avg_volatility': np.mean([p.volatility_score for p in self.patterns]) if self.patterns else 0,
                'max_shadow_ratio': max([max(p.upper_shadow_ratio, p.lower_shadow_ratio) for p in self.patterns]) if self.patterns else 0
            }
        }