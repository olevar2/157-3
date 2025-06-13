"""
Three White Soldiers Candlestick Pattern Implementation
CCI Gold Standard - Production Ready

The Three White Soldiers pattern consists of three consecutive long bullish candles
with progressively higher highs and higher lows, indicating strong upward momentum.
This pattern suggests a potential reversal from bearish to bullish trend.

Key Features:
- Three consecutive bullish (white/green) candles
- Each candle should have a relatively large real body
- Each candle opens within the previous candle's real body
- Each candle closes progressively higher
- Limited upper shadows preferred
- Pattern appears after a downtrend or consolidation

Market Psychology:
- Represents increasing buying pressure over three sessions
- Bulls gaining control and pushing prices consistently higher
- Strong momentum continuation signal
- High reliability when appearing after significant decline

Author: Platform3 AI Enhancement Engine
Version: 1.0.0
Category: Candlestick Pattern Recognition
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from decimal import Decimal
import numpy as np
import pandas as pd

from engines.indicator_base import IndicatorBase


@dataclass
class SoldiersSignalResult:
    """
    Result data class for Three White Soldiers pattern detection.
    
    Attributes:
        is_detected: Whether the Three White Soldiers pattern is detected
        pattern_strength: Strength score (0.0 to 1.0)
        candle1_data: First soldier candle analysis
        candle2_data: Second soldier candle analysis  
        candle3_data: Third soldier candle analysis
        body_progression: Progressive body size analysis
        shadow_analysis: Upper shadow analysis for all three candles
        volume_confirmation: Volume trend analysis if available
        reliability_score: Pattern reliability assessment
        breakout_potential: Potential for continued upward movement
        entry_level: Suggested entry price level
        target_levels: Potential target price levels
        stop_loss: Suggested stop-loss level
        pattern_context: Market context analysis
        formation_quality: Quality assessment of pattern formation
        market_conditions: Overall market environment assessment
    """
    is_detected: bool
    pattern_strength: float
    candle1_data: Dict[str, Any]
    candle2_data: Dict[str, Any]
    candle3_data: Dict[str, Any]
    body_progression: Dict[str, float]
    shadow_analysis: Dict[str, float]
    volume_confirmation: Optional[Dict[str, Any]]
    reliability_score: float
    breakout_potential: float
    entry_level: Optional[float]
    target_levels: List[float]
    stop_loss: Optional[float]
    pattern_context: str
    formation_quality: float
    market_conditions: str


class SoldiersSignal(IndicatorBase):
    """
    Three White Soldiers Pattern Detector
    
    Implements advanced detection for the Three White Soldiers bullish reversal pattern
    with sophisticated validation criteria and market context analysis.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.6,
                 max_upper_shadow_ratio: float = 0.3,
                 min_progression_ratio: float = 0.1,
                 volume_confirmation: bool = True,
                 lookback_periods: int = 20):
        """
        Initialize Three White Soldiers pattern detector.
        
        Args:
            min_body_ratio: Minimum body-to-range ratio for each candle
            max_upper_shadow_ratio: Maximum upper shadow ratio allowed
            min_progression_ratio: Minimum progression in closing prices
            volume_confirmation: Whether to use volume for confirmation
            lookback_periods: Periods to analyze for context
        """
        super().__init__()
        self.min_body_ratio = min_body_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio
        self.min_progression_ratio = min_progression_ratio
        self.volume_confirmation = volume_confirmation
        self.lookback_periods = lookback_periods
        
        # Pattern detection thresholds
        self.min_pattern_strength = 0.65
        self.strong_pattern_threshold = 0.80
        
        # Market context parameters
        self.trend_analysis_periods = 14
        self.volatility_periods = 10
        
    def calculate(self, data: pd.DataFrame) -> SoldiersSignalResult:
        """
        Calculate Three White Soldiers pattern detection.
        
        Args:
            data: OHLCV DataFrame with at least 3 periods
            
        Returns:
            SoldiersSignalResult with comprehensive pattern analysis
        """
        try:
            if len(data) < 3:
                return self._create_no_pattern_result("Insufficient data")
                
            # Get the last three candles
            candles = data.tail(3).copy()
            
            # Basic candle analysis
            candle_analysis = self._analyze_individual_candles(candles)
            
            # Pattern validation
            pattern_detected, strength = self._validate_soldiers_pattern(candles, candle_analysis)
            
            if not pattern_detected:
                return self._create_no_pattern_result("Pattern criteria not met")
                
            # Advanced analysis
            body_progression = self._analyze_body_progression(candles)
            shadow_analysis = self._analyze_shadows(candles)
            volume_data = self._analyze_volume_confirmation(candles) if self.volume_confirmation else None
            
            # Market context analysis
            context_analysis = self._analyze_market_context(data)
            
            # Calculate reliability and targets
            reliability = self._calculate_reliability_score(candles, strength, context_analysis)
            breakout_potential = self._assess_breakout_potential(candles, context_analysis)
            
            # Generate trading levels
            entry_level = float(candles.iloc[-1]['close'])
            target_levels = self._calculate_target_levels(candles)
            stop_loss = self._calculate_stop_loss(candles)
            
            return SoldiersSignalResult(
                is_detected=True,
                pattern_strength=strength,
                candle1_data=candle_analysis[0],
                candle2_data=candle_analysis[1],
                candle3_data=candle_analysis[2],
                body_progression=body_progression,
                shadow_analysis=shadow_analysis,
                volume_confirmation=volume_data,
                reliability_score=reliability,
                breakout_potential=breakout_potential,
                entry_level=entry_level,
                target_levels=target_levels,
                stop_loss=stop_loss,
                pattern_context=context_analysis['context_description'],
                formation_quality=self._assess_formation_quality(candles, strength),
                market_conditions=context_analysis['market_conditions']
            )
            
        except Exception as e:
            return self._create_error_result(f"Calculation error: {str(e)}")
    
    def _analyze_individual_candles(self, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each individual candle in the pattern."""
        analysis = []
        
        for i, (_, candle) in enumerate(candles.iterrows()):
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            
            # Basic measurements
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # Ratios
            body_ratio = body_size / total_range if total_range > 0 else 0
            upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
            
            # Candle characteristics
            is_bullish = close_price > open_price
            is_strong_body = body_ratio >= self.min_body_ratio
            has_small_upper_shadow = upper_shadow_ratio <= self.max_upper_shadow_ratio
            
            analysis.append({
                'candle_index': i + 1,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'body_size': body_size,
                'total_range': total_range,
                'upper_shadow': upper_shadow,
                'lower_shadow': lower_shadow,
                'body_ratio': body_ratio,
                'upper_shadow_ratio': upper_shadow_ratio,
                'is_bullish': is_bullish,
                'is_strong_body': is_strong_body,
                'has_small_upper_shadow': has_small_upper_shadow,
                'candle_quality': body_ratio * (1 - upper_shadow_ratio)
            })
            
        return analysis
    
    def _validate_soldiers_pattern(self, candles: pd.DataFrame, 
                                 candle_analysis: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """Validate the Three White Soldiers pattern criteria."""
        strength_components = []
        
        # Check all candles are bullish
        all_bullish = all(analysis['is_bullish'] for analysis in candle_analysis)
        if not all_bullish:
            return False, 0.0
        strength_components.append(0.3)  # Base score for all bullish
        
        # Check body strength
        strong_bodies = all(analysis['is_strong_body'] for analysis in candle_analysis)
        if strong_bodies:
            strength_components.append(0.25)
        
        # Check upper shadows
        small_shadows = all(analysis['has_small_upper_shadow'] for analysis in candle_analysis)
        if small_shadows:
            strength_components.append(0.2)
        
        # Check progressive highs and lows
        closes = [analysis['close'] for analysis in candle_analysis]
        highs = [analysis['high'] for analysis in candle_analysis]
        lows = [analysis['low'] for analysis in candle_analysis]
        
        progressive_closes = closes[0] < closes[1] < closes[2]
        progressive_highs = highs[0] < highs[1] < highs[2]
        
        if progressive_closes and progressive_highs:
            strength_components.append(0.15)
        
        # Check opening within previous body
        opens = [analysis['open'] for analysis in candle_analysis]
        
        open_within_body_2 = (candle_analysis[0]['close'] > opens[1] > candle_analysis[0]['open'])
        open_within_body_3 = (candle_analysis[1]['close'] > opens[2] > candle_analysis[1]['open'])
        
        if open_within_body_2 and open_within_body_3:
            strength_components.append(0.1)
        
        # Calculate total strength
        total_strength = sum(strength_components)
        
        # Pattern is valid if meets minimum criteria
        pattern_valid = (all_bullish and progressive_closes and 
                        total_strength >= self.min_pattern_strength)
        
        return pattern_valid, total_strength
    
    def _analyze_body_progression(self, candles: pd.DataFrame) -> Dict[str, float]:
        """Analyze the progression of candle bodies."""
        bodies = []
        
        for _, candle in candles.iterrows():
            body_size = abs(float(candle['close']) - float(candle['open']))
            bodies.append(body_size)
        
        # Calculate progression metrics
        progression_1_to_2 = (bodies[1] - bodies[0]) / bodies[0] if bodies[0] > 0 else 0
        progression_2_to_3 = (bodies[2] - bodies[1]) / bodies[1] if bodies[1] > 0 else 0
        
        avg_progression = (progression_1_to_2 + progression_2_to_3) / 2
        consistency = 1.0 - abs(progression_1_to_2 - progression_2_to_3) / 2
        
        return {
            'body_sizes': bodies,
            'progression_1_to_2': progression_1_to_2,
            'progression_2_to_3': progression_2_to_3,
            'average_progression': avg_progression,
            'progression_consistency': consistency,
            'total_progression': (bodies[2] - bodies[0]) / bodies[0] if bodies[0] > 0 else 0
        }
    
    def _analyze_shadows(self, candles: pd.DataFrame) -> Dict[str, float]:
        """Analyze upper and lower shadows of all candles."""
        upper_shadows = []
        lower_shadows = []
        shadow_ratios = []
        
        for _, candle in candles.iterrows():
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            upper_shadows.append(upper_shadow)
            lower_shadows.append(lower_shadow)
            
            if total_range > 0:
                shadow_ratios.append(upper_shadow / total_range)
            else:
                shadow_ratios.append(0)
        
        return {
            'upper_shadows': upper_shadows,
            'lower_shadows': lower_shadows,
            'upper_shadow_ratios': shadow_ratios,
            'average_upper_shadow_ratio': sum(shadow_ratios) / len(shadow_ratios),
            'max_upper_shadow_ratio': max(shadow_ratios),
            'shadow_uniformity': 1.0 - np.std(shadow_ratios) if len(shadow_ratios) > 1 else 1.0
        }
    
    def _analyze_volume_confirmation(self, candles: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze volume confirmation for the pattern."""
        if 'volume' not in candles.columns:
            return None
            
        volumes = candles['volume'].tolist()
        
        # Check for increasing volume trend
        volume_increasing = volumes[0] < volumes[1] < volumes[2]
        
        # Calculate volume progression
        vol_prog_1_to_2 = (volumes[1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
        vol_prog_2_to_3 = (volumes[2] - volumes[1]) / volumes[1] if volumes[1] > 0 else 0
        
        return {
            'volumes': volumes,
            'volume_increasing': volume_increasing,
            'volume_progression_1_to_2': vol_prog_1_to_2,
            'volume_progression_2_to_3': vol_prog_2_to_3,
            'average_volume_progression': (vol_prog_1_to_2 + vol_prog_2_to_3) / 2,
            'volume_confirmation_strength': 0.8 if volume_increasing else 0.3
        }
    
    def _analyze_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze broader market context for pattern reliability."""
        if len(data) < self.trend_analysis_periods:
            return {
                'context_description': 'Insufficient data for context analysis',
                'market_conditions': 'Unknown',
                'trend_strength': 0.5,
                'volatility_level': 0.5
            }
        
        # Calculate trend context
        recent_data = data.tail(self.trend_analysis_periods)
        closes = recent_data['close'].values
        
        # Simple trend analysis
        trend_slope = (closes[-1] - closes[0]) / len(closes)
        trend_strength = abs(trend_slope) / np.mean(closes)
        
        # Volatility analysis
        volatility_data = data.tail(self.volatility_periods)
        price_changes = volatility_data['close'].pct_change().dropna()
        volatility = price_changes.std() if len(price_changes) > 1 else 0
        
        # Determine market conditions
        if trend_slope < -0.001:
            market_condition = 'Bearish trend - Good reversal context'
            context_description = 'Pattern appears after bearish trend - High reversal potential'
        elif trend_slope > 0.001:
            market_condition = 'Bullish trend - Continuation pattern'
            context_description = 'Pattern appears in uptrend - Momentum continuation'
        else:
            market_condition = 'Sideways market - Breakout potential'
            context_description = 'Pattern appears in consolidation - Potential breakout'
        
        return {
            'context_description': context_description,
            'market_conditions': market_condition,
            'trend_strength': min(trend_strength * 10, 1.0),
            'volatility_level': min(volatility * 100, 1.0),
            'trend_slope': trend_slope
        }
    
    def _calculate_reliability_score(self, candles: pd.DataFrame, pattern_strength: float,
                                   context: Dict[str, Any]) -> float:
        """Calculate overall pattern reliability score."""
        # Base reliability from pattern strength
        reliability = pattern_strength * 0.6
        
        # Context adjustment
        if 'Bearish trend' in context['market_conditions']:
            reliability += 0.2  # Better reliability in bearish context
        elif 'Sideways' in context['market_conditions']:
            reliability += 0.1  # Moderate reliability in sideways market
        
        # Trend strength adjustment
        reliability += context['trend_strength'] * 0.1
        
        # Volatility adjustment (prefer moderate volatility)
        volatility = context['volatility_level']
        if 0.3 <= volatility <= 0.7:
            reliability += 0.1
        
        return min(reliability, 1.0)
    
    def _assess_breakout_potential(self, candles: pd.DataFrame, 
                                 context: Dict[str, Any]) -> float:
        """Assess potential for continued upward movement."""
        # Base potential from pattern characteristics
        closes = candles['close'].values
        price_momentum = (closes[-1] - closes[0]) / closes[0]
        
        base_potential = min(price_momentum * 10, 0.6)
        
        # Context adjustments
        if 'Bearish trend' in context['market_conditions']:
            base_potential += 0.3  # High breakout potential from reversal
        elif 'Bullish trend' in context['market_conditions']:
            base_potential += 0.2  # Good continuation potential
        
        # Trend strength bonus
        base_potential += context['trend_strength'] * 0.1
        
        return min(base_potential, 1.0)
    
    def _calculate_target_levels(self, candles: pd.DataFrame) -> List[float]:
        """Calculate potential target levels."""
        current_close = float(candles.iloc[-1]['close'])
        pattern_range = float(candles.iloc[-1]['close']) - float(candles.iloc[0]['open'])
        
        # Conservative and aggressive targets
        targets = [
            current_close + pattern_range * 0.5,  # Conservative target
            current_close + pattern_range * 1.0,  # Pattern height target
            current_close + pattern_range * 1.618  # Fibonacci extension
        ]
        
        return targets
    
    def _calculate_stop_loss(self, candles: pd.DataFrame) -> float:
        """Calculate suggested stop-loss level."""
        # Stop below the lowest low of the pattern with small buffer
        pattern_low = float(candles['low'].min())
        stop_buffer = (float(candles.iloc[-1]['close']) - pattern_low) * 0.1
        
        return pattern_low - stop_buffer
    
    def _assess_formation_quality(self, candles: pd.DataFrame, strength: float) -> float:
        """Assess the overall quality of pattern formation."""
        # Base quality from pattern strength
        quality = strength * 0.7
        
        # Size consistency bonus
        bodies = [abs(float(candle['close']) - float(candle['open'])) 
                 for _, candle in candles.iterrows()]
        
        body_consistency = 1.0 - (np.std(bodies) / np.mean(bodies)) if np.mean(bodies) > 0 else 0
        quality += body_consistency * 0.2
        
        # Progressive improvement bonus
        closes = candles['close'].values
        if all(closes[i] < closes[i+1] for i in range(len(closes)-1)):
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _create_no_pattern_result(self, reason: str) -> SoldiersSignalResult:
        """Create result object when no pattern is detected."""
        return SoldiersSignalResult(
            is_detected=False,
            pattern_strength=0.0,
            candle1_data={},
            candle2_data={},
            candle3_data={},
            body_progression={},
            shadow_analysis={},
            volume_confirmation=None,
            reliability_score=0.0,
            breakout_potential=0.0,
            entry_level=None,
            target_levels=[],
            stop_loss=None,
            pattern_context=reason,
            formation_quality=0.0,
            market_conditions='No pattern detected'
        )
    
    def _create_error_result(self, error_msg: str) -> SoldiersSignalResult:
        """Create result object when an error occurs."""
        return SoldiersSignalResult(
            is_detected=False,
            pattern_strength=0.0,
            candle1_data={},
            candle2_data={},
            candle3_data={},
            body_progression={},
            shadow_analysis={},
            volume_confirmation=None,
            reliability_score=0.0,
            breakout_potential=0.0,
            entry_level=None,
            target_levels=[],
            stop_loss=None,
            pattern_context=f'Error: {error_msg}',
            formation_quality=0.0,
            market_conditions='Error in calculation'
        )


# Test and demonstration code
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'open': [100.0, 101.5, 103.2, 104.8, 106.5],
        'high': [101.2, 103.0, 104.8, 106.2, 108.1],
        'low': [99.8, 101.0, 102.9, 104.5, 106.0],
        'close': [101.0, 102.8, 104.5, 106.0, 107.8],
        'volume': [10000, 12000, 15000, 18000, 20000]
    })
    
    # Initialize detector
    detector = SoldiersSignal(
        min_body_ratio=0.6,
        max_upper_shadow_ratio=0.3,
        volume_confirmation=True
    )
    
    # Test pattern detection
    result = detector.calculate(sample_data)
    
    print("=== Three White Soldiers Pattern Detection Test ===")
    print(f"Pattern Detected: {result.is_detected}")
    print(f"Pattern Strength: {result.pattern_strength:.3f}")
    print(f"Reliability Score: {result.reliability_score:.3f}")
    print(f"Breakout Potential: {result.breakout_potential:.3f}")
    print(f"Formation Quality: {result.formation_quality:.3f}")
    print(f"Pattern Context: {result.pattern_context}")
    print(f"Market Conditions: {result.market_conditions}")
    
    if result.is_detected:
        print(f"\nEntry Level: {result.entry_level}")
        print(f"Target Levels: {result.target_levels}")
        print(f"Stop Loss: {result.stop_loss}")
        
        print(f"\nCandle Analysis:")
        print(f"Candle 1 Quality: {result.candle1_data.get('candle_quality', 0):.3f}")
        print(f"Candle 2 Quality: {result.candle2_data.get('candle_quality', 0):.3f}")
        print(f"Candle 3 Quality: {result.candle3_data.get('candle_quality', 0):.3f}")
        
        if result.volume_confirmation:
            print(f"\nVolume Confirmation: {result.volume_confirmation['volume_increasing']}")
            print(f"Volume Strength: {result.volume_confirmation['volume_confirmation_strength']:.3f}")
    
    print("\n=== Test Complete ===")