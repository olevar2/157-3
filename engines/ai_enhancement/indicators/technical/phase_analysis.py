"""
Phase Analysis Indicator

Analyzes market phases using cyclical analysis and dominant cycle identification
to determine the current market regime and predict phase transitions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class PhaseAnalysisResult:
    current_phase: str                  # "accumulation", "markup", "distribution", "markdown"
    phase_strength: float               # Strength of current phase (0-1)
    cycle_period: Optional[int]         # Dominant cycle length in bars
    phase_progress: float               # Progress within current phase (0-1)
    next_phase: str                     # Expected next phase
    transition_probability: float       # Probability of phase transition (0-1)
    volume_confirmation: bool           # Volume supports phase analysis
    momentum_alignment: bool            # Momentum aligns with phase
    phase_duration: int                 # Bars in current phase
    timestamp: Optional[str] = None


class PhaseAnalysis(StandardIndicatorInterface):
    """
    Market Phase Analysis Indicator
    
    Identifies market phases using Wyckoff methodology combined with
    cyclical analysis to determine accumulation, markup, distribution,
    and markdown phases for strategic positioning.
    """
    
    CATEGORY = "technical"
    
    def __init__(self,
                 lookback: int = 100,
                 cycle_min: int = 10,
                 cycle_max: int = 50,
                 volume_ma_period: int = 20,
                 price_ma_period: int = 20,
                 momentum_period: int = 14,
                 phase_confirmation_bars: int = 5,
                 **kwargs):
        """
        Initialize Phase Analysis indicator.
        
        Args:
            lookback: Number of periods for phase analysis
            cycle_min: Minimum cycle length to detect
            cycle_max: Maximum cycle length to detect
            volume_ma_period: Moving average period for volume analysis
            price_ma_period: Moving average period for price trend
            momentum_period: Period for momentum calculation
            phase_confirmation_bars: Bars needed to confirm phase change
        """
        super().__init__(**kwargs)
        self.lookback = lookback
        self.cycle_min = cycle_min
        self.cycle_max = cycle_max
        self.volume_ma_period = volume_ma_period
        self.price_ma_period = price_ma_period
        self.momentum_period = momentum_period
        self.phase_confirmation_bars = phase_confirmation_bars
    
    def calculate(self, data: pd.DataFrame) -> PhaseAnalysisResult:
        """
        Analyze market phases.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            PhaseAnalysisResult with phase analysis
        """
        try:
            if len(data) < self.lookback:
                return PhaseAnalysisResult(
                    current_phase="insufficient_data",
                    phase_strength=0.0,
                    cycle_period=None,
                    phase_progress=0.0,
                    next_phase="unknown",
                    transition_probability=0.0,
                    volume_confirmation=False,
                    momentum_alignment=False,
                    phase_duration=0
                )
            
            # Get recent data
            recent_data = data.tail(self.lookback).copy()
            
            # Calculate technical indicators needed for phase analysis
            indicators = self._calculate_indicators(recent_data)
            
            # Detect dominant cycle
            cycle_period = self._detect_dominant_cycle(recent_data)
            
            # Analyze current phase
            current_phase, phase_strength = self._analyze_current_phase(recent_data, indicators)
            
            # Calculate phase progress and duration
            phase_progress, phase_duration = self._calculate_phase_metrics(recent_data, current_phase)
            
            # Determine next expected phase
            next_phase = self._determine_next_phase(current_phase, phase_progress)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(
                recent_data, current_phase, phase_progress, indicators
            )
            
            # Check confirmations
            volume_confirmation = self._check_volume_confirmation(recent_data, current_phase, indicators)
            momentum_alignment = self._check_momentum_alignment(current_phase, indicators)
            
            return PhaseAnalysisResult(
                current_phase=current_phase,
                phase_strength=phase_strength,
                cycle_period=cycle_period,
                phase_progress=phase_progress,
                next_phase=next_phase,
                transition_probability=transition_prob,
                volume_confirmation=volume_confirmation,
                momentum_alignment=momentum_alignment,
                phase_duration=phase_duration,
                timestamp=recent_data.index[-1].isoformat() if hasattr(recent_data.index[-1], 'isoformat') else None
            )
            
        except Exception as e:
            return PhaseAnalysisResult(
                current_phase="error",
                phase_strength=0.0,
                cycle_period=None,
                phase_progress=0.0,
                next_phase="unknown",
                transition_probability=0.0,
                volume_confirmation=False,
                momentum_alignment=False,
                phase_duration=0
            )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for phase analysis."""
        indicators = {}
        
        # Price moving averages
        indicators['price_ma'] = data['close'].rolling(window=self.price_ma_period).mean()
        indicators['price_trend'] = (data['close'] / indicators['price_ma'] - 1) * 100
        
        # Volume analysis
        indicators['volume_ma'] = data['volume'].rolling(window=self.volume_ma_period).mean()
        indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        
        # Momentum indicators
        indicators['roc'] = data['close'].pct_change(self.momentum_period) * 100
        
        # Price range analysis
        indicators['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        indicators['atr'] = indicators['true_range'].rolling(window=14).mean()
        
        # Relative strength vs. moving average
        indicators['relative_strength'] = (data['close'] - indicators['price_ma']) / indicators['atr']
        
        return indicators
    
    def _detect_dominant_cycle(self, data: pd.DataFrame) -> Optional[int]:
        """Detect dominant cycle using spectral analysis."""
        try:
            # Use log returns for cycle detection
            returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            
            if len(returns) < self.cycle_max * 2:
                return None
            
            # Simple autocorrelation-based cycle detection
            autocorr_values = []
            
            for lag in range(self.cycle_min, min(self.cycle_max, len(returns) // 2)):
                correlation = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                autocorr_values.append((lag, abs(correlation)))
            
            if not autocorr_values:
                return None
            
            # Find the lag with highest correlation
            best_cycle = max(autocorr_values, key=lambda x: x[1])
            
            # Only return if correlation is significant
            if best_cycle[1] > 0.1:
                return best_cycle[0]
            
            return None
            
        except Exception:
            return None
    
    def _analyze_current_phase(self, data: pd.DataFrame, indicators: Dict) -> Tuple[str, float]:
        """Analyze current market phase using Wyckoff methodology."""
        # Get recent values
        current_price = data['close'].iloc[-1]
        price_trend = indicators['price_trend'].iloc[-10:].mean()  # Average trend over last 10 bars
        volume_ratio = indicators['volume_ratio'].iloc[-10:].mean()
        momentum = indicators['roc'].iloc[-10:].mean()
        relative_strength = indicators['relative_strength'].iloc[-5:].mean()
        
        # Phase scoring
        phase_scores = {
            'accumulation': 0.0,
            'markup': 0.0,
            'distribution': 0.0,
            'markdown': 0.0
        }
        
        # Accumulation phase characteristics:
        # - Price below/near moving average
        # - High volume, low volatility
        # - Momentum turning positive
        if price_trend <= 0:  # Price at/below MA
            phase_scores['accumulation'] += 0.3
        if volume_ratio > 1.2:  # Above average volume
            phase_scores['accumulation'] += 0.3
        if momentum > -1 and relative_strength > -2:  # Momentum improving
            phase_scores['accumulation'] += 0.4
        
        # Markup phase characteristics:
        # - Price above moving average and rising
        # - Increasing volume and momentum
        # - Strong relative strength
        if price_trend > 2:  # Price well above MA
            phase_scores['markup'] += 0.4
        if volume_ratio > 1.0 and momentum > 1:  # Good volume and momentum
            phase_scores['markup'] += 0.3
        if relative_strength > 1:  # Strong relative performance
            phase_scores['markup'] += 0.3
        
        # Distribution phase characteristics:
        # - Price above MA but momentum declining
        # - High volume but poor price progress
        # - Relative strength weakening
        if price_trend > 0 and momentum < 1:  # Price up but momentum weak
            phase_scores['distribution'] += 0.4
        if volume_ratio > 1.2 and momentum < 0:  # High volume, negative momentum
            phase_scores['distribution'] += 0.3
        if relative_strength < 0 and price_trend > 0:  # Weakening relative strength
            phase_scores['distribution'] += 0.3
        
        # Markdown phase characteristics:
        # - Price below moving average and falling
        # - Increasing volume on declines
        # - Negative momentum and relative strength
        if price_trend < -2:  # Price well below MA
            phase_scores['markdown'] += 0.4
        if momentum < -1:  # Strong negative momentum
            phase_scores['markdown'] += 0.3
        if relative_strength < -1:  # Weak relative performance
            phase_scores['markdown'] += 0.3
        
        # Determine dominant phase
        best_phase = max(phase_scores.items(), key=lambda x: x[1])
        
        # Minimum threshold for phase identification
        if best_phase[1] < 0.5:
            return "transition", best_phase[1]
        
        return best_phase[0], best_phase[1]
    
    def _calculate_phase_metrics(self, data: pd.DataFrame, current_phase: str) -> Tuple[float, int]:
        """Calculate phase progress and duration."""
        # Simplified phase progress calculation
        # In a real implementation, this would track phase changes over time
        
        # Use price momentum to estimate phase progress
        momentum = data['close'].pct_change(5).iloc[-10:].mean() * 100
        
        if current_phase == "accumulation":
            # Progress based on momentum improvement
            progress = max(0, min(1, (momentum + 5) / 10))  # Normalize from -5 to +5
        elif current_phase == "markup":
            # Progress based on sustained momentum
            progress = max(0, min(1, momentum / 10))  # Normalize positive momentum
        elif current_phase == "distribution":
            # Progress based on momentum deterioration
            progress = max(0, min(1, (5 - momentum) / 10))  # Reverse momentum
        elif current_phase == "markdown":
            # Progress based on negative momentum
            progress = max(0, min(1, abs(momentum) / 10))  # Absolute negative momentum
        else:
            progress = 0.5  # Default for transition/unknown
        
        # Estimate phase duration (simplified)
        # Count bars where trend direction has been consistent
        price_changes = data['close'].diff()
        if current_phase in ["markup", "accumulation"]:
            consistent_bars = sum(price_changes.tail(20) >= 0)
        else:
            consistent_bars = sum(price_changes.tail(20) <= 0)
        
        phase_duration = min(consistent_bars, 20)  # Cap at 20 bars
        
        return progress, phase_duration
    
    def _determine_next_phase(self, current_phase: str, phase_progress: float) -> str:
        """Determine the expected next phase."""
        phase_sequence = {
            "accumulation": "markup",
            "markup": "distribution", 
            "distribution": "markdown",
            "markdown": "accumulation"
        }
        
        if current_phase in phase_sequence:
            return phase_sequence[current_phase]
        else:
            return "unknown"
    
    def _calculate_transition_probability(self, data: pd.DataFrame, current_phase: str, 
                                        phase_progress: float, indicators: Dict) -> float:
        """Calculate probability of phase transition."""
        # Base probability on phase progress
        base_prob = phase_progress
        
        # Adjust based on technical signals
        recent_momentum = indicators['roc'].iloc[-5:].mean()
        volume_trend = indicators['volume_ratio'].iloc[-5:].mean()
        
        # Higher probability if:
        # 1. Phase progress is high
        # 2. Momentum is changing direction
        # 3. Volume patterns support transition
        
        momentum_factor = 0.0
        if current_phase == "accumulation" and recent_momentum > 0:
            momentum_factor = 0.2
        elif current_phase == "markup" and recent_momentum < 1:
            momentum_factor = 0.2
        elif current_phase == "distribution" and recent_momentum < 0:
            momentum_factor = 0.2
        elif current_phase == "markdown" and recent_momentum > -2:
            momentum_factor = 0.2
        
        volume_factor = min(0.2, (volume_trend - 1.0) * 0.2) if volume_trend > 1.0 else 0.0
        
        total_prob = min(1.0, base_prob + momentum_factor + volume_factor)
        
        return total_prob
    
    def _check_volume_confirmation(self, data: pd.DataFrame, current_phase: str, 
                                  indicators: Dict) -> bool:
        """Check if volume confirms the current phase."""
        volume_ratio = indicators['volume_ratio'].iloc[-5:].mean()
        
        if current_phase == "accumulation":
            # Accumulation should have above-average volume
            return volume_ratio > 1.1
        elif current_phase == "markup":
            # Markup should have increasing volume
            return volume_ratio > 0.9
        elif current_phase == "distribution":
            # Distribution should have high volume but poor price progress
            price_progress = data['close'].iloc[-1] / data['close'].iloc[-10] - 1
            return volume_ratio > 1.1 and price_progress < 0.02
        elif current_phase == "markdown":
            # Markdown can have variable volume
            return True  # Less strict for markdown
        
        return False
    
    def _check_momentum_alignment(self, current_phase: str, indicators: Dict) -> bool:
        """Check if momentum aligns with the current phase."""
        momentum = indicators['roc'].iloc[-5:].mean()
        
        if current_phase == "accumulation":
            # Momentum should be stabilizing or improving
            return momentum > -2
        elif current_phase == "markup":
            # Momentum should be positive
            return momentum > 0
        elif current_phase == "distribution":
            # Momentum should be weakening
            return momentum < 2
        elif current_phase == "markdown":
            # Momentum should be negative
            return momentum < 0
        
        return True  # Default to aligned for unknown phases
    
    def get_display_name(self) -> str:
        return "Market Phase Analysis"
    
    def get_parameters(self) -> Dict:
        return {
            "lookback": self.lookback,
            "cycle_min": self.cycle_min,
            "cycle_max": self.cycle_max,
            "volume_ma_period": self.volume_ma_period,
            "price_ma_period": self.price_ma_period,
            "momentum_period": self.momentum_period,
            "phase_confirmation_bars": self.phase_confirmation_bars
        }