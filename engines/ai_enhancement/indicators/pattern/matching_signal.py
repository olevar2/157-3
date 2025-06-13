"""
Matching Signal Pattern Indicator

Identifies matching high/low patterns in candlestick data where multiple candles
show similar high or low prices within a specified tolerance. This pattern can
indicate potential support/resistance levels or consolidation phases.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class MatchingSignalResult:
    """Result structure for Matching Signal calculations"""
    signal_type: str  # "matching_high", "matching_low", "none"
    strength: float
    matches: List[int]  # Indices of matching candles
    confidence: float
    match_price: float
    timestamp: Optional[str] = None


class MatchingSignal(StandardIndicatorInterface):
    """
    Matching Signal Pattern Detector
    
    Identifies patterns where multiple candles show similar high or low prices,
    indicating potential support/resistance levels or market consolidation.
    """
    
    CATEGORY = "pattern"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, lookback: int = 5, tolerance: float = 0.001, 
                 min_matches: int = 3, **kwargs):
        """
        Initialize Matching Signal indicator
        
        Args:
            lookback: Number of periods to look back for matches
            tolerance: Price tolerance as percentage (0.001 = 0.1%)
            min_matches: Minimum number of matching candles required
        """
        self.lookback = lookback
        self.tolerance = tolerance
        self.min_matches = min_matches
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'lookback'):
            self.lookback = 5
        if not hasattr(self, 'tolerance'):
            self.tolerance = 0.001
        if not hasattr(self, 'min_matches'):
            self.min_matches = 3
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.lookback < 2:
            raise IndicatorValidationError("Lookback period must be at least 2")
        if self.tolerance <= 0 or self.tolerance > 0.1:
            raise IndicatorValidationError("Tolerance must be between 0 and 0.1 (10%)")
        if self.min_matches < 2:
            raise IndicatorValidationError("Minimum matches must be at least 2")
        if self.min_matches > self.lookback:
            raise IndicatorValidationError("Minimum matches cannot exceed lookback period")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return self.lookback + 1
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate matching signal patterns
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with matching signal results
        """
        try:
            self.validate_input_data(data)
            
            results = []
            
            for i in range(self.lookback, len(data)):
                window = data.iloc[i-self.lookback:i+1]
                
                # Check for matching highs
                high_matches = self._find_matches(window['high'].values, self.tolerance)
                high_signal = self._evaluate_matches(high_matches, "matching_high")
                
                # Check for matching lows
                low_matches = self._find_matches(window['low'].values, self.tolerance)
                low_signal = self._evaluate_matches(low_matches, "matching_low")
                
                # Select the stronger signal
                if high_signal.strength > low_signal.strength:
                    final_signal = high_signal
                    final_signal.match_price = window['high'].iloc[-1]
                elif low_signal.strength > 0:
                    final_signal = low_signal
                    final_signal.match_price = window['low'].iloc[-1]
                else:
                    final_signal = MatchingSignalResult(
                        signal_type="none",
                        strength=0.0,
                        matches=[],
                        confidence=0.0,
                        match_price=data['close'].iloc[i]
                    )
                
                results.append({
                    'signal_type': final_signal.signal_type,
                    'strength': final_signal.strength,
                    'confidence': final_signal.confidence,
                    'match_count': len(final_signal.matches),
                    'match_price': final_signal.match_price
                })
            
            # Create result DataFrame
            result_df = pd.DataFrame(results, index=data.index[self.lookback:])
            
            # Fill initial periods with neutral values
            initial_data = pd.DataFrame({
                'signal_type': ['none'] * self.lookback,
                'strength': [0.0] * self.lookback,
                'confidence': [0.0] * self.lookback,
                'match_count': [0] * self.lookback,
                'match_price': data['close'].iloc[:self.lookback].values
            }, index=data.index[:self.lookback])
            
            final_result = pd.concat([initial_data, result_df])
            self._last_calculation = final_result
            
            return final_result
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _find_matches(self, prices: np.ndarray, tolerance: float) -> List[int]:
        """Find indices of prices that match within tolerance"""
        matches = []
        reference_price = prices[-1]  # Use latest price as reference
        
        for i, price in enumerate(prices):
            if abs(price - reference_price) / reference_price <= tolerance:
                matches.append(i)
        
        return matches
    
    def _evaluate_matches(self, matches: List[int], signal_type: str) -> MatchingSignalResult:
        """Evaluate the quality of matches and create signal"""
        if len(matches) < self.min_matches:
            return MatchingSignalResult(
                signal_type="none",
                strength=0.0,
                matches=[],
                confidence=0.0,
                match_price=0.0
            )
        
        # Calculate strength based on number of matches and distribution
        match_count_score = min(len(matches) / self.lookback, 1.0)
        distribution_score = self._calculate_distribution_score(matches)
        
        strength = (match_count_score + distribution_score) / 2
        confidence = strength * (len(matches) / self.lookback)
        
        return MatchingSignalResult(
            signal_type=signal_type,
            strength=strength,
            matches=matches,
            confidence=confidence,
            match_price=0.0  # Will be set by caller
        )
    
    def _calculate_distribution_score(self, matches: List[int]) -> float:
        """Calculate how well distributed the matches are across the window"""
        if len(matches) <= 1:
            return 0.0
        
        # Calculate spacing between matches
        spacings = [matches[i+1] - matches[i] for i in range(len(matches)-1)]
        ideal_spacing = self.lookback / len(matches)
        
        # Score based on how close spacings are to ideal
        spacing_variance = np.var(spacings) if spacings else 0
        max_variance = (ideal_spacing / 2) ** 2
        
        distribution_score = max(0, 1 - (spacing_variance / max_variance))
        return min(distribution_score, 1.0)
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Matching Signal Pattern",
            category=self.CATEGORY,
            description="Identifies matching high/low patterns indicating support/resistance levels",
            parameters={
                "lookback": self.lookback,
                "tolerance": self.tolerance,
                "min_matches": self.min_matches
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Matching Signal ({self.lookback}, {self.tolerance:.1%})"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "lookback": self.lookback,
            "tolerance": self.tolerance,
            "min_matches": self.min_matches
        }