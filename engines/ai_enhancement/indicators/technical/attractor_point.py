"""
Attractor Point Indicator

Identifies market attractor points using dynamic systems theory concepts
to find price levels that repeatedly attract market activity.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class AttractorPointResult:
    """Result structure for Attractor Point analysis"""
    attractor_price: float  # Current attractor price level
    attraction_strength: float  # Strength of attraction (0-1)
    attractor_type: str  # "magnetic", "repulsive", "neutral"
    distance_to_attractor: float  # Current distance from attractor
    stability_score: float  # Stability of attractor point
    time_at_attractor: int  # Periods spent near attractor
    convergence_rate: float  # Rate of convergence to attractor


class AttractorPoint(StandardIndicatorInterface):
    """
    Market Attractor Point Analysis
    
    Identifies attractor points in market dynamics:
    - Price levels that repeatedly attract market activity
    - Strength and stability of attractors
    - Distance and convergence analysis
    - Magnetic vs repulsive behavior
    - Time-based attraction patterns
    """
    
    CATEGORY = "technical"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, lookback_period: int = 50, attraction_threshold: float = 0.02,
                 stability_period: int = 20, convergence_sensitivity: float = 0.5, **kwargs):
        """
        Initialize Attractor Point indicator
        
        Args:
            lookback_period: Period for attractor identification
            attraction_threshold: Price threshold for attraction zone
            stability_period: Period for stability analysis
            convergence_sensitivity: Sensitivity for convergence detection
        """
        self.lookback_period = lookback_period
        self.attraction_threshold = attraction_threshold
        self.stability_period = stability_period
        self.convergence_sensitivity = convergence_sensitivity
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'lookback_period'):
            self.lookback_period = 50
        if not hasattr(self, 'attraction_threshold'):
            self.attraction_threshold = 0.02
        if not hasattr(self, 'stability_period'):
            self.stability_period = 20
        if not hasattr(self, 'convergence_sensitivity'):
            self.convergence_sensitivity = 0.5
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.lookback_period < 10:
            raise IndicatorValidationError("Lookback period must be at least 10")
        if self.attraction_threshold <= 0 or self.attraction_threshold > 0.2:
            raise IndicatorValidationError("Attraction threshold must be between 0 and 0.2")
        if self.stability_period < 5:
            raise IndicatorValidationError("Stability period must be at least 5")
        if self.convergence_sensitivity <= 0 or self.convergence_sensitivity > 2:
            raise IndicatorValidationError("Convergence sensitivity must be between 0 and 2")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return self.lookback_period + 10
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate attractor point analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with attractor analysis
        """
        try:
            self.validate_input_data(data)
            
            results = []
            for i in range(len(data)):
                if i < self.lookback_period:
                    result = {
                        'attractor_price': data['close'].iloc[i],
                        'attraction_strength': 0.0,
                        'attractor_type': 'neutral',
                        'distance_to_attractor': 0.0,
                        'stability_score': 0.0,
                        'time_at_attractor': 0,
                        'convergence_rate': 0.0
                    }
                else:
                    # Analyze attractor at current position
                    attractor_analysis = self._analyze_attractor_point(i, data)
                    
                    result = {
                        'attractor_price': attractor_analysis.attractor_price,
                        'attraction_strength': attractor_analysis.attraction_strength,
                        'attractor_type': attractor_analysis.attractor_type,
                        'distance_to_attractor': attractor_analysis.distance_to_attractor,
                        'stability_score': attractor_analysis.stability_score,
                        'time_at_attractor': attractor_analysis.time_at_attractor,
                        'convergence_rate': attractor_analysis.convergence_rate
                    }
                
                results.append(result)
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _analyze_attractor_point(self, index: int, data: pd.DataFrame) -> AttractorPointResult:
        """Analyze attractor point at specific index"""
        
        # Get analysis window
        start_idx = max(0, index - self.lookback_period + 1)
        window_data = data.iloc[start_idx:index+1]
        
        # Identify potential attractor points
        attractor_candidates = self._identify_attractor_candidates(window_data)
        
        # Select primary attractor
        primary_attractor = self._select_primary_attractor(
            window_data, attractor_candidates, data['close'].iloc[index]
        )
        
        # Calculate attraction strength
        attraction_strength = self._calculate_attraction_strength(
            window_data, primary_attractor
        )
        
        # Determine attractor type
        attractor_type = self._determine_attractor_type(
            window_data, primary_attractor, data['close'].iloc[index]
        )
        
        # Calculate distance to attractor
        distance_to_attractor = abs(data['close'].iloc[index] - primary_attractor) / primary_attractor
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            window_data, primary_attractor
        )
        
        # Calculate time at attractor
        time_at_attractor = self._calculate_time_at_attractor(
            window_data, primary_attractor
        )
        
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate(
            window_data, primary_attractor
        )
        
        return AttractorPointResult(
            attractor_price=primary_attractor,
            attraction_strength=attraction_strength,
            attractor_type=attractor_type,
            distance_to_attractor=distance_to_attractor,
            stability_score=stability_score,
            time_at_attractor=time_at_attractor,
            convergence_rate=convergence_rate
        )
    
    def _identify_attractor_candidates(self, window_data: pd.DataFrame) -> List[float]:
        """Identify potential attractor price levels"""
        
        # Use multiple approaches to find attractors
        candidates = []
        
        # 1. Volume-weighted average price
        if 'volume' in window_data.columns:
            vwap = (window_data['close'] * window_data['volume']).sum() / window_data['volume'].sum()
            candidates.append(vwap)
        
        # 2. Price levels with high revisit frequency
        price_levels = self._find_high_frequency_levels(window_data)
        candidates.extend(price_levels)
        
        # 3. Support and resistance levels
        support_resistance = self._find_support_resistance_levels(window_data)
        candidates.extend(support_resistance)
        
        # 4. Moving average levels
        for period in [10, 20, 50]:
            if len(window_data) >= period:
                ma = window_data['close'].rolling(window=period).mean().iloc[-1]
                candidates.append(ma)
        
        # Remove duplicates and invalid values
        valid_candidates = [c for c in candidates if not np.isnan(c) and c > 0]
        unique_candidates = list(set([round(c, 4) for c in valid_candidates]))
        
        return unique_candidates[:10]  # Limit to top 10 candidates
    
    def _find_high_frequency_levels(self, window_data: pd.DataFrame) -> List[float]:
        """Find price levels with high revisit frequency"""
        
        # Create price bins
        price_min = window_data['low'].min()
        price_max = window_data['high'].max()
        num_bins = min(50, len(window_data))
        
        if price_max <= price_min:
            return []
        
        bin_edges = np.linspace(price_min, price_max, num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Count visits to each price level
        visit_counts = np.zeros(len(bin_centers))
        
        for _, row in window_data.iterrows():
            # Check which bins are covered by the candle's range
            low_bin = np.digitize(row['low'], bin_edges) - 1
            high_bin = np.digitize(row['high'], bin_edges) - 1
            
            # Increment visit count for covered bins
            for bin_idx in range(max(0, low_bin), min(len(bin_centers), high_bin + 1)):
                visit_counts[bin_idx] += 1
        
        # Find high-frequency levels
        threshold = np.percentile(visit_counts, 80)  # Top 20% of levels
        high_freq_indices = np.where(visit_counts >= threshold)[0]
        
        return [bin_centers[i] for i in high_freq_indices]
    
    def _find_support_resistance_levels(self, window_data: pd.DataFrame) -> List[float]:
        """Find support and resistance levels"""
        
        levels = []
        
        # Rolling highs and lows
        for period in [5, 10, 20]:
            if len(window_data) >= period:
                rolling_high = window_data['high'].rolling(window=period).max()
                rolling_low = window_data['low'].rolling(window=period).min()
                
                # Recent levels
                levels.append(rolling_high.iloc[-1])
                levels.append(rolling_low.iloc[-1])
        
        return levels
    
    def _select_primary_attractor(self, window_data: pd.DataFrame, 
                                candidates: List[float], current_price: float) -> float:
        """Select the primary attractor from candidates"""
        
        if not candidates:
            return current_price
        
        # Calculate attraction score for each candidate
        scores = []
        for candidate in candidates:
            score = self._calculate_candidate_score(window_data, candidate, current_price)
            scores.append(score)
        
        # Select candidate with highest score
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def _calculate_candidate_score(self, window_data: pd.DataFrame, 
                                 candidate: float, current_price: float) -> float:
        """Calculate attraction score for a candidate level"""
        
        score = 0.0
        
        # Distance factor (closer = better, but not too close)
        distance = abs(current_price - candidate) / current_price
        if 0.001 < distance < 0.1:  # Sweet spot for attraction
            score += (0.1 - distance) / 0.1
        
        # Frequency factor (how often price visits this level)
        visits = 0
        for _, row in window_data.iterrows():
            if row['low'] <= candidate <= row['high']:
                visits += 1
        
        frequency_score = visits / len(window_data)
        score += frequency_score
        
        # Volume factor (if available)
        if 'volume' in window_data.columns:
            volume_at_level = 0
            total_volume = 0
            
            for _, row in window_data.iterrows():
                total_volume += row['volume']
                if row['low'] <= candidate <= row['high']:
                    volume_at_level += row['volume']
            
            if total_volume > 0:
                volume_score = volume_at_level / total_volume
                score += volume_score
        
        return score
    
    def _calculate_attraction_strength(self, window_data: pd.DataFrame, 
                                     attractor: float) -> float:
        """Calculate strength of attraction to the attractor point"""
        
        # Calculate how often price moves toward the attractor
        attraction_events = 0
        total_events = 0
        
        for i in range(1, len(window_data)):
            prev_distance = abs(window_data['close'].iloc[i-1] - attractor)
            curr_distance = abs(window_data['close'].iloc[i] - attractor)
            
            total_events += 1
            
            # Check if price moved toward attractor
            if curr_distance < prev_distance:
                attraction_events += 1
        
        if total_events == 0:
            return 0.0
        
        attraction_ratio = attraction_events / total_events
        
        # Adjust for proximity (stronger when price is near attractor)
        current_distance = abs(window_data['close'].iloc[-1] - attractor) / attractor
        proximity_factor = max(0, 1 - current_distance / self.attraction_threshold)
        
        strength = attraction_ratio * (0.7 + proximity_factor * 0.3)
        
        return min(strength, 1.0)
    
    def _determine_attractor_type(self, window_data: pd.DataFrame, 
                                attractor: float, current_price: float) -> str:
        """Determine if attractor is magnetic or repulsive"""
        
        # Analyze recent price behavior around attractor
        recent_data = window_data.iloc[-self.stability_period:]
        
        approaches = 0
        departures = 0
        
        for i in range(1, len(recent_data)):
            prev_distance = abs(recent_data['close'].iloc[i-1] - attractor)
            curr_distance = abs(recent_data['close'].iloc[i] - attractor)
            
            attraction_zone = attractor * self.attraction_threshold
            
            if prev_distance > attraction_zone and curr_distance <= attraction_zone:
                approaches += 1
            elif prev_distance <= attraction_zone and curr_distance > attraction_zone:
                departures += 1
        
        if approaches > departures * 1.5:
            return 'magnetic'
        elif departures > approaches * 1.5:
            return 'repulsive'
        else:
            return 'neutral'
    
    def _calculate_stability_score(self, window_data: pd.DataFrame, 
                                 attractor: float) -> float:
        """Calculate stability score of the attractor"""
        
        # Measure consistency of attraction over time
        stability_window = min(self.stability_period, len(window_data))
        recent_data = window_data.iloc[-stability_window:]
        
        # Calculate variance in distance from attractor
        distances = [abs(price - attractor) / attractor for price in recent_data['close']]
        distance_variance = np.var(distances)
        
        # Lower variance = higher stability
        stability = max(0, 1 - distance_variance * 10)  # Scale appropriately
        
        return min(stability, 1.0)
    
    def _calculate_time_at_attractor(self, window_data: pd.DataFrame, 
                                   attractor: float) -> int:
        """Calculate periods spent near the attractor"""
        
        attraction_zone = attractor * self.attraction_threshold
        
        time_at_attractor = 0
        for price in window_data['close']:
            if abs(price - attractor) <= attraction_zone:
                time_at_attractor += 1
        
        return time_at_attractor
    
    def _calculate_convergence_rate(self, window_data: pd.DataFrame, 
                                  attractor: float) -> float:
        """Calculate rate of convergence to attractor"""
        
        if len(window_data) < 2:
            return 0.0
        
        # Calculate distances over time
        distances = [abs(price - attractor) / attractor for price in window_data['close']]
        
        # Linear regression on distances
        x = np.arange(len(distances))
        y = np.array(distances)
        
        if np.std(y) == 0:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Negative slope = convergence, positive = divergence
        convergence_rate = -slope * self.convergence_sensitivity
        
        return np.clip(convergence_rate, -2.0, 2.0)
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Attractor Point Analysis",
            category=self.CATEGORY,
            description="Identifies market attractor points using dynamic systems theory",
            parameters={
                "lookback_period": self.lookback_period,
                "attraction_threshold": self.attraction_threshold,
                "stability_period": self.stability_period,
                "convergence_sensitivity": self.convergence_sensitivity
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Attractor Point ({self.lookback_period})"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "lookback_period": self.lookback_period,
            "attraction_threshold": self.attraction_threshold,
            "stability_period": self.stability_period,
            "convergence_sensitivity": self.convergence_sensitivity
        }