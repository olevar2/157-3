"""
Price Volume Rank Indicator

Ranks volume relative to price movements to identify significant price levels
and volume anomalies that may indicate important support/resistance areas.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class PriceVolumeRankResult:
    """Result structure for Price Volume Rank analysis"""
    volume_rank: float  # Volume rank percentile (0-100)
    price_rank: float  # Price movement rank percentile (0-100)
    efficiency_score: float  # Price movement efficiency relative to volume
    significance_level: str  # "high", "medium", "low"
    volume_anomaly: bool  # Whether volume is anomalous for price movement
    relative_strength: float  # Strength relative to historical patterns


class PriceVolumeRank(StandardIndicatorInterface):
    """
    Price Volume Ranking System
    
    Ranks current volume and price movements against historical patterns:
    - Volume percentile ranking
    - Price movement percentile ranking  
    - Efficiency analysis (price movement per unit volume)
    - Anomaly detection
    - Relative strength calculations
    """
    
    CATEGORY = "volume"
    VERSION = "1.0.0" 
    AUTHOR = "Platform3"
    
    def __init__(self, ranking_period: int = 50, price_threshold: float = 0.01,
                 anomaly_threshold: float = 2.0, **kwargs):
        """
        Initialize Price Volume Rank indicator
        
        Args:
            ranking_period: Period for percentile calculations
            price_threshold: Minimum price movement for ranking
            anomaly_threshold: Standard deviation threshold for anomalies
        """
        self.ranking_period = ranking_period
        self.price_threshold = price_threshold
        self.anomaly_threshold = anomaly_threshold
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'ranking_period'):
            self.ranking_period = 50
        if not hasattr(self, 'price_threshold'):
            self.price_threshold = 0.01
        if not hasattr(self, 'anomaly_threshold'):
            self.anomaly_threshold = 2.0
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.ranking_period < 10:
            raise IndicatorValidationError("Ranking period must be at least 10")
        if self.price_threshold <= 0:
            raise IndicatorValidationError("Price threshold must be positive")
        if self.anomaly_threshold <= 0:
            raise IndicatorValidationError("Anomaly threshold must be positive")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return self.ranking_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price volume rankings
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ranking analysis
        """
        try:
            self.validate_input_data(data)
            
            # Calculate base metrics
            price_movements = self._calculate_price_movements(data)
            volume_metrics = self._calculate_volume_metrics(data)
            
            # Calculate rankings
            results = []
            for i in range(len(data)):
                if i < self.ranking_period:
                    result = {
                        'volume_rank': 50.0,  # Neutral rank
                        'price_rank': 50.0,
                        'efficiency_score': 0.5,
                        'significance_level': 'low',
                        'volume_anomaly': False,
                        'relative_strength': 0.5
                    }
                else:
                    # Calculate rankings for current position
                    ranking_analysis = self._calculate_rankings(
                        i, data, price_movements, volume_metrics
                    )
                    
                    result = {
                        'volume_rank': ranking_analysis.volume_rank,
                        'price_rank': ranking_analysis.price_rank,
                        'efficiency_score': ranking_analysis.efficiency_score,
                        'significance_level': ranking_analysis.significance_level,
                        'volume_anomaly': ranking_analysis.volume_anomaly,
                        'relative_strength': ranking_analysis.relative_strength
                    }
                
                results.append(result)
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_price_movements(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various price movement metrics"""
        # True range
        tr = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Price change
        price_change = abs(data['close'] - data['close'].shift(1))
        
        # Relative price change
        relative_change = price_change / data['close'].shift(1)
        
        # Intraday movement
        intraday_movement = (data['high'] - data['low']) / data['close']
        
        return pd.DataFrame({
            'true_range': tr,
            'price_change': price_change,
            'relative_change': relative_change,
            'intraday_movement': intraday_movement
        }).fillna(0)
    
    def _calculate_volume_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-related metrics"""
        # Volume change
        volume_change = data['volume'].pct_change()
        
        # Volume MA ratio
        volume_ma = data['volume'].rolling(window=20).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Volume momentum
        volume_momentum = data['volume'].rolling(window=5).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1)
        )
        
        return pd.DataFrame({
            'volume_change': volume_change,
            'volume_ratio': volume_ratio,
            'volume_momentum': volume_momentum
        }).fillna(0)
    
    def _calculate_rankings(self, index: int, data: pd.DataFrame,
                          price_movements: pd.DataFrame, volume_metrics: pd.DataFrame) -> PriceVolumeRankResult:
        """Calculate rankings at specific index"""
        
        # Define ranking window
        start_idx = max(0, index - self.ranking_period + 1)
        window_data = data.iloc[start_idx:index+1]
        window_price = price_movements.iloc[start_idx:index+1]
        window_volume = volume_metrics.iloc[start_idx:index+1]
        
        # Current values
        current_volume = data['volume'].iloc[index]
        current_price_change = price_movements['relative_change'].iloc[index]
        current_tr = price_movements['true_range'].iloc[index]
        
        # Calculate volume rank
        volume_rank = self._calculate_percentile_rank(
            current_volume, window_data['volume'].values
        )
        
        # Calculate price movement rank
        price_rank = self._calculate_percentile_rank(
            current_price_change, window_price['relative_change'].values
        )
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(
            index, window_data, window_price, window_volume
        )
        
        # Detect volume anomalies
        volume_anomaly = self._detect_volume_anomaly(
            current_volume, current_price_change, window_data, window_price
        )
        
        # Calculate relative strength
        relative_strength = self._calculate_relative_strength(
            volume_rank, price_rank, efficiency_score
        )
        
        # Determine significance level
        significance_level = self._determine_significance_level(
            volume_rank, price_rank, efficiency_score, volume_anomaly
        )
        
        return PriceVolumeRankResult(
            volume_rank=volume_rank,
            price_rank=price_rank,
            efficiency_score=efficiency_score,
            significance_level=significance_level,
            volume_anomaly=volume_anomaly,
            relative_strength=relative_strength
        )
    
    def _calculate_percentile_rank(self, current_value: float, historical_values: np.ndarray) -> float:
        """Calculate percentile rank of current value"""
        if len(historical_values) == 0:
            return 50.0
        
        # Remove NaN values
        valid_values = historical_values[~np.isnan(historical_values)]
        
        if len(valid_values) == 0:
            return 50.0
        
        # Calculate percentile rank
        rank = (valid_values < current_value).sum() / len(valid_values) * 100
        
        return rank
    
    def _calculate_efficiency_score(self, index: int, window_data: pd.DataFrame,
                                  window_price: pd.DataFrame, window_volume: pd.DataFrame) -> float:
        """Calculate price movement efficiency relative to volume"""
        
        current_price_change = window_price['relative_change'].iloc[-1]
        current_volume_ratio = window_volume['volume_ratio'].iloc[-1]
        
        if current_volume_ratio <= 0:
            return 0.5
        
        # Efficiency = price movement per unit of volume
        efficiency = current_price_change / current_volume_ratio
        
        # Normalize against historical efficiency
        historical_efficiency = []
        for i in range(1, len(window_data)):
            hist_price = window_price['relative_change'].iloc[i]
            hist_volume = window_volume['volume_ratio'].iloc[i]
            
            if hist_volume > 0 and hist_price > self.price_threshold:
                historical_efficiency.append(hist_price / hist_volume)
        
        if not historical_efficiency:
            return 0.5
        
        # Percentile rank of current efficiency
        efficiency_rank = self._calculate_percentile_rank(efficiency, np.array(historical_efficiency))
        
        return efficiency_rank / 100
    
    def _detect_volume_anomaly(self, current_volume: float, current_price_change: float,
                             window_data: pd.DataFrame, window_price: pd.DataFrame) -> bool:
        """Detect if current volume is anomalous for the price movement"""
        
        # Filter similar price movements
        similar_moves = []
        price_tolerance = current_price_change * 0.5  # 50% tolerance
        
        for i in range(len(window_data) - 1):
            hist_price_change = window_price['relative_change'].iloc[i]
            
            if abs(hist_price_change - current_price_change) <= price_tolerance:
                similar_moves.append(window_data['volume'].iloc[i])
        
        if len(similar_moves) < 3:
            return False
        
        # Calculate statistics for similar moves
        similar_volume_mean = np.mean(similar_moves)
        similar_volume_std = np.std(similar_moves)
        
        if similar_volume_std == 0:
            return False
        
        # Check if current volume is anomalous
        z_score = abs(current_volume - similar_volume_mean) / similar_volume_std
        
        return z_score > self.anomaly_threshold
    
    def _calculate_relative_strength(self, volume_rank: float, price_rank: float,
                                   efficiency_score: float) -> float:
        """Calculate relative strength score"""
        
        # Combine different aspects of strength
        strength_components = [
            volume_rank / 100,  # Volume strength
            price_rank / 100,   # Price movement strength
            efficiency_score,   # Efficiency strength
        ]
        
        # Weighted average
        weights = [0.4, 0.3, 0.3]
        relative_strength = sum(w * s for w, s in zip(weights, strength_components))
        
        return relative_strength
    
    def _determine_significance_level(self, volume_rank: float, price_rank: float,
                                    efficiency_score: float, volume_anomaly: bool) -> str:
        """Determine significance level of current reading"""
        
        # High significance criteria
        high_criteria = [
            volume_rank >= 80 or volume_rank <= 20,  # Extreme volume
            price_rank >= 80 or price_rank <= 20,    # Extreme price movement
            efficiency_score >= 0.8 or efficiency_score <= 0.2,  # Extreme efficiency
            volume_anomaly  # Volume anomaly detected
        ]
        
        # Medium significance criteria
        medium_criteria = [
            volume_rank >= 70 or volume_rank <= 30,
            price_rank >= 70 or price_rank <= 30,
            efficiency_score >= 0.6 or efficiency_score <= 0.4
        ]
        
        if sum(high_criteria) >= 2:
            return 'high'
        elif sum(medium_criteria) >= 2:
            return 'medium'
        else:
            return 'low'
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Price Volume Rank",
            category=self.CATEGORY,
            description="Ranks volume relative to price movements for significance analysis",
            parameters={
                "ranking_period": self.ranking_period,
                "price_threshold": self.price_threshold,
                "anomaly_threshold": self.anomaly_threshold
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Price Volume Rank ({self.ranking_period})"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "ranking_period": self.ranking_period,
            "price_threshold": self.price_threshold,
            "anomaly_threshold": self.anomaly_threshold
        }