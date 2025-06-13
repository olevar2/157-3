"""
Order Flow Sequence Signal Indicator

Analyzes sequential order flow patterns to identify market momentum shifts,
institutional order patterns, and algorithmic trading sequences.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class OrderFlowSequenceResult:
    """Result structure for Order Flow Sequence analysis"""
    sequence_type: str  # "accumulation", "distribution", "momentum", "reversal", "neutral"
    pattern_strength: float  # Strength of the identified pattern
    sequence_length: int  # Length of current sequence
    momentum_score: float  # Momentum within sequence
    institutional_score: float  # Likelihood of institutional activity
    algorithmic_score: float  # Likelihood of algorithmic trading
    next_expected: str  # Expected next move direction
    confidence: float


class OrderFlowSequenceSignal(StandardIndicatorInterface):
    """
    Order Flow Sequence Pattern Detector
    
    Identifies patterns in order flow sequences:
    - Accumulation/Distribution sequences
    - Momentum continuation patterns
    - Reversal sequence patterns
    - Institutional order sequences
    - Algorithmic trading patterns
    """
    
    CATEGORY = "microstructure"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, sequence_length: int = 10, momentum_threshold: float = 0.6,
                 institutional_threshold: float = 2.0, algo_sensitivity: float = 0.8, **kwargs):
        """
        Initialize Order Flow Sequence indicator
        
        Args:
            sequence_length: Length of sequence to analyze
            momentum_threshold: Threshold for momentum detection
            institutional_threshold: Volume threshold for institutional detection
            algo_sensitivity: Sensitivity for algorithmic pattern detection
        """
        self.sequence_length = sequence_length
        self.momentum_threshold = momentum_threshold
        self.institutional_threshold = institutional_threshold
        self.algo_sensitivity = algo_sensitivity
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'sequence_length'):
            self.sequence_length = 10
        if not hasattr(self, 'momentum_threshold'):
            self.momentum_threshold = 0.6
        if not hasattr(self, 'institutional_threshold'):
            self.institutional_threshold = 2.0
        if not hasattr(self, 'algo_sensitivity'):
            self.algo_sensitivity = 0.8
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.sequence_length < 3:
            raise IndicatorValidationError("Sequence length must be at least 3")
        if self.momentum_threshold <= 0 or self.momentum_threshold > 1:
            raise IndicatorValidationError("Momentum threshold must be between 0 and 1")
        if self.institutional_threshold <= 1:
            raise IndicatorValidationError("Institutional threshold must be > 1")
        if self.algo_sensitivity <= 0 or self.algo_sensitivity > 1:
            raise IndicatorValidationError("Algo sensitivity must be between 0 and 1")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return self.sequence_length * 2
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow sequence patterns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with sequence analysis
        """
        try:
            self.validate_input_data(data)
            
            # Calculate order flow metrics
            order_flow_metrics = self._calculate_order_flow_metrics(data)
            
            # Analyze sequences
            results = []
            for i in range(len(data)):
                if i < self.sequence_length:
                    result = {
                        'sequence_type': 'neutral',
                        'pattern_strength': 0.0,
                        'sequence_length': 0,
                        'momentum_score': 0.0,
                        'institutional_score': 0.0,
                        'algorithmic_score': 0.0,
                        'next_expected': 'neutral',
                        'confidence': 0.0
                    }
                else:
                    # Analyze sequence at current position
                    sequence_analysis = self._analyze_sequence(
                        i, data, order_flow_metrics
                    )
                    
                    result = {
                        'sequence_type': sequence_analysis.sequence_type,
                        'pattern_strength': sequence_analysis.pattern_strength,
                        'sequence_length': sequence_analysis.sequence_length,
                        'momentum_score': sequence_analysis.momentum_score,
                        'institutional_score': sequence_analysis.institutional_score,
                        'algorithmic_score': sequence_analysis.algorithmic_score,
                        'next_expected': sequence_analysis.next_expected,
                        'confidence': sequence_analysis.confidence
                    }
                
                results.append(result)
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_order_flow_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate base order flow metrics"""
        # Money flow multiplier
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.fillna(0)
        
        # Money flow volume
        mfv = mfm * data['volume']
        
        # Volume ratio (current vs average)
        volume_sma = data['volume'].rolling(window=self.sequence_length).mean()
        volume_ratio = data['volume'] / volume_sma
        
        # Price momentum
        price_momentum = data['close'].pct_change()
        
        # Volume-weighted momentum
        vw_momentum = price_momentum * volume_ratio
        
        return pd.DataFrame({
            'money_flow_volume': mfv,
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum,
            'vw_momentum': vw_momentum
        }).fillna(0)
    
    def _analyze_sequence(self, index: int, data: pd.DataFrame, 
                        metrics: pd.DataFrame) -> OrderFlowSequenceResult:
        """Analyze order flow sequence at specific index"""
        
        start_idx = max(0, index - self.sequence_length + 1)
        sequence_data = metrics.iloc[start_idx:index+1]
        price_data = data.iloc[start_idx:index+1]
        
        # Identify sequence patterns
        sequence_type, pattern_strength = self._identify_sequence_pattern(sequence_data, price_data)
        
        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(sequence_data)
        
        # Calculate institutional score
        institutional_score = self._calculate_institutional_score(sequence_data)
        
        # Calculate algorithmic score
        algorithmic_score = self._calculate_algorithmic_score(sequence_data)
        
        # Predict next expected move
        next_expected = self._predict_next_move(sequence_type, momentum_score, sequence_data)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            pattern_strength, momentum_score, institutional_score, algorithmic_score
        )
        
        return OrderFlowSequenceResult(
            sequence_type=sequence_type,
            pattern_strength=pattern_strength,
            sequence_length=len(sequence_data),
            momentum_score=momentum_score,
            institutional_score=institutional_score,
            algorithmic_score=algorithmic_score,
            next_expected=next_expected,
            confidence=confidence
        )
    
    def _identify_sequence_pattern(self, sequence_data: pd.DataFrame, 
                                 price_data: pd.DataFrame) -> Tuple[str, float]:
        """Identify the type of sequence pattern"""
        
        mfv_trend = self._calculate_trend_strength(sequence_data['money_flow_volume'])
        price_trend = self._calculate_trend_strength(price_data['close'].pct_change())
        volume_trend = self._calculate_trend_strength(sequence_data['volume_ratio'])
        
        # Accumulation: Positive MFV trend with stable/rising prices
        if mfv_trend > 0.3 and price_trend >= -0.1:
            return 'accumulation', abs(mfv_trend) + volume_trend * 0.3
        
        # Distribution: Negative MFV trend with stable/falling prices
        elif mfv_trend < -0.3 and price_trend <= 0.1:
            return 'distribution', abs(mfv_trend) + volume_trend * 0.3
        
        # Momentum: Strong price and volume trends in same direction
        elif abs(price_trend) > self.momentum_threshold and abs(volume_trend) > 0.3:
            if price_trend * volume_trend > 0:
                return 'momentum', (abs(price_trend) + abs(volume_trend)) / 2
        
        # Reversal: Divergence between price and volume/MFV
        elif abs(price_trend) > 0.3 and price_trend * mfv_trend < -0.2:
            return 'reversal', abs(price_trend - mfv_trend) / 2
        
        return 'neutral', 0.0
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(series) < 3:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        valid_idx = ~np.isnan(y)
        if valid_idx.sum() < 3:
            return 0.0
        
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        
        # Linear regression
        slope = np.polyfit(x_valid, y_valid, 1)[0]
        
        # Normalize slope
        y_range = np.ptp(y_valid)
        if y_range > 0:
            normalized_slope = slope * len(x_valid) / y_range
        else:
            normalized_slope = 0.0
        
        return np.clip(normalized_slope, -1.0, 1.0)
    
    def _calculate_momentum_score(self, sequence_data: pd.DataFrame) -> float:
        """Calculate momentum score for the sequence"""
        vw_momentum = sequence_data['vw_momentum']
        
        # Calculate consistency of momentum direction
        positive_momentum = (vw_momentum > 0).sum()
        negative_momentum = (vw_momentum < 0).sum()
        total_periods = len(vw_momentum)
        
        if total_periods == 0:
            return 0.0
        
        # Momentum score based on directional consistency
        consistency = max(positive_momentum, negative_momentum) / total_periods
        
        # Average momentum magnitude
        magnitude = abs(vw_momentum).mean()
        
        return min(consistency * magnitude * 2, 1.0)
    
    def _calculate_institutional_score(self, sequence_data: pd.DataFrame) -> float:
        """Calculate likelihood of institutional activity"""
        volume_ratio = sequence_data['volume_ratio']
        
        # High volume periods
        high_volume_periods = (volume_ratio > self.institutional_threshold).sum()
        total_periods = len(volume_ratio)
        
        if total_periods == 0:
            return 0.0
        
        high_volume_ratio = high_volume_periods / total_periods
        
        # Consistent high volume suggests institutional activity
        avg_volume_ratio = volume_ratio.mean()
        volume_consistency = 1 - volume_ratio.std() / (avg_volume_ratio + 0.001)
        
        institutional_score = (high_volume_ratio + volume_consistency * 0.5) / 1.5
        
        return min(max(institutional_score, 0.0), 1.0)
    
    def _calculate_algorithmic_score(self, sequence_data: pd.DataFrame) -> float:
        """Calculate likelihood of algorithmic trading"""
        # Algorithmic trading often shows regular patterns
        volume_ratio = sequence_data['volume_ratio']
        mfv = sequence_data['money_flow_volume']
        
        # Calculate regularity in volume patterns
        volume_std = volume_ratio.std()
        volume_mean = volume_ratio.mean()
        
        if volume_mean > 0:
            volume_cv = volume_std / volume_mean  # Coefficient of variation
        else:
            volume_cv = 1.0
        
        # Low coefficient of variation suggests algorithmic activity
        regularity_score = max(0, 1 - volume_cv)
        
        # Check for small, consistent order flow
        small_consistent_flow = (abs(mfv) < abs(mfv).mean() * 0.5).sum() / len(mfv)
        
        algorithmic_score = (regularity_score * 0.6 + small_consistent_flow * 0.4)
        
        return min(max(algorithmic_score, 0.0), 1.0)
    
    def _predict_next_move(self, sequence_type: str, momentum_score: float, 
                         sequence_data: pd.DataFrame) -> str:
        """Predict next expected move based on sequence"""
        
        if sequence_type == 'accumulation':
            return 'up'
        elif sequence_type == 'distribution':
            return 'down'
        elif sequence_type == 'momentum':
            # Continue in momentum direction
            recent_momentum = sequence_data['vw_momentum'].iloc[-3:].mean()
            return 'up' if recent_momentum > 0 else 'down'
        elif sequence_type == 'reversal':
            # Opposite to recent price direction
            recent_price_change = sequence_data['price_momentum'].iloc[-3:].mean()
            return 'down' if recent_price_change > 0 else 'up'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, pattern_strength: float, momentum_score: float,
                            institutional_score: float, algorithmic_score: float) -> float:
        """Calculate overall confidence in sequence analysis"""
        
        # Weighted combination of scores
        confidence = (
            pattern_strength * 0.4 +
            momentum_score * 0.3 +
            institutional_score * 0.2 +
            algorithmic_score * 0.1
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Order Flow Sequence Signal",
            category=self.CATEGORY,
            description="Analyzes sequential order flow patterns and institutional activity",
            parameters={
                "sequence_length": self.sequence_length,
                "momentum_threshold": self.momentum_threshold,
                "institutional_threshold": self.institutional_threshold,
                "algo_sensitivity": self.algo_sensitivity
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Order Flow Sequence ({self.sequence_length})"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "sequence_length": self.sequence_length,
            "momentum_threshold": self.momentum_threshold,
            "institutional_threshold": self.institutional_threshold,
            "algo_sensitivity": self.algo_sensitivity
        }