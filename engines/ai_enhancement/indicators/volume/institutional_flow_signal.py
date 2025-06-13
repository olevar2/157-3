"""
Institutional Flow Signal Indicator

Detects large institutional order flow by analyzing volume patterns, price impact,
and order imbalances that typically indicate institutional trading activity.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class InstitutionalFlowResult:
    """Result structure for Institutional Flow analysis"""
    flow_type: str  # "accumulation", "distribution", "neutral"
    intensity: float  # Flow intensity (0-1)
    volume_ratio: float  # Current volume vs average
    price_impact: float  # Price impact per unit volume
    order_size_estimate: str  # "small", "medium", "large", "institutional"
    confidence: float
    direction: str  # "buying", "selling", "neutral"
    stealth_score: float  # How stealthily orders are being executed


class InstitutionalFlowSignal(StandardIndicatorInterface):
    """
    Institutional Flow Detection
    
    Identifies institutional trading activity through:
    - Volume surge analysis
    - Price impact assessment
    - Order size estimation
    - Stealth trading detection
    - Accumulation/Distribution patterns
    """
    
    CATEGORY = "microstructure"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, volume_period: int = 20, volume_threshold: float = 2.0,
                 price_impact_period: int = 5, stealth_sensitivity: float = 0.3, **kwargs):
        """
        Initialize Institutional Flow indicator
        
        Args:
            volume_period: Period for volume analysis
            volume_threshold: Volume threshold multiplier for detection
            price_impact_period: Period for price impact calculation
            stealth_sensitivity: Sensitivity for stealth trading detection
        """
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.price_impact_period = price_impact_period
        self.stealth_sensitivity = stealth_sensitivity
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'volume_period'):
            self.volume_period = 20
        if not hasattr(self, 'volume_threshold'):
            self.volume_threshold = 2.0
        if not hasattr(self, 'price_impact_period'):
            self.price_impact_period = 5
        if not hasattr(self, 'stealth_sensitivity'):
            self.stealth_sensitivity = 0.3
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.volume_period < 5:
            raise IndicatorValidationError("Volume period must be at least 5")
        if self.volume_threshold <= 1.0:
            raise IndicatorValidationError("Volume threshold must be greater than 1.0")
        if self.price_impact_period < 1:
            raise IndicatorValidationError("Price impact period must be at least 1")
        if self.stealth_sensitivity <= 0 or self.stealth_sensitivity > 1:
            raise IndicatorValidationError("Stealth sensitivity must be between 0 and 1")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return max(self.volume_period, self.price_impact_period) + 10
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate institutional flow signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with institutional flow analysis
        """
        try:
            self.validate_input_data(data)
            
            # Calculate volume metrics
            volume_sma = data['volume'].rolling(window=self.volume_period).mean()
            volume_ratio = data['volume'] / volume_sma
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(data)
            
            # Calculate order flow metrics
            order_flow = self._calculate_order_flow(data)
            
            # Detect stealth trading
            stealth_score = self._calculate_stealth_score(data, volume_ratio)
            
            # Classify institutional activity
            results = []
            for i in range(len(data)):
                if i < self.volume_period:
                    # Initial periods with default values
                    result = {
                        'flow_type': 'neutral',
                        'intensity': 0.0,
                        'volume_ratio': 1.0,
                        'price_impact': 0.0,
                        'order_size_estimate': 'small',
                        'confidence': 0.0,
                        'direction': 'neutral',
                        'stealth_score': 0.0
                    }
                else:
                    # Analyze institutional flow
                    flow_analysis = self._analyze_institutional_flow(
                        i, data, volume_ratio, price_impact, order_flow, stealth_score
                    )
                    
                    result = {
                        'flow_type': flow_analysis.flow_type,
                        'intensity': flow_analysis.intensity,
                        'volume_ratio': flow_analysis.volume_ratio,
                        'price_impact': flow_analysis.price_impact,
                        'order_size_estimate': flow_analysis.order_size_estimate,
                        'confidence': flow_analysis.confidence,
                        'direction': flow_analysis.direction,
                        'stealth_score': flow_analysis.stealth_score
                    }
                
                results.append(result)
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_price_impact(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price impact per unit volume"""
        # Calculate price movement
        price_change = abs(data['close'] - data['close'].shift(1))
        
        # Normalize by average true range
        tr = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=self.price_impact_period).mean()
        
        # Calculate normalized price impact
        normalized_price_change = price_change / atr
        
        # Price impact per unit volume
        volume_normalized = data['volume'] / data['volume'].rolling(window=self.volume_period).mean()
        
        price_impact = normalized_price_change / (volume_normalized + 0.001)  # Avoid division by zero
        
        return price_impact.fillna(0)
    
    def _calculate_order_flow(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow metrics"""
        # Estimate buying vs selling pressure
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Money flow multiplier
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.fillna(0)
        
        # Money flow volume
        mfv = mfm * data['volume']
        
        # Accumulation/Distribution line
        ad_line = mfv.cumsum()
        
        # Volume-weighted average price
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Price vs VWAP (indicates institutional bias)
        price_vs_vwap = (data['close'] - vwap) / vwap
        
        return pd.DataFrame({
            'money_flow_volume': mfv,
            'ad_line': ad_line,
            'vwap': vwap,
            'price_vs_vwap': price_vs_vwap
        })
    
    def _calculate_stealth_score(self, data: pd.DataFrame, volume_ratio: pd.Series) -> pd.Series:
        """Calculate stealth trading score"""
        # Low volatility with high volume suggests stealth trading
        volatility = data['close'].pct_change().rolling(window=self.price_impact_period).std()
        
        # Normalize volatility
        avg_volatility = volatility.rolling(window=self.volume_period).mean()
        normalized_volatility = volatility / (avg_volatility + 0.0001)
        
        # Stealth score: high volume with low volatility
        stealth_score = (volume_ratio - 1) / (normalized_volatility + 1)
        stealth_score = stealth_score.clip(0, 2)  # Cap the score
        
        return stealth_score.fillna(0)
    
    def _analyze_institutional_flow(self, index: int, data: pd.DataFrame, 
                                  volume_ratio: pd.Series, price_impact: pd.Series,
                                  order_flow: pd.DataFrame, stealth_score: pd.Series) -> InstitutionalFlowResult:
        """Analyze institutional flow at specific index"""
        
        current_volume_ratio = volume_ratio.iloc[index]
        current_price_impact = price_impact.iloc[index]
        current_stealth = stealth_score.iloc[index]
        
        # Determine flow intensity
        intensity = self._calculate_flow_intensity(
            current_volume_ratio, current_price_impact, current_stealth
        )
        
        # Determine flow type (accumulation/distribution)
        flow_type = self._determine_flow_type(index, data, order_flow)
        
        # Estimate order size
        order_size = self._estimate_order_size(current_volume_ratio, current_stealth)
        
        # Determine direction
        direction = self._determine_flow_direction(index, data, order_flow)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            current_volume_ratio, current_price_impact, current_stealth, intensity
        )
        
        return InstitutionalFlowResult(
            flow_type=flow_type,
            intensity=intensity,
            volume_ratio=current_volume_ratio,
            price_impact=current_price_impact,
            order_size_estimate=order_size,
            confidence=confidence,
            direction=direction,
            stealth_score=current_stealth
        )
    
    def _calculate_flow_intensity(self, volume_ratio: float, price_impact: float, 
                                stealth_score: float) -> float:
        """Calculate flow intensity score"""
        # Combine volume, price impact, and stealth factors
        volume_component = min((volume_ratio - 1) / 2, 1.0)  # Normalize excess volume
        impact_component = min(abs(price_impact) * 10, 1.0)  # Scale price impact
        stealth_component = min(stealth_score / 2, 1.0)  # Normalize stealth score
        
        # Weighted combination
        intensity = (volume_component * 0.4 + impact_component * 0.3 + stealth_component * 0.3)
        
        return max(0.0, min(intensity, 1.0))
    
    def _determine_flow_type(self, index: int, data: pd.DataFrame, 
                           order_flow: pd.DataFrame) -> str:
        """Determine if flow is accumulation or distribution"""
        if index < self.volume_period:
            return 'neutral'
        
        # Look at recent trend in A/D line
        recent_ad = order_flow['ad_line'].iloc[index-self.volume_period:index+1]
        
        if len(recent_ad) < 2:
            return 'neutral'
        
        ad_trend = (recent_ad.iloc[-1] - recent_ad.iloc[0]) / len(recent_ad)
        
        # Look at price trend
        recent_prices = data['close'].iloc[index-self.volume_period:index+1]
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
        
        # Determine flow type
        if ad_trend > 0 and price_trend >= 0:
            return 'accumulation'
        elif ad_trend < 0 and price_trend <= 0:
            return 'distribution'
        elif ad_trend > 0 and price_trend < 0:
            return 'accumulation'  # Hidden accumulation
        elif ad_trend < 0 and price_trend > 0:
            return 'distribution'  # Hidden distribution
        else:
            return 'neutral'
    
    def _estimate_order_size(self, volume_ratio: float, stealth_score: float) -> str:
        """Estimate order size based on volume and stealth metrics"""
        # High stealth score with moderate volume suggests institutional
        if stealth_score > 1.0 and volume_ratio > 1.5:
            return 'institutional'
        elif volume_ratio > self.volume_threshold * 2:
            return 'large'
        elif volume_ratio > self.volume_threshold:
            return 'medium'
        else:
            return 'small'
    
    def _determine_flow_direction(self, index: int, data: pd.DataFrame, 
                                order_flow: pd.DataFrame) -> str:
        """Determine flow direction (buying/selling)"""
        if index < 1:
            return 'neutral'
        
        # Use money flow volume and price vs VWAP
        current_mfv = order_flow['money_flow_volume'].iloc[index]
        current_price_vs_vwap = order_flow['price_vs_vwap'].iloc[index]
        
        # Combine signals
        if current_mfv > 0 and current_price_vs_vwap > 0:
            return 'buying'
        elif current_mfv < 0 and current_price_vs_vwap < 0:
            return 'selling'
        elif abs(current_mfv) > abs(current_price_vs_vwap * 1000):  # Scale VWAP signal
            return 'buying' if current_mfv > 0 else 'selling'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, volume_ratio: float, price_impact: float,
                            stealth_score: float, intensity: float) -> float:
        """Calculate confidence in institutional flow detection"""
        confidence = 0.0
        
        # High volume confidence
        if volume_ratio > self.volume_threshold:
            confidence += 0.3
        
        # Price impact confidence
        if abs(price_impact) > 0.1:
            confidence += 0.2
        
        # Stealth trading confidence
        if stealth_score > self.stealth_sensitivity:
            confidence += 0.3
        
        # Overall intensity confidence
        confidence += intensity * 0.2
        
        return min(confidence, 1.0)
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Institutional Flow Signal",
            category=self.CATEGORY,
            description="Detects institutional trading activity and order flow patterns",
            parameters={
                "volume_period": self.volume_period,
                "volume_threshold": self.volume_threshold,
                "price_impact_period": self.price_impact_period,
                "stealth_sensitivity": self.stealth_sensitivity
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Institutional Flow ({self.volume_period}, {self.volume_threshold}x)"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "volume_period": self.volume_period,
            "volume_threshold": self.volume_threshold,
            "price_impact_period": self.price_impact_period,
            "stealth_sensitivity": self.stealth_sensitivity
        }