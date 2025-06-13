"""
Composite Signal Indicator

Combines multiple technical indicators into a unified composite signal
for comprehensive market analysis and trading decision support.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class CompositeSignalResult:
    """Result structure for Composite Signal analysis"""
    composite_score: float  # Overall composite signal (-1 to 1)
    signal_strength: str  # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    trend_component: float  # Trend analysis component
    momentum_component: float  # Momentum component
    volume_component: float  # Volume component
    volatility_component: float  # Volatility component
    confidence_level: float  # Confidence in signal


class CompositeSignal(StandardIndicatorInterface):
    """
    Multi-Indicator Composite Signal
    
    Combines signals from:
    - Trend indicators (SMA, EMA)
    - Momentum indicators (RSI, MACD)
    - Volume indicators
    - Volatility measures
    - Price action patterns
    """
    
    CATEGORY = "technical"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, trend_period: int = 20, momentum_period: int = 14,
                 volume_period: int = 20, **kwargs):
        self.trend_period = trend_period
        self.momentum_period = momentum_period
        self.volume_period = volume_period
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        if not hasattr(self, 'trend_period'):
            self.trend_period = 20
        if not hasattr(self, 'momentum_period'):
            self.momentum_period = 14
        if not hasattr(self, 'volume_period'):
            self.volume_period = 20
    
    def validate_parameters(self) -> bool:
        if self.trend_period < 5:
            raise IndicatorValidationError("Trend period must be at least 5")
        if self.momentum_period < 5:
            raise IndicatorValidationError("Momentum period must be at least 5")
        if self.volume_period < 5:
            raise IndicatorValidationError("Volume period must be at least 5")
        return True
    
    def _get_required_columns(self) -> List[str]:
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        return max(self.trend_period, self.momentum_period, self.volume_period) + 10
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            self.validate_input_data(data)
            
            # Calculate component signals
            trend_signals = self._calculate_trend_signals(data)
            momentum_signals = self._calculate_momentum_signals(data)
            volume_signals = self._calculate_volume_signals(data)
            volatility_signals = self._calculate_volatility_signals(data)
            
            # Combine into composite signal
            results = []
            for i in range(len(data)):
                composite_analysis = self._create_composite_signal(
                    i, trend_signals, momentum_signals, volume_signals, volatility_signals
                )
                
                results.append({
                    'composite_score': composite_analysis.composite_score,
                    'signal_strength': composite_analysis.signal_strength,
                    'trend_component': composite_analysis.trend_component,
                    'momentum_component': composite_analysis.momentum_component,
                    'volume_component': composite_analysis.volume_component,
                    'volatility_component': composite_analysis.volatility_component,
                    'confidence_level': composite_analysis.confidence_level
                })
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_trend_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        sma = data['close'].rolling(window=self.trend_period).mean()
        ema = data['close'].ewm(span=self.trend_period).mean()
        
        # Trend strength
        trend_strength = (data['close'] - sma) / sma
        
        return pd.DataFrame({
            'sma': sma,
            'ema': ema,
            'trend_strength': trend_strength
        }).fillna(0)
    
    def _calculate_momentum_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Simple RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.momentum_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.momentum_period).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        # Price momentum
        price_momentum = data['close'].pct_change(periods=self.momentum_period)
        
        return pd.DataFrame({
            'rsi': rsi,
            'price_momentum': price_momentum
        }).fillna(50)
    
    def _calculate_volume_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        volume_sma = data['volume'].rolling(window=self.volume_period).mean()
        volume_ratio = data['volume'] / volume_sma
        
        # Volume trend
        volume_trend = data['volume'].rolling(window=self.volume_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        return pd.DataFrame({
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend
        }).fillna(1)
    
    def _calculate_volatility_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        volatility = data['close'].rolling(window=self.trend_period).std()
        avg_volatility = volatility.rolling(window=self.trend_period).mean()
        volatility_ratio = volatility / (avg_volatility + 0.0001)
        
        return pd.DataFrame({
            'volatility': volatility,
            'volatility_ratio': volatility_ratio
        }).fillna(1)
    
    def _create_composite_signal(self, index: int, trend_signals: pd.DataFrame,
                               momentum_signals: pd.DataFrame, volume_signals: pd.DataFrame,
                               volatility_signals: pd.DataFrame) -> CompositeSignalResult:
        
        # Extract components
        trend_component = trend_signals['trend_strength'].iloc[index]
        
        rsi = momentum_signals['rsi'].iloc[index]
        momentum_component = (rsi - 50) / 50  # Normalize RSI to -1 to 1
        
        volume_component = min((volume_signals['volume_ratio'].iloc[index] - 1) / 2, 1)
        
        volatility_component = min((volatility_signals['volatility_ratio'].iloc[index] - 1) / 2, 1)
        
        # Weighted composite score
        composite_score = (
            trend_component * 0.4 +
            momentum_component * 0.3 +
            volume_component * 0.2 +
            volatility_component * 0.1
        )
        
        composite_score = np.clip(composite_score, -1, 1)
        
        # Determine signal strength
        signal_strength = self._determine_signal_strength(composite_score)
        
        # Calculate confidence
        confidence_level = self._calculate_confidence(
            trend_component, momentum_component, volume_component, volatility_component
        )
        
        return CompositeSignalResult(
            composite_score=composite_score,
            signal_strength=signal_strength,
            trend_component=trend_component,
            momentum_component=momentum_component,
            volume_component=volume_component,
            volatility_component=volatility_component,
            confidence_level=confidence_level
        )
    
    def _determine_signal_strength(self, composite_score: float) -> str:
        if composite_score > 0.6:
            return "strong_buy"
        elif composite_score > 0.2:
            return "buy"
        elif composite_score < -0.6:
            return "strong_sell"
        elif composite_score < -0.2:
            return "sell"
        else:
            return "neutral"
    
    def _calculate_confidence(self, trend: float, momentum: float, 
                            volume: float, volatility: float) -> float:
        # Confidence based on component agreement
        components = [trend, momentum, volume, volatility]
        avg_component = np.mean(components)
        
        # Agreement score
        agreement = 1 - np.std(components) / (abs(avg_component) + 0.5)
        
        return min(max(agreement, 0), 1)
    
    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Composite Signal",
            category=self.CATEGORY,
            description="Multi-indicator composite signal for comprehensive analysis",
            parameters={
                "trend_period": self.trend_period,
                "momentum_period": self.momentum_period,
                "volume_period": self.volume_period
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        return f"Composite Signal ({self.trend_period})"
    
    def get_parameters(self) -> Dict:
        return {
            "trend_period": self.trend_period,
            "momentum_period": self.momentum_period,
            "volume_period": self.volume_period
        }