"""
Mass Index - Platform3 Financial Indicator

The Mass Index is a volatility indicator developed by Donald Dorsey that identifies
potential reversal points by analyzing the narrowing and widening of the range
between high and low prices. It uses the concept that reversals often occur
after periods of volatility compression.

Platform3 compliant implementation with CCI proven patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import logging
import sys
import os

# Add the base path for imports when running as script
if __name__ == "__main__":
    
try:
    from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
except ImportError:
    # For direct script execution
        from base_indicator import StandardIndicatorInterface


class MassIndex(StandardIndicatorInterface):
    """
    Mass Index - Platform3 Implementation
    
    The Mass Index identifies potential reversal points by measuring the rate of change
    in the trading range. When the Mass Index rises above 27 and then drops below 26.5,
    it suggests a potential reversal in the current trend.
    
    Formula:
    1. Single EMA = EMA(High - Low, ema_period)
    2. Double EMA = EMA(Single EMA, ema_period)
    3. EMA Ratio = Single EMA / Double EMA
    4. Mass Index = SUM(EMA Ratio, mass_period)
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Performance Optimization  
    - Robust Error Handling
    """
    
    def __init__(self, ema_period: int = 9, mass_period: int = 25):
        """
        Initialize Mass Index.
        
        Args:
            ema_period (int): Period for EMA calculations (default: 9)
            mass_period (int): Period for Mass Index sum (default: 25)
        """
        super().__init__()
        self.ema_period = ema_period
        self.mass_period = mass_period
        self.name = "MassIndex"
        self.version = "1.0.0"
        self.logger = logging.getLogger(f"Platform3.{self.name}")
        self.logger.info(f"{self.name} initialized with ema_period={ema_period}, mass_period={mass_period}")

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return {
            "ema_period": self.ema_period,
            "mass_period": self.mass_period
        }

    def validate_parameters(self) -> bool:
        """Validate parameters."""
        return (isinstance(self.ema_period, int) and self.ema_period > 0 and
                isinstance(self.mass_period, int) and self.mass_period > 0)

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[period - 1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema

    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate Mass Index.
        
        Args:
            data: Market data (DataFrame with 'high', 'low' columns or 2D array [high, low])
            
        Returns:
            Dict containing Mass Index values and analysis
        """
        try:
            # Extract high and low prices
            if isinstance(data, pd.DataFrame):
                if 'high' in data.columns and 'low' in data.columns:
                    high_prices = data['high'].values
                    low_prices = data['low'].values
                elif len(data.columns) >= 2:
                    high_prices = data.iloc[:, 1].values  # Assuming [low, high] or [open, high, low, close]
                    low_prices = data.iloc[:, 2].values if len(data.columns) > 2 else data.iloc[:, 0].values
                else:
                    return {"error": "Insufficient price data columns"}
            else:
                if data.ndim == 2 and data.shape[1] >= 2:
                    high_prices = data[:, 1]
                    low_prices = data[:, 0] if data.shape[1] == 2 else data[:, 2]
                else:
                    return {"error": "Data must have high and low prices"}
            
            if len(high_prices) < self.minimum_periods:
                return {"error": f"Insufficient data. Need at least {self.minimum_periods} periods"}
            
            # Calculate range (High - Low)
            price_range = high_prices - low_prices
            
            # Avoid division by zero
            price_range = np.where(price_range == 0, 0.0001, price_range)
            
            # Calculate single EMA of the range
            single_ema = self._calculate_ema(price_range, self.ema_period)
            
            # Calculate double EMA (EMA of the single EMA)
            double_ema = self._calculate_ema(single_ema, self.ema_period)
            
            # Calculate EMA ratio
            ema_ratio = np.where(double_ema != 0, single_ema / double_ema, 1.0)
            
            # Calculate Mass Index (sum of EMA ratios over mass_period)
            mass_index = np.full(len(price_range), np.nan)
            
            start_idx = (self.ema_period - 1) * 2 + self.mass_period - 1
            for i in range(start_idx, len(price_range)):
                mass_index[i] = np.sum(ema_ratio[i - self.mass_period + 1:i + 1])
            
            # Generate signals
            signals = self._generate_signals(mass_index)
            
            # Quality assessment
            quality_score = self._assess_quality(mass_index, price_range)
            
            return {
                "mass_index": mass_index,
                "values": mass_index[~np.isnan(mass_index)].tolist(),
                "current_value": float(mass_index[-1]) if not np.isnan(mass_index[-1]) else None,
                "ema_ratio": ema_ratio,
                "single_ema": single_ema,
                "double_ema": double_ema,
                "signals": signals,
                "quality_score": quality_score,
                "signal_strength": self._calculate_signal_strength(mass_index),
                "reversal_zones": self._identify_reversal_zones(mass_index)
            }
            
        except Exception as e:
            self.logger.error(f"Mass Index calculation error: {e}")
            return {"error": str(e)}

    def _generate_signals(self, mass_index: np.ndarray) -> Dict[str, List[int]]:
        """Generate trading signals based on Mass Index levels."""
        signals = {
            "potential_reversal": [],
            "high_volatility": [],
            "low_volatility": []
        }
        
        for i in range(1, len(mass_index)):
            if np.isnan(mass_index[i]) or np.isnan(mass_index[i-1]):
                continue
                
            # Potential reversal: Mass Index crosses above 27 then below 26.5
            if mass_index[i-1] > 27 and mass_index[i] < 26.5:
                signals["potential_reversal"].append(i)
            
            # High volatility: Mass Index above 26.5
            if mass_index[i] > 26.5:
                signals["high_volatility"].append(i)
            
            # Low volatility: Mass Index below 22
            if mass_index[i] < 22:
                signals["low_volatility"].append(i)
        
        return signals

    def _assess_quality(self, mass_index: np.ndarray, price_range: np.ndarray) -> float:
        """Assess the quality of Mass Index calculation."""
        try:
            valid_mass = mass_index[~np.isnan(mass_index)]
            if len(valid_mass) == 0:
                return 0.0
            
            # Check for reasonable values
            reasonable_range = np.logical_and(valid_mass > 10, valid_mass < 50)
            range_quality = np.mean(reasonable_range)
            
            # Check volatility consistency
            volatility = np.std(price_range[-min(50, len(price_range)):])
            volatility_quality = min(1.0, volatility / (np.mean(price_range[-min(50, len(price_range)):]) * 0.1))
            
            # Data completeness
            completeness = len(valid_mass) / len(mass_index)
            
            return float(np.mean([range_quality, volatility_quality, completeness]))
            
        except Exception:
            return 0.5

    def _calculate_signal_strength(self, mass_index: np.ndarray) -> float:
        """Calculate signal strength based on Mass Index behavior."""
        try:
            valid_mass = mass_index[~np.isnan(mass_index)]
            if len(valid_mass) < 10:
                return 0.0
            
            recent_mass = valid_mass[-10:]
            
            # Signal strength based on current level and recent trend
            current_level = recent_mass[-1]
            trend_strength = abs(np.mean(np.diff(recent_mass)))
            
            # Normalize signal strength
            level_factor = min(1.0, abs(current_level - 25) / 10)  # 25 is neutral level
            trend_factor = min(1.0, trend_strength)
            
            return float((level_factor + trend_factor) / 2)
            
        except Exception:
            return 0.5

    def _identify_reversal_zones(self, mass_index: np.ndarray) -> List[Dict[str, Any]]:
        """Identify potential reversal zones."""
        zones = []
        try:
            valid_indices = ~np.isnan(mass_index)
            if not np.any(valid_indices):
                return zones
            
            # Find peaks above 27
            for i in range(1, len(mass_index) - 1):
                if (valid_indices[i] and 
                    mass_index[i] > 27 and 
                    mass_index[i] > mass_index[i-1] and 
                    mass_index[i] > mass_index[i+1]):
                    
                    zones.append({
                        "index": i,
                        "value": float(mass_index[i]),
                        "type": "potential_reversal_peak",
                        "strength": min(1.0, (mass_index[i] - 27) / 10)
                    })
            
            return zones
            
        except Exception:
            return []

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required for calculation."""
        return (self.ema_period - 1) * 2 + self.mass_period

    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "category": "volume",
            "subcategory": "volatility",
            "parameters": self.parameters,
            "output_keys": [
                "mass_index", "values", "current_value", "ema_ratio",
                "single_ema", "double_ema", "signals", "quality_score",
                "signal_strength", "reversal_zones"
            ],
            "minimum_periods": self.minimum_periods,
            "platform3_compliant": True,
            "description": "Mass Index volatility indicator for reversal detection"
        }


def export_indicator():
    """Export the indicator for registry discovery."""
    return MassIndex


# Test code
if __name__ == "__main__":
    print("Testing Mass Index Indicator...")
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create realistic OHLC data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some volatility to high/low
    high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    
    test_data = pd.DataFrame({
        'date': dates,
        'high': high_prices,
        'low': low_prices,
        'close': prices
    })
    
    # Test the indicator
    mass_idx = MassIndex(ema_period=9, mass_period=25)
    result = mass_idx.calculate(test_data)
    
    if "error" not in result:
        print(f"✓ Mass Index calculation successful")
        print(f"✓ Current Mass Index: {result.get('current_value', 'N/A'):.4f}")
        print(f"✓ Quality Score: {result['quality_score']:.4f}")
        print(f"✓ Signal Strength: {result['signal_strength']:.4f}")
        print(f"✓ Potential Reversals: {len(result['signals']['potential_reversal'])}")
        print(f"✓ Reversal Zones: {len(result['reversal_zones'])}")
        
        # Show some recent values
        if result['values']:
            recent_values = result['values'][-5:]
            print(f"✓ Recent Mass Index values: {[f'{v:.4f}' for v in recent_values]}")
    else:
        print(f"✗ Error: {result['error']}")
    
    # Test metadata
    metadata = mass_idx.get_metadata()
    print(f"✓ Metadata: {metadata['name']} v{metadata['version']}")
    print(f"✓ Minimum periods: {metadata['minimum_periods']}")
    
    print("\nMass Index test completed!")