"""
Ichimoku Kinko Hyo (Ichimoku Cloud) Indicator

Ichimoku Kinko Hyo, often referred to as the Ichimoku Cloud, is a comprehensive technical 
analysis system that provides information about support and resistance levels, trend direction, 
and momentum all in one view. It consists of five main components.

Key Features:
- Tenkan-sen (Conversion Line): 9-period high-low average
- Kijun-sen (Base Line): 26-period high-low average  
- Senkou Span A (Leading Span A): Future cloud boundary
- Senkou Span B (Leading Span B): Future cloud boundary
- Chikou Span (Lagging Span): Price displaced backward
- Kumo (Cloud): Support/resistance area between Span A and B

Mathematical Formulas:
- Tenkan-sen = (Highest High + Lowest Low) / 2 over 9 periods
- Kijun-sen = (Highest High + Lowest Low) / 2 over 26 periods  
- Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
- Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, plotted 26 periods ahead
- Chikou Span = Current close price, plotted 26 periods behind

Author: Platform3.AI Engine
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Temporarily import for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_indicator import StandardIndicatorInterface


@dataclass
class IchimokuConfig:
    """Configuration for Ichimoku Kinko Hyo calculation."""
    
    # Standard Ichimoku periods
    tenkan_period: int = 9  # Conversion line period
    kijun_period: int = 26  # Base line period
    senkou_b_period: int = 52  # Leading span B period
    displacement: int = 26  # Future displacement for cloud
    
    # Advanced features
    enable_signals: bool = True  # Generate trading signals
    enable_cloud_analysis: bool = True  # Detailed cloud analysis
    
    # Performance settings
    max_history: int = 200  # Maximum historical data to keep
    precision: int = 6  # Decimal precision for calculations


class IchimokuIndicator(StandardIndicatorInterface):
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud) Indicator
    
    A comprehensive technical analysis system providing trend direction,
    momentum, and support/resistance levels through five key components.
    """
    
    def __init__(self, config: Optional[IchimokuConfig] = None):
        """
        Initialize Ichimoku Indicator.
        
        Args:
            config: Ichimoku configuration parameters
        """
        self.config = config or IchimokuConfig()
        self.logger = logging.getLogger(__name__)
        
        # Historical data storage
        self.price_history: List[Dict] = []
        
        # Ichimoku components
        self.tenkan_sen: float = 0.0
        self.kijun_sen: float = 0.0
        self.senkou_span_a: float = 0.0
        self.senkou_span_b: float = 0.0
        self.chikou_span: float = 0.0
        
        # Historical values for plotting
        self.tenkan_history: List[float] = []
        self.kijun_history: List[float] = []
        self.senkou_a_history: List[float] = []
        self.senkou_b_history: List[float] = []
        self.chikou_history: List[float] = []
        
        # Cloud analysis
        self.cloud_color: str = "neutral"  # "bullish", "bearish", "neutral"
        self.cloud_thickness: float = 0.0
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Ichimoku Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Ichimoku Kinko Hyo components.
        
        Args:
            data: Market data containing OHLCV
            
        Returns:
            Dictionary containing Ichimoku calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Add to price history
            self._update_price_history(ohlc_data)
            
            # Calculate all Ichimoku components
            self._calculate_tenkan_sen()
            self._calculate_kijun_sen()
            self._calculate_senkou_spans()
            self._calculate_chikou_span()
            
            # Perform cloud analysis
            if self.config.enable_cloud_analysis:
                self._analyze_cloud()
            
            # Update historical values
            self._update_history()
            
            # Generate signals
            signals = {}
            if self.config.enable_signals:
                signals = self._generate_signals(ohlc_data['close'])
            
            # Mark as initialized
            if not self.is_initialized and len(self.price_history) >= self.config.senkou_b_period:
                self.is_initialized = True
            
            self._last_calculation_time = datetime.now()
            
            return self._format_results(ohlc_data, signals)
            
        except Exception as e:
            self.logger.error(f"Error in Ichimoku calculation: {str(e)}")
            return self._get_error_result(str(e))
    
    def _validate_and_extract_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """Validate input data and extract OHLC values."""
        try:
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return None
                
                latest = data.iloc[-1]
                return {
                    'high': float(latest.get('high', latest.get('High', 0))),
                    'low': float(latest.get('low', latest.get('Low', 0))),
                    'close': float(latest.get('close', latest.get('Close', 0))),
                    'timestamp': latest.get('timestamp', datetime.now())
                }
            
            elif isinstance(data, dict):
                return {
                    'high': float(data.get('high', data.get('High', 0))),
                    'low': float(data.get('low', data.get('Low', 0))),
                    'close': float(data.get('close', data.get('Close', 0))),
                    'timestamp': data.get('timestamp', datetime.now())
                }
            
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return None
    
    def _update_price_history(self, ohlc_data: Dict):
        """Update internal price history."""
        self.price_history.append(ohlc_data)
        
        # Maintain maximum history size
        if len(self.price_history) > self.config.max_history:
            self.price_history.pop(0)
    
    def _calculate_tenkan_sen(self):
        """Calculate Tenkan-sen (Conversion Line)."""
        if len(self.price_history) < self.config.tenkan_period:
            self.tenkan_sen = 0.0
            return
        
        recent_data = self.price_history[-self.config.tenkan_period:]
        highest_high = max(item['high'] for item in recent_data)
        lowest_low = min(item['low'] for item in recent_data)
        
        self.tenkan_sen = (highest_high + lowest_low) / 2
    
    def _calculate_kijun_sen(self):
        """Calculate Kijun-sen (Base Line)."""
        if len(self.price_history) < self.config.kijun_period:
            self.kijun_sen = 0.0
            return
        
        recent_data = self.price_history[-self.config.kijun_period:]
        highest_high = max(item['high'] for item in recent_data)
        lowest_low = min(item['low'] for item in recent_data)
        
        self.kijun_sen = (highest_high + lowest_low) / 2
    
    def _calculate_senkou_spans(self):
        """Calculate Senkou Span A and B (Leading Spans)."""
        # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2
        if self.tenkan_sen > 0 and self.kijun_sen > 0:
            self.senkou_span_a = (self.tenkan_sen + self.kijun_sen) / 2
        else:
            self.senkou_span_a = 0.0
        
        # Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods
        if len(self.price_history) < self.config.senkou_b_period:
            self.senkou_span_b = 0.0
            return
        
        recent_data = self.price_history[-self.config.senkou_b_period:]
        highest_high = max(item['high'] for item in recent_data)
        lowest_low = min(item['low'] for item in recent_data)
        
        self.senkou_span_b = (highest_high + lowest_low) / 2
    
    def _calculate_chikou_span(self):
        """Calculate Chikou Span (Lagging Span)."""
        # Chikou Span is current close price plotted 26 periods behind
        if len(self.price_history) > 0:
            self.chikou_span = self.price_history[-1]['close']
        else:
            self.chikou_span = 0.0
    
    def _analyze_cloud(self):
        """Analyze cloud (Kumo) characteristics."""
        if self.senkou_span_a > 0 and self.senkou_span_b > 0:
            # Determine cloud color (bullish/bearish)
            if self.senkou_span_a > self.senkou_span_b:
                self.cloud_color = "bullish"
            elif self.senkou_span_a < self.senkou_span_b:
                self.cloud_color = "bearish"
            else:
                self.cloud_color = "neutral"
            
            # Calculate cloud thickness
            self.cloud_thickness = abs(self.senkou_span_a - self.senkou_span_b)
        else:
            self.cloud_color = "neutral"
            self.cloud_thickness = 0.0
    
    def _update_history(self):
        """Update historical values for all components."""
        self.tenkan_history.append(self.tenkan_sen)
        self.kijun_history.append(self.kijun_sen)
        self.senkou_a_history.append(self.senkou_span_a)
        self.senkou_b_history.append(self.senkou_span_b)
        self.chikou_history.append(self.chikou_span)
        
        # Maintain maximum history
        max_keep = self.config.max_history
        if len(self.tenkan_history) > max_keep:
            self.tenkan_history.pop(0)
            self.kijun_history.pop(0)
            self.senkou_a_history.pop(0)
            self.senkou_b_history.pop(0)
            self.chikou_history.pop(0)
    
    def _generate_signals(self, current_price: float) -> Dict:
        """Generate trading signals based on Ichimoku analysis."""
        signals = {
            'trend_direction': 'neutral',
            'signal_strength': 0.0,
            'tenkan_kijun_cross': 'none',
            'price_vs_cloud': 'neutral',
            'chikou_confirmation': 'neutral'
        }
        
        if not self.is_initialized:
            return signals
        
        # 1. Tenkan-Kijun cross signal
        if len(self.tenkan_history) >= 2 and len(self.kijun_history) >= 2:
            prev_tenkan = self.tenkan_history[-2]
            prev_kijun = self.kijun_history[-2]
            
            # Current cross
            if self.tenkan_sen > self.kijun_sen and prev_tenkan <= prev_kijun:
                signals['tenkan_kijun_cross'] = 'bullish'
            elif self.tenkan_sen < self.kijun_sen and prev_tenkan >= prev_kijun:
                signals['tenkan_kijun_cross'] = 'bearish'
        
        # 2. Price vs Cloud position
        cloud_top = max(self.senkou_span_a, self.senkou_span_b)
        cloud_bottom = min(self.senkou_span_a, self.senkou_span_b)
        
        if current_price > cloud_top:
            signals['price_vs_cloud'] = 'above_cloud'
        elif current_price < cloud_bottom:
            signals['price_vs_cloud'] = 'below_cloud'
        else:
            signals['price_vs_cloud'] = 'in_cloud'
        
        # 3. Chikou confirmation (simplified)
        if len(self.price_history) >= self.config.displacement:
            past_price = self.price_history[-self.config.displacement]['close']
            if self.chikou_span > past_price:
                signals['chikou_confirmation'] = 'bullish'
            elif self.chikou_span < past_price:
                signals['chikou_confirmation'] = 'bearish'
        
        # 4. Overall trend direction
        bullish_signals = 0
        bearish_signals = 0
        
        if signals['tenkan_kijun_cross'] == 'bullish':
            bullish_signals += 1
        elif signals['tenkan_kijun_cross'] == 'bearish':
            bearish_signals += 1
            
        if signals['price_vs_cloud'] == 'above_cloud':
            bullish_signals += 1
        elif signals['price_vs_cloud'] == 'below_cloud':
            bearish_signals += 1
            
        if signals['chikou_confirmation'] == 'bullish':
            bullish_signals += 1
        elif signals['chikou_confirmation'] == 'bearish':
            bearish_signals += 1
            
        if self.cloud_color == 'bullish':
            bullish_signals += 1
        elif self.cloud_color == 'bearish':
            bearish_signals += 1
        
        # Determine overall trend
        if bullish_signals > bearish_signals:
            signals['trend_direction'] = 'bullish'
            signals['signal_strength'] = min(bullish_signals / 4.0, 1.0)
        elif bearish_signals > bullish_signals:
            signals['trend_direction'] = 'bearish'
            signals['signal_strength'] = min(bearish_signals / 4.0, 1.0)
        
        return signals
    
    def _format_results(self, ohlc_data: Dict, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core Ichimoku components
            'tenkan_sen': round(self.tenkan_sen, self.config.precision),
            'kijun_sen': round(self.kijun_sen, self.config.precision),
            'senkou_span_a': round(self.senkou_span_a, self.config.precision),
            'senkou_span_b': round(self.senkou_span_b, self.config.precision),
            'chikou_span': round(self.chikou_span, self.config.precision),
            
            # Cloud analysis
            'cloud_color': self.cloud_color,
            'cloud_thickness': round(self.cloud_thickness, self.config.precision),
            'cloud_top': round(max(self.senkou_span_a, self.senkou_span_b), self.config.precision),
            'cloud_bottom': round(min(self.senkou_span_a, self.senkou_span_b), self.config.precision),
            
            # Status
            'is_initialized': self.is_initialized,
            'data_points': len(self.price_history),
            'calculation_time': self._last_calculation_time.isoformat() if self._last_calculation_time else None,
            'status': 'active' if self.is_initialized else 'initializing'
        }
        
        # Add signals if enabled
        if self.config.enable_signals:
            result.update(signals)
        
        return result
    
    def _get_empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'tenkan_sen': 0.0,
            'kijun_sen': 0.0,
            'senkou_span_a': 0.0,
            'senkou_span_b': 0.0,
            'chikou_span': 0.0,
            'cloud_color': 'neutral',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'tenkan_sen': 0.0,
            'kijun_sen': 0.0,
            'senkou_span_a': 0.0,
            'senkou_span_b': 0.0,
            'chikou_span': 0.0,
            'cloud_color': 'neutral',
            'status': 'error',
            'error': error_message
        }
    
    def get_cloud_projection(self, periods_ahead: int = 26) -> List[Dict]:
        """
        Get future cloud projection.
        
        Args:
            periods_ahead: Number of periods to project ahead
            
        Returns:
            List of future cloud values
        """
        if not self.is_initialized:
            return []
        
        # For simplicity, project current cloud values forward
        projection = []
        for i in range(periods_ahead):
            projection.append({
                'period': i + 1,
                'senkou_span_a': self.senkou_span_a,
                'senkou_span_b': self.senkou_span_b,
                'cloud_color': self.cloud_color
            })
        
        return projection
    
    def get_historical_components(self, periods: int = 50) -> Dict:
        """Get historical values for all Ichimoku components."""
        return {
            'tenkan_sen': self.tenkan_history[-periods:] if self.tenkan_history else [],
            'kijun_sen': self.kijun_history[-periods:] if self.kijun_history else [],
            'senkou_span_a': self.senkou_a_history[-periods:] if self.senkou_a_history else [],
            'senkou_span_b': self.senkou_b_history[-periods:] if self.senkou_b_history else [],
            'chikou_span': self.chikou_history[-periods:] if self.chikou_history else []
        }
    
    def reset(self):
        """Reset indicator state."""
        self.price_history.clear()
        self.tenkan_sen = 0.0
        self.kijun_sen = 0.0
        self.senkou_span_a = 0.0
        self.senkou_span_b = 0.0
        self.chikou_span = 0.0
        self.tenkan_history.clear()
        self.kijun_history.clear()
        self.senkou_a_history.clear()
        self.senkou_b_history.clear()
        self.chikou_history.clear()
        self.cloud_color = "neutral"
        self.cloud_thickness = 0.0
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Ichimoku Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'Ichimoku',
            'full_name': 'Ichimoku Kinko Hyo (Ichimoku Cloud)',
            'description': 'Comprehensive technical analysis system with trend, momentum, and support/resistance',
            'category': 'Trend',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['trend', 'momentum', 'support', 'resistance', 'cloud', 'comprehensive'],
            'inputs': ['high', 'low', 'close'],
            'outputs': ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'signals'],
            'parameters': {
                'tenkan_period': self.config.tenkan_period,
                'kijun_period': self.config.kijun_period,
                'senkou_b_period': self.config.senkou_b_period,
                'displacement': self.config.displacement
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate periods
            if self.config.tenkan_period <= 0:
                self.logger.error("Invalid tenkan_period: must be positive")
                return False
            
            if self.config.kijun_period <= 0:
                self.logger.error("Invalid kijun_period: must be positive")
                return False
            
            if self.config.senkou_b_period <= 0:
                self.logger.error("Invalid senkou_b_period: must be positive")
                return False
            
            if self.config.displacement <= 0:
                self.logger.error("Invalid displacement: must be positive")
                return False
            
            # Validate period relationships
            if self.config.tenkan_period >= self.config.kijun_period:
                self.logger.warning("tenkan_period should be less than kijun_period for standard Ichimoku")
            
            if self.config.kijun_period >= self.config.senkou_b_period:
                self.logger.warning("kijun_period should be less than senkou_b_period for standard Ichimoku")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test data - extended dataset for Ichimoku
    test_data = []
    base_price = 100.0
    
    # Generate 60 periods of test data
    for i in range(60):
        trend = 0.02 * i  # Upward trend
        noise = np.random.normal(0, 0.5)
        
        high = base_price + trend + noise + 0.5
        low = base_price + trend + noise - 0.5
        close = base_price + trend + noise + np.random.uniform(-0.3, 0.3)
        
        test_data.append({
            'high': high,
            'low': low,
            'close': close,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = IchimokuConfig(enable_signals=True, enable_cloud_analysis=True)
    ichimoku = IchimokuIndicator(config)
    
    print("=== Ichimoku Kinko Hyo Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = ichimoku.calculate(data_point)
        
        # Only print every 10th period to avoid too much output
        if i % 10 == 0 or i >= 52:  # Show more detail after initialization
            print(f"Period {i}:")
            print(f"  Price: {data_point['close']:.2f}")
            print(f"  Tenkan-sen: {result['tenkan_sen']}")
            print(f"  Kijun-sen: {result['kijun_sen']}")
            print(f"  Senkou Span A: {result['senkou_span_a']}")
            print(f"  Senkou Span B: {result['senkou_span_b']}")
            print(f"  Cloud Color: {result['cloud_color']}")
            print(f"  Cloud Thickness: {result['cloud_thickness']}")
            
            if 'trend_direction' in result:
                print(f"  Trend: {result['trend_direction']}")
                print(f"  Signal Strength: {result['signal_strength']}")
                print(f"  Price vs Cloud: {result['price_vs_cloud']}")
            
            print(f"  Status: {result['status']}")
            print()
    
    # Test cloud projection
    print("=== Cloud Projection (next 5 periods) ===")
    projection = ichimoku.get_cloud_projection(5)
    for proj in projection:
        print(f"Period +{proj['period']}: Span A={proj['senkou_span_a']:.2f}, "
              f"Span B={proj['senkou_span_b']:.2f}, Color={proj['cloud_color']}")
    
    # Test historical components
    print(f"\n=== Historical Components (last 3 periods) ===")
    historical = ichimoku.get_historical_components(3)
    for component, values in historical.items():
        print(f"{component}: {[round(v, 2) for v in values]}")