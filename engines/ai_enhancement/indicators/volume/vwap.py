"""
Volume Weighted Average Price (VWAP) Indicator

The Volume Weighted Average Price (VWAP) is a trading benchmark that gives the average price 
a security has traded at throughout the day, based on both volume and price. VWAP is important 
because it provides traders with insight into both the trend and value of a security.

Key Features:
- Institutional benchmark for trade execution
- Intraday support/resistance levels
- Volume-weighted price calculation
- Reset daily or by session
- Real-time calculation capability

Mathematical Formula:
VWAP = Σ(Price × Volume) / Σ(Volume)
where Price is typically (High + Low + Close) / 3

Author: Platform3.AI Engine
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

# from ..base_indicator import StandardIndicatorInterface

# Temporarily import for testing
import sys
import os
from base_indicator import StandardIndicatorInterface


@dataclass
class VWAPConfig:
    """Configuration for Volume Weighted Average Price calculation."""
    
    # Session management
    session_start_hour: int = 9  # Market open hour (UTC)
    session_start_minute: int = 30  # Market open minute
    session_end_hour: int = 16  # Market close hour (UTC)
    session_end_minute: int = 0  # Market close minute
    reset_daily: bool = True  # Reset VWAP daily
    
    # Price calculation method
    typical_price_method: str = "hlc3"  # "hlc3", "ohlc4", "close"
    
    # Advanced features
    enable_bands: bool = True  # Calculate VWAP bands
    band_multiplier: float = 1.0  # Standard deviation multiplier for bands
    
    # Performance settings
    max_periods: int = 1000  # Maximum periods to keep in memory
    precision: int = 6  # Decimal precision for calculations


class VWAPIndicator(StandardIndicatorInterface):
    """
    Volume Weighted Average Price (VWAP) Indicator
    
    A volume-weighted average price indicator that provides institutional-grade
    benchmarking and support/resistance levels based on volume-weighted pricing.
    """
    
    def __init__(self, config: Optional[VWAPConfig] = None):
        """
        Initialize VWAP Indicator.
        
        Args:
            config: VWAP configuration parameters
        """
        self.config = config or VWAPConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core calculation arrays
        self.cumulative_volume: float = 0.0
        self.cumulative_pv: float = 0.0  # Price × Volume
        self.session_data: List[Dict] = []
        
        # VWAP values
        self.current_vwap: float = 0.0
        self.vwap_history: List[float] = []
        
        # Band calculations (if enabled)
        self.upper_band: float = 0.0
        self.lower_band: float = 0.0
        self.variance_sum: float = 0.0
        
        # Session tracking
        self.current_session_start: Optional[datetime] = None
        self.last_reset_time: Optional[datetime] = None
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("VWAP Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Volume Weighted Average Price.
        
        Args:
            data: Market data containing OHLCV
            
        Returns:
            Dictionary containing VWAP calculation results
        """
        try:
            # Validate and extract data
            ohlcv_data = self._validate_and_extract_data(data)
            if not ohlcv_data:
                return self._get_empty_result()
            
            # Check for session reset
            current_time = self._get_current_time(data)
            if self._should_reset_session(current_time):
                self._reset_session(current_time)
            
            # Calculate typical price
            typical_price = self._calculate_typical_price(ohlcv_data)
            volume = ohlcv_data['volume']
            
            # Update cumulative values
            price_volume = typical_price * volume
            self.cumulative_pv += price_volume
            self.cumulative_volume += volume
            
            # Calculate VWAP
            if self.cumulative_volume > 0:
                self.current_vwap = self.cumulative_pv / self.cumulative_volume
            else:
                self.current_vwap = typical_price
            
            # Store session data for variance calculation
            self.session_data.append({
                'price': typical_price,
                'volume': volume,
                'price_volume': price_volume,
                'timestamp': current_time
            })
            
            # Calculate bands if enabled
            if self.config.enable_bands:
                self._calculate_bands()
            
            # Update history
            self.vwap_history.append(self.current_vwap)
            if len(self.vwap_history) > self.config.max_periods:
                self.vwap_history.pop(0)
            
            # Mark as initialized
            if not self.is_initialized:
                self.is_initialized = True
            
            self._last_calculation_time = current_time
            
            return self._format_results(ohlcv_data, typical_price)
            
        except Exception as e:
            self.logger.error(f"Error in VWAP calculation: {str(e)}")
            return self._get_error_result(str(e))
    
    def _validate_and_extract_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """Validate input data and extract OHLCV values."""
        try:
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return None
                
                latest = data.iloc[-1]
                return {
                    'open': float(latest.get('open', latest.get('Open', 0))),
                    'high': float(latest.get('high', latest.get('High', 0))),
                    'low': float(latest.get('low', latest.get('Low', 0))),
                    'close': float(latest.get('close', latest.get('Close', 0))),
                    'volume': float(latest.get('volume', latest.get('Volume', 0)))
                }
            
            elif isinstance(data, dict):
                return {
                    'open': float(data.get('open', data.get('Open', 0))),
                    'high': float(data.get('high', data.get('High', 0))),
                    'low': float(data.get('low', data.get('Low', 0))),
                    'close': float(data.get('close', data.get('Close', 0))),
                    'volume': float(data.get('volume', data.get('Volume', 0)))
                }
            
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return None
    
    def _calculate_typical_price(self, ohlcv_data: Dict) -> float:
        """Calculate typical price based on configuration method."""
        if self.config.typical_price_method == "hlc3":
            return (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        elif self.config.typical_price_method == "ohlc4":
            return (ohlcv_data['open'] + ohlcv_data['high'] + 
                   ohlcv_data['low'] + ohlcv_data['close']) / 4
        else:  # "close"
            return ohlcv_data['close']
    
    def _get_current_time(self, data: Union[pd.DataFrame, Dict]) -> datetime:
        """Extract or generate current timestamp."""
        try:
            if isinstance(data, pd.DataFrame) and 'timestamp' in data.columns:
                return pd.to_datetime(data.iloc[-1]['timestamp'])
            elif isinstance(data, dict) and 'timestamp' in data:
                return pd.to_datetime(data['timestamp'])
            else:
                return datetime.now(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    
    def _should_reset_session(self, current_time: datetime) -> bool:
        """Determine if VWAP should reset for new session."""
        if not self.config.reset_daily:
            return False
        
        if self.last_reset_time is None:
            return True
        
        # Check if we've crossed into a new trading day
        last_reset_date = self.last_reset_time.date()
        current_date = current_time.date()
        
        if current_date > last_reset_date:
            # Check if we're in trading hours
            session_start = current_time.replace(
                hour=self.config.session_start_hour,
                minute=self.config.session_start_minute,
                second=0,
                microsecond=0
            )
            
            return current_time >= session_start
        
        return False
    
    def _reset_session(self, current_time: datetime):
        """Reset VWAP calculations for new session."""
        self.cumulative_volume = 0.0
        self.cumulative_pv = 0.0
        self.session_data.clear()
        self.variance_sum = 0.0
        self.current_session_start = current_time
        self.last_reset_time = current_time
        
        self.logger.info(f"VWAP session reset at {current_time}")
    
    def _calculate_bands(self):
        """Calculate VWAP standard deviation bands."""
        if len(self.session_data) < 2:
            self.upper_band = self.current_vwap
            self.lower_band = self.current_vwap
            return
        
        # Calculate volume-weighted variance
        total_volume = sum(item['volume'] for item in self.session_data)
        if total_volume == 0:
            self.upper_band = self.current_vwap
            self.lower_band = self.current_vwap
            return
        
        variance_sum = 0.0
        for item in self.session_data:
            price_diff = item['price'] - self.current_vwap
            variance_sum += (price_diff ** 2) * item['volume']
        
        variance = variance_sum / total_volume
        std_dev = np.sqrt(variance)
        
        band_offset = std_dev * self.config.band_multiplier
        self.upper_band = self.current_vwap + band_offset
        self.lower_band = self.current_vwap - band_offset
    
    def _format_results(self, ohlcv_data: Dict, typical_price: float) -> Dict:
        """Format calculation results."""
        result = {
            'vwap': round(self.current_vwap, self.config.precision),
            'typical_price': round(typical_price, self.config.precision),
            'cumulative_volume': self.cumulative_volume,
            'session_trades': len(self.session_data),
            'session_start': self.current_session_start.isoformat() if self.current_session_start else None,
            'calculation_time': self._last_calculation_time.isoformat() if self._last_calculation_time else None,
            'status': 'active' if self.cumulative_volume > 0 else 'initializing'
        }
        
        # Add bands if enabled
        if self.config.enable_bands:
            result.update({
                'upper_band': round(self.upper_band, self.config.precision),
                'lower_band': round(self.lower_band, self.config.precision),
                'band_width': round(self.upper_band - self.lower_band, self.config.precision)
            })
        
        # Add signals
        result.update(self._generate_signals(ohlcv_data['close']))
        
        return result
    
    def _generate_signals(self, current_price: float) -> Dict:
        """Generate trading signals based on VWAP analysis."""
        signals = {
            'price_vs_vwap': 'neutral',
            'signal_strength': 0.0,
            'support_resistance_level': self.current_vwap
        }
        
        if self.current_vwap > 0:
            price_diff_pct = ((current_price - self.current_vwap) / self.current_vwap) * 100
            
            if price_diff_pct > 0.1:
                signals['price_vs_vwap'] = 'above_vwap'
                signals['signal_strength'] = min(abs(price_diff_pct) / 2.0, 1.0)
            elif price_diff_pct < -0.1:
                signals['price_vs_vwap'] = 'below_vwap'
                signals['signal_strength'] = min(abs(price_diff_pct) / 2.0, 1.0)
            
            # Band signals if available
            if self.config.enable_bands:
                if current_price > self.upper_band:
                    signals['band_signal'] = 'overbought'
                elif current_price < self.lower_band:
                    signals['band_signal'] = 'oversold'
                else:
                    signals['band_signal'] = 'normal'
        
        return signals
    
    def _get_empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'vwap': 0.0,
            'typical_price': 0.0,
            'cumulative_volume': 0.0,
            'session_trades': 0,
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'vwap': 0.0,
            'typical_price': 0.0,
            'cumulative_volume': 0.0,
            'session_trades': 0,
            'status': 'error',
            'error': error_message
        }
    
    def get_historical_values(self, periods: int = 50) -> List[float]:
        """
        Get historical VWAP values.
        
        Args:
            periods: Number of historical periods to return
            
        Returns:
            List of historical VWAP values
        """
        return self.vwap_history[-periods:] if self.vwap_history else []
    
    def get_session_statistics(self) -> Dict:
        """Get current session statistics."""
        return {
            'session_start': self.current_session_start.isoformat() if self.current_session_start else None,
            'trades_count': len(self.session_data),
            'total_volume': self.cumulative_volume,
            'current_vwap': self.current_vwap,
            'session_high': max((item['price'] for item in self.session_data), default=0),
            'session_low': min((item['price'] for item in self.session_data), default=0)
        }
    
    def reset(self):
        """Reset indicator state."""
        self.cumulative_volume = 0.0
        self.cumulative_pv = 0.0
        self.session_data.clear()
        self.current_vwap = 0.0
        self.vwap_history.clear()
        self.upper_band = 0.0
        self.lower_band = 0.0
        self.variance_sum = 0.0
        self.current_session_start = None
        self.last_reset_time = None
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("VWAP Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'VWAP',
            'full_name': 'Volume Weighted Average Price',
            'description': 'Institutional benchmark providing volume-weighted average price with support/resistance levels',
            'category': 'Volume',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['volume', 'benchmark', 'institutional', 'support', 'resistance'],
            'inputs': ['high', 'low', 'close', 'volume'],
            'outputs': ['vwap', 'upper_band', 'lower_band', 'signals'],
            'parameters': {
                'session_start_hour': self.config.session_start_hour,
                'session_start_minute': self.config.session_start_minute,
                'reset_daily': self.config.reset_daily,
                'enable_bands': self.config.enable_bands,
                'band_multiplier': self.config.band_multiplier
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate session hours
            if not (0 <= self.config.session_start_hour <= 23):
                self.logger.error("Invalid session_start_hour: must be 0-23")
                return False
            
            if not (0 <= self.config.session_start_minute <= 59):
                self.logger.error("Invalid session_start_minute: must be 0-59")
                return False
            
            # Validate band multiplier
            if self.config.band_multiplier <= 0:
                self.logger.error("Invalid band_multiplier: must be positive")
                return False
            
            # Validate max periods
            if self.config.max_periods <= 0:
                self.logger.error("Invalid max_periods: must be positive")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_data = [
        {'high': 101.5, 'low': 100.0, 'close': 101.2, 'volume': 10000, 'timestamp': '2024-01-01 09:30:00'},
        {'high': 102.0, 'low': 101.0, 'close': 101.8, 'volume': 15000, 'timestamp': '2024-01-01 09:31:00'},
        {'high': 102.5, 'low': 101.5, 'close': 102.1, 'volume': 12000, 'timestamp': '2024-01-01 09:32:00'},
        {'high': 102.2, 'low': 101.8, 'close': 102.0, 'volume': 8000, 'timestamp': '2024-01-01 09:33:00'},
        {'high': 103.0, 'low': 102.0, 'close': 102.8, 'volume': 20000, 'timestamp': '2024-01-01 09:34:00'},
    ]
    
    # Initialize indicator
    config = VWAPConfig(enable_bands=True, band_multiplier=1.0)
    vwap = VWAPIndicator(config)
    
    print("=== VWAP Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = vwap.calculate(data_point)
        print(f"Period {i}:")
        print(f"  Price: {data_point['close']}")
        print(f"  Volume: {data_point['volume']}")
        print(f"  VWAP: {result['vwap']}")
        print(f"  Typical Price: {result['typical_price']}")
        print(f"  Signal: {result.get('price_vs_vwap', 'N/A')}")
        if 'upper_band' in result:
            print(f"  Upper Band: {result['upper_band']}")
            print(f"  Lower Band: {result['lower_band']}")
        print(f"  Status: {result['status']}")
        print()
    
    # Test session statistics
    print("=== Session Statistics ===")
    stats = vwap.get_session_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test historical values
    print(f"\n=== Historical Values (last 3) ===")
    historical = vwap.get_historical_values(3)
    for i, value in enumerate(historical, 1):
        print(f"Period {i}: {value}")