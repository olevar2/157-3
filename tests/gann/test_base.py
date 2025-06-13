"""
Gann Test Base Infrastructure
============================

Base test classes and utilities for comprehensive Gann indicator testing.
Provides shared test data generators, validation methods, and base test patterns
following Platform3 testing standards.

Created following Fibonacci test patterns from:
tests/fibonacci/test_fibonacci_indicators.py
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import random
import time
from unittest.mock import Mock, patch

# Platform3 imports
from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
from engines.ai_enhancement.indicators.gann import GANN_INDICATORS


class GannTestBase(unittest.TestCase):
    """
    Base test class for all Gann indicator tests.
    
    Provides:
    - Geometric precision validation (6+ decimal places)
    - Test data generators for various market conditions
    - Performance benchmarking utilities
    - Signal validation helpers
    - Mathematical calculation verification
    """
    
    def setUp(self):
        """Set up test environment with precision requirements and test data."""
        # Geometric precision requirement for Gann calculations
        self.geometric_precision = 1e-6
        self.angle_precision_places = 6
        
        # Performance benchmarks (Platform3 standards)
        self.max_time_1k = 0.01    # 10ms for 1K data points
        self.max_time_10k = 0.1    # 100ms for 10K data points  
        self.max_time_100k = 1.0   # 1s for 100K data points
        
        # Test data sizes
        self.small_dataset = 100
        self.medium_dataset = 1000
        self.large_dataset = 10000
        self.stress_dataset = 100000
        
        # Generate base test data
        self.test_data = self.generate_realistic_ohlc_data(self.medium_dataset)
        self.trend_up_data = self.generate_trending_data(self.medium_dataset, "up")
        self.trend_down_data = self.generate_trending_data(self.medium_dataset, "down")
        self.sideways_data = self.generate_trending_data(self.medium_dataset, "sideways")
        
    def validate_angle_precision(self, calculated: float, expected: float, 
                                message: str = "Angle calculation precision"):
        """
        Validate angle calculations meet geometric precision requirements.
        
        Args:
            calculated: Calculated angle value
            expected: Expected angle value  
            message: Assertion message
        """
        self.assertAlmostEqual(
            calculated, expected, 
            places=self.angle_precision_places,
            msg=f"{message}: Expected {expected}, got {calculated}"
        )
        
    def validate_ratio_precision(self, calculated: float, expected: float,
                                message: str = "Ratio calculation precision"):
        """
        Validate Gann ratio calculations meet precision requirements.
        
        Args:
            calculated: Calculated ratio value
            expected: Expected ratio value
            message: Assertion message  
        """
        self.assertAlmostEqual(
            calculated, expected,
            delta=self.geometric_precision,
            msg=f"{message}: Expected {expected}, got {calculated}"
        )
        
    def generate_realistic_ohlc_data(self, size: int, 
                                   base_price: float = 100.0,
                                   volatility: float = 0.02) -> pd.DataFrame:
        """
        Generate realistic OHLC data with controlled characteristics.
        
        Args:
            size: Number of data points
            base_price: Starting price level
            volatility: Price volatility factor
            
        Returns:
            DataFrame with OHLC data and timestamps
        """
        np.random.seed(42)  # Reproducible test data
        
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        
        # Generate price movements with geometric brownian motion
        returns = np.random.normal(0, volatility, size)
        price_path = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC from price path
        data = []
        for i, price in enumerate(price_path):
            # Add intraday volatility
            daily_vol = volatility * 0.5
            high = price * (1 + abs(np.random.normal(0, daily_vol)))
            low = price * (1 - abs(np.random.normal(0, daily_vol)))
            
            # Ensure OHLC logic
            open_price = price_path[i-1] if i > 0 else price
            close_price = price
            
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            data.append({
                'timestamp': dates[i],
                'open': round(open_price, 4),
                'high': round(high, 4), 
                'low': round(low, 4),
                'close': round(close_price, 4),
                'volume': random.randint(1000000, 5000000)
            })
            
        return pd.DataFrame(data)
        
    def generate_trending_data(self, size: int, trend_type: str = "up",
                              base_price: float = 100.0) -> pd.DataFrame:
        """
        Generate data with specific trend characteristics for Gann testing.
        
        Args:
            size: Number of data points
            trend_type: "up", "down", or "sideways"
            base_price: Starting price level
            
        Returns:
            DataFrame with trending OHLC data
        """
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        
        if trend_type == "up":
            # Consistent upward trend with Gann angle characteristics
            trend_rate = 0.001  # ~45 degree angle equivalent
            base_trend = base_price * (1 + trend_rate) ** np.arange(size)
            
        elif trend_type == "down":
            # Consistent downward trend  
            trend_rate = -0.001
            base_trend = base_price * (1 + trend_rate) ** np.arange(size)
            
        else:  # sideways
            # Sideways movement around base price
            base_trend = base_price + np.random.normal(0, base_price * 0.01, size)
            
        # Add some noise but maintain trend
        noise = np.random.normal(0, base_price * 0.005, size)
        prices = base_trend + noise
        
        data = []
        for i, price in enumerate(prices):
            # Minimal intraday variation for clean trend testing
            high = price * 1.005
            low = price * 0.995
            open_price = prices[i-1] if i > 0 else price
            
            data.append({
                'timestamp': dates[i],
                'open': round(open_price, 4),
                'high': round(high, 4),
                'low': round(low, 4), 
                'close': round(price, 4),
                'volume': 1000000
            })
            
        return pd.DataFrame(data)
        
    def generate_gann_test_data(self, trend_type: str = "geometric") -> pd.DataFrame:
        """
        Generate data specifically designed for Gann geometric validation.
        
        Args:
            trend_type: Type of geometric progression to create
            
        Returns:
            DataFrame optimized for Gann calculations
        """
        size = 100
        base_price = 100.0
        
        if trend_type == "geometric":
            # Perfect geometric progression for angle testing
            ratio = 1.01  # 1% growth per period
            prices = [base_price * (ratio ** i) for i in range(size)]
            
        elif trend_type == "square_of_nine":
            # Square of 9 number sequence
            prices = []
            center = base_price
            for i in range(size):
                # Create square of 9 progression
                level = int(np.sqrt(i)) + 1
                price = center * (1 + 0.01 * level)
                prices.append(price)
                
        else:  # harmonic
            # Harmonic price relationships
            prices = []
            for i in range(size):
                harmonic = base_price * (1 + 0.618 * np.sin(i * 0.1))  # Golden ratio
                prices.append(harmonic)
        
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        
        data = []
        for i, price in enumerate(prices):
            data.append({
                'timestamp': dates[i],
                'open': round(price, 4),
                'high': round(price * 1.001, 4),
                'low': round(price * 0.999, 4),
                'close': round(price, 4),
                'volume': 1000000
            })
            
        return pd.DataFrame(data)
        
    def benchmark_indicator_performance(self, indicator_class, data_sizes: List[int],
                                      parameters: Dict = None) -> Dict[str, float]:
        """
        Benchmark indicator performance across different data sizes.
        
        Args:
            indicator_class: Gann indicator class to test
            data_sizes: List of data sizes to benchmark
            parameters: Optional indicator parameters
            
        Returns:
            Dictionary of execution times by data size
        """
        if parameters is None:
            parameters = {}
            
        results = {}
        
        for size in data_sizes:
            test_data = self.generate_realistic_ohlc_data(size)
            
            # Create indicator instance
            indicator = indicator_class(parameters)
            
            # Benchmark calculation time
            start_time = time.time()
            result = indicator.calculate(test_data)
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[f"{size}_points"] = execution_time
            
            # Validate result is not None
            self.assertIsNotNone(result, f"Indicator returned None for {size} data points")
            
        return results
        
    def validate_performance_requirements(self, execution_times: Dict[str, float]):
        """
        Validate that indicator meets Platform3 performance requirements.
        
        Args:
            execution_times: Dictionary of execution times by data size
        """
        if "1000_points" in execution_times:
            self.assertLess(
                execution_times["1000_points"], self.max_time_1k,
                f"1K data processing too slow: {execution_times['1000_points']:.4f}s > {self.max_time_1k}s"
            )
            
        if "10000_points" in execution_times:
            self.assertLess(
                execution_times["10000_points"], self.max_time_10k,
                f"10K data processing too slow: {execution_times['10000_points']:.4f}s > {self.max_time_10k}s"
            )
            
        if "100000_points" in execution_times:
            self.assertLess(
                execution_times["100000_points"], self.max_time_100k,
                f"100K data processing too slow: {execution_times['100000_points']:.4f}s > {self.max_time_100k}s"
            )
            
    def validate_gann_angles(self, angles: List[float], expected_angles: List[float]):
        """
        Validate calculated Gann angles against expected sacred angles.
        
        Args:
            angles: Calculated angle values
            expected_angles: Expected Gann sacred angles
        """
        self.assertEqual(len(angles), len(expected_angles), 
                        "Angle count mismatch")
        
        for i, (calc, expected) in enumerate(zip(angles, expected_angles)):
            self.validate_angle_precision(
                calc, expected,
                f"Gann angle {i+1} validation"
            )
            
    def validate_square_levels(self, levels: List[float], price_center: float):
        """
        Validate Gann Square of 9 levels are correctly calculated.
        
        Args:
            levels: Calculated square levels
            price_center: Center price for square calculation
        """
        self.assertGreater(len(levels), 0, "No square levels calculated")
        
        # Validate levels are in ascending order
        for i in range(1, len(levels)):
            self.assertGreater(levels[i], levels[i-1],
                             f"Square levels not in ascending order at index {i}")
            
        # Validate center price is included or close
        distances = [abs(level - price_center) for level in levels]
        min_distance = min(distances)
        self.assertLess(min_distance, price_center * 0.01,
                       "Center price not found in square levels")
                       
    def validate_signal_structure(self, signals: Dict[str, Any]):
        """
        Validate that signal output has required structure and types.
        
        Args:
            signals: Signal dictionary from indicator
        """
        # Required signal fields
        required_fields = ['buy_signals', 'sell_signals', 'signal_strength', 'timestamp']
        
        for field in required_fields:
            self.assertIn(field, signals, f"Missing required signal field: {field}")
            
        # Validate signal types
        self.assertIsInstance(signals['buy_signals'], (list, pd.Series, np.ndarray),
                            "buy_signals must be array-like")
        self.assertIsInstance(signals['sell_signals'], (list, pd.Series, np.ndarray),
                            "sell_signals must be array-like")
        self.assertIsInstance(signals['signal_strength'], (list, pd.Series, np.ndarray, float),
                            "signal_strength must be numeric")
                            
    def create_mock_parameters(self, **kwargs) -> Mock:
        """
        Create mock parameters object for testing.
        
        Args:
            **kwargs: Parameter key-value pairs
            
        Returns:
            Mock object with get method
        """
        mock_params = Mock()
        mock_params.get.side_effect = lambda key, default=None: kwargs.get(key, default)
        return mock_params
        
    def assert_indicator_interface_compliance(self, indicator_instance):
        """
        Validate that indicator properly implements StandardIndicatorInterface.
        
        Args:
            indicator_instance: Instance of Gann indicator to validate
        """
        # Check inheritance
        self.assertIsInstance(indicator_instance, StandardIndicatorInterface,
                            "Indicator must inherit from StandardIndicatorInterface")
        
        # Check required methods exist
        required_methods = ['calculate', 'get_signals', 'get_support_resistance', 
                          'validate_parameters', 'get_debug_info']
        
        for method in required_methods:
            self.assertTrue(hasattr(indicator_instance, method),
                          f"Indicator missing required method: {method}")
            self.assertTrue(callable(getattr(indicator_instance, method)),
                          f"Indicator method {method} is not callable")
        
        # Check class metadata
        self.assertTrue(hasattr(indicator_instance.__class__, 'CATEGORY'),
                       "Indicator missing CATEGORY metadata")
        self.assertTrue(hasattr(indicator_instance.__class__, 'VERSION'),
                       "Indicator missing VERSION metadata")
        self.assertTrue(hasattr(indicator_instance.__class__, 'AUTHOR'),
                       "Indicator missing AUTHOR metadata")


class GannMathValidation:
    """
    Mathematical validation utilities for Gann calculations.
    
    Provides static methods for validating geometric calculations,
    angle computations, and sacred number relationships.
    """
    
    # Gann Sacred Angles (in degrees)
    SACRED_ANGLES = [15, 18.75, 26.25, 45, 63.75, 71.25, 75]
    
    # Golden Ratio and Fibonacci Constants
    GOLDEN_RATIO = 1.618033988749
    FIBONACCI_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618]
    
    @staticmethod
    def calculate_angle_degrees(price_change: float, time_change: float, 
                               scale_factor: float = 1.0) -> float:
        """
        Calculate angle in degrees from price and time changes.
        
        Args:
            price_change: Change in price
            time_change: Change in time periods
            scale_factor: Gann scale factor
            
        Returns:
            Angle in degrees
        """
        if time_change == 0:
            return 90.0 if price_change > 0 else -90.0
            
        slope = (price_change * scale_factor) / time_change
        angle_radians = np.arctan(slope)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees
    
    @staticmethod  
    def validate_sacred_angle(angle: float, tolerance: float = 0.1) -> bool:
        """
        Check if angle is close to a Gann sacred angle.
        
        Args:
            angle: Angle to validate
            tolerance: Tolerance in degrees
            
        Returns:
            True if angle is sacred, False otherwise
        """
        abs_angle = abs(angle)
        
        for sacred in GannMathValidation.SACRED_ANGLES:
            if abs(abs_angle - sacred) <= tolerance:
                return True
                
        return False
    
    @staticmethod
    def calculate_square_of_nine_level(center: float, level: int) -> float:
        """
        Calculate price level for given Square of 9 position.
        
        Args:
            center: Center price
            level: Square level (0 = center, 1 = first ring, etc.)
            
        Returns:
            Price level for the position
        """
        if level == 0:
            return center
            
        # Square of 9 expansion formula
        ring_size = 8 * level
        level_factor = np.sqrt(1 + ring_size)
        
        return center * level_factor
    
    @staticmethod
    def validate_fibonacci_relationship(value1: float, value2: float, 
                                      tolerance: float = 0.01) -> bool:
        """
        Check if two values have a Fibonacci ratio relationship.
        
        Args:
            value1: First value
            value2: Second value  
            tolerance: Tolerance for ratio matching
            
        Returns:
            True if values have Fibonacci relationship
        """
        if value2 == 0:
            return False
            
        ratio = value1 / value2
        
        for fib_ratio in GannMathValidation.FIBONACCI_RATIOS:
            if abs(ratio - fib_ratio) <= tolerance:
                return True
            if abs(ratio - (1/fib_ratio)) <= tolerance:
                return True
                
        return False


# Test data constants for standardized testing
GANN_TEST_CONSTANTS = {
    'BASE_PRICE': 100.0,
    'VOLATILITY_LOW': 0.01,
    'VOLATILITY_MEDIUM': 0.02, 
    'VOLATILITY_HIGH': 0.05,
    'TEST_PERIODS': [100, 1000, 10000],
    'ANGLE_TOLERANCE': 0.1,
    'PRICE_TOLERANCE': 0.01,
    'PERFORMANCE_LIMITS': {
        'small': 0.01,   # 10ms for small datasets
        'medium': 0.1,   # 100ms for medium datasets
        'large': 1.0     # 1s for large datasets  
    }
}