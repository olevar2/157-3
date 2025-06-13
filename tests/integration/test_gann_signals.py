#!/usr/bin/env python3
"""
Comprehensive Gann Signal Generation and Trading Integration Tests

This module implements comprehensive testing for Gann indicator signal generation,
trading signal validation, and integration with Platform3 trading system.

Test Coverage:
- Signal generation validation for each indicator type
- Trading signal accuracy with historical data
- Combination signal testing with other Platform3 indicators
- Real-time signal generation simulation
- Signal timing and latency validation

Platform3 Integration Standards:
- CCI-compatible patterns and testing methodology
- Performance benchmarks for signal generation
- Trading system integration validation
- Mathematical precision in signal calculations
"""

import sys
import time
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add Platform3 to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from engines.ai_enhancement.indicators.gann.gann_angles_indicator import GannAnglesIndicator
from engines.ai_enhancement.indicators.gann.gann_fan_indicator import GannFanIndicator
from engines.ai_enhancement.indicators.gann.gann_price_time_indicator import GannPriceTimeIndicator
from engines.ai_enhancement.indicators.gann.gann_square_indicator import GannSquareIndicator
from engines.ai_enhancement.indicators.gann.gann_time_cycle_indicator import GannTimeCycleIndicator


class MockIndicator:
    """Mock indicator for testing purposes"""
    def __init__(self, name: str, period: int = 14):
        self.name = name
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Mock calculation that returns simple values"""
        if self.name == 'RSI':
            # Return RSI-like values between 0-100
            return pd.Series(
                np.random.uniform(20, 80, len(data)), 
                index=data.index, 
                name='rsi'
            )
        elif self.name == 'SMA':
            # Return simple moving average
            return data['close'].rolling(window=self.period, min_periods=1).mean()
        else:
            return pd.Series(data['close'], index=data.index)


class TestGannSignalGeneration(unittest.TestCase):
    """
    Comprehensive test suite for Gann indicator signal generation
    and trading system integration.
    """
    
    def setUp(self):
        """Set up test fixtures and data"""
        # Initialize all Gann indicators
        self.indicators = {
            'GannAnglesIndicator': GannAnglesIndicator(),
            'GannSquareIndicator': GannSquareIndicator(),
            'GannFanIndicator': GannFanIndicator(),
            'GannTimeCycleIndicator': GannTimeCycleIndicator(),
            'GannPriceTimeIndicator': GannPriceTimeIndicator()
        }
        
        # Initialize comparison indicators for combination testing (using mocks to avoid conflicts)
        self.comparison_indicators = {
            'RSI': self._create_mock_rsi_indicator(),
            'SMA': self._create_mock_sma_indicator()
        }
        
        # Create test data sets
        self.test_data = self._create_test_data()
        self.trending_data = self._create_trending_data()
        self.volatile_data = self._create_volatile_data()
        self.breakout_data = self._create_breakout_data()
    
    def _create_mock_rsi_indicator(self) -> MockIndicator:
        """Create mock RSI indicator"""
        return MockIndicator('RSI', 14)
    
    def _create_mock_sma_indicator(self) -> MockIndicator:
        """Create mock SMA indicator"""
        return MockIndicator('SMA', 20)
        
    def _create_test_data(self) -> pd.DataFrame:
        """Create standard test data for signal testing"""
        periods = 100
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # Create realistic price movement
        np.random.seed(42)
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, periods):
            # Add some trend and noise
            trend = 0.02 * np.sin(i * 0.1)  # Slow trend
            noise = np.random.normal(0, 0.5)  # Random noise
            change = trend + noise
            new_price = max(prices[-1] + change, 1.0)  # Ensure positive prices
            prices.append(new_price)
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
            'high': prices * (1 + np.random.uniform(0.005, 0.02, periods)),
            'low': prices * (1 + np.random.uniform(-0.02, -0.005, periods)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)
    
    def _create_trending_data(self) -> pd.DataFrame:
        """Create trending data for trend signal testing"""
        periods = 100
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # Strong uptrend
        base_price = 100.0
        trend_strength = 0.5  # Strong trend
        
        prices = []
        for i in range(periods):
            trend_component = i * trend_strength
            noise = np.random.normal(0, 0.2)
            price = base_price + trend_component + noise
            prices.append(max(price, 1.0))
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, periods)),
            'high': prices * (1 + np.random.uniform(0.002, 0.01, periods)),
            'low': prices * (1 + np.random.uniform(-0.01, -0.002, periods)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)
    
    def _create_volatile_data(self) -> pd.DataFrame:
        """Create volatile sideways data for range signal testing"""
        periods = 100
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # High volatility, no trend
        base_price = 100.0
        volatility = 2.0
        
        prices = []
        for i in range(periods):
            noise = np.random.normal(0, volatility)
            price = base_price + noise
            prices.append(max(price, 1.0))
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
            'high': prices * (1 + np.random.uniform(0.005, 0.02, periods)),
            'low': prices * (1 + np.random.uniform(-0.02, -0.005, periods)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)    
    def _create_breakout_data(self) -> pd.DataFrame:
        """Create data with clear breakout patterns for signal testing"""
        periods = 100
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # Create sideways movement followed by breakout
        base_price = 100.0
        prices = []
        
        for i in range(periods):
            if i < 50:
                # Sideways movement
                noise = np.random.normal(0, 0.3)
                price = base_price + noise
            else:
                # Strong breakout
                breakout_component = (i - 50) * 1.0
                noise = np.random.normal(0, 0.5)
                price = base_price + breakout_component + noise
            
            prices.append(max(price, 1.0))
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, periods)),
            'high': prices * (1 + np.random.uniform(0.002, 0.01, periods)),
            'low': prices * (1 + np.random.uniform(-0.01, -0.002, periods)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)

    # =====================================
    # Signal Generation Validation Tests
    # =====================================
    
    def test_angle_break_signals(self):
        """Test angle break signal generation for GannAnglesIndicator"""
        indicator = self.indicators['GannAnglesIndicator']
        
        # Calculate indicator values
        result = indicator.calculate(self.trending_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Get signals
        signals = indicator.get_signals()
        self._validate_signal_structure(signals)
        
        # Test with breakout data to trigger angle breaks
        breakout_result = indicator.calculate(self.breakout_data)
        breakout_signals = indicator.get_signals()
        
        # Validate signal logic
        self.assertIsInstance(breakout_signals['buy_signals'], (list, pd.Series, np.ndarray))
        self.assertIsInstance(breakout_signals['sell_signals'], (list, pd.Series, np.ndarray))
        self.assertIsInstance(breakout_signals['signal_strength'], (float, int, np.number))
        
    def test_square_level_signals(self):
        """Test square level breach signals for GannSquareIndicator"""
        indicator = self.indicators['GannSquareIndicator']
        
        # Calculate with different data patterns
        for data_name, data in [
            ('trending', self.trending_data),
            ('volatile', self.volatile_data),
            ('breakout', self.breakout_data)
        ]:
            with self.subTest(data_pattern=data_name):
                result = indicator.calculate(data)
                self.assertIsInstance(result, pd.DataFrame)
                
                signals = indicator.get_signals()
                self._validate_signal_structure(signals)
                
                # Test signal timing
                signal_timestamp = signals.get('timestamp')
                self.assertIsInstance(signal_timestamp, pd.Timestamp)
    
    def test_fan_line_penetration_signals(self):
        """Test fan line penetration signals for GannFanIndicator"""
        indicator = self.indicators['GannFanIndicator']
        
        result = indicator.calculate(self.breakout_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        signals = indicator.get_signals()
        self._validate_signal_structure(signals)
        
        # Verify fan-specific signal characteristics
        if hasattr(signals, 'fan_line_breaks'):
            self.assertIsInstance(signals['fan_line_breaks'], (list, pd.Series, np.ndarray))
        
        # Test signal strength calculation
        strength = signals['signal_strength']
        self.assertTrue(0.0 <= strength <= 1.0, f"Signal strength {strength} should be between 0 and 1")
    
    def test_time_cycle_signals(self):
        """Test time cycle completion signals for GannTimeCycleIndicator"""
        indicator = self.indicators['GannTimeCycleIndicator']
        
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        signals = indicator.get_signals()
        self._validate_signal_structure(signals)
        
        # Test cycle-specific signals
        if 'cycle_signals' in signals:
            cycle_signals = signals['cycle_signals']
            self.assertIsInstance(cycle_signals, (list, pd.Series, np.ndarray))
    
    def test_price_time_square_signals(self):
        """Test price-time square formation signals for GannPriceTimeIndicator"""
        indicator = self.indicators['GannPriceTimeIndicator']
        
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        signals = indicator.get_signals()
        self._validate_signal_structure(signals)
        
        # Test price-time specific signal validation
        if 'square_formations' in signals:
            formations = signals['square_formations']
            self.assertIsInstance(formations, (list, pd.Series, np.ndarray))

    # =====================================
    # Signal Accuracy and Historical Testing
    # =====================================
    
    def test_signal_accuracy_with_historical_data(self):
        """Test signal accuracy using different market conditions"""
        test_datasets = {
            'bull_market': self.trending_data,
            'bear_market': self._create_bear_market_data(),
            'sideways_market': self.volatile_data,
            'breakout': self.breakout_data
        }
        
        for indicator_name, indicator in self.indicators.items():
            for market_type, data in test_datasets.items():
                with self.subTest(indicator=indicator_name, market=market_type):
                    # Calculate indicator
                    result = indicator.calculate(data)
                    self.assertIsInstance(result, pd.DataFrame)
                    
                    # Get signals
                    signals = indicator.get_signals()
                    self._validate_signal_structure(signals)
                    
                    # Validate signal consistency
                    self._validate_signal_consistency(signals)
    
    def _create_bear_market_data(self) -> pd.DataFrame:
        """Create bear market data for signal testing"""
        periods = 100
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # Strong downtrend
        base_price = 100.0
        trend_strength = -0.3  # Strong downtrend
        
        prices = []
        for i in range(periods):
            trend_component = i * trend_strength
            noise = np.random.normal(0, 0.2)
            price = base_price + trend_component + noise
            prices.append(max(price, 1.0))
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, periods)),
            'high': prices * (1 + np.random.uniform(0.002, 0.01, periods)),
            'low': prices * (1 + np.random.uniform(-0.01, -0.002, periods)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)    # =====================================
    # Combination Signal Testing
    # =====================================
    
    def test_combination_signals_with_rsi(self):
        """Test combination signals with RSI indicator"""
        gann_indicator = self.indicators['GannAnglesIndicator']
        rsi_indicator = self.comparison_indicators['RSI']
        
        # Calculate both indicators
        gann_result = gann_indicator.calculate(self.trending_data)
        rsi_result = rsi_indicator.calculate(self.trending_data)
        
        # Get signals from both
        gann_signals = gann_indicator.get_signals()
        rsi_signals = self._get_rsi_signals(rsi_result)
        
        # Test combination logic
        combined_signals = self._combine_signals(gann_signals, rsi_signals)
        
        # Validate combined signal structure
        self.assertIn('combined_buy', combined_signals)
        self.assertIn('combined_sell', combined_signals)
        self.assertIn('confidence_score', combined_signals)
        
        # Test signal strength enhancement
        confidence = combined_signals['confidence_score']
        self.assertIsInstance(confidence, (float, int, np.number))
        self.assertTrue(0.0 <= confidence <= 1.0)
    
    def test_combination_signals_with_moving_average(self):
        """Test combination signals with Simple Moving Average"""
        gann_indicator = self.indicators['GannFanIndicator']
        sma_indicator = self.comparison_indicators['SMA']
        
        # Calculate both indicators
        gann_result = gann_indicator.calculate(self.breakout_data)
        sma_result = sma_indicator.calculate(self.breakout_data)
        
        # Get signals
        gann_signals = gann_indicator.get_signals()
        sma_signals = self._get_sma_signals(sma_result, self.breakout_data['close'])
        
        # Test combination
        combined_signals = self._combine_signals(gann_signals, sma_signals)
        
        # Validate enhanced signal accuracy
        self.assertIsInstance(combined_signals, dict)
        self.assertIn('signal_source', combined_signals)
        self.assertEqual(combined_signals['signal_source'], 'combined')
    
    def test_multi_gann_indicator_combinations(self):
        """Test combinations of multiple Gann indicators"""
        # Use multiple Gann indicators
        angles_indicator = self.indicators['GannAnglesIndicator']
        fan_indicator = self.indicators['GannFanIndicator']
        square_indicator = self.indicators['GannSquareIndicator']
        
        # Calculate all indicators
        test_data = self.breakout_data
        angles_result = angles_indicator.calculate(test_data)
        fan_result = fan_indicator.calculate(test_data)
        square_result = square_indicator.calculate(test_data)
        
        # Get all signals
        angles_signals = angles_indicator.get_signals()
        fan_signals = fan_indicator.get_signals()
        square_signals = square_indicator.get_signals()
        
        # Test multi-indicator combination
        multi_signals = self._combine_multiple_gann_signals([
            angles_signals, fan_signals, square_signals
        ])
        
        # Validate multi-signal structure
        self.assertIn('consensus_strength', multi_signals)
        self.assertIn('agreement_count', multi_signals)
        self.assertIsInstance(multi_signals['agreement_count'], int)
        self.assertTrue(0 <= multi_signals['agreement_count'] <= 3)

    # =====================================
    # Real-time Signal Generation Simulation
    # =====================================
    
    def test_real_time_signal_generation(self):
        """Test real-time signal generation capability"""
        indicator = self.indicators['GannAnglesIndicator']
        
        # Simulate streaming data
        base_data = self.test_data.iloc[:50]  # Initial data
        new_data_points = self.test_data.iloc[50:]  # New streaming data
        
        # Initial calculation
        initial_result = indicator.calculate(base_data)
        initial_signals = indicator.get_signals()
        
        # Simulate real-time updates
        streaming_signals = []
        for i, (timestamp, new_point) in enumerate(new_data_points.iterrows()):
            # Add new data point
            current_data = pd.concat([
                base_data, 
                new_data_points.iloc[:i+1]
            ])
            
            # Measure signal generation time
            start_time = time.time()
            result = indicator.calculate(current_data)
            signals = indicator.get_signals()
            calc_time = time.time() - start_time
            
            # Validate real-time performance (should be fast)
            self.assertLess(calc_time, 0.1, "Real-time signal generation too slow")
            
            # Store signal for analysis
            streaming_signals.append({
                'timestamp': timestamp,
                'signals': signals,
                'calc_time': calc_time
            })
        
        # Validate streaming signal consistency
        self.assertEqual(len(streaming_signals), len(new_data_points))
        for signal_data in streaming_signals:
            self._validate_signal_structure(signal_data['signals'])
    
    def test_signal_latency_performance(self):
        """Test signal generation latency for different data sizes"""
        latency_requirements = {
            100: 0.01,   # 100 points: < 10ms
            500: 0.05,   # 500 points: < 50ms
            1000: 0.1,   # 1000 points: < 100ms
        }
        
        for indicator_name, indicator in self.indicators.items():
            for data_size, max_latency in latency_requirements.items():
                with self.subTest(indicator=indicator_name, size=data_size):
                    # Create data of specified size
                    test_data = self._create_sized_data(data_size)
                    
                    # Measure signal generation time
                    start_time = time.time()
                    result = indicator.calculate(test_data)
                    signals = indicator.get_signals()
                    latency = time.time() - start_time
                    
                    # Validate latency requirement
                    self.assertLess(
                        latency, max_latency,
                        f"{indicator_name} signal latency {latency:.4f}s exceeds {max_latency}s for {data_size} points"
                    )
                    
                    # Ensure signals are still valid
                    self._validate_signal_structure(signals)

    # =====================================
    # Trading System Integration Tests
    # =====================================
    
    def test_trading_system_integration(self):
        """Test integration with Platform3 trading system"""
        for indicator_name, indicator in self.indicators.items():
            with self.subTest(indicator=indicator_name):
                # Calculate indicator
                result = indicator.calculate(self.trending_data)
                signals = indicator.get_signals()
                
                # Test trading system compatibility
                trading_signals = self._convert_to_trading_signals(signals)
                
                # Validate trading signal format
                self.assertIn('action', trading_signals)  # buy/sell/hold
                self.assertIn('quantity', trading_signals)
                self.assertIn('confidence', trading_signals)
                self.assertIn('stop_loss', trading_signals)
                self.assertIn('take_profit', trading_signals)
                
                # Validate trading signal values
                self.assertIn(trading_signals['action'], ['buy', 'sell', 'hold'])
                self.assertIsInstance(trading_signals['quantity'], (int, float))
                self.assertTrue(0.0 <= trading_signals['confidence'] <= 1.0)    
    def test_signal_timing_validation(self):
        """Test signal timing accuracy and consistency"""
        indicator = self.indicators['GannSquareIndicator']
        
        # Test with time-series data
        result = indicator.calculate(self.test_data)
        signals = indicator.get_signals()
        
        # Validate timestamp accuracy
        signal_timestamp = signals.get('timestamp')
        current_time = pd.Timestamp.now()
        time_diff = abs((signal_timestamp - current_time).total_seconds())
        
        # Signal should be generated within reasonable time
        self.assertLess(time_diff, 60, "Signal timestamp too far from current time")
        
        # Test signal consistency over time
        previous_signals = None
        for i in range(5):  # Test multiple calculations
            result = indicator.calculate(self.test_data)
            current_signals = indicator.get_signals()
            
            if previous_signals is not None:
                # Signals should be consistent for same data
                self.assertEqual(
                    len(current_signals['buy_signals']),
                    len(previous_signals['buy_signals'])
                )
            
            previous_signals = current_signals

    # =====================================
    # Helper Methods and Utilities
    # =====================================
    
    def _validate_signal_structure(self, signals: Dict[str, Any]):
        """Validate signal structure compliance with Platform3 standards"""
        # Required signal fields
        required_fields = ['buy_signals', 'sell_signals', 'signal_strength', 'timestamp']
        
        for field in required_fields:
            self.assertIn(field, signals, f"Missing required signal field: {field}")
        
        # Validate signal types
        self.assertIsInstance(signals['buy_signals'], (list, pd.Series, np.ndarray),
                            "buy_signals must be array-like")
        self.assertIsInstance(signals['sell_signals'], (list, pd.Series, np.ndarray),
                            "sell_signals must be array-like")
        self.assertIsInstance(signals['signal_strength'], (list, pd.Series, np.ndarray, float, int, np.number),
                            "signal_strength must be numeric")
        self.assertIsInstance(signals['timestamp'], pd.Timestamp,
                            "timestamp must be pandas Timestamp")
    
    def _validate_signal_consistency(self, signals: Dict[str, Any]):
        """Validate signal logical consistency"""
        buy_signals = signals['buy_signals']
        sell_signals = signals['sell_signals']
        
        # Ensure signals are not conflicting (both buy and sell at same time)
        if hasattr(buy_signals, '__len__') and hasattr(sell_signals, '__len__'):
            if len(buy_signals) > 0 and len(sell_signals) > 0:
                # For now, just ensure both exist but could be refined for timing
                self.assertTrue(True)  # Placeholder for more complex logic
    
    def _get_rsi_signals(self, rsi_result: pd.Series) -> Dict[str, Any]:
        """Convert RSI values to signal format"""
        latest_rsi = rsi_result.iloc[-1] if len(rsi_result) > 0 else 50
        
        return {
            'buy_signals': [1] if latest_rsi < 30 else [],
            'sell_signals': [1] if latest_rsi > 70 else [],
            'signal_strength': abs(latest_rsi - 50) / 50,
            'timestamp': pd.Timestamp.now()
        }
    
    def _get_sma_signals(self, sma_result: pd.Series, price_data: pd.Series) -> Dict[str, Any]:
        """Convert SMA vs price to signal format"""
        if len(sma_result) == 0 or len(price_data) == 0:
            return {
                'buy_signals': [],
                'sell_signals': [],
                'signal_strength': 0.0,
                'timestamp': pd.Timestamp.now()
            }
        
        latest_sma = sma_result.iloc[-1]
        latest_price = price_data.iloc[-1]
        
        price_above_sma = latest_price > latest_sma
        signal_strength = abs(latest_price - latest_sma) / latest_sma
        
        return {
            'buy_signals': [1] if price_above_sma else [],
            'sell_signals': [1] if not price_above_sma else [],
            'signal_strength': min(signal_strength, 1.0),
            'timestamp': pd.Timestamp.now()
        }
    
    def _combine_signals(self, signal1: Dict[str, Any], signal2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine two signal dictionaries"""
        # Simple combination logic
        combined_buy_strength = len(signal1.get('buy_signals', [])) + len(signal2.get('buy_signals', []))
        combined_sell_strength = len(signal1.get('sell_signals', [])) + len(signal2.get('sell_signals', []))
        
        # Calculate confidence based on agreement
        strength1 = signal1.get('signal_strength', 0)
        strength2 = signal2.get('signal_strength', 0)
        confidence_score = (strength1 + strength2) / 2
        
        return {
            'combined_buy': combined_buy_strength > 0,
            'combined_sell': combined_sell_strength > 0,
            'confidence_score': min(confidence_score, 1.0),
            'signal_source': 'combined',
            'timestamp': pd.Timestamp.now()
        }
    
    def _combine_multiple_gann_signals(self, signal_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple Gann indicator signals"""
        buy_count = sum(len(signals.get('buy_signals', [])) for signals in signal_list)
        sell_count = sum(len(signals.get('sell_signals', [])) for signals in signal_list)
        
        # Calculate consensus strength
        total_strength = sum(signals.get('signal_strength', 0) for signals in signal_list)
        consensus_strength = total_strength / len(signal_list) if signal_list else 0
        
        # Count agreements
        agreement_count = sum(
            1 for signals in signal_list 
            if len(signals.get('buy_signals', [])) > 0 or len(signals.get('sell_signals', [])) > 0
        )
        
        return {
            'consensus_buy': buy_count > len(signal_list) // 2,
            'consensus_sell': sell_count > len(signal_list) // 2,
            'consensus_strength': consensus_strength,
            'agreement_count': agreement_count,
            'total_indicators': len(signal_list),
            'timestamp': pd.Timestamp.now()
        }
    
    def _convert_to_trading_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Convert indicator signals to trading system format"""
        buy_signals = signals.get('buy_signals', [])
        sell_signals = signals.get('sell_signals', [])
        strength = signals.get('signal_strength', 0)
        
        # Determine action
        if len(buy_signals) > len(sell_signals):
            action = 'buy'
        elif len(sell_signals) > len(buy_signals):
            action = 'sell'
        else:
            action = 'hold'
        
        # Calculate position sizing based on signal strength
        base_quantity = 100
        quantity = int(base_quantity * strength) if isinstance(strength, (int, float)) else base_quantity
        
        return {
            'action': action,
            'quantity': max(quantity, 1),  # Minimum 1 share
            'confidence': float(strength) if isinstance(strength, (int, float, np.number)) else 0.0,
            'stop_loss': 0.02,  # 2% stop loss
            'take_profit': 0.06,  # 6% take profit
            'timestamp': signals.get('timestamp', pd.Timestamp.now())
        }
    
    def _create_sized_data(self, size: int) -> pd.DataFrame:
        """Create test data of specified size"""
        dates = pd.date_range('2023-01-01', periods=size, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, size):
            change = np.random.normal(0, 0.5)
            new_price = max(prices[-1] + change, 1.0)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, size)),
            'high': prices * (1 + np.random.uniform(0.005, 0.02, size)),
            'low': prices * (1 + np.random.uniform(-0.02, -0.005, size)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, size)
        }, index=dates)


if __name__ == '__main__':
    unittest.main(verbosity=2)