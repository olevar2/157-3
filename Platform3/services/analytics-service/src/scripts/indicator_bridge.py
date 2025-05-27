#!/usr/bin/env python3
"""
Python Bridge for ComprehensiveIndicatorAdapter_67
Handles requests from the TypeScript analytics service
"""

import sys
import json
import numpy as np
from datetime import datetime
import argparse
import logging
import warnings

# Suppress all warnings including pandas FutureWarnings
warnings.filterwarnings('ignore')

# Suppress logging to stderr to avoid interfering with JSON output
logging.basicConfig(level=logging.CRITICAL)  # Only show critical errors
logging.getLogger().setLevel(logging.CRITICAL)

# Disable all info/warning loggers that might interfere
for logger_name in ['momentum', 'trend', 'volatility', 'volume', 'cycle', 'advanced', 'ScalpingLSTM', 'TickClassifier', 'GannAnglesCalculator', 'GannFanAnalysis', 'GannPatternDetector', 'GannSquareOfNine', 'GannTimePrice', 'FractalGeometryIndicator', 'FibonacciExtension', 'FibonacciRetracement', 'ConflictResolver', 'SignalAggregator']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Add the indicators path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
platform3_dir = os.path.join(current_dir, '..', '..', '..', '..')
sys.path.append(platform3_dir)

try:
    from ComprehensiveIndicatorAdapter_67 import (
        ComprehensiveIndicatorAdapter_67,
        MarketData,
        IndicatorCategory
    )
except ImportError as e:
    print(f"ERROR: Failed to import ComprehensiveIndicatorAdapter_67: {e}", file=sys.stderr)
    sys.exit(1)

class PlatformIndicatorBridge:
    """Bridge between TypeScript service and Python indicator adapter"""

    def __init__(self):
        self.adapter = ComprehensiveIndicatorAdapter_67()

    def create_market_data_from_input(self, market_data):
        """Convert input market data to MarketData object"""
        try:
            return MarketData(
                open=np.array(market_data['open']),
                high=np.array(market_data['high']),
                low=np.array(market_data['low']),
                close=np.array(market_data['close']),
                volume=np.array(market_data['volume']),
                timestamp=np.array(market_data.get('timestamps', market_data.get('timestamp', [])))
            )
        except Exception as e:
            raise ValueError(f"Invalid market data format: {e}")

    def serialize_result(self, result):
        """Convert IndicatorResult to JSON-serializable dict"""
        return {
            'indicator_name': result.indicator_name,
            'category': result.category.value if hasattr(result.category, 'value') else str(result.category),
            'values': self._serialize_values(result.values),
            'signals': self._serialize_values(result.signals),
            'metadata': result.metadata,
            'calculation_time': result.calculation_time,
            'success': result.success,
            'error_message': result.error_message
        }

    def _serialize_values(self, values):
        """Convert numpy arrays and complex objects to JSON-serializable format"""
        if values is None:
            return None
        elif isinstance(values, np.ndarray):
            return values.tolist()
        elif isinstance(values, dict):
            return {k: self._serialize_values(v) for k, v in values.items()}
        elif isinstance(values, (list, tuple)):
            return [self._serialize_values(v) for v in values]
        elif isinstance(values, (np.integer, np.floating)):
            return float(values)
        elif hasattr(values, '__dict__'):
            # Handle custom objects by converting to dict
            try:
                return {k: self._serialize_values(v) for k, v in values.__dict__.items()}
            except:
                return str(values)
        elif hasattr(values, '_asdict'):
            # Handle namedtuples
            try:
                return self._serialize_values(values._asdict())
            except:
                return str(values)
        else:
            try:
                # Try to convert to basic types
                if isinstance(values, (int, float, str, bool)):
                    return values
                else:
                    return str(values)
            except:
                return str(values)

    def calculate_single_indicator(self, indicator_name, market_data, symbol, timeframe):
        """Calculate a single indicator"""
        try:
            market_data_obj = self.create_market_data_from_input(market_data)
            result = self.adapter.calculate_indicator(indicator_name, market_data_obj)
            return self.serialize_result(result)
        except Exception as e:
            return {
                'indicator_name': indicator_name,
                'category': 'unknown',
                'values': None,
                'signals': None,
                'metadata': {'symbol': symbol, 'timeframe': timeframe},
                'calculation_time': 0,
                'success': False,
                'error_message': str(e)
            }

    def calculate_all_indicators(self, market_data, symbol, timeframe):
        """Calculate all 67 indicators"""
        try:
            market_data_obj = self.create_market_data_from_input(market_data)
            all_indicator_names = self.adapter.get_all_indicator_names()

            results = {}
            categories = {}
            successful = 0
            failed = 0
            total_time = 0

            # Calculate all indicators
            for indicator_name in all_indicator_names:
                result = self.adapter.calculate_indicator(indicator_name, market_data_obj)
                serialized_result = self.serialize_result(result)
                results[indicator_name] = serialized_result

                # Update statistics
                if result.success:
                    successful += 1
                else:
                    failed += 1
                total_time += result.calculation_time

                # Group by category
                category = serialized_result['category']
                if category not in categories:
                    categories[category] = {
                        'indicators': [],
                        'success_count': 0,
                        'total_count': 0,
                        'success_rate': 0
                    }

                categories[category]['indicators'].append(serialized_result)
                categories[category]['total_count'] += 1
                if result.success:
                    categories[category]['success_count'] += 1

            # Calculate category success rates
            for category in categories:
                cat_data = categories[category]
                cat_data['success_rate'] = (cat_data['success_count'] / cat_data['total_count']) * 100

            # Create summary by category
            summary = {
                'momentum': [r for r in results.values() if r['category'] == 'momentum'],
                'trend': [r for r in results.values() if r['category'] == 'trend'],
                'volatility': [r for r in results.values() if r['category'] == 'volatility'],
                'volume': [r for r in results.values() if r['category'] == 'volume'],
                'cycle': [r for r in results.values() if r['category'] == 'cycle'],
                'advanced': [r for r in results.values() if r['category'] == 'advanced'],
                'gann': [r for r in results.values() if r['category'] == 'gann'],
                'scalping': [r for r in results.values() if r['category'] == 'scalping'],
                'daytrading': [r for r in results.values() if r['category'] == 'daytrading'],
                'swingtrading': [r for r in results.values() if r['category'] == 'swingtrading'],
                'signals': [r for r in results.values() if r['category'] == 'signals']
            }

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'total_indicators': len(all_indicator_names),
                'successful_indicators': successful,
                'failed_indicators': failed,
                'success_rate': (successful / len(all_indicator_names)) * 100,
                'total_calculation_time': total_time,
                'categories': categories,
                'summary': summary,
                'results': results
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'total_indicators': 0,
                'successful_indicators': 0,
                'failed_indicators': 0,
                'success_rate': 0,
                'total_calculation_time': 0,
                'categories': {},
                'summary': {},
                'error': str(e)
            }

    def calculate_batch_indicators(self, indicator_names, market_data, symbol, timeframe):
        """Calculate a batch of indicators"""
        try:
            market_data_obj = self.create_market_data_from_input(market_data)
            results = {}

            for indicator_name in indicator_names:
                result = self.adapter.calculate_indicator(indicator_name, market_data_obj)
                results[indicator_name] = self.serialize_result(result)

            return results

        except Exception as e:
            return {
                'error': str(e),
                'results': {}
            }

    def list_available_indicators(self):
        """Get list of all available indicators by category"""
        try:
            all_indicators = self.adapter.get_all_indicator_names()
            categories = {}

            for indicator_name in all_indicators:
                # Get category for each indicator
                _, _, category = self.adapter.all_indicators[indicator_name]
                category_name = category.value if hasattr(category, 'value') else str(category)

                if category_name not in categories:
                    categories[category_name] = []
                categories[category_name].append(indicator_name)

            return {
                'total_indicators': len(all_indicators),
                'categories': categories,
                'all_indicators': all_indicators
            }

        except Exception as e:
            return {
                'error': str(e),
                'total_indicators': 0,
                'categories': {},
                'all_indicators': []
            }

    def test_adapter(self):
        """Test the adapter functionality"""
        try:
            # Simple test - just check if adapter is initialized
            indicator_count = len(self.adapter.get_all_indicator_names())
            if indicator_count > 0:
                print("TEST_SUCCESS")
                return True
            else:
                print(f"TEST_FAILED: No indicators available (count: {indicator_count})")
                return False

        except Exception as e:
            print(f"TEST_ERROR: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Platform3 Indicator Bridge')
    parser.add_argument('--test', action='store_true', help='Run adapter test')
    args = parser.parse_args()

    bridge = PlatformIndicatorBridge()

    if args.test:
        bridge.test_adapter()
        return

    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())
        action = input_data.get('action')

        if action == 'calculate_single':
            result = bridge.calculate_single_indicator(
                input_data['indicator_name'],
                input_data['market_data'],
                input_data['symbol'],
                input_data['timeframe']
            )

        elif action == 'calculate_all':
            result = bridge.calculate_all_indicators(
                input_data['market_data'],
                input_data['symbol'],
                input_data['timeframe']
            )

        elif action == 'calculate_batch':
            result = bridge.calculate_batch_indicators(
                input_data['indicator_names'],
                input_data['market_data'],
                input_data['symbol'],
                input_data['timeframe']
            )

        elif action == 'list_indicators':
            result = bridge.list_available_indicators()

        else:
            result = {'error': f'Unknown action: {action}'}

        # Output result as JSON
        print(json.dumps(result, indent=2))

    except Exception as e:
        error_result = {
            'error': str(e),
            'success': False
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()
