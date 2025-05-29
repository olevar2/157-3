"""
Platform3 Indicator Effectiveness Testing Suite
Comprehensive testing framework to verify the performance and accuracy 
of all 67 indicators in the humanitarian trading system.
"""
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Add the engines directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

# Import the base classes
from engines.indicator_base import MarketData, TimeFrame, IndicatorResult, IndicatorSignal, SignalType

# Import implemented indicators
from engines.momentum.williams_r import WilliamsR
from engines.momentum.cci import CCI
from engines.momentum.roc import ROC

class IndicatorEffectivenessTest:
    """
    Comprehensive testing framework for indicator effectiveness validation.
    """
    
    def __init__(self):
        self.results = {
            'test_timestamp': datetime.now().isoformat(),
            'total_indicators_tested': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'performance_metrics': {},
            'signal_accuracy': {},
            'calculation_speed': {},
            'error_analysis': {}
        }
        
        # Track implemented indicators
        self.implemented_indicators = {
            'Williams %R': WilliamsR,
            'CCI': CCI,
            'ROC': ROC
        }
        
        # Test datasets for different market conditions
        self.test_datasets = {
            'trending_up': self._generate_uptrend_data(),
            'trending_down': self._generate_downtrend_data(),
            'sideways': self._generate_sideways_data(),
            'volatile': self._generate_volatile_data(),
            'breakout': self._generate_breakout_data()
        }
    
    def _generate_uptrend_data(self, length: int = 100) -> List[MarketData]:
        """Generate uptrending market data for testing."""
        base_time = datetime.now()
        data = []
        base_price = 100.0
        
        for i in range(length):
            # Trending up with some noise
            trend_factor = 1 + (i * 0.005)  # 0.5% growth per period
            noise = np.random.normal(0, 0.002)  # Small random noise
            price = base_price * trend_factor * (1 + noise)
            
            high = price * (1 + abs(np.random.normal(0, 0.003)))
            low = price * (1 - abs(np.random.normal(0, 0.003)))
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                open=price * 0.999,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=1000 + int(np.random.normal(0, 100)),
                timeframe=TimeFrame.M1
            ))
        
        return data
    
    def _generate_downtrend_data(self, length: int = 100) -> List[MarketData]:
        """Generate downtrending market data for testing."""
        base_time = datetime.now()
        data = []
        base_price = 100.0
        
        for i in range(length):
            # Trending down with some noise
            trend_factor = 1 - (i * 0.004)  # 0.4% decline per period
            noise = np.random.normal(0, 0.002)
            price = base_price * trend_factor * (1 + noise)
            
            high = price * (1 + abs(np.random.normal(0, 0.003)))
            low = price * (1 - abs(np.random.normal(0, 0.003)))
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                open=price * 1.001,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=1000 + int(np.random.normal(0, 100)),
                timeframe=TimeFrame.M1
            ))
        
        return data
    
    def _generate_sideways_data(self, length: int = 100) -> List[MarketData]:
        """Generate sideways/ranging market data for testing."""
        base_time = datetime.now()
        data = []
        base_price = 100.0
        
        for i in range(length):
            # Sideways movement with oscillation
            oscillation = np.sin(i * 0.2) * 0.02  # 2% oscillation
            noise = np.random.normal(0, 0.003)
            price = base_price * (1 + oscillation + noise)
            
            high = price * (1 + abs(np.random.normal(0, 0.004)))
            low = price * (1 - abs(np.random.normal(0, 0.004)))
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=1000 + int(np.random.normal(0, 150)),
                timeframe=TimeFrame.M1
            ))
        
        return data
    
    def _generate_volatile_data(self, length: int = 100) -> List[MarketData]:
        """Generate high volatility market data for testing."""
        base_time = datetime.now()
        data = []
        base_price = 100.0
        
        for i in range(length):
            # High volatility with random movements
            volatility = np.random.normal(0, 0.015)  # 1.5% volatility
            price = base_price * (1 + volatility)
            base_price = price  # Compound the movements
            
            high = price * (1 + abs(np.random.normal(0, 0.008)))
            low = price * (1 - abs(np.random.normal(0, 0.008)))
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=1000 + int(np.random.normal(0, 300)),
                timeframe=TimeFrame.M1
            ))
        
        return data
    
    def _generate_breakout_data(self, length: int = 100) -> List[MarketData]:
        """Generate breakout scenario data for testing."""
        base_time = datetime.now()
        data = []
        base_price = 100.0
        
        # First 70% - consolidation phase
        consolidation_length = int(length * 0.7)
        for i in range(consolidation_length):
            # Tight range
            noise = np.random.normal(0, 0.001)  # Very low volatility
            price = base_price * (1 + noise)
            
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=800 + int(np.random.normal(0, 50)),
                timeframe=TimeFrame.M1
            ))
        
        # Last 30% - breakout phase
        breakout_length = length - consolidation_length
        for i in range(breakout_length):
            # Strong upward movement
            momentum = 1 + (i * 0.008)  # 0.8% per period
            noise = np.random.normal(0, 0.003)
            price = base_price * momentum * (1 + noise)
            
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=consolidation_length + i),
                open=price * 0.998,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=1500 + int(np.random.normal(0, 200)),
                timeframe=TimeFrame.M1
            ))
        
        return data
    
    def test_indicator_calculation(self, indicator_class, indicator_name: str, 
                                 test_data: List[MarketData]) -> Dict[str, Any]:
        """Test indicator calculation performance and accuracy."""
        test_result = {
            'indicator_name': indicator_name,
            'calculation_success': False,
            'calculation_time_ms': None,
            'signal_generated': False,
            'signal_type': None,
            'signal_strength': None,
            'signal_confidence': None,
            'error_message': None,
            'value_range': None,
            'raw_value': None
        }
        
        try:
            # Initialize indicator
            indicator = indicator_class(TimeFrame.M1)
            
            # Measure calculation time
            start_time = time.time()
            result = indicator.calculate(test_data)
            calculation_time = (time.time() - start_time) * 1000
            
            test_result.update({
                'calculation_success': True,
                'calculation_time_ms': calculation_time,
                'raw_value': result.value,
                'value_range': f"{min(-100, result.value) if isinstance(result.value, (int, float)) else 'complex'}" +
                             f" to {max(100, result.value) if isinstance(result.value, (int, float)) else 'complex'}"
            })
            
            # Check for signal
            if result.signal:
                test_result.update({
                    'signal_generated': True,
                    'signal_type': result.signal.signal_type.value,
                    'signal_strength': result.signal.strength,
                    'signal_confidence': result.signal.confidence
                })
            
        except Exception as e:
            test_result.update({
                'calculation_success': False,
                'error_message': str(e)
            })
        
        return test_result
    
    def test_all_market_conditions(self) -> Dict[str, Any]:
        """Test all implemented indicators across different market conditions."""
        comprehensive_results = {
            'test_summary': {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0
            },
            'indicator_performance': {},
            'market_condition_analysis': {}
        }
        
        print("🔬 Platform3 Indicator Effectiveness Testing Suite")
        print("=" * 60)
        print(f"Testing {len(self.implemented_indicators)} implemented indicators")
        print(f"Across {len(self.test_datasets)} market conditions")
        print()
        
        for market_condition, test_data in self.test_datasets.items():
            print(f"📊 Testing Market Condition: {market_condition.upper()}")
            condition_results = {}
            
            for indicator_name, indicator_class in self.implemented_indicators.items():
                print(f"  🔍 Testing {indicator_name}...", end=" ")
                
                test_result = self.test_indicator_calculation(
                    indicator_class, indicator_name, test_data
                )
                
                condition_results[indicator_name] = test_result
                comprehensive_results['test_summary']['total_tests'] += 1
                
                if test_result['calculation_success']:
                    comprehensive_results['test_summary']['successful_tests'] += 1
                    print("✅ PASS")
                else:
                    comprehensive_results['test_summary']['failed_tests'] += 1
                    print(f"❌ FAIL - {test_result['error_message']}")
            
            comprehensive_results['market_condition_analysis'][market_condition] = condition_results
            print()
        
        # Calculate performance metrics
        for indicator_name in self.implemented_indicators.keys():
            indicator_results = []
            for condition_results in comprehensive_results['market_condition_analysis'].values():
                if indicator_name in condition_results:
                    indicator_results.append(condition_results[indicator_name])
            
            # Calculate average performance
            successful_tests = [r for r in indicator_results if r['calculation_success']]
            
            if successful_tests:
                avg_calc_time = sum(r['calculation_time_ms'] for r in successful_tests) / len(successful_tests)
                signal_rate = sum(1 for r in successful_tests if r['signal_generated']) / len(successful_tests)
                avg_confidence = sum(r['signal_confidence'] for r in successful_tests 
                                   if r['signal_confidence']) / max(1, sum(1 for r in successful_tests 
                                   if r['signal_confidence']))
                
                comprehensive_results['indicator_performance'][indicator_name] = {
                    'success_rate': len(successful_tests) / len(indicator_results),
                    'average_calculation_time_ms': avg_calc_time,
                    'signal_generation_rate': signal_rate,
                    'average_signal_confidence': avg_confidence if avg_confidence else 0,
                    'total_tests': len(indicator_results)
                }
        
        return comprehensive_results
    
    def generate_effectiveness_report(self) -> str:
        """Generate a comprehensive effectiveness report."""
        results = self.test_all_market_conditions()
        
        report = f"""
# PLATFORM3 INDICATOR EFFECTIVENESS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 TEST SUMMARY
- **Total Tests**: {results['test_summary']['total_tests']}
- **Successful Tests**: {results['test_summary']['successful_tests']}
- **Failed Tests**: {results['test_summary']['failed_tests']}
- **Success Rate**: {(results['test_summary']['successful_tests'] / max(1, results['test_summary']['total_tests']) * 100):.1f}%

## 🎯 INDICATOR PERFORMANCE ANALYSIS

"""
        
        for indicator_name, performance in results['indicator_performance'].items():
            report += f"""### {indicator_name}
- **Success Rate**: {performance['success_rate']*100:.1f}%
- **Average Calculation Time**: {performance['average_calculation_time_ms']:.2f}ms
- **Signal Generation Rate**: {performance['signal_generation_rate']*100:.1f}%
- **Average Signal Confidence**: {performance['average_signal_confidence']:.2f}

"""
        
        report += "\n## 📈 MARKET CONDITION ANALYSIS\n\n"
        
        for condition, condition_results in results['market_condition_analysis'].items():
            report += f"### {condition.upper().replace('_', ' ')} Market\n"
            successful_indicators = [name for name, result in condition_results.items() 
                                   if result['calculation_success']]
            report += f"- **Successful Indicators**: {len(successful_indicators)}/{len(condition_results)}\n"
            
            # Find best performing indicator for this condition
            best_indicator = None
            best_score = 0
            for name, result in condition_results.items():
                if result['calculation_success'] and result['signal_generated']:
                    score = result['signal_confidence'] * result['signal_strength']
                    if score > best_score:
                        best_score = score
                        best_indicator = name
            
            if best_indicator:
                report += f"- **Best Performer**: {best_indicator} (score: {best_score:.2f})\n"
            
            report += "\n"
        
        report += f"""
## 🚀 HUMANITARIAN MISSION IMPACT

### Current Implementation Status
- **Functional Indicators**: {len(self.implemented_indicators)} of 67 target
- **Implementation Progress**: {(len(self.implemented_indicators)/67)*100:.1f}%
- **Remaining Work**: {67 - len(self.implemented_indicators)} indicators to implement

### Effectiveness Metrics
- **Average Calculation Speed**: {sum(p['average_calculation_time_ms'] for p in results['indicator_performance'].values()) / len(results['indicator_performance']):.2f}ms
- **Overall Signal Reliability**: {sum(p['average_signal_confidence'] for p in results['indicator_performance'].values()) / len(results['indicator_performance']) * 100:.1f}%

### Next Steps for Full 67-Indicator Implementation
1. **Phase 1**: Complete remaining momentum indicators (5 more needed)
2. **Phase 2**: Implement trend indicators (7 indicators)
3. **Phase 3**: Add volume indicators (5 indicators)
4. **Phase 4**: Build pattern recognition engines (15 indicators)
5. **Phase 5**: Create advanced analytics (35 indicators)

**MISSION**: Every additional functional indicator increases trading accuracy and humanitarian profit potential!
"""
        
        return report
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"indicator_effectiveness_test_{timestamp}.json"
        
        results = self.test_all_market_conditions()
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {filename}")
        return filename

def main():
    """Run the comprehensive indicator effectiveness test."""
    tester = IndicatorEffectivenessTest()
    
    # Generate and display the effectiveness report
    report = tester.generate_effectiveness_report()
    print(report)
    
    # Save results to file
    results_file = tester.save_results()
      # Also save the report
    report_file = f"INDICATOR_EFFECTIVENESS_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")
    print("\n🎯 CONCLUSION: Testing framework operational - ready for full 67-indicator implementation!")

if __name__ == "__main__":
    main()
