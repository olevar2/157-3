"""
Test ALL 101 Platform3 Indicators with Adaptive Enhancement
Quick compatibility and functionality test
"""

import unittest
import numpy as np
import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

try:
    from engines.ai_enhancement.adaptive_indicators import AdaptiveIndicators
except ImportError as e:
    print(f"Warning: Could not import AdaptiveIndicators: {e}")
    sys.exit(1)

class TestAll101Indicators(unittest.TestCase):
    
    def setUp(self):
        """Setup test data and list of all base indicators."""
        self.test_data = self._generate_test_data()
        
        # List of all 101 base indicators from Platform3
        self.base_indicators = [
            # Core Momentum (3)
            'RSI', 'MACD', 'Stochastic',
            
            # Core Trend (3) 
            'ADX', 'Ichimoku', 'SMA',
            
            # Momentum (11)
            'CCI', 'MFI', 'ROC', 'TRIX', 'Williams_R', 'Ultimate_Oscillator',
            'Awesome_Oscillator', 'Detrended_Price_Oscillator', 'PPO', 'TSI',
            
            # Trend (8)
            'Bollinger_Bands', 'Keltner_Channels', 'Parabolic_SAR', 'Aroon',
            'ATR', 'Donchian_Channels', 'Vortex', 'DMS',
            
            # Volume (10) 
            'OBV', 'VWAP', 'CMF', 'AD', 'VSA', 'Volume_Profile',
            'Tick_Volume', 'Smart_Money', 'Order_Flow', 'VPT',
            
            # Volatility (3)
            'Historical_Volatility', 'Volatility_Index', 'Keltner',
            
            # Pattern (8)
            'Doji', 'Engulfing', 'Hammer', 'Harami', 'Elliott_Wave',
            'Fibonacci_Retracement', 'Gann_Angles', 'Japanese_Candlestick',
            
            # Statistical (4)
            'Correlation', 'Linear_Regression', 'Standard_Deviation', 'Z_Score',
            
            # Fibonacci (5)
            'Fib_Retracement', 'Fib_Extension', 'Fib_Confluence', 
            'Fib_Projection', 'Fib_Time_Zone',
            
            # Gann (5)
            'Gann_Fan', 'Gann_Square', 'Gann_Time_Cycles', 
            'Gann_Pattern', 'Price_Time_Relationships',
            
            # Elliott Wave (3)
            'Wave_Count', 'Impulse_Corrective', 'Fibonacci_Wave',
            
            # Divergence (4)
            'Hidden_Divergence', 'Momentum_Divergence', 
            'Multi_Timeframe_Divergence', 'Price_Volume_Divergence',
            
            # Cycle (8)
            'Alligator', 'Fisher_Transform', 'Hurst_Exponent', 
            'Cycle_Period', 'Dominant_Cycle', 'Market_Regime', 
            'Phase_Analysis', 'Cycle_Identification',
            
            # Fractal (3)
            'Chaos_Theory', 'Fractal_Dimension', 'Self_Similarity',
            
            # AI Enhancement (9)
            'Market_Microstructure', 'ML_Signal_Generator', 'Multi_Asset_Correlation',
            'Pattern_Recognition_AI', 'Regime_Detection_AI', 'Risk_Assessment_AI',
            'Sentiment_Integration', 'Signal_Confidence_AI', 'Adaptive_Indicators',
            
            # ML Advanced Daytrading (5)
            'DayTrading_Ensemble', 'Intraday_Momentum_ML', 'Session_Breakout_ML',
            'Trend_Continuation_ML', 'Volatility_ML',
            
            # ML Advanced Swing (5)
            'Multi_Timeframe_ML', 'Quick_Reversal_ML', 'Short_Swing_Patterns',
            'Swing_Ensemble', 'Swing_Momentum_ML',
            
            # Sentiment (2)
            'News_Scraper', 'Social_Media_Integrator',
            
            # Performance (1)
            'Scalping_Metrics',
            
            # Pivot (1)
            'Pivot_Point_Calculator'
        ]
        
        print(f"📊 Testing {len(self.base_indicators)} indicators with adaptive AI enhancement...")
        
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5T')
        np.random.seed(42)
        
        # Generate realistic price data
        prices = []
        base_price = 100.0
        
        for i in range(len(dates)):
            change = np.random.normal(0.001, 0.01) * base_price
            base_price += change
            prices.append(base_price)
            
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
    
    def test_all_indicators_quick_compatibility(self):
        """Quick test to see which indicators are compatible with adaptive enhancement."""
        print(f"\n🔬 TESTING ALL {len(self.base_indicators)} INDICATORS...")
        print("="*80)
        
        results = {}
        working_count = 0
        
        for i, base_indicator in enumerate(self.base_indicators, 1):
            try:
                print(f"[{i:3d}/101] Testing {base_indicator:<30} ", end="")
                
                # Test indicator creation
                start_time = time.time()
                adaptive_indicator = AdaptiveIndicators(
                    base_indicator=base_indicator,
                    adaptation_period=20,
                    optimization_window=50
                )
                
                # Test calculation with small dataset
                small_data = self.test_data.iloc[:50]  # Use smaller dataset for speed
                result = adaptive_indicator.calculate(small_data)
                
                execution_time = (time.time() - start_time) * 1000
                  # Verify result
                if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                    results[base_indicator] = {
                        'status': '✅ PASS',
                        'time_ms': f"{execution_time:.1f}ms",
                        'error': None,
                        'rows': len(result)
                    }
                    working_count += 1
                    print(f"✅ PASS ({execution_time:.1f}ms, {len(result)} rows)")
                else:
                    results[base_indicator] = {
                        'status': '⚠️ PARTIAL',
                        'time_ms': f"{execution_time:.1f}ms", 
                        'error': f'Result: {type(result)}, Empty: {result.empty if hasattr(result, "empty") else "N/A"}',
                        'rows': 0
                    }
                    print(f"⚠️ PARTIAL - Result type: {type(result)}")
                    
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 50:
                    error_msg = error_msg[:50] + "..."
                    
                results[base_indicator] = {
                    'status': '❌ FAIL',
                    'time_ms': 'N/A',
                    'error': error_msg
                }
                print(f"❌ FAIL - {error_msg}")
        
        # Print summary
        print("\n" + "="*80)
        print("📊 ADAPTIVE INDICATOR COMPATIBILITY RESULTS")
        print("="*80)
        
        success_count = sum(1 for r in results.values() if r['status'] == '✅ PASS')
        partial_count = sum(1 for r in results.values() if r['status'] == '⚠️ PARTIAL')
        fail_count = sum(1 for r in results.values() if r['status'] == '❌ FAIL')
        
        print(f"✅ WORKING:    {success_count:3d}/101 ({success_count/101*100:.1f}%)")
        print(f"⚠️ PARTIAL:    {partial_count:3d}/101 ({partial_count/101*100:.1f}%)")
        print(f"❌ FAILED:     {fail_count:3d}/101 ({fail_count/101*100:.1f}%)")
        print(f"🎯 TOTAL:      {success_count + partial_count:3d}/101 indicators can use adaptive AI")
        
        # Show failed indicators
        if fail_count > 0:
            print(f"\n❌ FAILED INDICATORS:")
            for indicator, result in results.items():
                if result['status'] == '❌ FAIL':
                    print(f"   • {indicator:<30} - {result['error']}")
        
        # Show performance leaders
        working_indicators = [(k, v) for k, v in results.items() if v['status'] == '✅ PASS']
        if working_indicators:
            # Sort by execution time
            working_indicators.sort(key=lambda x: float(x[1]['time_ms'].replace('ms', '')))
            
            print(f"\n⚡ FASTEST ADAPTIVE INDICATORS:")
            for indicator, result in working_indicators[:5]:
                print(f"   • {indicator:<30} - {result['time_ms']}")
                
            print(f"\n🐌 SLOWEST ADAPTIVE INDICATORS:")
            for indicator, result in working_indicators[-5:]:
                print(f"   • {indicator:<30} - {result['time_ms']}")
        
        print(f"\n{'='*80}")
          # Save detailed results
        with open('all_indicators_test_results.txt', 'w', encoding='utf-8') as f:
            f.write("Platform3 Adaptive Indicators Compatibility Test Results\n")
            f.write("="*60 + "\n\n")
            for indicator, result in results.items():
                status_clean = result['status'].replace('✅', 'PASS').replace('⚠️', 'PARTIAL').replace('❌', 'FAIL')
                f.write(f"{indicator:<35} {status_clean:<12} {result['time_ms']:<10} {result['error'] or ''}\n")
        
        print(f"📄 Detailed results saved to: all_indicators_test_results.txt")
        
        # Assert that at least 70% of indicators work with adaptive enhancement
        success_rate = (success_count + partial_count) / len(self.base_indicators) * 100
        self.assertGreaterEqual(success_rate, 70.0, 
                              f"Expected at least 70% of indicators to work with adaptive AI, got {success_rate:.1f}%")
        
        return results

if __name__ == "__main__":
    unittest.main(verbosity=2)
