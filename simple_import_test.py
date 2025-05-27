"""
Simple import test for Platform3 indicators
"""

import sys
import os

# Add the services directory to the path
sys.path.append('services/analytics-service/src')

def test_imports():
    """Test all indicator imports"""
    print('🚀 Testing Platform3 Indicator Imports...')
    print('=' * 50)
    
    results = {}
    
    # Test Pivot indicators
    try:
        from engines.indicators.pivot import PivotPointCalculator, PivotType, TimeFrame
        print('✅ Pivot indicators imported successfully')
        results['Pivot'] = True
    except Exception as e:
        print(f'❌ Pivot import failed: {e}')
        results['Pivot'] = False
    
    # Test Trend indicators
    try:
        from engines.indicators.trend import ADX, Ichimoku, MovingAverages
        print('✅ Trend indicators imported successfully')
        results['Trend'] = True
    except Exception as e:
        print(f'❌ Trend import failed: {e}')
        results['Trend'] = False
    
    # Test Gann indicators
    try:
        from engines.gann import GannAnglesCalculator, GannSquareOfNine, GannFanAnalysis
        print('✅ Gann indicators imported successfully')
        results['Gann'] = True
    except Exception as e:
        print(f'❌ Gann import failed: {e}')
        results['Gann'] = False
    
    # Test Fibonacci indicators
    try:
        from engines.fibonacci import FibonacciRetracement, FibonacciExtension, ConfluenceDetector
        print('✅ Fibonacci indicators imported successfully')
        results['Fibonacci'] = True
    except Exception as e:
        print(f'❌ Fibonacci import failed: {e}')
        results['Fibonacci'] = False
    
    # Test Fractal Geometry indicators
    try:
        from engines.fractal_geometry import FractalGeometryIndicator
        print('✅ Fractal Geometry indicators imported successfully')
        results['Fractal Geometry'] = True
    except Exception as e:
        print(f'❌ Fractal Geometry import failed: {e}')
        results['Fractal Geometry'] = False
    
    # Test Elliott Wave indicators
    try:
        from engines.swingtrading import ShortTermElliottWaves
        print('✅ Elliott Wave indicators imported successfully')
        results['Elliott Wave'] = True
    except Exception as e:
        print(f'❌ Elliott Wave import failed: {e}')
        results['Elliott Wave'] = False
    
    # Test basic technical indicators
    try:
        from engines.indicators.momentum import RSI, MACD, Stochastic
        print('✅ Technical indicators imported successfully')
        results['Technical'] = True
    except Exception as e:
        print(f'❌ Technical import failed: {e}')
        results['Technical'] = False
    
    print('=' * 50)
    print('📊 IMPORT TEST RESULTS:')
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for indicator_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f'{indicator_type}: {status}')
    
    print('=' * 50)
    success_rate = (passed / total) * 100
    print(f'🎯 SUCCESS RATE: {passed}/{total} ({success_rate:.1f}%)')
    
    if passed == total:
        print('🎉 ALL INDICATOR TYPES SUCCESSFULLY IMPORTED!')
        print('✅ Platform3 has COMPLETE indicator coverage:')
        print('   • Gann indicators (all types)')
        print('   • Fibonacci indicators (all types)')
        print('   • Pivot indicators (all types)')
        print('   • Elliott Wave indicators')
        print('   • Fractal Geometry analysis')
        print('   • Technical indicators suite')
        return True
    else:
        print(f'❌ {total - passed} indicator type(s) failed import')
        return False

if __name__ == "__main__":
    success = test_imports()
    print('\n🏁 Import test completed!')
    sys.exit(0 if success else 1)
