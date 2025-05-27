"""
Simple test for Pivot Indicators to verify they work correctly
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio

# Add the services directory to the path
sys.path.append('services/analytics-service/src')

def create_sample_data():
    """Create sample OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')

    data = []
    base_price = 1.1000
    for i, date in enumerate(dates):
        price = base_price + (i * 0.0001) + np.random.normal(0, 0.0005)
        high = price + np.random.uniform(0, 0.001)
        low = price - np.random.uniform(0, 0.001)

        data.append({
            'open': price,
            'high': max(high, price),
            'low': min(low, price),
            'close': price + np.random.normal(0, 0.0002),
            'volume': np.random.randint(1000, 5000)
        })

    df = pd.DataFrame(data, index=dates)
    return df

async def test_pivot_indicators():
    """Test pivot indicators"""
    print("Testing Pivot Indicators...")

    try:
        from engines.indicators.pivot import PivotPointCalculator, PivotType, TimeFrame

        # Create test data
        data = create_sample_data()
        print(f"Created sample data with {len(data)} periods")

        # Initialize calculator
        calculator = PivotPointCalculator()
        print("Pivot calculator initialized")

        # Test each pivot type
        for pivot_type in PivotType:
            try:
                result = await calculator.calculate_pivot_points(
                    symbol='EURUSD',
                    price_data=data,
                    pivot_type=pivot_type,
                    timeframe=TimeFrame.DAILY
                )

                print(f"✅ {pivot_type.value} Pivots:")
                print(f"   Pivot Point: {result.pivot_point:.5f}")
                print(f"   R1: {result.resistance_1:.5f}, S1: {result.support_1:.5f}")
                print(f"   Trading Bias: {result.trading_bias}")
                print(f"   Confidence: {result.confidence:.2f}")

            except Exception as e:
                print(f"❌ Error testing {pivot_type.value}: {e}")
                return False

        print("🎉 All Pivot Indicators working correctly!")
        return True

    except Exception as e:
        print(f"❌ Import or setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_trend_indicators():
    """Test trend indicators"""
    print("\nTesting Trend Indicators...")

    try:
        from engines.indicators.trend import ADX, Ichimoku

        # Create test data
        data = create_sample_data()
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Test ADX
        adx = ADX(period=14)
        adx_result = adx.calculate(high, low, close)
        print(f"✅ ADX: {adx_result.adx:.2f}, Trend: {adx_result.trend_direction.value}")

        # Test Ichimoku
        ichimoku = Ichimoku()
        ichimoku_result = ichimoku.calculate(high, low, close)
        print(f"✅ Ichimoku: Cloud Position: {ichimoku_result.cloud_position.value}")

        print("🎉 All Trend Indicators working correctly!")
        return True

    except Exception as e:
        print(f"❌ Trend indicators error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the tests"""
    print("🚀 Starting Platform3 Indicators Test")
    print("=" * 50)

    # Test pivot indicators
    pivot_success = await test_pivot_indicators()

    # Test trend indicators
    trend_success = await test_trend_indicators()

    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")

    if pivot_success and trend_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Pivot indicators are working correctly")
        print("✅ Trend indicators are working correctly")
        print("✅ Platform3 indicators integration successful!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
