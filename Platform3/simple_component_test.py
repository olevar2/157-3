"""
Simple Component Test for Platform3

Tests each implemented component individually to verify functionality.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_time_weighted_volatility():
    """Test TimeWeightedVolatility component"""
    logger.info("Testing TimeWeightedVolatility...")
    
    try:
        sys.path.append('services/analytics-service/src/engines/indicators/advanced')
        from TimeWeightedVolatility import TimeWeightedVolatility
        
        # Create analyzer
        volatility_analyzer = TimeWeightedVolatility(lookback_periods=20)
        
        # Generate sample price data
        np.random.seed(42)
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, 50))
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(50, 0, -1)]
        
        # Analyze
        result = volatility_analyzer.analyze(prices, timestamps)
        
        # Verify results
        assert result.current_volatility >= 0, "Volatility should be non-negative"
        assert hasattr(result, 'regime'), "Should have regime"
        assert hasattr(result, 'session'), "Should have session"
        
        logger.info(f"✅ TimeWeightedVolatility: Volatility={result.current_volatility:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ TimeWeightedVolatility test failed: {str(e)}")
        return False

def test_adx_indicator():
    """Test ADX Indicator component"""
    logger.info("Testing ADX Indicator...")
    
    try:
        sys.path.append('services/analytics-service/src/engines/indicators/trend')
        from ADX import ADXIndicator
        
        # Create indicator
        adx_indicator = ADXIndicator(period=14)
        
        # Generate sample OHLC data
        np.random.seed(42)
        base_price = 1.1000
        
        for i in range(30):
            change = np.random.normal(0, 0.001)
            close = base_price + change
            high = close + abs(np.random.normal(0, 0.0005))
            low = close - abs(np.random.normal(0, 0.0005))
            
            result = adx_indicator.update(high, low, close)
            base_price = close
        
        # Verify results
        assert 0 <= result.adx <= 100, f"ADX should be 0-100, got {result.adx}"
        assert 0 <= result.plus_di <= 100, f"+DI should be 0-100, got {result.plus_di}"
        assert 0 <= result.minus_di <= 100, f"-DI should be 0-100, got {result.minus_di}"
        
        logger.info(f"✅ ADX: ADX={result.adx:.2f}, +DI={result.plus_di:.2f}, -DI={result.minus_di:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ADX test failed: {str(e)}")
        return False

def test_ichimoku_indicator():
    """Test Ichimoku Indicator component"""
    logger.info("Testing Ichimoku Indicator...")
    
    try:
        sys.path.append('services/analytics-service/src/engines/indicators/trend')
        from Ichimoku import IchimokuIndicator
        
        # Create indicator
        ichimoku_indicator = IchimokuIndicator()
        
        # Generate sample OHLC data
        np.random.seed(42)
        base_price = 1.1000
        
        for i in range(60):  # Need more data for Ichimoku
            change = np.random.normal(0, 0.001)
            close = base_price + change
            high = close + abs(np.random.normal(0, 0.0005))
            low = close - abs(np.random.normal(0, 0.0005))
            
            result = ichimoku_indicator.update(high, low, close)
            base_price = close
        
        # Verify results
        assert result.tenkan_sen > 0, "Tenkan-sen should be positive"
        assert result.kijun_sen > 0, "Kijun-sen should be positive"
        assert hasattr(result, 'cloud_position'), "Should have cloud position"
        assert hasattr(result, 'overall_signal'), "Should have overall signal"
        
        logger.info(f"✅ Ichimoku: Tenkan={result.tenkan_sen:.4f}, Kijun={result.kijun_sen:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ichimoku test failed: {str(e)}")
        return False

def test_day_trading_analytics():
    """Test DayTradingAnalytics component"""
    logger.info("Testing DayTradingAnalytics...")
    
    try:
        sys.path.append('services/analytics-service/src/performance')
        from DayTradingAnalytics import DayTradingAnalytics
        
        # Create analytics
        day_analytics = DayTradingAnalytics(account_balance=10000.0)
        
        # Add sample trades
        base_time = datetime.now() - timedelta(days=10)
        for i in range(10):
            entry_time = base_time + timedelta(hours=i*2)
            exit_time = entry_time + timedelta(hours=1)
            
            entry_price = 1.1000 + np.random.normal(0, 0.01)
            exit_price = entry_price + np.random.normal(0, 0.005)
            quantity = 10000
            direction = 'long' if np.random.random() > 0.5 else 'short'
            
            day_analytics.add_trade(entry_time, exit_time, entry_price, exit_price, quantity, direction)
        
        # Calculate metrics
        metrics = day_analytics.calculate_metrics()
        
        # Verify results
        assert metrics.total_trades == 10, f"Should have 10 trades, got {metrics.total_trades}"
        assert 0 <= metrics.win_rate <= 100, f"Win rate should be 0-100, got {metrics.win_rate}"
        
        logger.info(f"✅ DayTradingAnalytics: {metrics.total_trades} trades, Win rate: {metrics.win_rate:.1f}%")
        return True
        
    except Exception as e:
        logger.error(f"❌ DayTradingAnalytics test failed: {str(e)}")
        return False

def test_sentiment_scores():
    """Test SentimentScores component"""
    logger.info("Testing SentimentScores...")
    
    try:
        sys.path.append('services/analytics-service/src/engines/indicators/advanced')
        from SentimentScores import SentimentScores, SentimentSource
        
        # Create analyzer
        sentiment_analyzer = SentimentScores(lookback_periods=20)
        
        # Add sample sentiment data
        sentiment_analyzer.add_sentiment_data(SentimentSource.NEWS, 0.5, confidence=0.8)
        sentiment_analyzer.add_sentiment_data(SentimentSource.SOCIAL_MEDIA, -0.2, confidence=0.6)
        sentiment_analyzer.add_price_data(1.1050)
        
        # Calculate sentiment
        result = sentiment_analyzer.calculate_sentiment_scores()
        
        # Verify results
        assert -1 <= result.overall_sentiment <= 1, f"Sentiment should be -1 to 1, got {result.overall_sentiment}"
        assert hasattr(result, 'sentiment_level'), "Should have sentiment level"
        
        logger.info(f"✅ SentimentScores: Overall sentiment={result.overall_sentiment:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ SentimentScores test failed: {str(e)}")
        return False

def test_profit_optimizer():
    """Test ProfitOptimizer component"""
    logger.info("Testing ProfitOptimizer...")
    
    try:
        sys.path.append('services/analytics-service/src/performance')
        from ProfitOptimizer import ProfitOptimizer, PositionSizingMethod
        
        # Create optimizer
        optimizer = ProfitOptimizer(initial_capital=100000.0)
        
        # Add sample trade results
        base_time = datetime.now() - timedelta(days=30)
        for i in range(20):
            entry_time = base_time + timedelta(hours=i*2)
            exit_time = entry_time + timedelta(hours=1)
            
            pnl = np.random.normal(50, 200)  # Random P&L
            position_size = 0.02
            
            optimizer.add_trade_result(entry_time, exit_time, pnl, position_size)
        
        # Test Kelly fraction
        kelly_fraction = optimizer.calculate_kelly_fraction()
        assert 0 <= kelly_fraction <= 1, f"Kelly fraction should be 0-1, got {kelly_fraction}"
        
        # Test position sizing
        position_opt = optimizer.optimize_position_sizing(PositionSizingMethod.KELLY)
        assert 'optimal_size' in position_opt, "Should return optimal size"
        
        logger.info(f"✅ ProfitOptimizer: Kelly fraction={kelly_fraction:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ProfitOptimizer test failed: {str(e)}")
        return False

def run_simple_tests():
    """Run all simple component tests"""
    logger.info("🚀 Starting Simple Component Tests")
    logger.info("=" * 50)
    
    tests = [
        ("TimeWeightedVolatility", test_time_weighted_volatility),
        ("ADX Indicator", test_adx_indicator),
        ("Ichimoku Indicator", test_ichimoku_indicator),
        ("DayTradingAnalytics", test_day_trading_analytics),
        ("SentimentScores", test_sentiment_scores),
        ("ProfitOptimizer", test_profit_optimizer)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info("=" * 50)
    logger.info(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"⚠️ {total - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)
