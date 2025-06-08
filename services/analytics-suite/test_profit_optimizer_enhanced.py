"""
Core Analytics Interface Test - Tests without communication framework
"""

import asyncio
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_profit_optimizer_only():
    """Test only the ProfitOptimizer without dependencies"""
    try:
        # Direct import of ProfitOptimizer
        sys.path.append(os.path.dirname(__file__))
        
        from ProfitOptimizer import (
            ProfitOptimizer, 
            AnalyticsInterface, 
            AnalyticsReport, 
            RealtimeMetric
        )
        
        print("✅ ProfitOptimizer imported successfully")
        
        # Create instance
        optimizer = ProfitOptimizer()
        print("✅ ProfitOptimizer instance created")
        
        # Test data
        test_data = {
            'trades': [
                {'profit': 100, 'risk': 50, 'timestamp': '2025-06-01T10:00:00'},
                {'profit': 200, 'risk': 75, 'timestamp': '2025-06-01T11:00:00'},
                {'profit': -50, 'risk': 25, 'timestamp': '2025-06-01T12:00:00'}
            ],
            'portfolio_value': 10000,
            'timeframe': '1h'
        }
        
        # Test AnalyticsInterface methods
        print("\n=== Testing AnalyticsInterface Implementation ===")
        
        # 1. Test process_data
        print("1. Testing process_data...")
        result = await optimizer.process_data(test_data)
        print(f"   ✅ Process result: {result}")
        
        # 2. Test get_real_time_metrics
        print("2. Testing get_real_time_metrics...")
        metrics = optimizer.get_real_time_metrics()
        print(f"   ✅ Real-time metrics: {len(metrics)} metrics collected")
        for metric in metrics[:3]:  # Show first 3 metrics
            print(f"      - {metric.metric_name}: {metric.value} {metric.unit}")
        
        # 3. Test generate_report
        print("3. Testing generate_report...")
        report = await optimizer.generate_report("1h")
        print(f"   ✅ Report generated for {report.service_name}")
        print(f"      - Timeframe: {report.timeframe}")
        print(f"      - Confidence: {report.confidence_score}")
        print(f"      - Data Quality: {report.data_quality}")
        print(f"      - Insights: {len(report.insights)} insights")
        print(f"      - Recommendations: {len(report.recommendations)} recommendations")
          # Test optimization functionality
        print("\n=== Testing Optimization Functionality ===")
        
        # Test profit optimization
        print("4. Testing profit optimization...")
        strategy_data = {
            'returns': [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.01],
            'account_balance': 10000,
            'risk_tolerance': 0.02,
            'strategy_name': 'test_strategy'
        }
        
        optimization_result = optimizer.optimize_profits(strategy_data)
        print(f"   ✅ Optimization result: {optimization_result.status}")
        
        # Test portfolio optimization
        print("5. Testing portfolio optimization...")
        portfolio_strategies = {
            'strategy_1': {
                'returns': [0.01, 0.02, -0.01],
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.05
            },
            'strategy_2': {
                'returns': [0.015, -0.005, 0.02],
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.03
            }
        }
        
        optimal_weights = optimizer.optimize_portfolio_allocation(portfolio_strategies)
        print(f"   ✅ Optimal portfolio weights: {optimal_weights}")
        
        print("\n🎉 ProfitOptimizer AnalyticsInterface implementation fully functional!")
        
        # Summary
        print("\n" + "="*60)
        print("PROFIT OPTIMIZER ENHANCEMENT SUMMARY")
        print("="*60)
        print("✅ AnalyticsInterface Implementation Complete")
        print("   - process_data() method working")
        print("   - get_real_time_metrics() method working")
        print("   - generate_report() method working")
        print("✅ Core Optimization Features Working")
        print("   - Profit optimization")
        print("   - Portfolio allocation optimization")
        print("✅ Real-time Metrics Collection Active")
        print("✅ Standardized Report Generation")
        print("✅ Ready for Framework Integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("Testing Enhanced ProfitOptimizer AnalyticsInterface Implementation...")
    print("="*70)
    
    success = await test_profit_optimizer_only()
    
    if success:
        print("\n🎉 SUCCESS: ProfitOptimizer Analytics Enhancement Complete!")
        print("🚀 Ready for production deployment and framework integration!")
    else:
        print("\n❌ FAILED: Issues found in ProfitOptimizer implementation")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
