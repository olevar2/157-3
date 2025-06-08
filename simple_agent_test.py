#!/usr/bin/env python3
"""
Simple Agent Integration Test
Test the Genius Agent Integration without complex dependencies
"""

import sys
import os
sys.path.append('.')

def test_agent_integration():
    """Test the agent integration system"""
    print("Testing Genius Agent Integration...")
    
    try:
        # Test basic import
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        print("✅ GeniusAgentIntegration imported successfully")
        
        # Test initialization
        agent = GeniusAgentIntegration()
        print("✅ GeniusAgentIntegration initialized successfully")
        
        # Create simple test data
        market_data = {
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'data': [
                {'timestamp': '2024-01-01', 'open': 1.1000, 'high': 1.1020, 'low': 1.0980, 'close': 1.1010, 'volume': 1000},
                {'timestamp': '2024-01-02', 'open': 1.1010, 'high': 1.1030, 'low': 1.0990, 'close': 1.1020, 'volume': 1100},
                {'timestamp': '2024-01-03', 'open': 1.1020, 'high': 1.1040, 'low': 1.1000, 'close': 1.1030, 'volume': 1200}
            ],
            'current_price': 1.1030
        }
        
        # Test analysis execution
        print("Executing full analysis...")
        result = agent.execute_full_analysis(market_data)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Metadata: {result.get('analysis_metadata', {})}")
        
        # Check if we got meaningful results
        if 'trading_decision' in result:
            print("✅ Trading decision generated")
        
        if 'multi_agent_signals' in result:
            print("✅ Multi-agent signals generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during agent integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_integration()
    if success:
        print("\n🎉 Agent integration is working!")
    else:
        print("\n⚠️ Agent integration needs fixes")
