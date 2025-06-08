#!/usr/bin/env python3
"""
Enhanced Phase 4B Diagnostic Tool
Identifies and fixes the indicator package generation issues
"""

import asyncio
import time
import traceback
import numpy as np
from datetime import datetime
from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge, GeniusAgentType

class Phase4BDiagnosticTool:
    def __init__(self):
        self.bridge = AdaptiveIndicatorBridge()
        # Generate longer test data to meet period requirements
        data_length = 200  # Increased from 100 to 200 for indicators needing 54+ periods
        self.test_market_data = {
            'timestamp': datetime.now(),
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'open': np.random.uniform(1.0500, 1.0600, data_length),
            'high': np.random.uniform(1.0600, 1.0700, data_length),            'low': np.random.uniform(1.0400, 1.0500, data_length),
            'close': np.random.uniform(1.0500, 1.0600, data_length),
            'volume': np.random.uniform(1000, 10000, data_length),
            'regime': 'trending'
        }
    
    async def diagnostic_test_agent(self, agent_type: GeniusAgentType, max_indicators: int = 5):
        """Run detailed diagnostic test for a specific agent"""
        print(f"\n🔍 Diagnosing {agent_type.value}...")
        
        try:
            # Get agent configuration first
            agent_config = self.bridge.agent_indicator_mapping.get(agent_type)
            if not agent_config:
                print(f"   ❌ No configuration found for {agent_type.value}")
                return None
            
            primary_count = len(agent_config.get('primary_indicators', []))
            secondary_count = len(agent_config.get('secondary_indicators', []))
            total_available = primary_count + secondary_count
            
            print(f"   📊 Available: {primary_count} primary + {secondary_count} secondary = {total_available} total")
            print(f"   🎯 Requesting: {max_indicators} indicators")
            
            # Test the indicator package generation
            start_time = time.time()
            
            try:
                indicator_package = await self.bridge.get_comprehensive_indicator_package(
                    self.test_market_data, agent_type, max_indicators=max_indicators
                )
                
                calculation_time = (time.time() - start_time) * 1000
                
                # Analyze results
                indicators_received = len(indicator_package.indicators)
                is_fallback = indicator_package.metadata.get('fallback_mode', False)
                optimization_score = indicator_package.optimization_score
                
                print(f"   ⏱️  Calculation time: {calculation_time:.1f}ms")
                print(f"   📦 Indicators received: {indicators_received}")
                print(f"   🔄 Fallback mode: {is_fallback}")
                print(f"   📈 Optimization score: {optimization_score:.3f}")
                
                # Check if we got the expected indicators
                if indicators_received < min(max_indicators, total_available) and not is_fallback:
                    print(f"   ⚠️  Expected {min(max_indicators, total_available)} but got {indicators_received}")
                
                if is_fallback:
                    print(f"   🚨 FALLBACK TRIGGERED - Main calculation failed")
                    print(f"      Fallback indicators: {list(indicator_package.indicators.keys())}")
                
                return {
                    'agent_type': agent_type.value,
                    'calculation_time_ms': calculation_time,
                    'indicators_received': indicators_received,
                    'indicators_expected': min(max_indicators, total_available),
                    'fallback_mode': is_fallback,
                    'optimization_score': optimization_score,
                    'success': not is_fallback and indicators_received >= min(5, total_available)
                }
                
            except Exception as inner_e:
                print(f"   ❌ Exception in package generation: {str(inner_e)}")
                print(f"   📜 Traceback: {traceback.format_exc()}")
                return {
                    'agent_type': agent_type.value,
                    'error': str(inner_e),
                    'success': False
                }
          except Exception as e:
            print(f"   ❌ Diagnostic error: {str(e)}")
            return {
                'agent_type': agent_type.value,
                'error': str(e),
                'success': False
            }
    
    async def run_comprehensive_diagnostic(self):
        """Run comprehensive diagnostic for all agents"""
        print("🚀 Starting Phase 4B Comprehensive Diagnostic")
        print("=" * 60)
        
        results = {}
        successful_agents = 0
        
        for agent_type in GeniusAgentType:
            result = await self.diagnostic_test_agent(agent_type, max_indicators=5)
            if result:
                results[agent_type.value] = result
                if result.get('success', False):
                    successful_agents += 1
        
        print(f"\n" + "=" * 60)
        print("🎯 DIAGNOSTIC SUMMARY")
        print(f"   Total Agents: {len(GeniusAgentType)}")
        print(f"   Successful: {successful_agents}")
        print(f"   Success Rate: {(successful_agents/len(GeniusAgentType))*100:.1f}%")
        
        print(f"\n📊 AGENT PERFORMANCE:")
        for agent_name, result in results.items():
            if result.get('success', False):
                print(f"   ✅ {agent_name:30}: {result.get('indicators_received', 0):2d} indicators in {result.get('calculation_time_ms', 0):.1f}ms")
            else:
                error = result.get('error', 'Unknown error')
                print(f"   ❌ {agent_name:30}: {error}")
        
        # Identify issues
        print(f"\n🔧 IDENTIFIED ISSUES:")
        fallback_agents = [name for name, result in results.items() if result.get('fallback_mode', False)]
        if fallback_agents:
            print(f"   🚨 Agents using fallback: {', '.join(fallback_agents)}")
        
        low_indicator_agents = [name for name, result in results.items() 
                              if result.get('indicators_received', 0) < 5 and not result.get('fallback_mode', False)]
        if low_indicator_agents:
            print(f"   📉 Agents with low indicator count: {', '.join(low_indicator_agents)}")
        
        error_agents = [name for name, result in results.items() if 'error' in result]
        if error_agents:
            print(f"   💥 Agents with errors: {', '.join(error_agents)}")
        
        return results

async def main():
    """Main diagnostic execution"""
    print("🚀 Starting Phase 4B Diagnostic...")
    try:
        diagnostic = Phase4BDiagnosticTool()
        print("✅ Diagnostic tool initialized")
        results = await diagnostic.run_comprehensive_diagnostic()
        print("✅ Diagnostic completed")
        return results
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())
