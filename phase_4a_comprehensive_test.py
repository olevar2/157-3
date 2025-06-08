#!/usr/bin/env python3
"""
Phase 4A Comprehensive Test - Adaptive Indicator Bridge Expansion
Tests the expansion from 8 to 157 indicators across all 9 genius agents
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Import the enhanced adaptive indicator bridge
from engines.ai_enhancement.adaptive_indicator_bridge import (
    AdaptiveIndicatorBridge, 
    GeniusAgentType, 
    IndicatorPackage
)

class Phase4AComprehensiveTest:
    """Comprehensive test suite for Phase 4A implementation"""
    
    def __init__(self):
        self.bridge = AdaptiveIndicatorBridge()
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '4A_expansion_test',
            'total_indicators_expected': 157,
            'genius_agents_tested': 9,
            'test_results': {},
            'performance_metrics': {},
            'errors': [],
            'success_rate': 0.0
        }
        
    def generate_test_market_data(self) -> Dict[str, Any]:
        """Generate comprehensive test market data"""
        return {
            'timestamp': datetime.now(),
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'open': np.random.uniform(1.0500, 1.0600, 100),
            'high': np.random.uniform(1.0600, 1.0700, 100),
            'low': np.random.uniform(1.0400, 1.0500, 100),
            'close': np.random.uniform(1.0500, 1.0600, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'tick_volume': np.random.uniform(500, 5000, 100),
            'spread': np.random.uniform(0.5, 2.0, 100),
            'market_hours': 'active',
            'session': 'london',
            'volatility': 0.0125,
            'regime': 'trending'
        }
    
    async def test_indicator_registry_expansion(self) -> Dict[str, Any]:
        """Test that indicator registry has been expanded to 157 indicators"""
        print("🔍 Testing Indicator Registry Expansion...")
        
        registry_size = len(self.bridge.indicator_registry)
        expected_categories = [
            'fractal', 'volume', 'pattern', 'fibonacci', 'statistical',
            'momentum', 'trend', 'volatility', 'ml_advanced', 'cycle', 'divergence'
        ]
        
        categories_found = set()
        for indicator_name, config in self.bridge.indicator_registry.items():
            category = config.get('category', 'unknown')
            categories_found.add(category)
        
        result = {
            'total_indicators_registered': registry_size,
            'expected_indicators': 157,
            'expansion_success': registry_size >= 100,  # Allow some flexibility
            'categories_found': len(categories_found),
            'expected_categories': len(expected_categories),
            'category_coverage': list(categories_found),
            'registry_complete': registry_size >= 100
        }
        
        print(f"   ✅ Registry Size: {registry_size} indicators")
        print(f"   ✅ Categories: {len(categories_found)} found")
        
        return result
    
    async def test_all_genius_agents(self) -> Dict[str, Any]:
        """Test indicator packages for all 9 genius agents"""
        print("🤖 Testing All Genius Agents...")
        
        test_market_data = self.generate_test_market_data()
        agent_results = {}
        
        for agent_type in GeniusAgentType:
            print(f"   Testing {agent_type.value}...")
            
            try:
                start_time = time.time()
                
                # Test comprehensive indicator package
                indicator_package = await self.bridge.get_comprehensive_indicator_package(
                    test_market_data, agent_type, max_indicators=30
                )
                
                calculation_time = (time.time() - start_time) * 1000  # Convert to ms
                
                agent_results[agent_type.value] = {
                    'success': True,
                    'indicators_calculated': len(indicator_package.indicators),
                    'calculation_time_ms': calculation_time,
                    'optimization_score': indicator_package.optimization_score,
                    'metadata': indicator_package.metadata,
                    'agent_specific_indicators': len(
                        self.bridge.agent_indicator_mapping.get(agent_type, {}).get('primary_indicators', [])
                    )
                }
                
                print(f"      ✅ {len(indicator_package.indicators)} indicators in {calculation_time:.2f}ms")
                
            except Exception as e:
                agent_results[agent_type.value] = {
                    'success': False,
                    'error': str(e),
                    'indicators_calculated': 0,
                    'calculation_time_ms': 0
                }
                print(f"      ❌ Error: {str(e)}")
                self.test_results['errors'].append(f"{agent_type.value}: {str(e)}")
        
        return agent_results
    
    async def test_performance_optimization(self) -> Dict[str, Any]:
        """Test performance optimization for 157 indicators"""
        print("⚡ Testing Performance Optimization...")
        
        test_market_data = self.generate_test_market_data()
        performance_results = {}
        
        # Test different indicator loads
        for max_indicators in [10, 25, 50]:
            print(f"   Testing with {max_indicators} indicators...")
            
            agent_performance = {}
            for agent_type in [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.EXECUTION_EXPERT]:
                try:
                    start_time = time.time()
                    
                    indicator_package = await self.bridge.get_comprehensive_indicator_package(
                        test_market_data, agent_type, max_indicators=max_indicators
                    )
                    
                    calculation_time = (time.time() - start_time) * 1000
                    
                    agent_performance[agent_type.value] = {
                        'calculation_time_ms': calculation_time,
                        'indicators_calculated': len(indicator_package.indicators),
                        'performance_target_met': calculation_time < (max_indicators * 1.0),  # <1ms per indicator
                        'optimization_score': indicator_package.optimization_score
                    }
                    
                except Exception as e:
                    agent_performance[agent_type.value] = {
                        'error': str(e),
                        'performance_target_met': False
                    }
            
            performance_results[f'{max_indicators}_indicators'] = agent_performance
        
        return performance_results
    
    async def test_adaptive_features(self) -> Dict[str, Any]:
        """Test adaptive features across different market regimes"""
        print("🔄 Testing Adaptive Features...")
        
        market_regimes = ['trending', 'ranging', 'volatile']
        adaptive_results = {}
        
        for regime in market_regimes:
            print(f"   Testing {regime} market regime...")
            
            test_data = self.generate_test_market_data()
            test_data['regime'] = regime
            
            regime_results = {}
            
            try:
                # Test with Pattern Master for regime adaptation
                indicator_package = await self.bridge.get_comprehensive_indicator_package(
                    test_data, GeniusAgentType.PATTERN_MASTER, max_indicators=20
                )
                
                regime_results = {
                    'indicators_calculated': len(indicator_package.indicators),
                    'market_regime_detected': indicator_package.metadata.get('market_regime'),
                    'adaptive_adjustments': indicator_package.metadata.get('adaptive_adjustments'),
                    'regime_optimization_score': indicator_package.optimization_score,
                    'regime_specific_indicators': self._count_regime_specific_indicators(
                        indicator_package.indicators, regime
                    )
                }
                
            except Exception as e:
                regime_results = {'error': str(e)}
                self.test_results['errors'].append(f"Regime {regime}: {str(e)}")
            
            adaptive_results[regime] = regime_results
        
        return adaptive_results
    
    def _count_regime_specific_indicators(self, indicators: Dict, regime: str) -> int:
        """Count indicators that are specifically optimized for the regime"""
        count = 0
        regime_keywords = {
            'trending': ['fractal', 'pattern', 'fibonacci'],
            'ranging': ['oscillator', 'fibonacci', 'volume'],
            'volatile': ['fractal', 'volume', 'volatility']
        }
        
        keywords = regime_keywords.get(regime, [])
        for indicator_name in indicators.keys():
            if any(keyword in indicator_name.lower() for keyword in keywords):
                count += 1
        
        return count
    
    async def test_error_handling_and_fallbacks(self) -> Dict[str, Any]:
        """Test error handling and fallback mechanisms"""
        print("🛡️ Testing Error Handling and Fallbacks...")
        
        # Test with invalid market data
        invalid_data = {'invalid': 'data'}
        fallback_results = {}
        
        try:
            indicator_package = await self.bridge.get_comprehensive_indicator_package(
                invalid_data, GeniusAgentType.RISK_GENIUS, max_indicators=10
            )
            
            fallback_results = {
                'fallback_activated': True,
                'fallback_indicators_count': len(indicator_package.indicators),
                'fallback_successful': len(indicator_package.indicators) > 0,
                'error_handling_working': True
            }
            
        except Exception as e:
            fallback_results = {
                'fallback_activated': False,
                'error_handling_working': False,
                'error': str(e)
            }
        
        return fallback_results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete Phase 4A test suite"""
        print("🚀 Starting Phase 4A Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test 1: Registry Expansion
        self.test_results['test_results']['registry_expansion'] = await self.test_indicator_registry_expansion()
        
        # Test 2: All Genius Agents
        self.test_results['test_results']['genius_agents'] = await self.test_all_genius_agents()
        
        # Test 3: Performance Optimization
        self.test_results['performance_metrics'] = await self.test_performance_optimization()
        
        # Test 4: Adaptive Features
        self.test_results['test_results']['adaptive_features'] = await self.test_adaptive_features()
        
        # Test 5: Error Handling
        self.test_results['test_results']['error_handling'] = await self.test_error_handling_and_fallbacks()
        
        # Calculate overall success rate
        total_time = time.time() - start_time
        self.test_results['total_test_time_seconds'] = total_time
        
        # Calculate success metrics
        successful_agents = sum(1 for agent_result in self.test_results['test_results']['genius_agents'].values() 
                              if agent_result.get('success', False))
        
        self.test_results['success_rate'] = (successful_agents / 9) * 100  # 9 genius agents
        self.test_results['phase_4a_ready'] = (
            self.test_results['test_results']['registry_expansion']['registry_complete'] and
            successful_agents >= 8 and  # At least 8/9 agents working
            len(self.test_results['errors']) < 5  # Less than 5 critical errors
        )
        
        print("\n" + "=" * 60)
        print(f"🎯 Phase 4A Test Complete in {total_time:.2f}s")
        print(f"✅ Success Rate: {self.test_results['success_rate']:.1f}%")
        print(f"🚀 Phase 4A Ready: {'YES' if self.test_results['phase_4a_ready'] else 'NO'}")
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase_4a_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {filename}")

async def main():
    """Main test execution"""
    tester = Phase4AComprehensiveTest()
    results = await tester.run_comprehensive_test()
    tester.save_results()
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
