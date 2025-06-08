#!/usr/bin/env python3
"""
Phase 4B Implementation: Genius Agent Optimization for 157 Indicators
Optimizes all 9 genius agent mappings for best performance with complete indicator set
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Set
import numpy as np

# Import the adaptive indicator bridge
from engines.ai_enhancement.adaptive_indicator_bridge import (
    AdaptiveIndicatorBridge, 
    GeniusAgentType, 
    IndicatorPackage
)

class Phase4BGeniusAgentOptimizer:
    """Phase 4B: Optimize genius agent mappings for 157 indicators"""
    
    def __init__(self):
        self.bridge = AdaptiveIndicatorBridge()
        self.optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '4B_genius_agent_optimization',
            'total_indicators': len(self.bridge.indicator_registry),
            'genius_agents': 9,
            'optimization_results': {},
            'performance_improvements': {},
            'errors': []
        }
        
    def analyze_indicator_categories(self) -> Dict[str, List[str]]:
        """Analyze all indicators by category for optimal distribution"""
        categories = {}
        
        for indicator_name, config in self.bridge.indicator_registry.items():
            category = config.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(indicator_name)
        
        return categories
    
    def calculate_agent_expertise_scores(self) -> Dict[GeniusAgentType, Dict[str, float]]:
        """Calculate expertise scores for each agent across indicator categories"""
        
        expertise_matrix = {
            GeniusAgentType.RISK_GENIUS: {
                'statistical': 1.0, 'volatility': 1.0, 'fractal': 0.8, 'correlation': 1.0,
                'momentum': 0.6, 'trend': 0.7, 'volume': 0.5, 'pattern': 0.4
            },
            GeniusAgentType.PATTERN_MASTER: {
                'pattern': 1.0, 'fractal': 1.0, 'fibonacci': 1.0, 'elliott_wave': 1.0,
                'gann': 0.9, 'cycle': 0.8, 'harmonic': 1.0, 'trend': 0.7
            },
            GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: {
                'volume': 1.0, 'flow': 1.0, 'institutional': 1.0, 'tick': 1.0,
                'order_flow': 1.0, 'liquidity': 1.0, 'spread': 1.0, 'depth': 1.0
            },
            GeniusAgentType.EXECUTION_EXPERT: {
                'volume': 0.9, 'momentum': 1.0, 'trend': 0.8, 'breakout': 1.0,
                'entry_exit': 1.0, 'timing': 1.0, 'fractal': 0.7, 'volatility': 0.6
            },
            GeniusAgentType.SESSION_EXPERT: {
                'pivot': 1.0, 'support_resistance': 1.0, 'session': 1.0, 'time': 1.0,
                'fibonacci': 0.8, 'gann': 0.7, 'volume': 0.6, 'profile': 1.0
            },
            GeniusAgentType.PAIR_SPECIALIST: {
                'correlation': 1.0, 'statistical': 0.9, 'relative': 1.0, 'spread': 1.0,
                'cointegration': 1.0, 'hedge': 1.0, 'arbitrage': 1.0, 'divergence': 0.8
            },
            GeniusAgentType.DECISION_MASTER: {
                'consensus': 1.0, 'signal': 1.0, 'confirmation': 1.0, 'filter': 1.0,
                'trend': 0.8, 'momentum': 0.7, 'pattern': 0.6, 'volume': 0.5
            },
            GeniusAgentType.AI_MODEL_COORDINATOR: {
                'ml_advanced': 1.0, 'neural': 1.0, 'ensemble': 1.0, 'prediction': 1.0,
                'adaptive': 1.0, 'learning': 1.0, 'optimization': 1.0, 'fractal': 0.8
            },
            GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS: {
                'sentiment': 1.0, 'behavioral': 1.0, 'psychological': 1.0, 'news': 1.0,
                'social': 1.0, 'fear_greed': 1.0, 'institutional': 0.7, 'flow': 0.6
            }
        }
        
        return expertise_matrix
    
    def optimize_indicator_assignments(self) -> Dict[GeniusAgentType, Dict]:
        """Optimize indicator assignments for each genius agent"""
        
        categories = self.analyze_indicator_categories()
        expertise_scores = self.calculate_agent_expertise_scores()
        optimized_mappings = {}
        
        for agent_type in GeniusAgentType:
            print(f"🔧 Optimizing {agent_type.value}...")
            
            # Get agent expertise
            agent_expertise = expertise_scores.get(agent_type, {})
            
            # Score all indicators for this agent
            indicator_scores = {}
            
            for indicator_name, config in self.bridge.indicator_registry.items():
                category = config.get('category', 'unknown')
                priority = config.get('priority', 2)
                
                # Base score from category expertise
                category_score = 0.0
                for expertise_cat, score in agent_expertise.items():
                    if expertise_cat in category or expertise_cat in indicator_name:
                        category_score = max(category_score, score)
                
                # If no direct expertise match, check if agent is listed for this indicator
                agent_affinity = 1.0
                if agent_type in config.get('agents', []):
                    agent_affinity = 1.5
                
                # Priority factor (lower priority number = higher importance)
                priority_factor = 1.0 / priority
                
                # Final score
                final_score = category_score * agent_affinity * priority_factor
                
                if final_score > 0:
                    indicator_scores[indicator_name] = final_score
            
            # Select top indicators for this agent
            sorted_indicators = sorted(
                indicator_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Primary indicators (top performers)
            primary_count = min(15, len(sorted_indicators) // 2)
            primary_indicators = [name for name, score in sorted_indicators[:primary_count]]
            
            # Secondary indicators (good performers)
            secondary_count = min(20, len(sorted_indicators) - primary_count)
            secondary_indicators = [name for name, score in sorted_indicators[primary_count:primary_count + secondary_count]]
            
            # Fallback indicators (minimal set for emergencies)
            fallback_indicators = [name for name, score in sorted_indicators[:5]]
            
            optimized_mappings[agent_type] = {
                'primary_indicators': primary_indicators,
                'secondary_indicators': secondary_indicators,
                'fallback_indicators': fallback_indicators,
                'total_available': len(sorted_indicators),
                'optimization_score': sum(score for name, score in sorted_indicators[:primary_count]),
                'coverage_percentage': (len(primary_indicators + secondary_indicators) / len(self.bridge.indicator_registry)) * 100
            }
            
            print(f"   ✅ {len(primary_indicators)} primary + {len(secondary_indicators)} secondary indicators")        
        return optimized_mappings
    
    async def test_optimized_performance(self, optimized_mappings: Dict) -> Dict[str, Any]:
        """Test performance with optimized mappings"""
        print("⚡ Testing Optimized Performance...")
        
        test_market_data = {
            'timestamp': datetime.now(),
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'open': np.random.uniform(1.0500, 1.0600, 100),
            'high': np.random.uniform(1.0600, 1.0700, 100),
            'low': np.random.uniform(1.0400, 1.0500, 100),
            'close': np.random.uniform(1.0500, 1.0600, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'regime': 'trending'
        }
        
        performance_results = {}
        
        for agent_type in GeniusAgentType:
            try:
                # Test with the optimized indicator count for this agent
                primary_count = len(optimized_mappings[agent_type]['primary_indicators'])
                secondary_count = len(optimized_mappings[agent_type]['secondary_indicators'])
                total_optimized_count = primary_count + secondary_count
                
                start_time = time.time()
                
                # Test with the full optimized indicator set
                indicator_package = await self.bridge.get_comprehensive_indicator_package(
                    test_market_data, agent_type, max_indicators=total_optimized_count
                )
                    
                calculation_time = (time.time() - start_time) * 1000
                
                # Performance target: <5ms per indicator for full optimization test
                performance_target_met = calculation_time < (total_optimized_count * 5.0)
                
                performance_results[agent_type.value] = {
                    'calculation_time_ms': calculation_time,
                    'indicators_calculated': len(indicator_package.indicators),
                    'optimization_score': indicator_package.optimization_score,
                    'performance_target_met': performance_target_met,
                    'available_indicators': optimized_mappings[agent_type]['total_available'],
                    'coverage_percentage': optimized_mappings[agent_type]['coverage_percentage'],
                    'primary_indicators_count': primary_count,
                    'secondary_indicators_count': secondary_count,
                    'total_optimized_count': total_optimized_count,
                    'ms_per_indicator': calculation_time / max(len(indicator_package.indicators), 1)
                }
                
                print(f"   ✅ {agent_type.value}: {calculation_time:.1f}ms, {len(indicator_package.indicators)} indicators ({calculation_time/max(len(indicator_package.indicators), 1):.1f}ms/indicator)")
                
            except Exception as e:
                performance_results[agent_type.value] = {
                    'error': str(e),
                    'performance_target_met': False,
                    'available_indicators': optimized_mappings.get(agent_type, {}).get('total_available', 0),
                    'coverage_percentage': optimized_mappings.get(agent_type, {}).get('coverage_percentage', 0)
                }
                print(f"   ❌ {agent_type.value}: {str(e)}")
        
        return performance_results
    
    def generate_optimization_report(self, optimized_mappings: Dict, performance_results: Dict) -> Dict:
        """Generate comprehensive optimization report"""
        
        # Convert optimized_mappings to serializable format
        serializable_mappings = {}
        for agent_type, mapping in optimized_mappings.items():
            agent_name = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
            serializable_mappings[agent_name] = mapping
        
        # Convert performance_results to serializable format  
        serializable_performance = {}
        for agent_key, result in performance_results.items():
            # Handle both string and enum keys
            agent_name = agent_key.value if hasattr(agent_key, 'value') else str(agent_key)
            serializable_performance[agent_name] = result
        
        total_indicators = len(self.bridge.indicator_registry)
        successful_agents = sum(1 for result in performance_results.values() 
                              if result.get('performance_target_met', False))
        
        # Calculate improvement metrics
        avg_coverage = np.mean([mapping['coverage_percentage'] for mapping in optimized_mappings.values()])
        avg_optimization_score = np.mean([mapping['optimization_score'] for mapping in optimized_mappings.values()])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'phase': '4B',
            'optimization_type': 'genius_agent_mapping',
            'phase_4b_summary': {
                'total_indicators_optimized': total_indicators,
                'genius_agents_optimized': len(optimized_mappings),
                'successful_agents': successful_agents,
                'success_rate_percentage': (successful_agents / 9) * 100,
                'average_coverage_percentage': float(avg_coverage),
                'average_optimization_score': float(avg_optimization_score)
            },
            'optimization_details': serializable_mappings,
            'performance_results': serializable_performance,
            'recommendations': {
                'phase_4b_complete': successful_agents >= 8,
                'ready_for_phase_4c': successful_agents >= 8 and avg_coverage > 15.0,
                'next_actions': [
                    'Implement performance caching for high-frequency calculations',
                    'Add intelligent indicator selection based on market conditions',
                    'Optimize parallel processing for large indicator sets',
                    'Implement adaptive timeout management'
                ]
            }
        }
        
        return report
    
    async def run_phase_4b_optimization(self) -> Dict[str, Any]:
        """Run complete Phase 4B optimization"""
        print("🚀 Starting Phase 4B: Genius Agent Optimization")
        print("=" * 60)
        print(f"📊 Total Indicators: {len(self.bridge.indicator_registry)}")
        print(f"🤖 Genius Agents: 9")
        print()
        
        start_time = time.time()
        
        # Step 1: Optimize indicator assignments
        print("1️⃣ Optimizing Indicator Assignments...")
        optimized_mappings = self.optimize_indicator_assignments()
        
        # Step 2: Test performance
        print("\n2️⃣ Testing Optimized Performance...")
        performance_results = await self.test_optimized_performance(optimized_mappings)
        
        # Step 3: Generate report
        print("\n3️⃣ Generating Optimization Report...")
        report = self.generate_optimization_report(optimized_mappings, performance_results)
        
        total_time = time.time() - start_time
        report['execution_time_seconds'] = total_time
        
        print(f"\n" + "=" * 60)
        print(f"🎯 Phase 4B Complete in {total_time:.2f}s")
        print(f"✅ Success Rate: {report['phase_4b_summary']['success_rate_percentage']:.1f}%")
        print(f"📈 Average Coverage: {report['phase_4b_summary']['average_coverage_percentage']:.1f}%")
        print(f"🚀 Ready for Phase 4C: {'YES' if report['recommendations']['ready_for_phase_4c'] else 'NO'}")
        
        return report
    
    def save_optimization_results(self, report: Dict, filename: str = None):
        """Save optimization results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase_4b_optimization_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Optimization results saved to: {filename}")

async def main():
    """Main optimization execution"""
    optimizer = Phase4BGeniusAgentOptimizer()
    results = await optimizer.run_phase_4b_optimization()
    optimizer.save_optimization_results(results)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
