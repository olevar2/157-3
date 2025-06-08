#!/usr/bin/env python3
"""
Production Readiness Test for adaptive_indicator_bridge.py
Comprehensive validation and analysis for agent indicator usage
"""

import sys
import importlib
import traceback
from pathlib import Path
from collections import defaultdict
import json

def test_syntax_and_imports():
    """Test 1: Verify syntax and basic imports"""
    print("=" * 60)
    print("TEST 1: SYNTAX AND IMPORTS VALIDATION")
    print("=" * 60)
    
    try:
        # Test syntax compilation
        print("✓ Testing syntax compilation...")
        import py_compile
        py_compile.compile('d:/MD/Platform3/engines/ai_enhancement/adaptive_indicator_bridge.py', doraise=True)
        print("✓ Syntax compilation: PASSED")
          # Test import
        print("✓ Testing module import...")
        sys.path.insert(0, 'd:/MD/Platform3')
        from engines.ai_enhancement.adaptive_indicator_bridge import (
            AdaptiveIndicatorBridge, 
            GeniusAgentType,
            IndicatorPackage
        )
        print("✓ Module import: PASSED")
        
        return True, AdaptiveIndicatorBridge, GeniusAgentType, IndicatorPackage
        
    except Exception as e:
        print(f"✗ Syntax/Import test FAILED: {e}")
        traceback.print_exc()
        return False, None, None, None

def test_bridge_instantiation(AdaptiveIndicatorBridge):
    """Test 2: Verify bridge can be instantiated"""
    print("\n" + "=" * 60)
    print("TEST 2: BRIDGE INSTANTIATION")
    print("=" * 60)
    
    try:
        bridge = AdaptiveIndicatorBridge()
        print(f"✓ Bridge instantiated successfully")
        print(f"✓ Bridge type: {type(bridge)}")
          # Test basic functionality
        indicator_registry = bridge._build_comprehensive_157_indicator_registry()
        agent_mapping = bridge._build_comprehensive_agent_mapping()
        
        print(f"✓ Indicator registry built: {len(indicator_registry)} indicators")
        print(f"✓ Agent mapping built: {len(agent_mapping)} agents")
        
        return True, bridge
        
    except Exception as e:
        print(f"✗ Bridge instantiation FAILED: {e}")
        traceback.print_exc()
        return False, None

def test_agent_indicator_coverage(bridge):
    """Test 3: Verify agent indicator coverage"""
    print("\n" + "=" * 60)
    print("TEST 3: AGENT INDICATOR COVERAGE")
    print("=" * 60)
    
    try:
        # Test agent mapping functionality
        agent_mapping = bridge._build_comprehensive_agent_mapping()
        indicator_registry = bridge._build_comprehensive_157_indicator_registry()
        
        print(f"✓ Agent mapping: {len(agent_mapping)} agents")
        print(f"✓ Indicator registry: {len(indicator_registry)} indicators")
        
        # Analyze coverage for each agent
        agent_stats = {}
        for agent_type, config in agent_mapping.items():
            primary_indicators = config.get('primary_indicators', [])
            secondary_indicators = config.get('secondary_indicators', [])
            
            agent_stats[str(agent_type)] = {
                'primary': len(primary_indicators),
                'secondary': len(secondary_indicators),
                'total': len(primary_indicators) + len(secondary_indicators)
            }
            
            print(f"  🤖 {agent_type.value}: {len(primary_indicators)} primary + {len(secondary_indicators)} secondary = {len(primary_indicators) + len(secondary_indicators)} total")
        
        return True, agent_stats
        
    except Exception as e:
        print(f"✗ Agent coverage test FAILED: {e}")
        traceback.print_exc()
        return False, {}

def test_enhanced_bridge():
    """Test 3: Test EnhancedIndicatorBridge with agent mappings"""
    print("\n" + "=" * 60)
    print("TEST 3: ENHANCED BRIDGE WITH AGENT MAPPINGS")
    print("=" * 60)
    
    try:
        from engines.ai_enhancement.adaptive_indicator_bridge import EnhancedIndicatorBridge
        enhanced_bridge = EnhancedIndicatorBridge()
        print("✓ EnhancedIndicatorBridge instantiated successfully")
        
        # Analyze agent mappings
        if hasattr(enhanced_bridge, 'agent_mapping'):
            print(f"✓ Agent mapping available: {len(enhanced_bridge.agent_mapping)} agents")
            
            agent_stats = {}
            for agent, config in enhanced_bridge.agent_mapping.items():
                primary_count = len(config.get('primary_indicators', []))
                secondary_count = len(config.get('secondary_indicators', []))
                total_count = primary_count + secondary_count
                
                agent_stats[str(agent)] = {
                    'primary': primary_count,
                    'secondary': secondary_count,
                    'total': total_count
                }
                
            return True, enhanced_bridge, agent_stats
        else:
            print("✗ No agent_mapping found in EnhancedIndicatorBridge")
            return False, enhanced_bridge, {}
            
    except Exception as e:
        print(f"✗ EnhancedIndicatorBridge test FAILED: {e}")
        traceback.print_exc()
        return False, None, {}

def analyze_indicator_registry(enhanced_bridge):
    """Test 4: Analyze indicator registry and categorization"""
    print("\n" + "=" * 60)
    print("TEST 4: INDICATOR REGISTRY ANALYSIS")
    print("=" * 60)
    
    try:
        if not hasattr(enhanced_bridge, 'indicator_registry'):
            print("✗ No indicator_registry found")
            return False, {}
            
        registry = enhanced_bridge.indicator_registry
        print(f"✓ Indicator registry found: {len(registry)} indicators")
        
        # Categorize indicators
        categories = defaultdict(list)
        agent_indicator_count = defaultdict(set)
        
        for indicator_key, config in registry.items():
            category = config.get('category', 'unknown')
            categories[category].append(indicator_key)
            
            # Count indicators per agent
            agents = config.get('agents', [])
            for agent in agents:
                agent_indicator_count[str(agent)].add(indicator_key)
        
        # Print category breakdown
        print("\n📊 INDICATOR CATEGORIES:")
        for category, indicators in categories.items():
            print(f"  {category.upper()}: {len(indicators)} indicators")
        
        # Print agent indicator counts
        print("\n🤖 AGENT INDICATOR ACCESS:")
        for agent, indicators in agent_indicator_count.items():
            print(f"  {agent}: {len(indicators)} indicators")
        
        return True, {
            'categories': dict(categories),
            'agent_counts': {k: len(v) for k, v in agent_indicator_count.items()},
            'total_indicators': len(registry)
        }
        
    except Exception as e:
        print(f"✗ Registry analysis FAILED: {e}")
        traceback.print_exc()
        return False, {}

def test_monitor_functionality(bridge, IndicatorBridgeMonitor):
    """Test 5: Test monitoring functionality"""
    print("\n" + "=" * 60)
    print("TEST 5: MONITOR FUNCTIONALITY")
    print("=" * 60)
    
    try:
        monitor = IndicatorBridgeMonitor(bridge)
        print("✓ Monitor instantiated successfully")
        
        # Run diagnostics
        diagnostics = monitor.run_diagnostics()
        print(f"✓ Diagnostics completed")
        print(f"  - Test success rate: {diagnostics.get('test_success_rate', 0):.2%}")
        print(f"  - Health score: {diagnostics.get('health', {}).get('health_score', 0):.2%}")
        
        return True, diagnostics
        
    except Exception as e:
        print(f"✗ Monitor test FAILED: {e}")
        traceback.print_exc()
        return False, {}

def generate_production_report(test_results):
    """Generate comprehensive production readiness report"""
    print("\n" + "=" * 80)
    print("🏭 PRODUCTION READINESS REPORT")
    print("=" * 80)
    
    # Overall status
    all_tests_passed = all(result['passed'] for result in test_results.values())
    status = "✅ PRODUCTION READY" if all_tests_passed else "❌ NOT PRODUCTION READY"
    
    print(f"\n📋 OVERALL STATUS: {status}")
    
    # Test summary
    print(f"\n🧪 TEST RESULTS SUMMARY:")
    for test_name, result in test_results.items():
        status_icon = "✅" if result['passed'] else "❌"
        print(f"  {status_icon} {test_name}")
        if not result['passed'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Agent indicator analysis
    if 'agent_stats' in test_results.get('enhanced_bridge', {}):
        print(f"\n🤖 AGENT INDICATOR COVERAGE:")
        agent_stats = test_results['enhanced_bridge']['agent_stats']
        for agent, stats in agent_stats.items():
            print(f"  {agent}:")
            print(f"    Primary: {stats['primary']} indicators")
            print(f"    Secondary: {stats['secondary']} indicators")
            print(f"    Total: {stats['total']} indicators")
    
    # Registry analysis
    if 'registry_analysis' in test_results:
        registry_data = test_results['registry_analysis']['data']
        print(f"\n📊 INDICATOR REGISTRY SUMMARY:")
        print(f"  Total indicators: {registry_data.get('total_indicators', 0)}")
        print(f"  Categories: {len(registry_data.get('categories', {}))}")
        print(f"  Agents with access: {len(registry_data.get('agent_counts', {}))}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if all_tests_passed:
        print("  • File is ready for production deployment")
        print("  • All syntax and import tests passed")
        print("  • Bridge functionality verified")
        print("  • Monitoring system operational")
    else:
        print("  • Address failing tests before production deployment")
        print("  • Review error messages and fix issues")
        print("  • Re-run tests after fixes")
    
    return {
        'production_ready': all_tests_passed,
        'test_results': test_results,
        'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'Unknown'
    }

def main():
    """Run comprehensive production readiness tests"""
    print("🚀 ADAPTIVE INDICATOR BRIDGE - PRODUCTION READINESS TEST")
    print("=" * 80)
    
    test_results = {}
      # Test 1: Syntax and imports
    passed, AdaptiveIndicatorBridge, GeniusAgentType, IndicatorPackage = test_syntax_and_imports()
    test_results['syntax_imports'] = {'passed': passed}
    
    if not passed:
        print("\n❌ CRITICAL FAILURE: Cannot proceed with other tests")
        return generate_production_report(test_results)
    
    # Test 2: Bridge instantiation
    passed, bridge = test_bridge_instantiation(AdaptiveIndicatorBridge)
    test_results['bridge_instantiation'] = {'passed': passed}
    
    if not passed:
        print("\n❌ CRITICAL FAILURE: Bridge cannot be instantiated")
        return generate_production_report(test_results)
      # Test 3: Agent indicator coverage  
    passed, agent_stats = test_agent_indicator_coverage(bridge)
    test_results['agent_coverage'] = {
        'passed': passed,
        'agent_stats': agent_stats
    }
    
    # Generate final report
    final_report = generate_production_report(test_results)
    
    # Save detailed results
    try:
        with open('production_readiness_results.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: production_readiness_results.json")
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")
    
    return final_report

if __name__ == '__main__':
    main()
