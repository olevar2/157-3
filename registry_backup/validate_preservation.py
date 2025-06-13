#!/usr/bin/env python3
"""
Registry Preservation Validation Script
Ensures no functionality is lost during indicator transformation
"""

import sys
import os
import importlib
import json
from typing import Dict, List, Any, Set
from datetime import datetime

# Add the Platform3 directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def validate_registry_preservation():
    """Comprehensive validation of registry preservation"""
    print("=" * 60)
    print("Registry Preservation Validation")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'indicator_count': 0,
        'ai_agent_count': 0,
        'category_count': 0,
        'alias_count': 0,
        'validation_errors': [],
        'preservation_status': 'UNKNOWN'
    }
    
    try:
        # Import the registry
        from engines.ai_enhancement.registry import (
            INDICATOR_REGISTRY,
            AI_AGENTS_REGISTRY,
            validate_registry,
            get_indicator,
            get_ai_agent,
            list_ai_agents,
            validate_ai_agents,
            get_indicator_categories
        )
        
        print("\n1. Indicator Registry Validation")
        print("-" * 40)
        
        # Validate indicator registry
        unique_count = validate_registry()
        results['indicator_count'] = unique_count
        print(f"✓ Registry validation passed: {unique_count} unique indicators")
        
        # Test indicator categories
        categories = get_indicator_categories()
        results['category_count'] = len(categories)
        print(f"✓ Categories identified: {len(categories)}")
        
        # Count aliases
        total_entries = len(INDICATOR_REGISTRY)
        alias_count = total_entries - unique_count
        results['alias_count'] = alias_count
        print(f"✓ Aliases preserved: {alias_count}")
        
        # Test sample indicators
        sample_indicators = [
            'relativestrengthindex',
            'bollinger_bands',
            'movingaverageconvergencedivergence',
            'stochasticoscillator',
            'awesomeoscillator'
        ]
        
        working_indicators = 0
        for indicator_name in sample_indicators:
            try:
                indicator = get_indicator(indicator_name)
                if callable(indicator):
                    working_indicators += 1
                    print(f"  ✓ {indicator_name}: {indicator.__name__}")
                else:
                    results['validation_errors'].append(f"Indicator {indicator_name} not callable")
            except Exception as e:
                results['validation_errors'].append(f"Failed to get {indicator_name}: {str(e)}")
        
        print(f"✓ Sample indicators working: {working_indicators}/{len(sample_indicators)}")
        
        print("\n2. AI Agents Registry Validation")
        print("-" * 40)
        
        # Validate AI agents
        agent_count = validate_ai_agents()
        results['ai_agent_count'] = agent_count
        print(f"✓ AI agents validated: {agent_count}")
        
        # Test agent listing
        agent_info = list_ai_agents()
        expected_agents = [
            'risk_genius',
            'session_expert', 
            'pattern_master',
            'execution_expert',
            'pair_specialist',
            'decision_master',
            'ai_model_coordinator',
            'market_microstructure_genius',
            'sentiment_integration_genius'
        ]
        
        working_agents = 0
        for agent_name in expected_agents:
            try:
                agent = get_ai_agent(agent_name)
                if 'type' in agent and 'class' in agent:
                    working_agents += 1
                    print(f"  ✓ {agent_name}: {agent['specialization']}")
                else:
                    results['validation_errors'].append(f"Agent {agent_name} missing required fields")
            except Exception as e:
                results['validation_errors'].append(f"Failed to get agent {agent_name}: {str(e)}")
        
        print(f"✓ Expected agents working: {working_agents}/{len(expected_agents)}")
        
        print("\n3. Critical Components Check")
        print("-" * 40)
        
        # Check for dummy indicators
        dummy_indicators = []
        for name, indicator in INDICATOR_REGISTRY.items():
            if hasattr(indicator, '__name__') and 'dummy' in indicator.__name__.lower():
                dummy_indicators.append(name)
        
        if dummy_indicators:
            results['validation_errors'].append(f"Found dummy indicators: {dummy_indicators}")
            print(f"✗ Found {len(dummy_indicators)} dummy indicators")
        else:
            print("✓ No dummy indicators found")
        
        # Check category coverage
        expected_categories = [
            'momentum', 'trend', 'volume', 'volatility', 'pattern',
            'fractal', 'fibonacci', 'gann', 'statistical', 'cycle',
            'divergence', 'sentiment', 'ai_enhancement'
        ]
        
        missing_categories = []
        for cat in expected_categories:
            if cat not in categories:
                missing_categories.append(cat)
        
        if missing_categories:
            results['validation_errors'].append(f"Missing categories: {missing_categories}")
            print(f"✗ Missing categories: {missing_categories}")
        else:
            print("✓ All expected categories present")
        
        print("\n4. Sub-Indicator Verification")
        print("-" * 40)
        
        # Check for key sub-indicators
        key_sub_indicators = [
            'abandoned_baby_signal',
            'doji_type',
            'fibonacci_type',
            'gann_angles_time_cycles',
            'fractalchannelresult',
            'chaos_fractal_dimension'
        ]
        
        found_sub_indicators = 0
        for sub_indicator in key_sub_indicators:
            if sub_indicator in INDICATOR_REGISTRY:
                found_sub_indicators += 1
                print(f"  ✓ {sub_indicator}")
            else:
                print(f"  ✗ {sub_indicator}")
        
        print(f"✓ Sub-indicators found: {found_sub_indicators}/{len(key_sub_indicators)}")
        
        # Determine overall status
        if len(results['validation_errors']) == 0:
            results['preservation_status'] = 'PRESERVED'
            print("\n" + "=" * 60)
            print("✓ REGISTRY PRESERVATION: SUCCESSFUL")
            print("All functionality preserved and validated")
        else:
            results['preservation_status'] = 'COMPROMISED'
            print("\n" + "=" * 60)
            print("✗ REGISTRY PRESERVATION: ISSUES FOUND")
            print("Validation errors detected:")
            for error in results['validation_errors']:
                print(f"  - {error}")
        
    except Exception as e:
        results['validation_errors'].append(f"Critical error during validation: {str(e)}")
        results['preservation_status'] = 'FAILED'
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
    
    # Save results
    with open('registry_backup/preservation_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: registry_backup/preservation_validation_results.json")
    print("=" * 60)
    
    return results['preservation_status'] == 'PRESERVED'

if __name__ == "__main__":
    success = validate_registry_preservation()
    sys.exit(0 if success else 1)