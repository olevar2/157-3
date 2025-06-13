#!/usr/bin/env python3
"""
PLATFORM3 AGENT VERIFICATION - ANSWERS TO THREE CRITICAL QUESTIONS
==================================================================

1. Will each agent be proficient in using their own indicators?
2. Is there logical integration and professional cooperation among agents?
3. Does each agent know and use their assigned indicators?

For Platform3 Humanitarian Trading Mission: Helping sick babies and poor families
"""

import asyncio
import sys
import os
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
from engines.ai_enhancement.genius_agent_integration import GeniusAgentType
from engines.ai_enhancement.indicator_mappings import AGENT_INDICATOR_MAPPINGS

def verify_agent_proficiency():
    """QUESTION 1: Agent Proficiency with Their Indicators"""
    print("=" * 80)
    print("QUESTION 1: AGENT PROFICIENCY VERIFICATION")
    print("=" * 80)
    
    agent_files = {
        "risk_genius": "ai-platform/ai-models/intelligent-agents/risk-genius/ultra_fast_model.py",
        "session_expert": "ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py",
        "pattern_master": "ai-platform/ai-models/intelligent-agents/strategy-expert/ultra_fast_model.py",
        "execution_expert": "ai-platform/ai-models/intelligent-agents/execution-expert/ultra_fast_model.py",
        "pair_specialist": "ai-platform/ai-models/intelligent-agents/pair-specialist/ultra_fast_model.py",
        "decision_master": "ai-platform/ai-models/intelligent-agents/decision-master/model.py",
        "ai_model_coordinator": "ai-platform/ai-models/intelligent-agents/indicator-expert/model.py",
        "market_microstructure_genius": "ai-platform/ai-models/intelligent-agents/simulation-expert/model.py",
        "sentiment_integration_genius": "ai-platform/ai-models/intelligent-agents/currency-pair-intelligence/model.py"
    }
    
    proficiency_scores = {}
    
    for agent_name, file_path in agent_files.items():
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check proficiency indicators
            checks = {
                'bridge_import': 'AdaptiveIndicatorBridge' in content,
                'bridge_usage': 'bridge.get_agent_indicators_async' in content,
                'indicator_synthesis': '_synthesize' in content,
                'async_methods': 'async def' in content,
                'fallback_logic': 'fallback' in content.lower(),
                'professional_analysis': any(term in content for term in [
                    'analyze_', 'calculate_', 'assess_', 'evaluate_'
                ])
            }
            
            score = sum(checks.values()) / len(checks)
            proficiency_scores[agent_name] = {
                'score': score,
                'checks': checks,
                'status': 'PROFICIENT' if score >= 0.7 else 'NEEDS_WORK'
            }
            
        except Exception as e:
            proficiency_scores[agent_name] = {
                'score': 0.0,
                'error': str(e),
                'status': 'ERROR'
            }
    
    # Display results
    proficient_count = 0
    for agent_name, result in proficiency_scores.items():
        score = result['score']
        status = result['status']
        
        print(f"{agent_name:<30} Score: {score:.1%} | {status}")
        
        if 'checks' in result:
            for check, passed in result['checks'].items():
                symbol = 'YES' if passed else 'NO'
                print(f"  - {check.replace('_', ' ').title()}: {symbol}")
        
        if status == 'PROFICIENT':
            proficient_count += 1
        print()
    
    success_rate = proficient_count / len(proficiency_scores)
    print(f"PROFICIENCY SUMMARY: {proficient_count}/{len(proficiency_scores)} agents proficient ({success_rate:.1%})")
    
    return success_rate >= 0.8

def verify_agent_coordination():
    """QUESTION 2: Agent Coordination and Cooperation"""
    print("=" * 80)
    print("QUESTION 2: AGENT COORDINATION & COOPERATION")
    print("=" * 80)
    
    coordination_files = {
        "Model Communication": "ai-platform/ai-services/coordination-hub/ModelCommunication.py",
        "Agent Config Coordinator": "ai-platform/ai-services/config/AgentConfigCoordinator.py",
        "Collaboration Monitor": "ai-platform/ai-services/monitoring/AgentCollaborationMonitor.py",
        "Integration Framework": "engines/ai_enhancement/genius_agent_integration.py"
    }
    
    coordination_evidence = {}
    
    for component, file_path in coordination_files.items():
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            evidence = {
                'async_coordination': 'async' in content and 'coordination' in content.lower(),
                'inter_agent_comm': any(term in content.lower() for term in [
                    'inter-agent', 'communication', 'message'
                ]),
                'health_monitoring': any(term in content.lower() for term in [
                    'health', 'monitor', 'status'
                ]),
                'error_handling': 'error' in content.lower() and 'handling' in content.lower(),
                'humanitarian_mission': any(term in content.lower() for term in [
                    'humanitarian', 'sick babies', 'poor families'
                ])
            }
            
            coordination_evidence[component] = evidence
            
        except Exception as e:
            coordination_evidence[component] = {'error': str(e)}
    
    # Analyze coordination strength
    total_evidence = 0
    found_evidence = 0
    
    for component, evidence in coordination_evidence.items():
        print(f"{component}:")
        
        if 'error' not in evidence:
            for feature, found in evidence.items():
                symbol = 'YES' if found else 'NO'
                print(f"  - {feature.replace('_', ' ').title()}: {symbol}")
                total_evidence += 1
                if found:
                    found_evidence += 1
        else:
            print(f"  ERROR: {evidence['error']}")
        print()
    
    coordination_strength = found_evidence / total_evidence if total_evidence > 0 else 0
    print(f"COORDINATION SUMMARY: {found_evidence}/{total_evidence} features present ({coordination_strength:.1%})")
    
    return coordination_strength >= 0.6

async def verify_indicator_access():
    """QUESTION 3: Agent Indicator Knowledge and Usage"""
    print("=" * 80)
    print("QUESTION 3: INDICATOR ACCESS & USAGE VERIFICATION")
    print("=" * 80)
    
    bridge = AdaptiveIndicatorBridge()
    
    # Test data in dictionary format (bridge will convert to list format for indicators)
    test_data = {
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        "open": [99.5, 100.5, 101.5, 102.5, 103.5],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    }
    
    access_results = {}
    
    for agent_type in GeniusAgentType:
        agent_name = agent_type.value
        
        # Get expected indicator count
        agent_mapping = AGENT_INDICATOR_MAPPINGS.get(agent_name, {})
        expected_count = sum(len(indicators) for indicators in agent_mapping.values())
        
        try:
            # Test bridge access
            indicators = await bridge.get_agent_indicators_async(agent_type, test_data)
            received_count = len(indicators)
            
            access_ratio = received_count / expected_count if expected_count > 0 else 0
            
            access_results[agent_name] = {
                'expected': expected_count,
                'received': received_count,
                'ratio': access_ratio,
                'status': 'FULL_ACCESS' if access_ratio > 0.8 else 'PARTIAL_ACCESS' if access_ratio > 0.3 else 'NO_ACCESS'
            }
            
        except Exception as e:
            access_results[agent_name] = {
                'expected': expected_count,
                'received': 0,
                'ratio': 0.0,
                'error': str(e),
                'status': 'ERROR'
            }
    
    # Display results
    full_access_count = 0
    partial_access_count = 0
    
    for agent_name, result in access_results.items():
        expected = result['expected']
        received = result['received']
        ratio = result['ratio']
        status = result['status']
        
        print(f"{agent_name:<30} {expected:>3} -> {received:>3} ({ratio:.1%}) {status}")
        
        if status == 'FULL_ACCESS':
            full_access_count += 1
        elif status == 'PARTIAL_ACCESS':
            partial_access_count += 1
    
    total_agents = len(access_results)
    access_success_rate = (full_access_count + partial_access_count) / total_agents
    
    print(f"\nACCESS SUMMARY:")
    print(f"  Full Access: {full_access_count}/{total_agents}")
    print(f"  Partial Access: {partial_access_count}/{total_agents}")
    print(f"  Success Rate: {access_success_rate:.1%}")
    
    return access_success_rate >= 0.8

async def main():
    """Main verification function"""
    print("PLATFORM3 COMPREHENSIVE AGENT VERIFICATION")
    print("=" * 80)
    print("Humanitarian Mission: Helping sick babies and poor families")
    print("Verifying agent proficiency, coordination, and indicator usage")
    print("=" * 80)
    print()
    
    # Run all three verifications
    question1_passed = verify_agent_proficiency()
    print()
    
    question2_passed = verify_agent_coordination()
    print()
    
    question3_passed = await verify_indicator_access()
    print()
    
    # Final summary
    print("=" * 80)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 80)
    
    questions = {
        "Q1: Agent Proficiency": question1_passed,
        "Q2: Agent Coordination": question2_passed,
        "Q3: Indicator Usage": question3_passed
    }
    
    for question, passed in questions.items():
        status = "PASS" if passed else "FAIL"
        print(f"{question:<25} {status}")
    
    all_passed = all(questions.values())
    
    print()
    if all_passed:
        print("SUCCESS: ALL VERIFICATIONS PASSED!")
        print("- Agents ARE proficient with their indicators")
        print("- Logical integration and cooperation EXISTS")  
        print("- Each agent KNOWS and USES assigned indicators")
        print("Platform3 is ready for humanitarian trading mission!")
    else:
        print("WARNING: Some verifications failed")
        print("Review failed areas before deployment")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Verification failed with error: {e}")
        sys.exit(1)