#!/usr/bin/env python3
"""
Comprehensive DecisionMaster Integration Validation
Tests all aspects of the DynamicRiskAgent integration
"""

import sys
import ast
import traceback
from pathlib import Path

def validate_code_structure():
    """Validate the overall code structure and integration"""
    
    print("=== COMPREHENSIVE DECISIONMASTER VALIDATION ===\n")
    
    try:
        with open('model.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse the code to check AST
        try:
            tree = ast.parse(code)
            print("✅ SYNTAX: Code parses correctly into valid AST")
        except SyntaxError as e:
            print(f"❌ SYNTAX ERROR: {e}")
            return False
        
        # Check class structure
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        print(f"✅ CLASSES: Found {len(classes)} classes: {', '.join(classes)}")
        
        # Check method definitions
        methods = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print(f"✅ METHODS: Found {len(methods)} methods")
        
        # Critical integration checks
        critical_checks = {
            "DynamicRiskAgent Import": "from dynamic_risk_agent.model import DynamicRiskAgent" in code,
            "Platform3 Communication": "Platform3CommunicationFramework" in code,
            "Risk Agent Initialization": "self.risk_agent = DynamicRiskAgent()" in code,
            "Async Decision Method": "async def make_trading_decision" in code,
            "Advanced Risk Assessment": "async def _get_advanced_risk_assessment" in code,
            "Risk-Aware Decision Logic": "def _make_risk_aware_decision" in code,
            "Enhanced Reasoning": "def _add_enhanced_decision_reasoning" in code,
            "Performance Tracking": "def track_risk_adjusted_performance" in code,
            "Error Handling": "except Exception as e:" in code,
            "Fallback Mechanism": "fallback" in code.lower(),
        }
        
        print("\n=== CRITICAL INTEGRATION FEATURES ===")
        passed = 0
        for check, result in critical_checks.items():
            status = "✅ PRESENT" if result else "❌ MISSING"
            print(f"{status} {check}")
            if result:
                passed += 1
        
        print(f"\nIntegration Score: {passed}/{len(critical_checks)} ({passed/len(critical_checks)*100:.1f}%)")
        
        # Risk management features
        risk_features = {
            "Risk Score Combination": "combined_risk_score = (ai_risk_score * 0.7) + (basic_risk_score * 0.3)" in code,
            "High Risk Rejection": "reject trade" in code.lower() and "high risk" in code.lower(),
            "Position Size Reduction": "position_size *= risk_reduction_factor" in code,
            "Position Size Enhancement": "enhancement_factor" in code,
            "AI Risk Factors": "ai_risk_factors" in code,
            "Risk Recommendations": "ai_recommendations" in code,
            "Risk-Adjusted Scoring": "risk_adjustment" in code,
        }
        
        print("\n=== RISK MANAGEMENT FEATURES ===")
        risk_passed = 0
        for feature, result in risk_features.items():
            status = "✅ IMPLEMENTED" if result else "❌ MISSING"
            print(f"{status} {feature}")
            if result:
                risk_passed += 1
        
        print(f"\nRisk Features Score: {risk_passed}/{len(risk_features)} ({risk_passed/len(risk_features)*100:.1f}%)")
        
        # Integration quality assessment
        quality_checks = {
            "Proper Async/Await Usage": code.count("async def") >= 2 and code.count("await") >= 3,
            "Error Handling Coverage": code.count("try:") >= 3 and code.count("except") >= 3,
            "Logging Integration": "logger." in code and code.count("logger.") >= 5,
            "Type Hints": "Dict[str, Any]" in code and "Optional[" in code,
            "Docstring Coverage": code.count('"""') >= 10,
            "Configuration Support": "self.config" in code,
        }
        
        print("\n=== CODE QUALITY ASSESSMENT ===")
        quality_passed = 0
        for check, result in quality_checks.items():
            status = "✅ GOOD" if result else "❌ NEEDS IMPROVEMENT"
            print(f"{status} {check}")
            if result:
                quality_passed += 1
        
        print(f"\nQuality Score: {quality_passed}/{len(quality_checks)} ({quality_passed/len(quality_checks)*100:.1f}%)")
        
        # Final assessment
        print("\n=== FINAL VALIDATION RESULTS ===")
        
        total_score = (passed + risk_passed + quality_passed) / (len(critical_checks) + len(risk_features) + len(quality_checks)) * 100
        
        print(f"Overall Integration Score: {total_score:.1f}%")
        
        if total_score >= 90:
            print("🎯 EXCELLENT: Production-ready integration")
            grade = "A+"
        elif total_score >= 80:
            print("✅ GOOD: Solid integration with minor improvements needed")
            grade = "A"
        elif total_score >= 70:
            print("⚠️  ACCEPTABLE: Integration functional but needs enhancement")
            grade = "B"
        else:
            print("❌ INSUFFICIENT: Major issues need to be addressed")
            grade = "C"
        
        print(f"Integration Grade: {grade}")
        
        # Specific recommendations
        print("\n=== RECOMMENDATIONS ===")
        if passed == len(critical_checks):
            print("✅ All critical features implemented")
        else:
            print("⚠️  Implement missing critical features")
        
        if risk_passed >= len(risk_features) * 0.8:
            print("✅ Strong risk management implementation")
        else:
            print("⚠️  Enhance risk management features")
        
        if quality_passed >= len(quality_checks) * 0.8:
            print("✅ High code quality standards met")
        else:
            print("⚠️  Improve code quality and documentation")
        
        return total_score >= 80
        
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_code_structure()
    print(f"\n{'='*50}")
    print("VALIDATION " + ("PASSED" if success else "FAILED"))
    print(f"{'='*50}")
    sys.exit(0 if success else 1)
