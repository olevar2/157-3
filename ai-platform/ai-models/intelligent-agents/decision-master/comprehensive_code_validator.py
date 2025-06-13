#!/usr/bin/env python3
"""
Comprehensive Code Validation for DecisionMaster
Validates imports, class definitions, method signatures, and overall code health
"""

import ast
import os
import sys
import importlib.util
from pathlib import Path

# Get project root once at module level
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def validate_code_structure():
    """Validate the overall code structure"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    model_path = Path(f"{project_root}/ai-platform/ai-models/intelligent-agents/decision-master/model.py")
    
    if not model_path.exists():
        print("❌ ERROR: model.py file not found")
        return False
    
    print("✅ File exists")
    
    # Read the file content
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ File readable")
    except Exception as e:
        print(f"❌ ERROR: Cannot read file: {e}")
        return False
    
    # Parse AST
    try:
        tree = ast.parse(content)
        print("✅ Python syntax valid")
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        return False
    
    # Check for required imports
    required_imports = [
        'asyncio', 'logging', 'numpy', 'pandas', 'datetime',
        'typing', 'pathlib', 'json', 'os', 'sys'
    ]
    
    import_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_names.add(node.module)
    
    missing_imports = [imp for imp in required_imports if imp not in import_names]
    if missing_imports:
        print(f"⚠️  WARNING: Missing imports: {missing_imports}")
    else:
        print("✅ All required imports present")
    
    # Check for required classes
    required_classes = [
        'DecisionMaster', 'TradingDecision', 'MarketConditions',
        'SignalInput', 'PortfolioContext', 'DecisionType', 'ConfidenceLevel'
    ]
    
    class_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
    
    missing_classes = [cls for cls in required_classes if cls not in class_names]
    if missing_classes:
        print(f"❌ ERROR: Missing classes: {missing_classes}")
        return False
    else:
        print("✅ All required classes present")
      # Check DecisionMaster methods
    decision_master_methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'DecisionMaster':
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decision_master_methods.add(item.name)
    
    required_methods = [
        '__init__', 'make_trading_decision', '_analyze_signals',
        '_assess_market_conditions', '_evaluate_risk', '_generate_initial_proposal',
        '_get_advanced_risk_assessment', '_make_risk_aware_decision',
        '_add_enhanced_decision_reasoning'
    ]
    
    missing_methods = [method for method in required_methods if method not in decision_master_methods]
    if missing_methods:
        print(f"❌ ERROR: Missing DecisionMaster methods: {missing_methods}")
        return False
    else:
        print("✅ All required DecisionMaster methods present")
    
    # Check for async methods
    async_methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            async_methods.add(node.name)
    
    required_async_methods = ['make_trading_decision', '_get_advanced_risk_assessment']
    missing_async = [method for method in required_async_methods if method not in async_methods]
    if missing_async:
        print(f"❌ ERROR: Missing async methods: {missing_async}")
        return False
    else:
        print("✅ All required async methods present")
    
    return True

def check_import_issues():
    """Check for potential import issues"""
    
    print("\n🔍 CHECKING IMPORT DEPENDENCIES...")
    
    # Check if enum and dataclasses imports are at the top
    model_path = Path(f"{project_root}/ai-platform/ai-models/intelligent-agents/decision-master/model.py")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find enum and dataclass imports
    enum_line = None
    dataclass_line = None
    
    for i, line in enumerate(lines):
        if 'from enum import Enum' in line:
            enum_line = i + 1
        if 'from dataclasses import dataclass' in line:
            dataclass_line = i + 1
    
    if enum_line and enum_line > 30:
        print(f"⚠️  WARNING: enum import found at line {enum_line}, should be near top")
        return False
    
    if dataclass_line and dataclass_line > 30:
        print(f"⚠️  WARNING: dataclass import found at line {dataclass_line}, should be near top")
        return False
    
    print("✅ Import structure looks good")
    return True

def validate_integration_points():
    """Validate DynamicRiskAgent integration points"""
    
    print("\n🔍 CHECKING INTEGRATION POINTS...")
    
    model_path = Path(f"{project_root}/ai-platform/ai-models/intelligent-agents/decision-master/model.py")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for DynamicRiskAgent integration
    integration_checks = [
        'from dynamic_risk_agent.model import DynamicRiskAgent',
        'self.risk_agent = DynamicRiskAgent()',
        'self.risk_agent_available',
        'await self.risk_agent.assess_trade_risk',
        'await self.risk_agent.assess_portfolio_risk'
    ]
    
    for check in integration_checks:
        if check in content:
            print(f"✅ Found: {check}")
        else:
            print(f"❌ Missing: {check}")
            return False
    
    print("✅ All integration points present")
    return True

def check_code_quality():
    """Check for potential code quality issues"""
    
    print("\n🔍 CHECKING CODE QUALITY...")
    
    model_path = Path(f"{project_root}/ai-platform/ai-models/intelligent-agents/decision-master/model.py")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check for common issues
    if 'TODO' in content:
        issues.append("Contains TODO items")
    
    if 'FIXME' in content:
        issues.append("Contains FIXME items")
    
    if 'print(' in content and 'logger' in content:
        # Check if there are print statements when logger is available
        lines = content.split('\n')
        print_lines = [i+1 for i, line in enumerate(lines) if 'print(' in line and not line.strip().startswith('#')]
        if print_lines:
            issues.append(f"Contains print statements at lines: {print_lines[:5]} (should use logger)")
    
    # Check for proper error handling
    if 'except Exception as e:' in content:
        print("✅ Error handling present")
    else:
        issues.append("Missing comprehensive error handling")
    
    # Check for async/await usage
    if 'async def' in content and 'await' in content:
        print("✅ Proper async/await usage")
    else:
        issues.append("Inconsistent async/await usage")
    
    if issues:
        print("⚠️  Code quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Code quality looks good")
        return True

def main():
    """Run comprehensive validation"""
    
    print("🚀 COMPREHENSIVE CODE VALIDATION FOR DECISIONMASTER")
    print("=" * 60)
    
    all_passed = True
    
    # Structure validation
    print("\n1️⃣ VALIDATING CODE STRUCTURE...")
    if not validate_code_structure():
        all_passed = False
    
    # Import validation  
    print("\n2️⃣ VALIDATING IMPORTS...")
    if not check_import_issues():
        all_passed = False
    
    # Integration validation
    print("\n3️⃣ VALIDATING INTEGRATION POINTS...")
    if not validate_integration_points():
        all_passed = False
    
    # Code quality validation
    print("\n4️⃣ VALIDATING CODE QUALITY...")
    if not check_code_quality():
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED! Code is ready for production.")
    else:
        print("❌ SOME VALIDATIONS FAILED! Review issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
