"""
Final 157 Indicator Registry Expansion
Complete the adaptive indicator bridge with all available indicators
"""

import json
import os
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path

def discover_all_indicators():
    """Discover all available indicators across the platform"""
    indicators = {}
    base_path = Path("engines")
    
    categories = [
        "advanced", "ai_enhancement", "core_momentum", "core_trend", 
        "cycle", "divergence", "elliott_wave", "fibonacci", "fractal",
        "gann", "ml_advanced", "momentum", "pattern", "pivot",
        "sentiment", "statistical", "trend", "volatility", "volume"
    ]
    
    for category in categories:
        category_path = base_path / category
        if not category_path.exists():
            continue
            
        indicators[category] = []
        
        # Scan all Python files in the category
        for py_file in category_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                # Try to import and inspect the module
                module_name = f"engines.{category}.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find all classes that could be indicators
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (not name.startswith("_") and 
                            name not in ["Enum", "ABC", "Exception"] and
                            hasattr(obj, "__module__") and
                            obj.__module__ == module_name):
                            
                            indicators[category].append({
                                "name": name,
                                "file": py_file.name,
                                "module": module_name,
                                "class_name": name
                            })
                            
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
                continue
    
    return indicators

def generate_expanded_registry():
    """Generate the expanded registry code for adaptive_indicator_bridge.py"""
    indicators = discover_all_indicators()
    
    # Define agent priorities and mappings
    agent_mappings = {
        'fractal': ['RISK_GENIUS', 'PATTERN_MASTER', 'AI_MODEL_COORDINATOR'],
        'volume': ['SESSION_EXPERT', 'MARKET_MICROSTRUCTURE_GENIUS', 'EXECUTION_EXPERT'],
        'pattern': ['PATTERN_MASTER', 'DECISION_MASTER', 'AI_MODEL_COORDINATOR'],
        'fibonacci': ['PATTERN_MASTER', 'RISK_GENIUS', 'AI_MODEL_COORDINATOR'],
        'statistical': ['AI_MODEL_COORDINATOR', 'RISK_GENIUS', 'PAIR_SPECIALIST'],
        'momentum': ['EXECUTION_EXPERT', 'SESSION_EXPERT', 'DECISION_MASTER'],
        'trend': ['SESSION_EXPERT', 'DECISION_MASTER', 'PAIR_SPECIALIST'],
        'volatility': ['RISK_GENIUS', 'MARKET_MICROSTRUCTURE_GENIUS', 'AI_MODEL_COORDINATOR'],
        'ml_advanced': ['AI_MODEL_COORDINATOR', 'SENTIMENT_INTEGRATION_GENIUS', 'RISK_GENIUS'],
        'cycle': ['PATTERN_MASTER', 'SESSION_EXPERT', 'AI_MODEL_COORDINATOR'],
        'divergence': ['PATTERN_MASTER', 'EXECUTION_EXPERT', 'DECISION_MASTER'],
        'core_momentum': ['EXECUTION_EXPERT', 'SESSION_EXPERT', 'DECISION_MASTER'],
        'core_trend': ['SESSION_EXPERT', 'DECISION_MASTER', 'PAIR_SPECIALIST'],
        'elliott_wave': ['PATTERN_MASTER', 'AI_MODEL_COORDINATOR', 'DECISION_MASTER'],
        'gann': ['PATTERN_MASTER', 'SESSION_EXPERT', 'AI_MODEL_COORDINATOR'],
        'pivot': ['SESSION_EXPERT', 'EXECUTION_EXPERT', 'DECISION_MASTER'],
        'sentiment': ['SENTIMENT_INTEGRATION_GENIUS', 'AI_MODEL_COORDINATOR', 'DECISION_MASTER'],
        'advanced': ['AI_MODEL_COORDINATOR', 'RISK_GENIUS', 'MARKET_MICROSTRUCTURE_GENIUS'],
        'ai_enhancement': ['AI_MODEL_COORDINATOR', 'SENTIMENT_INTEGRATION_GENIUS', 'MARKET_MICROSTRUCTURE_GENIUS']
    }
    
    registry_code = ""
    total_count = 0
    
    for category, category_indicators in indicators.items():
        if not category_indicators:
            continue
            
        count = len(category_indicators)
        total_count += count
        
        registry_code += f"            # ====== {category.upper()} INDICATORS ({count} indicators) ======\n"
        
        for indicator in category_indicators:
            name = indicator['name']
            key = name.lower().replace('indicator', '').replace('calculator', '').replace('analysis', '').replace('analyzer', '')
            if key.endswith('_'):
                key = key[:-1]
            
            agents = agent_mappings.get(category, ['AI_MODEL_COORDINATOR', 'RISK_GENIUS'])
            agents_str = ', '.join([f'GeniusAgentType.{agent}' for agent in agents])
            
            registry_code += f"            '{key}': {{\n"
            registry_code += f"                'module': '{indicator['module']}',\n"
            registry_code += f"                'category': '{category}',\n"
            registry_code += f"                'agents': [{agents_str}],\n"
            registry_code += f"                'priority': 2,\n"
            registry_code += f"                'class_name': '{indicator['class_name']}'\n"
            registry_code += f"            }},\n"
        
        registry_code += "\n"
    
    return registry_code, total_count, indicators

def main():
    """Main execution function"""
    print("🚀 Final 157 Indicator Registry Expansion")
    print("=" * 50)
    
    # Discover all indicators
    print("📊 Discovering all available indicators...")
    registry_code, total_count, indicators = generate_expanded_registry()
    
    print(f"✅ Found {total_count} indicators across {len(indicators)} categories")
    
    # Save discovery report
    report = {
        "timestamp": str(datetime.now()),
        "total_indicators": total_count,
        "categories": indicators,
        "registry_code": registry_code
    }
    
    with open("final_157_indicator_discovery.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Show category breakdown
    print("\n📋 Category Breakdown:")
    for category, category_indicators in indicators.items():
        if category_indicators:
            print(f"  {category}: {len(category_indicators)} indicators")
    
    # Save the registry code to a file for easy insertion
    with open("expanded_registry_code.py", "w") as f:
        f.write("# Expanded Registry Code for adaptive_indicator_bridge.py\n")
        f.write("# Replace _build_indicator_registry return statement with this:\n\n")
        f.write("return {\n")
        f.write(registry_code)
        f.write("        }\n")
    
    print(f"\n✅ Registry expansion complete!")
    print(f"📁 Files created:")
    print(f"   - final_157_indicator_discovery.json (discovery report)")
    print(f"   - expanded_registry_code.py (registry code)")
    print(f"📊 Total indicators ready for integration: {total_count}")
    
    return total_count >= 157

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 SUCCESS: 157+ indicator target achieved!")
    else:
        print("\n⚠️  Target not reached, expanding existing categories...")
