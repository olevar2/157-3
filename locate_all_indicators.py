"""
Platform3 Indicator Locator and Catalog Builder
Finds and catalogs all 115+ indicators across the platform
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import re

class IndicatorLocator:
    def __init__(self):
        self.platform_root = Path(__file__).parent
        self.indicators_found = {}
        self.indicator_categories = {
            'momentum': [],
            'trend': [],
            'volume': [],
            'volatility': [],
            'pattern': [],
            'sentiment': [],
            'statistical': [],
            'gann': [],
            'fibonacci': [],
            'custom': [],
            'ml_enhanced': [],
            'hybrid': []
        }
        
        # Extended indicator keywords for better detection
        self.indicator_keywords = [
            'indicator', 'oscillator', 'index', 'average', 'band', 
            'momentum', 'trend', 'volume', 'volatility', 'pattern',
            'rsi', 'macd', 'ema', 'sma', 'bollinger', 'stochastic',
            'adx', 'atr', 'cci', 'roc', 'williams', 'obv', 'mfi',
            'ichimoku', 'parabolic', 'keltner', 'donchian', 'pivot',
            'fibonacci', 'gann', 'elliott', 'harmonic', 'candlestick'
        ]
        
    def scan_for_indicators(self):
        """Scan the entire platform for indicator implementations"""
        engines_path = self.platform_root / 'engines'
        
        print("🔍 Scanning for indicators in Platform3...")
        print(f"Root path: {self.platform_root}")
        print(f"Engines path: {engines_path}")
        
        # Scan each category directory
        for category in self.indicator_categories.keys():
            category_path = engines_path / category
            if category_path.exists():
                self._scan_category(category, category_path)
            else:
                print(f"⚠️  Category directory not found: {category_path}")
        
        # Scan additional locations with recursive search
        self._deep_scan_platform()
        
        # Look for indicator definitions in config files
        self._scan_config_files()
        
        return self.indicators_found
    
    def _scan_category(self, category: str, path: Path):
        """Scan a category directory for indicators"""
        print(f"\n📁 Scanning {category} indicators in: {path}")
        
        for file_path in path.glob("*.py"):
            if file_path.name.startswith("__") or file_path.name.startswith("test_"):
                continue
                
            indicators = self._extract_indicators_from_file(file_path)
            if indicators:
                self.indicator_categories[category].extend(indicators)
                for indicator in indicators:
                    self.indicators_found[indicator['name']] = {
                        'category': category,
                        'file': str(file_path),
                        'class': indicator['class'],
                        'implemented': indicator['implemented'],
                        'methods': indicator.get('methods', [])
                    }
    
    def _extract_indicators_from_file(self, file_path: Path) -> List[Dict]:
        """Extract indicator classes from a Python file"""
        indicators = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # First try to parse with AST
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if self._is_indicator_class(node, content):
                            indicator_info = {
                                'name': node.name,
                                'class': node.name,
                                'implemented': self._check_implementation(node),
                                'methods': self._get_class_methods(node)
                            }
                            indicators.append(indicator_info)
            except SyntaxError:
                # If AST parsing fails, use regex fallback
                indicators.extend(self._regex_extract_indicators(content))
                
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {e}")
            
        return indicators
    
    def _regex_extract_indicators(self, content: str) -> List[Dict]:
        """Fallback regex extraction when AST parsing fails"""
        indicators = []
        
        # Pattern to find class definitions
        class_pattern = r'class\s+(\w+)(?:\s*\([^)]*\))?:'
        matches = re.finditer(class_pattern, content)
        
        for match in matches:
            class_name = match.group(1)
            if any(keyword in class_name.lower() for keyword in self.indicator_keywords):
                indicators.append({
                    'name': class_name,
                    'class': class_name,
                    'implemented': 'calculate' in content[match.start():match.start()+2000],
                    'methods': []
                })
        
        return indicators
    
    def _is_indicator_class(self, node: ast.ClassDef, content: str) -> bool:
        """Enhanced check if a class is an indicator"""
        # Check inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                if any(keyword in base.id.lower() for keyword in self.indicator_keywords):
                    return True
            elif isinstance(base, ast.Attribute):
                if any(keyword in base.attr.lower() for keyword in self.indicator_keywords):
                    return True
        
        # Check class name
        return any(keyword in node.name.lower() for keyword in self.indicator_keywords)
    
    def _check_implementation(self, node: ast.ClassDef) -> bool:
        """Check if the indicator has key methods"""
        key_methods = ['calculate', 'compute', 'update', 'process']
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in key_methods:
                return True
        return False
    
    def _get_class_methods(self, node: ast.ClassDef) -> List[str]:
        """Get list of methods in a class"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                methods.append(item.name)
        return methods
    
    def _deep_scan_platform(self):
        """Deep scan the entire platform for indicators"""
        search_patterns = [
            "**/*indicator*.py",
            "**/*oscillator*.py",
            "**/indicators/**/*.py",
            "**/engines/**/*.py",
            "**/strategies/**/*.py",
            "**/analytics/**/*.py",
            "**/ai/**/*.py"
        ]
        
        print("\n🔍 Performing deep scan for indicators...")
        
        for pattern in search_patterns:
            for py_file in self.platform_root.glob(pattern):
                # Skip test files and __pycache__
                if ('test' in py_file.name or 
                    '__pycache__' in str(py_file) or
                    py_file.name.startswith('__')):
                    continue
                
                # Skip if already scanned
                if str(py_file) in [ind['file'] for ind in self.indicators_found.values()]:
                    continue
                
                indicators = self._extract_indicators_from_file(py_file)
                for indicator in indicators:
                    if indicator['name'] not in self.indicators_found:
                        # Try to determine category from path
                        category = self._determine_category(py_file)
                        self.indicators_found[indicator['name']] = {
                            'category': category,
                            'file': str(py_file),
                            'class': indicator['class'],
                            'implemented': indicator['implemented'],
                            'methods': indicator.get('methods', [])
                        }
    
    def _determine_category(self, file_path: Path) -> str:
        """Determine indicator category from file path or name"""
        path_str = str(file_path).lower()
        
        for category in self.indicator_categories.keys():
            if category in path_str:
                return category
        
        # Check file name for category hints
        file_name = file_path.stem.lower()
        if any(word in file_name for word in ['rsi', 'macd', 'momentum', 'stoch']):
            return 'momentum'
        elif any(word in file_name for word in ['ma', 'ema', 'sma', 'trend', 'adx']):
            return 'trend'
        elif any(word in file_name for word in ['volume', 'obv', 'mfi']):
            return 'volume'
        elif any(word in file_name for word in ['atr', 'volatility', 'vix']):
            return 'volatility'
        elif any(word in file_name for word in ['pattern', 'candlestick', 'chart']):
            return 'pattern'
        
        return 'uncategorized'
    
    def _scan_config_files(self):
        """Scan configuration files for indicator definitions"""
        config_patterns = ["**/*config*.json", "**/*config*.yaml", "**/*config*.yml"]
        
        print("\n🔍 Scanning configuration files for indicator definitions...")
        
        for pattern in config_patterns:
            for config_file in self.platform_root.glob(pattern):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for indicator references in config
                    for keyword in self.indicator_keywords:
                        if keyword in content.lower():
                            # Extract potential indicator names
                            self._extract_from_config(content, config_file)
                            break
                            
                except Exception as e:
                    continue
    
    def _extract_from_config(self, content: str, config_file: Path):
        """Extract indicator references from config files"""
        # This would need to be implemented based on your config structure
        pass
    
    def generate_report(self):
        """Generate a comprehensive report of all indicators"""
        report = {
            'scan_date': datetime.now().isoformat(),
            'total_indicators_found': len(self.indicators_found),
            'implemented_count': sum(1 for ind in self.indicators_found.values() if ind['implemented']),
            'categories': {},
            'missing_indicators': [],
            'all_indicators': self.indicators_found
        }
        
        # Count by category
        for category, indicators in self.indicator_categories.items():
            report['categories'][category] = len(indicators)
        
        # List of expected indicators that might be missing
        expected_indicators = [
            'RSI', 'MACD', 'Stochastic', 'WilliamsR', 'CCI', 'TSI', 'ROC',
            'SMA', 'EMA', 'WMA', 'TEMA', 'DEMA', 'HMA', 'KAMA',
            'BollingerBands', 'ATR', 'KeltnerChannel', 'DonchianChannel',
            'OBV', 'MFI', 'VWAP', 'VolumeProfile', 'ChaikinMF',
            'ADX', 'Aroon', 'Ichimoku', 'ParabolicSAR', 'SuperTrend',
            'FibonacciRetracement', 'GannAngles', 'ElliottWave'
        ]
        
        for expected in expected_indicators:
            found = False
            for indicator_name in self.indicators_found.keys():
                if expected.lower() in indicator_name.lower():
                    found = True
                    break
            if not found:
                report['missing_indicators'].append(expected)
        
        # Save report
        report_path = self.platform_root / 'indicator_audit_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {report_path}")
        return report
    
    def print_summary(self):
        """Print a summary of findings"""
        print("\n" + "="*60)
        print("PLATFORM3 INDICATOR AUDIT SUMMARY")
        print("="*60)
        
        total = len(self.indicators_found)
        implemented = sum(1 for ind in self.indicators_found.values() if ind['implemented'])
        
        print(f"\n✅ Total indicators found: {total}")
        print(f"✅ Implemented indicators: {implemented}")
        print(f"⚠️  Stub/Unimplemented: {total - implemented}")
        
        print("\n📊 Indicators by category:")
        for category, indicators in self.indicator_categories.items():
            if indicators:
                print(f"   {category}: {len(indicators)} indicators")
        
        print("\n🔍 Sample indicators found:")
        for i, (name, info) in enumerate(list(self.indicators_found.items())[:10]):
            status = "✅" if info['implemented'] else "❌"
            print(f"   {status} {name} ({info['category']})")
            
        if total < 115:
            print(f"\n⚠️  WARNING: Expected 115+ indicators, found only {total}")
            print("   Some indicators may be in different locations or not yet implemented.")

def main():
    """Run the indicator locator"""
    locator = IndicatorLocator()
    
    # Scan for indicators
    indicators = locator.scan_for_indicators()
    
    # Generate report
    report = locator.generate_report()
    
    # Print summary
    locator.print_summary()
    
    # Provide next steps
    print("\n📝 Next Steps:")
    print("1. Review indicator_audit_report.json for detailed findings")
    print("2. Check for indicators in unexpected locations")
    print("3. Implement any missing indicators")
    print("4. Ensure all indicators follow the standard interface")
    print("5. Update the indicator registry for AI agent access")

if __name__ == "__main__":
    main()
