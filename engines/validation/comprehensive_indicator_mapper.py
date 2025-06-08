# -*- coding: utf-8 -*-
"""
Platform3 Comprehensive Indicator Mapping System
===============================================

Systematically maps all indicator files to their actual class names
to resolve import path discrepancies across all 12 indicator categories.

For the humanitarian forex trading platform mission.
"""

import os
import ast
import sys
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Add the engines directory to the path
engines_path = Path(__file__).parent.parent
sys.path.insert(0, str(engines_path))

class IndicatorMapper:
    """Maps indicator files to their actual class names and validates imports."""
    
    def __init__(self, engines_root: str = None):
        if engines_root is None:
            self.engines_root = Path(__file__).parent.parent
        else:
            self.engines_root = Path(engines_root)
        
        # Define all indicator categories
        self.categories = [
            'momentum', 'trend', 'volume', 'volatility', 'statistical',
            'pattern', 'sentiment', 'fibonacci', 'elliott_wave', 'gann',
            'cycle', 'pivot'
        ]
        
        self.mapping = {}
        self.import_results = {}
        self.errors = {}
    
    def extract_classes_from_file(self, file_path: Path) -> List[str]:
        """Extract class names from a Python file using AST parsing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)            
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Only include classes that likely inherit from TechnicalIndicator
                    # or contain "Indicator" in the name
                    if ('Indicator' in node.name or 
                        'Oscillator' in node.name or
                        'Bands' in node.name or
                        'Channels' in node.name or
                        'SAR' in node.name or
                        'ADX' in node.name or
                        'MACD' in node.name or
                        'EMA' in node.name or
                        'SMA' in node.name or
                        'RSI' in node.name or
                        'CCI' in node.name or
                        'MFI' in node.name or
                        'ROC' in node.name or
                        'Williams' in node.name or
                        'Ultimate' in node.name or
                        'Awesome' in node.name or
                        'Bollinger' in node.name or
                        'Donchian' in node.name or
                        'Keltner' in node.name or
                        'Parabolic' in node.name or
                        'Aroon' in node.name or
                        'Vortex' in node.name or
                        'ATR' in node.name or
                        'DMS' in node.name or
                        'VWAP' in node.name or
                        'OBV' in node.name or
                        'Volume' in node.name or
                        'Profile' in node.name or
                        'Flow' in node.name or
                        'Accumulation' in node.name or
                        'Distribution' in node.name or
                        'Volatility' in node.name):
                        classes.append(node.name)
            
            return classes
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def scan_category(self, category: str) -> Dict[str, List[str]]:
        """Scan a category directory for indicator files and their classes."""
        category_path = self.engines_root / category
        if not category_path.exists():
            return {}
        
        file_class_mapping = {}
        
        # Get all Python files (excluding backups and __pycache__)
        python_files = [f for f in category_path.glob("*.py") 
                       if not f.name.startswith('__') 
                       and not 'backup' in f.name.lower()
                       and not 'unicode' in f.name.lower()]        
        for py_file in python_files:
            classes = self.extract_classes_from_file(py_file)
            if classes:
                file_class_mapping[py_file.stem] = classes
        
        return file_class_mapping
    
    def create_comprehensive_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """Create comprehensive mapping of all categories."""
        print("🔍 Scanning all indicator categories...")
        
        for category in self.categories:
            print(f"  📁 Scanning {category}...")
            self.mapping[category] = self.scan_category(category)
            
            # Count files and classes
            file_count = len(self.mapping[category])
            class_count = sum(len(classes) for classes in self.mapping[category].values())
            print(f"    ✅ Found {file_count} files with {class_count} classes")
        
        return self.mapping
    
    def test_imports(self) -> Dict[str, Dict[str, bool]]:
        """Test import success for all discovered classes."""
        print("\n🧪 Testing imports for all discovered classes...")
        
        for category, file_mapping in self.mapping.items():
            print(f"  📦 Testing {category} imports...")
            self.import_results[category] = {}
            
            for filename, classes in file_mapping.items():
                for class_name in classes:
                    try:
                        # Try to import the class
                        module_path = f"engines.{category}.{filename}"
                        exec(f"from {module_path} import {class_name}")
                        self.import_results[category][f"{filename}.{class_name}"] = True
                        print(f"    ✅ {filename}.{class_name}")
                        
                    except Exception as e:
                        self.import_results[category][f"{filename}.{class_name}"] = False
                        self.errors[f"{category}.{filename}.{class_name}"] = str(e)
                        print(f"    ❌ {filename}.{class_name}: {e}")
        
        return self.import_results    
    def generate_import_report(self) -> str:
        """Generate comprehensive import success report."""
        report = []
        report.append("=" * 80)
        report.append("PLATFORM3 INDICATOR IMPORT MAPPING REPORT")
        report.append("=" * 80)
        report.append(f"Generated for humanitarian forex trading platform")
        report.append(f"Total categories scanned: {len(self.categories)}")
        report.append("")
        
        # Summary statistics
        total_files = sum(len(file_mapping) for file_mapping in self.mapping.values())
        total_classes = sum(sum(len(classes) for classes in file_mapping.values()) 
                           for file_mapping in self.mapping.values())
        
        successful_imports = sum(sum(1 for success in category_results.values() if success)
                               for category_results in self.import_results.values())
        
        failed_imports = sum(sum(1 for success in category_results.values() if not success)
                           for category_results in self.import_results.values())
        
        success_rate = (successful_imports / (successful_imports + failed_imports)) * 100 if (successful_imports + failed_imports) > 0 else 0
        
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total indicator files: {total_files}")
        report.append(f"Total indicator classes: {total_classes}")
        report.append(f"Successful imports: {successful_imports}")
        report.append(f"Failed imports: {failed_imports}")
        report.append(f"Success rate: {success_rate:.1f}%")
        report.append("")
        
        # Category breakdown
        report.append("📁 CATEGORY BREAKDOWN")
        report.append("-" * 40)
        
        for category in self.categories:
            if category in self.mapping:
                file_count = len(self.mapping[category])
                class_count = sum(len(classes) for classes in self.mapping[category].values())
                
                if category in self.import_results:
                    success_count = sum(1 for success in self.import_results[category].values() if success)
                    fail_count = sum(1 for success in self.import_results[category].values() if not success)
                    cat_success_rate = (success_count / (success_count + fail_count)) * 100 if (success_count + fail_count) > 0 else 0
                    
                    report.append(f"{category:15} | Files: {file_count:2d} | Classes: {class_count:2d} | Success: {success_count:2d}/{success_count + fail_count:2d} ({cat_success_rate:5.1f}%)")
                else:
                    report.append(f"{category:15} | Files: {file_count:2d} | Classes: {class_count:2d} | Not tested")
        
        report.append("")        
        # Detailed mapping
        report.append("🗂️  DETAILED FILE-TO-CLASS MAPPING")
        report.append("-" * 40)
        
        for category, file_mapping in self.mapping.items():
            if file_mapping:
                report.append(f"\n[{category.upper()}]")
                for filename, classes in file_mapping.items():
                    report.append(f"  📄 {filename}.py:")
                    for class_name in classes:
                        import_key = f"{filename}.{class_name}"
                        if category in self.import_results and import_key in self.import_results[category]:
                            status = "✅" if self.import_results[category][import_key] else "❌"
                            report.append(f"    {status} {class_name}")
                        else:
                            report.append(f"    ❓ {class_name}")
        
        # Error details
        if self.errors:
            report.append("\n❌ IMPORT ERRORS")
            report.append("-" * 40)
            for error_key, error_msg in self.errors.items():
                report.append(f"{error_key}: {error_msg}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def generate_import_fixes(self) -> str:
        """Generate import fix suggestions."""
        fixes = []
        fixes.append("🔧 IMPORT FIX SUGGESTIONS")
        fixes.append("=" * 40)
        
        # Analyze common error patterns
        missing_files = []
        dependency_errors = []
        syntax_errors = []
        
        for error_key, error_msg in self.errors.items():
            if "No module named" in error_msg:
                missing_files.append(error_key)
            elif "TickVolumeIndicators" in error_msg:
                dependency_errors.append(error_key)
            elif "invalid syntax" in error_msg or "SyntaxError" in error_msg:
                syntax_errors.append(error_key)
        
        if missing_files:
            fixes.append("\n📁 MISSING FILES/MODULES:")
            for missing in missing_files:
                fixes.append(f"  - {missing}")
        
        if dependency_errors:
            fixes.append("\n🔗 DEPENDENCY ISSUES:")
            for dep_error in dependency_errors:
                fixes.append(f"  - {dep_error} (TickVolumeIndicators issue)")
        
        if syntax_errors:
            fixes.append("\n⚠️  SYNTAX ERRORS:")
            for syntax_error in syntax_errors:
                fixes.append(f"  - {syntax_error}")
        
        return "\n".join(fixes)
def main():
    """Main execution function."""
    print("🚀 Platform3 Comprehensive Indicator Mapping")
    print("=" * 50)
    
    # Initialize mapper
    mapper = IndicatorMapper()
    
    # Create comprehensive mapping
    mapping = mapper.create_comprehensive_mapping()
    
    # Test all imports
    import_results = mapper.test_imports()
    
    # Generate reports
    report = mapper.generate_import_report()
    fixes = mapper.generate_import_fixes()
    
    # Save reports to files
    report_file = Path(__file__).parent / "indicator_mapping_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n\n")
        f.write(fixes)
    
    print(f"\n📄 Report saved to: {report_file}")
    print("\n" + "=" * 50)
    print("✅ Comprehensive mapping complete!")
    
    return mapper, mapping, import_results

if __name__ == "__main__":
    mapper, mapping, results = main()