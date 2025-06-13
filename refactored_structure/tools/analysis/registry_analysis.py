"""
Registry Analysis Tool
Consolidates registry validation and analysis functionality
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegistryAnalyzer:
    """Comprehensive registry analysis and validation"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.registry_data = {}
        self.logger = logging.getLogger(__name__)
        
    def scan_registry_files(self) -> Dict[str, Any]:
        """Scan for all registry-related files"""
        registry_files = []
        registry_patterns = ['registry', 'REGISTRY', 'Registry']
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if any(pattern in file for pattern in registry_patterns):
                    file_path = os.path.join(root, file)
                    registry_files.append({
                        "file": file_path,
                        "name": file,
                        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat() if os.path.exists(file_path) else None
                    })
        
        return {
            "registry_files": registry_files,
            "count": len(registry_files),
            "scan_timestamp": datetime.now().isoformat()
        }
    
    def analyze_registry_completeness(self) -> Dict[str, Any]:
        """Analyze registry completeness and consistency"""
        try:
            # Look for indicator registry patterns
            indicator_patterns = []
            class_definitions = []
            
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.split('\n')
                                
                                for i, line in enumerate(lines):
                                    # Look for registry patterns
                                    if 'INDICATOR_REGISTRY' in line or 'indicator_registry' in line:
                                        indicator_patterns.append({
                                            "file": file_path,
                                            "line": i + 1,
                                            "content": line.strip()
                                        })
                                    
                                    # Look for class definitions
                                    if line.strip().startswith('class ') and any(term in line for term in ['Indicator', 'Agent']):
                                        class_name = line.split('class ')[1].split('(')[0].strip()
                                        class_definitions.append({
                                            "class": class_name,
                                            "file": file_path,
                                            "line": i + 1
                                        })
                        except Exception as e:
                            continue
            
            return {
                "indicator_registry_references": len(indicator_patterns),
                "class_definitions_found": len(class_definitions),
                "registry_patterns": indicator_patterns,
                "classes": class_definitions,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Registry completeness analysis failed: {e}")
            return {"error": str(e)}
    
    def find_registry_inconsistencies(self) -> Dict[str, Any]:
        """Find inconsistencies in registry implementations"""
        try:
            inconsistencies = []
            registry_formats = {}
            
            # Scan for different registry formats
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if file.endswith('.py') and 'registry' in file.lower():
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # Check for different registry patterns
                                if 'INDICATOR_REGISTRY = {' in content:
                                    registry_formats['dict_format'] = registry_formats.get('dict_format', 0) + 1
                                elif 'class IndicatorRegistry' in content:
                                    registry_formats['class_format'] = registry_formats.get('class_format', 0) + 1
                                elif 'def register_indicator' in content:
                                    registry_formats['function_format'] = registry_formats.get('function_format', 0) + 1
                                
                        except Exception:
                            continue
            
            return {
                "registry_formats": registry_formats,
                "format_count": len(registry_formats),
                "inconsistencies": inconsistencies,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Registry inconsistency analysis failed: {e}")
            return {"error": str(e)}
    
    def validate_registry_structure(self) -> Dict[str, Any]:
        """Validate registry structure and organization"""
        validation_results = {
            "valid_registries": [],
            "invalid_registries": [],
            "validation_errors": [],
            "recommendations": []
        }
        
        try:
            registry_scan = self.scan_registry_files()
            
            for registry_file in registry_scan['registry_files']:
                file_path = registry_file['file']
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Basic validation checks
                        has_registry_var = 'INDICATOR_REGISTRY' in content or 'indicator_registry' in content
                        has_imports = 'import' in content
                        has_classes = 'class ' in content
                        
                        if has_registry_var and has_imports:
                            validation_results['valid_registries'].append(file_path)
                        else:
                            validation_results['invalid_registries'].append({
                                "file": file_path,
                                "issues": {
                                    "missing_registry_var": not has_registry_var,
                                    "missing_imports": not has_imports,
                                    "missing_classes": not has_classes
                                }
                            })
                            
                except Exception as e:
                    validation_results['validation_errors'].append({
                        "file": file_path,
                        "error": str(e)
                    })
            
            # Generate recommendations
            if len(validation_results['invalid_registries']) > 0:
                validation_results['recommendations'].append("Fix invalid registry files")
            if len(validation_results['validation_errors']) > 0:
                validation_results['recommendations'].append("Resolve validation errors")
            
            validation_results['validation_timestamp'] = datetime.now().isoformat()
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Registry validation failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_registry_analysis(self) -> Dict[str, Any]:
        """Run all registry analysis methods"""
        self.logger.info("Starting comprehensive registry analysis")
        
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "base_path": self.base_path,
            "file_scan": self.scan_registry_files(),
            "completeness": self.analyze_registry_completeness(),
            "inconsistencies": self.find_registry_inconsistencies(),
            "validation": self.validate_registry_structure()
        }
        
        return results


def main():
    """Main execution function"""
    analyzer = RegistryAnalyzer()
    results = analyzer.run_comprehensive_registry_analysis()
    
    print("=== COMPREHENSIVE REGISTRY ANALYSIS ===")
    print(f"Analysis completed: {results['analysis_timestamp']}")
    print(f"Base path: {results['base_path']}")
    print()
    
    # Print summary
    if 'file_scan' in results:
        print(f"Registry files found: {results['file_scan'].get('count', 0)}")
    
    if 'completeness' in results:
        completeness = results['completeness']
        print(f"Registry references: {completeness.get('indicator_registry_references', 0)}")
        print(f"Class definitions: {completeness.get('class_definitions_found', 0)}")
    
    if 'validation' in results:
        validation = results['validation']
        print(f"Valid registries: {len(validation.get('valid_registries', []))}")
        print(f"Invalid registries: {len(validation.get('invalid_registries', []))}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"registry_analysis_report_{timestamp}.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")


if __name__ == "__main__":
    main()