"""
Comprehensive Validation Tool for Platform3
Consolidates multiple validation and testing scripts
"""

import os
import sys
import logging
import json
import importlib
from typing import Dict, List, Any, Optional
from datetime import datetime


class Platform3Validator:
    """Comprehensive validation for Platform3 components"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.validation_results = {}
        self.logger = logging.getLogger(__name__)
        
    def validate_core_structure(self) -> Dict[str, Any]:
        """Validate core directory structure"""
        try:
            required_dirs = [
                'core',
                'core/agents',
                'core/indicators', 
                'core/services',
                'tools',
                'docs'
            ]
            
            missing_dirs = []
            existing_dirs = []
            
            for dir_path in required_dirs:
                full_path = os.path.join(self.base_path, dir_path)
                if os.path.exists(full_path):
                    existing_dirs.append(dir_path)
                else:
                    missing_dirs.append(dir_path)
            
            return {
                "structure_valid": len(missing_dirs) == 0,
                "existing_dirs": existing_dirs,
                "missing_dirs": missing_dirs,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Structure validation failed: {e}")
            return {"error": str(e)}
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate Python imports and dependencies"""
        try:
            import_results = []
            test_imports = [
                'pandas',
                'numpy', 
                'logging',
                'datetime',
                'typing'
            ]
            
            for module_name in test_imports:
                try:
                    importlib.import_module(module_name)
                    import_results.append({
                        "module": module_name,
                        "status": "success"
                    })
                except ImportError as e:
                    import_results.append({
                        "module": module_name,
                        "status": "failed",
                        "error": str(e)
                    })
            
            success_count = sum(1 for r in import_results if r["status"] == "success")
            
            return {
                "imports_valid": success_count == len(test_imports),
                "success_count": success_count,
                "total_count": len(test_imports),
                "results": import_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Import validation failed: {e}")
            return {"error": str(e)}
    
    def validate_file_integrity(self) -> Dict[str, Any]:
        """Validate file integrity and syntax"""
        try:
            python_files = []
            syntax_errors = []
            
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        python_files.append(file_path)
                        
                        # Check syntax
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                compile(content, file_path, 'exec')
                        except SyntaxError as e:
                            syntax_errors.append({
                                "file": file_path,
                                "error": str(e),
                                "line": getattr(e, 'lineno', 'unknown')
                            })
                        except Exception as e:
                            syntax_errors.append({
                                "file": file_path,
                                "error": f"Read error: {str(e)}",
                                "line": "unknown"
                            })
            
            return {
                "files_checked": len(python_files),
                "syntax_errors": len(syntax_errors),
                "syntax_valid": len(syntax_errors) == 0,
                "errors": syntax_errors,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"File validation failed: {e}")
            return {"error": str(e)}
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files"""
        try:
            config_files = [
                'requirements.txt',
                'pyproject.toml',
                'docker-compose.yml'
            ]
            
            file_status = []
            for config_file in config_files:
                file_path = os.path.join(self.base_path, config_file)
                exists = os.path.exists(file_path)
                file_status.append({
                    "file": config_file,
                    "exists": exists,
                    "path": file_path
                })
            
            valid_count = sum(1 for f in file_status if f["exists"])
            
            return {
                "config_valid": valid_count == len(config_files),
                "valid_count": valid_count,
                "total_count": len(config_files),
                "files": file_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation methods"""
        self.logger.info("Starting comprehensive Platform3 validation")
        
        results = {
            "validation_timestamp": datetime.now().isoformat(),
            "base_path": self.base_path,
            "structure": self.validate_core_structure(),
            "imports": self.validate_imports(),
            "file_integrity": self.validate_file_integrity(),
            "configuration": self.validate_configuration()
        }
        
        # Calculate overall validation status
        validations = [
            results["structure"].get("structure_valid", False),
            results["imports"].get("imports_valid", False), 
            results["file_integrity"].get("syntax_valid", False),
            results["configuration"].get("config_valid", False)
        ]
        
        results["overall_valid"] = all(validations)
        results["validation_score"] = sum(validations) / len(validations) * 100
        
        self.validation_results = results
        return results
    
    def save_validation_report(self, output_file: str = None) -> str:
        """Save validation results to file"""
        if not self.validation_results:
            self.run_comprehensive_validation()
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"validation_report_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            self.logger.info(f"Validation report saved to: {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return ""


def main():
    """Main execution function"""
    validator = Platform3Validator()
    results = validator.run_comprehensive_validation()
    
    print("=== COMPREHENSIVE PLATFORM3 VALIDATION ===")
    print(f"Validation completed: {results['validation_timestamp']}")
    print(f"Base path: {results['base_path']}")
    print(f"Overall validation: {'PASSED' if results['overall_valid'] else 'FAILED'}")
    print(f"Validation score: {results['validation_score']:.1f}%")
    print()
    
    # Print detailed results
    for category, result in results.items():
        if isinstance(result, dict) and 'timestamp' in result:
            print(f"{category.upper()}:")
            for key, value in result.items():
                if key != 'timestamp':
                    print(f"  {key}: {value}")
            print()
    
    # Save report
    report_file = validator.save_validation_report()
    if report_file:
        print(f"Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()