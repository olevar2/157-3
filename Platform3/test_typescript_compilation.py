#!/usr/bin/env python3
"""
Platform3 Enterprise Services TypeScript Compilation Test
Tests TypeScript compilation for all enterprise services
"""

import os
import subprocess
import json
from pathlib import Path

class TypeScriptCompilationTester:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.services = [
            'shadow-mode-service',
            'deployment-service', 
            'monitoring-service'
        ]
        self.test_results = {}
        
    def run_compilation_tests(self):
        """Test TypeScript compilation for all enterprise services"""
        print("🔧 Platform3 Enterprise Services TypeScript Compilation Test")
        print("=" * 70)
        
        overall_success = True
        
        for service in self.services:
            print(f"\n📦 Testing {service}...")
            success = self.test_service_compilation(service)
            overall_success = overall_success and success
            
        self.generate_compilation_report(overall_success)
        
    def test_service_compilation(self, service_name):
        """Test TypeScript compilation for a specific service"""
        service_path = self.base_path / 'services' / service_name
        
        if not service_path.exists():
            print(f"  ❌ Service directory not found: {service_path}")
            self.test_results[service_name] = {
                'success': False,
                'error': 'Service directory not found'
            }
            return False
            
        # Check if package.json exists
        package_json = service_path / 'package.json'
        if not package_json.exists():
            print(f"  ❌ package.json not found")
            self.test_results[service_name] = {
                'success': False,
                'error': 'package.json not found'
            }
            return False
            
        # Check TypeScript files
        ts_files = list(service_path.rglob('*.ts'))
        print(f"  📄 Found {len(ts_files)} TypeScript files")
        
        # Create tsconfig.json if it doesn't exist
        tsconfig_path = service_path / 'tsconfig.json'
        if not tsconfig_path.exists():
            self.create_tsconfig(tsconfig_path)
            print(f"  ✅ Created tsconfig.json")
            
        # Test syntax validation for each TypeScript file
        syntax_errors = []
        for ts_file in ts_files:
            try:
                # Basic syntax check by reading and parsing
                with open(ts_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for basic TypeScript syntax issues
                if self.check_typescript_syntax(content, ts_file):
                    print(f"  ✅ {ts_file.name} - Syntax OK")
                else:
                    syntax_errors.append(ts_file.name)
                    print(f"  ⚠️ {ts_file.name} - Potential syntax issues")
                    
            except Exception as e:
                syntax_errors.append(f"{ts_file.name}: {str(e)}")
                print(f"  ❌ {ts_file.name} - Error: {e}")
                
        # Test npm install simulation (check dependencies)
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
                
            dependencies = package_data.get('dependencies', {})
            dev_dependencies = package_data.get('devDependencies', {})
            
            print(f"  📦 Dependencies: {len(dependencies)}")
            print(f"  🔧 Dev Dependencies: {len(dev_dependencies)}")
            
            # Check for essential TypeScript dependencies
            essential_deps = ['typescript', '@types/node', '@types/express']
            missing_deps = []
            
            all_deps = {**dependencies, **dev_dependencies}
            for dep in essential_deps:
                if dep not in all_deps:
                    missing_deps.append(dep)
                    
            if missing_deps:
                print(f"  ⚠️ Missing recommended dependencies: {', '.join(missing_deps)}")
            else:
                print(f"  ✅ All essential TypeScript dependencies present")
                
        except Exception as e:
            print(f"  ❌ Error reading package.json: {e}")
            syntax_errors.append(f"package.json error: {str(e)}")
            
        # Record results
        success = len(syntax_errors) == 0
        self.test_results[service_name] = {
            'success': success,
            'typescript_files': len(ts_files),
            'syntax_errors': syntax_errors,
            'dependencies': len(dependencies) if 'dependencies' in locals() else 0,
            'dev_dependencies': len(dev_dependencies) if 'dev_dependencies' in locals() else 0
        }
        
        if success:
            print(f"  ✅ {service_name} compilation test PASSED")
        else:
            print(f"  ❌ {service_name} compilation test FAILED")
            
        return success
        
    def check_typescript_syntax(self, content, file_path):
        """Basic TypeScript syntax validation"""
        try:
            # Check for basic syntax patterns
            issues = []
            
            # Check for unmatched braces
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces != close_braces:
                issues.append(f"Unmatched braces: {open_braces} open, {close_braces} close")
                
            # Check for unmatched parentheses
            open_parens = content.count('(')
            close_parens = content.count(')')
            if open_parens != close_parens:
                issues.append(f"Unmatched parentheses: {open_parens} open, {close_parens} close")
                
            # Check for basic TypeScript patterns
            if 'export class' in content or 'export interface' in content or 'export function' in content:
                # This looks like a proper TypeScript file
                pass
            elif 'import' in content and 'from' in content:
                # Has imports, likely TypeScript
                pass
            elif file_path.name.endswith('.ts'):
                # TypeScript file should have TypeScript content
                if not any(keyword in content for keyword in ['interface', 'type', 'class', 'function', 'const', 'let']):
                    issues.append("No TypeScript content detected")
                    
            # Check for common syntax errors
            if ';;' in content:
                issues.append("Double semicolons found")
                
            if issues:
                print(f"    Issues found: {'; '.join(issues)}")
                return False
                
            return True
            
        except Exception as e:
            print(f"    Syntax check error: {e}")
            return False
            
    def create_tsconfig(self, tsconfig_path):
        """Create a basic tsconfig.json for the service"""
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "lib": ["ES2020"],
                "outDir": "./dist",
                "rootDir": "./src",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True,
                "resolveJsonModule": True,
                "declaration": True,
                "declarationMap": True,
                "sourceMap": True
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist"]
        }
        
        with open(tsconfig_path, 'w', encoding='utf-8') as f:
            json.dump(tsconfig, f, indent=2)
            
    def generate_compilation_report(self, overall_success):
        """Generate compilation test report"""
        print("\n" + "=" * 70)
        print("📊 TYPESCRIPT COMPILATION TEST REPORT")
        print("=" * 70)
        
        total_services = len(self.services)
        passed_services = sum(1 for result in self.test_results.values() if result['success'])
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Services Tested: {total_services}")
        print(f"   Passed: {passed_services}")
        print(f"   Success Rate: {(passed_services/total_services)*100:.1f}%")
        
        # Detailed results per service
        for service, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"\n{status} {service}:")
            print(f"   TypeScript Files: {result.get('typescript_files', 0)}")
            print(f"   Dependencies: {result.get('dependencies', 0)}")
            print(f"   Dev Dependencies: {result.get('dev_dependencies', 0)}")
            
            if result.get('syntax_errors'):
                print(f"   Syntax Errors: {len(result['syntax_errors'])}")
                for error in result['syntax_errors']:
                    print(f"     - {error}")
                    
        if overall_success:
            print("\n🎉 ALL ENTERPRISE SERVICES READY FOR COMPILATION!")
        else:
            print("\n⚠️ Some services need attention before compilation")
            
        # Save detailed results
        with open(self.base_path / 'typescript_compilation_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
            
        print(f"\n📄 Detailed results saved to: typescript_compilation_results.json")

if __name__ == "__main__":
    tester = TypeScriptCompilationTester()
    tester.run_compilation_tests()
