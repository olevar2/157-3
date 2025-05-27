#!/usr/bin/env python3
"""
Platform3 Enterprise Deployment Framework Testing Suite
Tests all enterprise deployment components and configurations
"""

import os
import json
import yaml
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Any

class EnterpriseDeploymentTester:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.test_results = {
            'files_validation': {},
            'configuration_validation': {},
            'docker_validation': {},
            'kubernetes_validation': {},
            'service_structure': {},
            'deployment_scripts': {}
        }
        
    def run_all_tests(self):
        """Run comprehensive enterprise deployment tests"""
        print("🧪 Platform3 Enterprise Deployment Testing Suite")
        print("=" * 60)
        
        # Test 1: File Structure Validation
        print("\n📁 Testing File Structure...")
        self.test_file_structure()
        
        # Test 2: Configuration Validation
        print("\n⚙️ Testing Configuration Files...")
        self.test_configuration_files()
        
        # Test 3: Docker Configuration
        print("\n🐳 Testing Docker Configurations...")
        self.test_docker_configurations()
        
        # Test 4: Kubernetes Manifests
        print("\n☸️ Testing Kubernetes Manifests...")
        self.test_kubernetes_manifests()
        
        # Test 5: Service Structure
        print("\n🎯 Testing Service Structure...")
        self.test_service_structure()
        
        # Test 6: Deployment Scripts
        print("\n📜 Testing Deployment Scripts...")
        self.test_deployment_scripts()
        
        # Test 7: CI/CD Pipeline
        print("\n🚀 Testing CI/CD Pipeline...")
        self.test_cicd_pipeline()
        
        # Generate final report
        self.generate_test_report()
        
    def test_file_structure(self):
        """Test that all required enterprise files exist"""
        required_files = [
            # Shadow Mode Service
            'services/shadow-mode-service/src/ShadowModeOrchestrator.ts',
            'services/shadow-mode-service/src/server.ts',
            'services/shadow-mode-service/package.json',
            'services/shadow-mode-service/Dockerfile',
            
            # Deployment Service
            'services/deployment-service/src/RollbackManager.ts',
            'services/deployment-service/src/server.ts',
            'services/deployment-service/package.json',
            'services/deployment-service/Dockerfile',
            
            # Monitoring Service
            'services/monitoring-service/src/PerformanceMonitor.ts',
            'services/monitoring-service/src/server.ts',
            'services/monitoring-service/package.json',
            'services/monitoring-service/Dockerfile',
            
            # Infrastructure
            '.github/workflows/platform3-enterprise-deployment.yml',
            'k8s/enterprise-deployment.yaml',
            'config/enterprise-config.yaml',
            'scripts/deploy-enterprise.sh',
            
            # Documentation
            'enterprise-deployment-framework.md',
            'ENTERPRISE_DEPLOYMENT_COMPLETE.md',
            'ENTERPRISE_DEPLOYMENT_FILES.md'
        ]
        
        for file_path in required_files:
            full_path = self.base_path / file_path
            exists = full_path.exists()
            self.test_results['files_validation'][file_path] = {
                'exists': exists,
                'size': full_path.stat().st_size if exists else 0
            }
            
            status = "✅" if exists else "❌"
            print(f"  {status} {file_path}")
            
    def test_configuration_files(self):
        """Test configuration file validity"""
        config_files = {
            'config/enterprise-config.yaml': 'yaml',
            '.github/workflows/platform3-enterprise-deployment.yml': 'yaml',
            'k8s/enterprise-deployment.yaml': 'yaml'
        }
        
        for file_path, file_type in config_files.items():
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                self.test_results['configuration_validation'][file_path] = {
                    'valid': False,
                    'error': 'File not found'
                }
                print(f"  ❌ {file_path} - File not found")
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if file_type == 'yaml':
                    parsed = yaml.safe_load(content)
                elif file_type == 'json':
                    parsed = json.loads(content)
                    
                self.test_results['configuration_validation'][file_path] = {
                    'valid': True,
                    'size': len(content),
                    'structure': type(parsed).__name__
                }
                print(f"  ✅ {file_path} - Valid {file_type.upper()}")
                
            except Exception as e:
                self.test_results['configuration_validation'][file_path] = {
                    'valid': False,
                    'error': str(e)
                }
                print(f"  ❌ {file_path} - Invalid: {e}")
                
    def test_docker_configurations(self):
        """Test Docker configuration files"""
        docker_files = [
            'services/shadow-mode-service/Dockerfile',
            'services/deployment-service/Dockerfile',
            'services/monitoring-service/Dockerfile'
        ]
        
        for dockerfile in docker_files:
            full_path = self.base_path / dockerfile
            
            if not full_path.exists():
                self.test_results['docker_validation'][dockerfile] = {
                    'valid': False,
                    'error': 'File not found'
                }
                print(f"  ❌ {dockerfile} - File not found")
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for essential Dockerfile components
                required_components = ['FROM', 'WORKDIR', 'COPY', 'RUN', 'EXPOSE', 'CMD']
                missing_components = []
                
                for component in required_components:
                    if component not in content:
                        missing_components.append(component)
                        
                self.test_results['docker_validation'][dockerfile] = {
                    'valid': len(missing_components) == 0,
                    'missing_components': missing_components,
                    'size': len(content)
                }
                
                status = "✅" if len(missing_components) == 0 else "❌"
                print(f"  {status} {dockerfile}")
                if missing_components:
                    print(f"    Missing: {', '.join(missing_components)}")
                    
            except Exception as e:
                self.test_results['docker_validation'][dockerfile] = {
                    'valid': False,
                    'error': str(e)
                }
                print(f"  ❌ {dockerfile} - Error: {e}")
                
    def test_kubernetes_manifests(self):
        """Test Kubernetes manifest validity"""
        k8s_file = self.base_path / 'k8s/enterprise-deployment.yaml'
        
        if not k8s_file.exists():
            self.test_results['kubernetes_validation']['enterprise-deployment.yaml'] = {
                'valid': False,
                'error': 'File not found'
            }
            print("  ❌ enterprise-deployment.yaml - File not found")
            return
            
        try:
            with open(k8s_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse YAML documents
            documents = list(yaml.safe_load_all(content))
            
            # Check for required Kubernetes resources
            required_kinds = ['Namespace', 'Deployment', 'Service', 'ServiceAccount']
            found_kinds = [doc.get('kind') for doc in documents if doc]
            
            missing_kinds = [kind for kind in required_kinds if kind not in found_kinds]
            
            self.test_results['kubernetes_validation']['enterprise-deployment.yaml'] = {
                'valid': len(missing_kinds) == 0,
                'documents_count': len(documents),
                'found_kinds': found_kinds,
                'missing_kinds': missing_kinds
            }
            
            status = "✅" if len(missing_kinds) == 0 else "❌"
            print(f"  {status} enterprise-deployment.yaml")
            print(f"    Documents: {len(documents)}")
            print(f"    Kinds: {', '.join(set(found_kinds))}")
            if missing_kinds:
                print(f"    Missing: {', '.join(missing_kinds)}")
                
        except Exception as e:
            self.test_results['kubernetes_validation']['enterprise-deployment.yaml'] = {
                'valid': False,
                'error': str(e)
            }
            print(f"  ❌ enterprise-deployment.yaml - Error: {e}")
            
    def test_service_structure(self):
        """Test service directory structure and package.json files"""
        services = ['shadow-mode-service', 'deployment-service', 'monitoring-service']
        
        for service in services:
            service_path = self.base_path / 'services' / service
            
            if not service_path.exists():
                self.test_results['service_structure'][service] = {
                    'valid': False,
                    'error': 'Service directory not found'
                }
                print(f"  ❌ {service} - Directory not found")
                continue
                
            # Check package.json
            package_json = service_path / 'package.json'
            if package_json.exists():
                try:
                    with open(package_json, 'r', encoding='utf-8') as f:
                        package_data = json.load(f)
                        
                    required_fields = ['name', 'version', 'scripts', 'dependencies']
                    missing_fields = [field for field in required_fields if field not in package_data]
                    
                    self.test_results['service_structure'][service] = {
                        'valid': len(missing_fields) == 0,
                        'package_json_valid': True,
                        'missing_fields': missing_fields,
                        'dependencies_count': len(package_data.get('dependencies', {}))
                    }
                    
                    status = "✅" if len(missing_fields) == 0 else "❌"
                    print(f"  {status} {service}")
                    if missing_fields:
                        print(f"    Missing fields: {', '.join(missing_fields)}")
                        
                except Exception as e:
                    self.test_results['service_structure'][service] = {
                        'valid': False,
                        'package_json_valid': False,
                        'error': str(e)
                    }
                    print(f"  ❌ {service} - package.json error: {e}")
            else:
                self.test_results['service_structure'][service] = {
                    'valid': False,
                    'package_json_valid': False,
                    'error': 'package.json not found'
                }
                print(f"  ❌ {service} - package.json not found")
                
    def test_deployment_scripts(self):
        """Test deployment script validity"""
        script_path = self.base_path / 'scripts/deploy-enterprise.sh'
        
        if not script_path.exists():
            self.test_results['deployment_scripts']['deploy-enterprise.sh'] = {
                'valid': False,
                'error': 'Script not found'
            }
            print("  ❌ deploy-enterprise.sh - File not found")
            return
            
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for essential script components
            required_components = [
                '#!/bin/bash',
                'kubectl',
                'docker',
                'namespace',
                'deployment'
            ]
            
            missing_components = []
            for component in required_components:
                if component.lower() not in content.lower():
                    missing_components.append(component)
                    
            self.test_results['deployment_scripts']['deploy-enterprise.sh'] = {
                'valid': len(missing_components) == 0,
                'missing_components': missing_components,
                'size': len(content),
                'executable': os.access(script_path, os.X_OK)
            }
            
            status = "✅" if len(missing_components) == 0 else "❌"
            print(f"  {status} deploy-enterprise.sh")
            if missing_components:
                print(f"    Missing: {', '.join(missing_components)}")
                
        except Exception as e:
            self.test_results['deployment_scripts']['deploy-enterprise.sh'] = {
                'valid': False,
                'error': str(e)
            }
            print(f"  ❌ deploy-enterprise.sh - Error: {e}")
            
    def test_cicd_pipeline(self):
        """Test CI/CD pipeline configuration"""
        pipeline_path = self.base_path / '.github/workflows/platform3-enterprise-deployment.yml'
        
        if not pipeline_path.exists():
            print("  ❌ CI/CD pipeline file not found")
            return
            
        try:
            with open(pipeline_path, 'r', encoding='utf-8') as f:
                pipeline_config = yaml.safe_load(f)
                
            # Check for essential pipeline components
            required_jobs = ['security-scan', 'build-and-test', 'test-indicators']
            found_jobs = list(pipeline_config.get('jobs', {}).keys())
            
            missing_jobs = [job for job in required_jobs if job not in found_jobs]
            
            print(f"  ✅ CI/CD Pipeline Configuration")
            print(f"    Jobs found: {len(found_jobs)}")
            print(f"    Jobs: {', '.join(found_jobs)}")
            if missing_jobs:
                print(f"    Missing recommended jobs: {', '.join(missing_jobs)}")
                
        except Exception as e:
            print(f"  ❌ CI/CD pipeline error: {e}")
            
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 ENTERPRISE DEPLOYMENT TEST REPORT")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            category_passed = 0
            category_total = len(tests)
            
            for test_name, result in tests.items():
                total_tests += 1
                if result.get('valid', False) or result.get('exists', False):
                    passed_tests += 1
                    category_passed += 1
                    
            if category_total > 0:
                success_rate = (category_passed / category_total) * 100
                print(f"\n{category.replace('_', ' ').title()}: {category_passed}/{category_total} ({success_rate:.1f}%)")
                
        overall_success = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Success Rate: {overall_success:.1f}%")
        
        if overall_success >= 90:
            print("   Status: ✅ EXCELLENT - Enterprise deployment ready!")
        elif overall_success >= 75:
            print("   Status: ✅ GOOD - Minor issues to address")
        elif overall_success >= 50:
            print("   Status: ⚠️ NEEDS WORK - Several issues found")
        else:
            print("   Status: ❌ CRITICAL - Major issues require attention")
            
        # Save detailed results
        with open(self.base_path / 'enterprise_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
            
        print(f"\n📄 Detailed results saved to: enterprise_test_results.json")

if __name__ == "__main__":
    tester = EnterpriseDeploymentTester()
    tester.run_all_tests()
