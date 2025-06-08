"""
Platform3 Recovery Plan Generator
Analyzes broken indicators and generates specific remediation tasks
"""

import os
import json
import sys
import importlib.util
import traceback
from typing import Dict, List, Any, Set
from datetime import datetime

class RecoveryPlanGenerator:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.engines_path = os.path.join(self.base_path, 'engines')
        
    def analyze_broken_indicators(self) -> Dict[str, Any]:
        """Analyze each broken indicator file to understand specific issues"""
        print("🔧 ANALYZING BROKEN INDICATORS FOR RECOVERY PLAN")
        print("=" * 60)
        
        categories = [
            'fractal', 'pattern', 'momentum', 'trend', 'volatility', 
            'volume', 'statistical', 'fibonacci', 'gann', 'elliott_wave', 'ml_advanced'
        ]
        
        broken_analysis = {}
        
        for category in categories:
            category_path = os.path.join(self.engines_path, category)
            if os.path.exists(category_path):
                broken_files = self._analyze_category_issues(category, category_path)
                if broken_files:
                    broken_analysis[category] = broken_files
        
        return broken_analysis
    
    def _analyze_category_issues(self, category: str, category_path: str) -> List[Dict]:
        """Analyze issues in a specific category"""
        broken_files = []
        
        # Add paths for imports
        if category_path not in sys.path:
            sys.path.insert(0, category_path)
            sys.path.insert(0, os.path.join(category_path, '..'))
        
        for file in os.listdir(category_path):
            if file.endswith('.py') and not file.startswith('__'):
                module_name = file[:-3]
                file_path = os.path.join(category_path, file)
                
                # Try to load and analyze
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Check for indicator classes
                        has_indicator_class = False
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                hasattr(attr, 'calculate') and
                                attr_name not in ['BaseIndicator', 'object']):
                                has_indicator_class = True
                                break
                        
                        if not has_indicator_class:
                            # File loads but no proper indicator class
                            issue = self._analyze_file_content(file_path)
                            broken_files.append({
                                'file': file,
                                'issue_type': 'missing_indicator_class',
                                'details': issue
                            })
                            
                except Exception as e:
                    # File doesn't load
                    issue = self._categorize_error(str(e), file_path)
                    broken_files.append({
                        'file': file,
                        'issue_type': 'import_error',
                        'error': str(e),
                        'details': issue
                    })
        
        return broken_files
    
    def _analyze_file_content(self, file_path: str) -> Dict[str, Any]:
        """Analyze file content to understand what's missing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'has_class_definition': 'class ' in content,
                'has_calculate_method': 'def calculate' in content,
                'has_base_indicator_import': 'BaseIndicator' in content,
                'has_proper_inheritance': False,
                'estimated_completion': 0
            }
            
            # Check for proper inheritance pattern
            if 'class ' in content and 'BaseIndicator' in content:
                lines = content.split('\n')
                for line in lines:
                    if 'class ' in line and 'BaseIndicator' in line:
                        analysis['has_proper_inheritance'] = True
                        break
            
            # Estimate completion percentage
            completion_score = 0
            if analysis['has_class_definition']: completion_score += 25
            if analysis['has_calculate_method']: completion_score += 25
            if analysis['has_base_indicator_import']: completion_score += 25
            if analysis['has_proper_inheritance']: completion_score += 25
            
            analysis['estimated_completion'] = completion_score
            
            return analysis
            
        except Exception as e:
            return {'error': f"Could not read file: {e}"}
    
    def _categorize_error(self, error_msg: str, file_path: str) -> Dict[str, Any]:
        """Categorize the type of error"""
        error_category = 'unknown'
        fix_complexity = 'medium'
        
        if 'No module named' in error_msg:
            error_category = 'missing_dependency'
            fix_complexity = 'easy'
        elif 'SyntaxError' in error_msg:
            error_category = 'syntax_error'
            fix_complexity = 'easy'
        elif 'ImportError' in error_msg:
            error_category = 'import_error'
            fix_complexity = 'medium'
        elif 'AttributeError' in error_msg:
            error_category = 'missing_attribute'
            fix_complexity = 'medium'
        elif 'IndentationError' in error_msg:
            error_category = 'indentation_error'
            fix_complexity = 'easy'
        
        return {
            'category': error_category,
            'fix_complexity': fix_complexity,
            'requires_external_deps': 'No module named' in error_msg
        }
    
    def generate_recovery_plan(self) -> Dict[str, Any]:
        """Generate comprehensive recovery plan"""
        broken_analysis = self.analyze_broken_indicators()
        
        # Calculate statistics
        total_broken = sum(len(files) for files in broken_analysis.values())
        
        # Categorize by fix complexity
        easy_fixes = []
        medium_fixes = []
        hard_fixes = []
        
        for category, files in broken_analysis.items():
            for file_info in files:
                fix_item = {
                    'category': category,
                    'file': file_info['file'],
                    'issue': file_info['issue_type'],
                    'details': file_info.get('details', {})
                }
                
                complexity = file_info.get('details', {}).get('fix_complexity', 'medium')
                if complexity == 'easy':
                    easy_fixes.append(fix_item)
                elif complexity == 'medium':
                    medium_fixes.append(fix_item)
                else:
                    hard_fixes.append(fix_item)
        
        # Generate action plan
        action_plan = self._generate_action_plan(easy_fixes, medium_fixes, hard_fixes)
        
        # Adaptive layer requirements
        adaptive_requirements = self._analyze_adaptive_layer_requirements()
        
        recovery_plan = {
            'timestamp': datetime.now().isoformat(),
            'current_status': {
                'working_indicators': 94,
                'broken_indicators': total_broken,
                'target_indicators': 130,
                'gap_to_close': total_broken
            },
            'broken_indicator_analysis': broken_analysis,
            'fix_complexity_breakdown': {
                'easy_fixes': len(easy_fixes),
                'medium_fixes': len(medium_fixes),
                'hard_fixes': len(hard_fixes)
            },
            'prioritized_action_plan': action_plan,
            'adaptive_layer_requirements': adaptive_requirements,
            'estimated_timeline': self._estimate_timeline(easy_fixes, medium_fixes, hard_fixes),
            'resource_requirements': self._calculate_resource_requirements()
        }
        
        return recovery_plan
    
    def _generate_action_plan(self, easy_fixes: List, medium_fixes: List, hard_fixes: List) -> Dict[str, Any]:
        """Generate prioritized action plan"""
        return {
            'phase_1_immediate': {
                'description': 'Fix easy syntax and import errors',
                'tasks': easy_fixes[:10],  # Top 10 easy fixes
                'estimated_effort': '1-2 days',
                'impact': 'Quick wins, immediate indicator recovery'
            },
            'phase_2_medium': {
                'description': 'Complete missing implementations',
                'tasks': medium_fixes,
                'estimated_effort': '1-2 weeks',
                'impact': 'Major indicator recovery, system stability'
            },
            'phase_3_advanced': {
                'description': 'Implement complex missing indicators',
                'tasks': hard_fixes,
                'estimated_effort': '2-4 weeks',
                'impact': 'Full feature completion'
            },
            'phase_4_optimization': {
                'description': 'Optimize adaptive layer and genius agent integration',
                'tasks': [
                    'Update adaptive bridge registry with all 130 indicators',
                    'Enhance genius agent indicator mappings',
                    'Implement performance optimization for 130 indicators',
                    'Add real-time indicator health monitoring'
                ],
                'estimated_effort': '1-2 weeks',
                'impact': 'Maximum system performance and reliability'
            }
        }
    
    def _analyze_adaptive_layer_requirements(self) -> Dict[str, Any]:
        """Analyze what's needed for full adaptive layer integration"""
        return {
            'current_adaptive_registry': 8,  # From adaptive_indicators.py
            'target_adaptive_registry': 130,
            'genius_agent_mappings_needed': {
                'risk_genius': 'Needs access to all volatility, statistical, and correlation indicators',
                'pattern_master': 'Needs all pattern, fractal, and elliott_wave indicators',
                'session_expert': 'Needs time-based and session-specific indicators',
                'execution_expert': 'Needs volume and microstructure indicators',
                'pair_specialist': 'Needs correlation and statistical indicators',
                'decision_master': 'Needs aggregated signals from all indicators',
                'ai_model_coordinator': 'Needs ML-enhanced indicators',
                'market_microstructure_genius': 'Needs volume and tick-level indicators',
                'sentiment_integration_genius': 'Needs sentiment and news-based indicators'
            },
            'required_updates': [
                'Expand indicator_registry in adaptive_indicator_bridge.py to include all 130 indicators',
                'Update agent_indicator_mapping for all 9 genius agents',
                'Implement adaptive parameter adjustment for all indicator types',
                'Add performance caching for 130 indicators',
                'Create indicator correlation matrix for optimization',
                'Implement regime-based indicator selection algorithms'
            ]
        }
    
    def _estimate_timeline(self, easy_fixes: List, medium_fixes: List, hard_fixes: List) -> Dict[str, str]:
        """Estimate implementation timeline"""
        return {
            'phase_1_immediate': '1-2 days (10 easy fixes)',
            'phase_2_medium': f'1-2 weeks ({len(medium_fixes)} medium complexity)',
            'phase_3_advanced': f'2-4 weeks ({len(hard_fixes)} complex indicators)',
            'phase_4_optimization': '1-2 weeks (adaptive layer enhancement)',
            'total_timeline': '6-10 weeks for complete implementation',
            'quick_impact_milestone': '70% improvement in 2 weeks (phases 1-2)',
            'full_completion': '100% indicator coverage in 6-10 weeks'
        }
    
    def _calculate_resource_requirements(self) -> Dict[str, Any]:
        """Calculate required resources"""
        return {
            'human_resources': {
                'senior_developer': '2-3 weeks full-time for complex indicators',
                'mid_developer': '3-4 weeks for medium complexity fixes',
                'junior_developer': '1 week for syntax/import fixes'
            },
            'technical_requirements': [
                'Access to financial data APIs for testing',
                'Mathematical libraries (numpy, scipy, pandas)',
                'Machine learning libraries for ML indicators',
                'Comprehensive testing framework setup',
                'Continuous integration pipeline for indicator validation'
            ],
            'testing_requirements': [
                'Historical market data for backtesting',
                'Unit tests for each indicator class',
                'Integration tests for adaptive layer',
                'Performance benchmarking suite',
                'Genius agent validation framework'
            ]
        }
    
    def print_summary(self, recovery_plan: Dict) -> None:
        """Print executive summary of recovery plan"""
        print("\n📋 RECOVERY PLAN EXECUTIVE SUMMARY")
        print("=" * 50)
        
        status = recovery_plan['current_status']
        print(f"Current Status: {status['working_indicators']}/{status['target_indicators']} indicators working")
        print(f"Recovery Gap: {status['gap_to_close']} indicators need fixing")
        
        breakdown = recovery_plan['fix_complexity_breakdown']
        print(f"\nFix Complexity:")
        print(f"  Easy fixes: {breakdown['easy_fixes']} indicators")
        print(f"  Medium fixes: {breakdown['medium_fixes']} indicators")
        print(f"  Hard fixes: {breakdown['hard_fixes']} indicators")
        
        timeline = recovery_plan['estimated_timeline']
        print(f"\nTimeline:")
        print(f"  Quick impact (70%): {timeline['quick_impact_milestone']}")
        print(f"  Full completion: {timeline['full_completion']}")
        
        print(f"\n🎯 KEY PHASES:")
        for phase_name, phase_info in recovery_plan['prioritized_action_plan'].items():
            print(f"  {phase_name}: {phase_info['estimated_effort']}")
        
        print(f"\n🔧 ADAPTIVE LAYER UPDATES NEEDED:")
        adaptive_reqs = recovery_plan['adaptive_layer_requirements']
        print(f"  Registry expansion: {adaptive_reqs['current_adaptive_registry']} → {adaptive_reqs['target_adaptive_registry']} indicators")
        print(f"  Genius agent mappings: {len(adaptive_reqs['genius_agent_mappings_needed'])} agents need updates")
        print(f"  Core updates: {len(adaptive_reqs['required_updates'])} major changes")
        
    def save_recovery_plan(self, recovery_plan: Dict) -> str:
        """Save detailed recovery plan"""
        filename = f"platform3_recovery_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(recovery_plan, f, indent=2, ensure_ascii=False)
        return filename

if __name__ == "__main__":
    generator = RecoveryPlanGenerator()
    recovery_plan = generator.generate_recovery_plan()
    generator.print_summary(recovery_plan)
    filename = generator.save_recovery_plan(recovery_plan)
    print(f"\n📄 Detailed recovery plan saved: {filename}")
