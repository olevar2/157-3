#!/usr/bin/env python3
"""
Platform3 Indicator Loadability Analysis
Systematic analysis of 290 discovered indicator classes vs validation results

This script cross-references:
1. Enhanced indicator discovery report (290 classes)
2. Comprehensive validation results (execution status)  
3. Identifies exact mapping between discovered classes and their execution status
4. Categorizes failure types and provides actionable remediation plans
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from datetime import datetime

def load_enhanced_discovery_report(file_path: str) -> Dict[str, Any]:
    """Load the enhanced indicator discovery report"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading enhanced discovery report: {e}")
        return {}

def load_validation_results(file_path: str) -> Dict[str, Any]:
    """Load the comprehensive validation results"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading validation results: {e}")
        return {}

def normalize_indicator_name(name: str) -> str:
    """Normalize indicator names for comparison"""
    return name.lower().replace('_', '').replace('-', '')

def categorize_execution_errors(execution_details: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Categorize execution errors by type for pattern analysis"""
    error_categories = {
        'dataframe_ambiguity': [],
        'missing_attributes': [],
        'constructor_issues': [],
        'abstract_class_issues': [],
        'insufficient_data': [],
        'missing_methods': [],
        'parameter_mismatches': [],
        'other_errors': []
    }
    
    for indicator, details in execution_details.items():
        if details['status'] == 'FAILED':
            error_msg = details.get('error', '').lower()
            
            if 'dataframe is ambiguous' in error_msg:
                error_categories['dataframe_ambiguity'].append(indicator)
            elif 'has no attribute' in error_msg:
                error_categories['missing_attributes'].append(indicator)
            elif 'unexpected keyword argument' in error_msg:
                error_categories['constructor_issues'].append(indicator)
            elif "can't instantiate abstract class" in error_msg:
                error_categories['abstract_class_issues'].append(indicator)
            elif 'insufficient data' in error_msg:
                error_categories['insufficient_data'].append(indicator)
            elif 'missing' in error_msg and 'required' in error_msg:
                error_categories['parameter_mismatches'].append(indicator)
            elif 'has no attribute' in error_msg and 'method' in error_msg:
                error_categories['missing_methods'].append(indicator)
            else:
                error_categories['other_errors'].append(indicator)
    
    return error_categories

def cross_reference_indicators(discovery_report: Dict, validation_results: Dict) -> Dict[str, Any]:
    """Cross-reference discovered indicators with validation results"""
    discovered_indicators = {}
    
    # Extract all discovered indicators with metadata
    for category, indicators in discovery_report.get('detailed_results', {}).items():
        for indicator in indicators:
            name = indicator['name']
            normalized_name = normalize_indicator_name(name)
            discovered_indicators[normalized_name] = {
                'original_name': name,
                'category': category,
                'file': indicator['file'],
                'path': indicator['path'],
                'validation_status': 'NOT_TESTED',
                'execution_status': 'UNKNOWN',
                'error_details': None
            }
    
    # Cross-reference with validation results
    execution_details = validation_results.get('indicator_execution', {}).get('execution_details', {})
    missing_indicators = validation_results.get('indicator_registry', {}).get('missing_indicators', [])
    
    # Map execution results
    for indicator_key, exec_details in execution_details.items():
        normalized_key = normalize_indicator_name(indicator_key)
        if normalized_key in discovered_indicators:
            discovered_indicators[normalized_key]['validation_status'] = 'TESTED'
            discovered_indicators[normalized_key]['execution_status'] = exec_details['status']
            if exec_details['status'] == 'FAILED':
                discovered_indicators[normalized_key]['error_details'] = exec_details.get('error')
    
    # Mark missing indicators
    for missing_indicator in missing_indicators:
        # Extract base name from module path (e.g., "momentum.knowsurething" -> "knowsurething")
        base_name = missing_indicator.split('.')[-1]
        normalized_missing = normalize_indicator_name(base_name)
        if normalized_missing in discovered_indicators:
            discovered_indicators[normalized_missing]['validation_status'] = 'MISSING_FROM_REGISTRY'
    
    return discovered_indicators

def generate_comprehensive_analysis() -> Dict[str, Any]:
    """Generate comprehensive loadability analysis"""
    
    # File paths
    discovery_report_path = "enhanced_indicator_discovery_report.json"
    validation_results_path = "comprehensive_validation_results.json"
    
    # Load reports
    discovery_report = load_enhanced_discovery_report(discovery_report_path)
    validation_results = load_validation_results(validation_results_path)
    
    if not discovery_report or not validation_results:
        return {"error": "Failed to load required reports"}
    
    # Cross-reference indicators
    cross_referenced = cross_reference_indicators(discovery_report, validation_results)
    
    # Categorize execution errors
    execution_details = validation_results.get('indicator_execution', {}).get('execution_details', {})
    error_categories = categorize_execution_errors(execution_details)
    
    # Generate statistics
    total_discovered = len(cross_referenced)
    tested_indicators = sum(1 for i in cross_referenced.values() if i['validation_status'] == 'TESTED')
    successful_indicators = sum(1 for i in cross_referenced.values() if i['execution_status'] == 'SUCCESS')
    failed_indicators = sum(1 for i in cross_referenced.values() if i['execution_status'] == 'FAILED')
    missing_from_registry = sum(1 for i in cross_referenced.values() if i['validation_status'] == 'MISSING_FROM_REGISTRY')
    not_tested = sum(1 for i in cross_referenced.values() if i['validation_status'] == 'NOT_TESTED')
    
    # Category breakdown
    category_analysis = {}
    for indicator_data in cross_referenced.values():
        category = indicator_data['category']
        if category not in category_analysis:
            category_analysis[category] = {
                'total': 0,
                'tested': 0,
                'successful': 0,
                'failed': 0,
                'missing_from_registry': 0,
                'not_tested': 0
            }
        
        category_analysis[category]['total'] += 1
        if indicator_data['validation_status'] == 'TESTED':
            category_analysis[category]['tested'] += 1
            if indicator_data['execution_status'] == 'SUCCESS':
                category_analysis[category]['successful'] += 1
            elif indicator_data['execution_status'] == 'FAILED':
                category_analysis[category]['failed'] += 1
        elif indicator_data['validation_status'] == 'MISSING_FROM_REGISTRY':
            category_analysis[category]['missing_from_registry'] += 1
        else:
            category_analysis[category]['not_tested'] += 1
    
    return {
        'timestamp': datetime.now().isoformat(),
        'analysis_metadata': {
            'discovery_report_indicators': discovery_report.get('total_indicators', 0),
            'validation_tested_indicators': len(execution_details),
            'cross_reference_matches': total_discovered
        },
        'overall_statistics': {
            'total_discovered_indicators': total_discovered,
            'indicators_tested': tested_indicators,
            'successful_executions': successful_indicators,
            'failed_executions': failed_indicators,
            'missing_from_registry': missing_from_registry,
            'not_tested': not_tested,
            'success_rate_of_tested': round((successful_indicators / tested_indicators * 100) if tested_indicators > 0 else 0, 2),
            'registry_coverage': round((tested_indicators / total_discovered * 100) if total_discovered > 0 else 0, 2)
        },
        'error_analysis': {
            'error_categories': error_categories,
            'error_counts': {category: len(indicators) for category, indicators in error_categories.items()},
            'most_common_errors': sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        },
        'category_breakdown': category_analysis,
        'detailed_cross_reference': cross_referenced,
        'remediation_priorities': {
            'high_priority': {
                'description': 'Indicators discovered but missing from registry',
                'count': missing_from_registry,
                'action': 'Add to indicator registry for testing'
            },
            'medium_priority': {
                'description': 'Indicators in registry but failing execution',
                'count': failed_indicators,
                'action': 'Fix execution errors by category'
            },
            'low_priority': {
                'description': 'Indicators discovered but never tested',
                'count': not_tested,
                'action': 'Add to validation test suite'
            }
        }
    }

def main():
    """Main execution function"""
    print("=== Platform3 Indicator Loadability Analysis ===")
    print("Cross-referencing 290 discovered indicators with validation results...")
    
    try:
        analysis_results = generate_comprehensive_analysis()
        
        if 'error' in analysis_results:
            print(f"Error: {analysis_results['error']}")
            return
        
        # Save detailed results
        output_file = 'indicator_loadability_analysis_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        stats = analysis_results['overall_statistics']
        print(f"\n=== SUMMARY RESULTS ===")
        print(f"Total Discovered Indicators: {stats['total_discovered_indicators']}")
        print(f"Indicators Tested: {stats['indicators_tested']}")
        print(f"Successful Executions: {stats['successful_executions']}")
        print(f"Failed Executions: {stats['failed_executions']}")
        print(f"Missing from Registry: {stats['missing_from_registry']}")
        print(f"Not Tested: {stats['not_tested']}")
        print(f"Success Rate (of tested): {stats['success_rate_of_tested']}%")
        print(f"Registry Coverage: {stats['registry_coverage']}%")
        
        print(f"\n=== ERROR ANALYSIS ===")
        error_counts = analysis_results['error_analysis']['error_counts']
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"{error_type.replace('_', ' ').title()}: {count} indicators")
        
        print(f"\n=== REMEDIATION PRIORITIES ===")
        for priority, details in analysis_results['remediation_priorities'].items():
            print(f"{priority.upper()}: {details['description']} ({details['count']} indicators)")
            print(f"  Action: {details['action']}")
        
        print(f"\nDetailed analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()