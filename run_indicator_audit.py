#!/usr/bin/env python3
"""
Run the Comprehensive Indicator Audit and Validation System
Platform3 - Humanitarian Trading System

This script runs the indicator audit system to validate all 115+ indicators
and their integration with AI agents and TypeScript trading engine.
"""

import asyncio
import json
from pathlib import Path
from engines.validation.indicator_audit_system import ComprehensiveIndicatorAuditSystem

async def run_audit():
    """Run the comprehensive audit"""
    print("=== PLATFORM3 COMPREHENSIVE INDICATOR AUDIT ===")
    print("Validating 115+ indicators across 12 categories...")
    print("Checking AI integration and TypeScript bridge...")
    
    # Initialize and run the audit system
    audit_system = ComprehensiveIndicatorAuditSystem()
    results = await audit_system.run_full_audit()
    
    # Print summary
    print("\n=== AUDIT RESULTS ===")
    print(f"Total Indicators: {results['summary']['total_indicators']}")
    print(f"Passed Indicators: {results['summary']['passed_indicators']}")
    print(f"Pass Rate: {results['summary']['pass_rate']:.1%}")
    print(f"Average Performance: {results['summary']['average_performance_ms']:.1f}ms")
    print(f"AI Integration: {results['summary']['ai_integration_status']}")
    print(f"TypeScript Integration: {results['summary']['typescript_integration_status']}")
    
    # Print recommendations
    print("\n=== RECOMMENDATIONS ===")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    print(f"\nDetailed report saved to: {Path('reports/indicator_audit_report.json').absolute()}")
    return results

if __name__ == "__main__":
    results = asyncio.run(run_audit())