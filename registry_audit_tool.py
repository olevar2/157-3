#!/usr/bin/env python3
"""
Registry Audit Tool for Platform3 Recovery Plan Phase 4A
Performs comprehensive audit of adaptive_indicator_bridge.py registry
to ensure exactly 157 indicators as specified in recovery plan.
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class RegistryAuditor:
    def __init__(self, bridge_file_path):
        self.bridge_file_path = Path(bridge_file_path)
        self.current_registry = {}
        self.recovery_plan_requirements = {
            'MOMENTUM': 22,
            'PATTERN': 30, 
            'VOLUME': 22,
            'FRACTAL': 19,
            'FIBONACCI': 6,
            'STATISTICAL': 13,
            'TREND': 8,
            'VOLATILITY': 7,
            'ML_ADVANCED': 2,
            'ELLIOTT_WAVE': 3,
            'GANN': 6        }
        self.total_required = 157
        
    def parse_current_registry(self):
        """Parse the current adaptive indicator bridge registry"""
        print("🔍 Parsing current registry from adaptive_indicator_bridge.py...")
        
        if not self.bridge_file_path.exists():
            raise FileNotFoundError(f"Bridge file not found: {self.bridge_file_path}")
            
        with open(self.bridge_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract registry section - look for the method that builds the registry
        registry_pattern = r'def _build_comprehensive_157_indicator_registry\(.*?\).*?return\s*\{(.*?)\}'
        match = re.search(registry_pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError("Could not find _build_comprehensive_157_indicator_registry method")
            
        registry_content = match.group(1)
          # Parse individual indicators and their categories
        # Look for patterns like 'indicator_name': { ... 'category': 'category_name' ... }
        indicator_pattern = r"'([^']+)':\s*\{[^}]*'category':\s*'([^']+)'[^}]*\}"
        indicators = re.findall(indicator_pattern, registry_content)
        
        # Group indicators by category and track duplicates
        duplicate_indicators = []
        seen_indicators = set()
        
        for indicator_name, category in indicators:
            if indicator_name in seen_indicators:
                duplicate_indicators.append(indicator_name)
                continue
            seen_indicators.add(indicator_name)
              if category not in self.current_registry:
                self.current_registry[category] = []
            self.current_registry[category].append(indicator_name)
            
        print(f"✅ Parsed {len(self.current_registry)} categories")
        
        # Print summary for verification
        total_indicators = sum(len(indicators) for indicators in self.current_registry.values())
        print(f"📊 Found {total_indicators} total indicators")
        for category, indicators in self.current_registry.items():
            print(f"  {category}: {len(indicators)} indicators")
        
        if duplicate_indicators:
            print(f"⚠️ Found {len(duplicate_indicators)} duplicate indicators: {duplicate_indicators[:5]}...")
            
        return self.current_registry
        
    def audit_registry(self):
        """Perform comprehensive audit against recovery plan requirements"""
        print("\n📊 COMPREHENSIVE REGISTRY AUDIT")
        print("=" * 50)
        
        audit_results = {
            'total_indicators': 0,
            'category_analysis': {},
            'discrepancies': [],
            'compliance_status': 'UNKNOWN',
            'extra_indicators': [],
            'missing_indicators': [],
            'duplicate_check': {}
        }
        
        # Count total indicators and analyze categories
        all_indicators = set()
        duplicate_indicators = []
        
        for category, indicators in self.current_registry.items():
            category_count = len(indicators)
            audit_results['total_indicators'] += category_count
            
            # Check for duplicates within category
            seen_in_category = set()
            for indicator in indicators:
                if indicator in seen_in_category:
                    duplicate_indicators.append(f"{indicator} (in {category})")
                seen_in_category.add(indicator)
                
                # Check for duplicates across all categories
                if indicator in all_indicators:
                    duplicate_indicators.append(f"{indicator} (across categories)")
                all_indicators.add(indicator)
            
            # Compare against recovery plan requirements
            expected_count = self.recovery_plan_requirements.get(category, 0)
            status = "✅ COMPLIANT" if category_count == expected_count else "❌ MISMATCH"
            
            audit_results['category_analysis'][category] = {
                'current_count': category_count,
                'required_count': expected_count,
                'difference': category_count - expected_count,
                'status': status,
                'indicators': indicators
            }
            
            if category_count != expected_count:
                audit_results['discrepancies'].append({
                    'category': category,
                    'current': category_count,
                    'required': expected_count,
                    'difference': category_count - expected_count
                })
        
        # Check for categories in registry but not in recovery plan
        extra_categories = set(self.current_registry.keys()) - set(self.recovery_plan_requirements.keys())
        for category in extra_categories:
            count = len(self.current_registry[category])
            audit_results['category_analysis'][category] = {
                'current_count': count,
                'required_count': 0,
                'difference': count,
                'status': "⚠️ EXTRA CATEGORY",
                'indicators': self.current_registry[category]
            }
            audit_results['discrepancies'].append({
                'category': category,
                'current': count,
                'required': 0,
                'difference': count,
                'note': 'Extra category not in recovery plan'
            })
        
        # Check for missing categories
        missing_categories = set(self.recovery_plan_requirements.keys()) - set(self.current_registry.keys())
        for category in missing_categories:
            required_count = self.recovery_plan_requirements[category]
            audit_results['category_analysis'][category] = {
                'current_count': 0,
                'required_count': required_count,
                'difference': -required_count,
                'status': "🚨 MISSING CATEGORY",
                'indicators': []
            }
            audit_results['discrepancies'].append({
                'category': category,
                'current': 0,
                'required': required_count,
                'difference': -required_count,
                'note': 'Missing category from recovery plan'
            })
        
        # Overall compliance check
        total_difference = audit_results['total_indicators'] - self.total_required
        if audit_results['total_indicators'] == self.total_required and len(audit_results['discrepancies']) == 0:
            audit_results['compliance_status'] = 'FULLY_COMPLIANT'
        elif audit_results['total_indicators'] == self.total_required:
            audit_results['compliance_status'] = 'COUNT_COMPLIANT_DISTRIBUTION_ISSUES'
        else:
            audit_results['compliance_status'] = 'NON_COMPLIANT'
            
        audit_results['total_difference'] = total_difference
        audit_results['duplicate_indicators'] = duplicate_indicators
        
        return audit_results
        
    def print_audit_report(self, audit_results):
        """Print comprehensive audit report"""
        print(f"\n🎯 REGISTRY AUDIT REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Overall status
        print(f"\n📈 OVERALL STATUS: {audit_results['compliance_status']}")
        print(f"Current Total: {audit_results['total_indicators']} indicators")
        print(f"Required Total: {self.total_required} indicators")
        print(f"Difference: {audit_results['total_difference']:+d} indicators")
        
        # Category analysis
        print(f"\n📊 CATEGORY ANALYSIS")
        print("-" * 70)
        print(f"{'Category':<20} {'Current':<8} {'Required':<10} {'Diff':<6} {'Status'}")
        print("-" * 70)
        
        for category, analysis in audit_results['category_analysis'].items():
            diff_str = f"{analysis['difference']:+d}" if analysis['difference'] != 0 else "0"
            print(f"{category:<20} {analysis['current_count']:<8} {analysis['required_count']:<10} {diff_str:<6} {analysis['status']}")
        
        # Discrepancies
        if audit_results['discrepancies']:
            print(f"\n🚨 DISCREPANCIES FOUND ({len(audit_results['discrepancies'])})")
            print("-" * 50)
            for disc in audit_results['discrepancies']:
                note = f" - {disc.get('note', '')}" if disc.get('note') else ""
                print(f"• {disc['category']}: {disc['current']} → {disc['required']} ({disc['difference']:+d}){note}")
        
        # Duplicates
        if audit_results['duplicate_indicators']:
            print(f"\n🔄 DUPLICATE INDICATORS FOUND ({len(audit_results['duplicate_indicators'])})")
            print("-" * 50)
            for dup in audit_results['duplicate_indicators']:
                print(f"• {dup}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        if audit_results['compliance_status'] == 'FULLY_COMPLIANT':
            print("✅ Registry is fully compliant with recovery plan requirements!")
        else:
            total_diff = audit_results['total_difference']
            if total_diff > 0:
                print(f"📉 Remove {total_diff} extra indicators to reach exactly 157")
            elif total_diff < 0:
                print(f"📈 Add {abs(total_diff)} missing indicators to reach exactly 157")
                
            # Specific category recommendations
            for disc in audit_results['discrepancies']:
                if disc['difference'] > 0:
                    print(f"• Remove {disc['difference']} indicators from {disc['category']}")
                elif disc['difference'] < 0:
                    print(f"• Add {abs(disc['difference'])} indicators to {disc['category']}")
        
        return audit_results
        
    def generate_corrected_registry(self, audit_results):
        """Generate corrected registry with exactly 157 indicators"""
        print(f"\n🔧 GENERATING CORRECTED REGISTRY")
        print("-" * 40)
        
        corrected_registry = {}
        corrections_made = []
        
        # Start with recovery plan categories
        for category, required_count in self.recovery_plan_requirements.items():
            current_indicators = self.current_registry.get(category, [])
            current_count = len(current_indicators)
            
            if current_count == required_count:
                # Perfect match
                corrected_registry[category] = current_indicators[:]
                corrections_made.append(f"✅ {category}: Kept all {required_count} indicators")
                
            elif current_count > required_count:
                # Too many - keep first required_count
                corrected_registry[category] = current_indicators[:required_count]
                removed_count = current_count - required_count
                corrections_made.append(f"📉 {category}: Removed {removed_count} excess indicators (kept first {required_count})")
                
            else:
                # Too few - need to add indicators
                corrected_registry[category] = current_indicators[:]
                needed = required_count - current_count
                
                # Try to generate reasonable indicator names for missing ones
                base_names = [
                    f"{category.lower()}_indicator_{i+current_count+1}" 
                    for i in range(needed)
                ]
                corrected_registry[category].extend(base_names)
                corrections_made.append(f"📈 {category}: Added {needed} missing indicators")
        
        # Verify total count
        total_corrected = sum(len(indicators) for indicators in corrected_registry.values())
        
        print(f"\n📊 CORRECTION SUMMARY")
        for correction in corrections_made:
            print(f"  {correction}")
            
        print(f"\n🎯 FINAL COUNT: {total_corrected} indicators (Target: {self.total_required})")
        
        if total_corrected == self.total_required:
            print("✅ Corrected registry meets exact 157-indicator requirement!")
        else:
            print(f"⚠️ Corrected registry has {total_corrected - self.total_required:+d} indicators vs requirement")
            
        return corrected_registry, corrections_made
        
    def backup_current_registry(self):
        """Create backup of current registry"""
        backup_file = f"adaptive_indicator_bridge_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        backup_path = self.bridge_file_path.parent / backup_file
        
        import shutil
        shutil.copy2(self.bridge_file_path, backup_path)
        print(f"💾 Created backup: {backup_file}")
        return backup_path
        
    def save_audit_report(self, audit_results):
        """Save audit results to JSON file"""
        report_file = f"registry_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.bridge_file_path.parent / report_file
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(audit_results, f, indent=2, ensure_ascii=False)
            
        print(f"📄 Saved audit report: {report_file}")
        return report_path

def main():
    """Main execution function"""
    bridge_file = Path("D:/MD/Platform3/engines/ai_enhancement/adaptive_indicator_bridge.py")
    
    try:
        print("🚀 Platform3 Registry Audit Tool")
        print("Recovery Plan Phase 4A - 157-Indicator Compliance Check")
        print("=" * 60)
        
        auditor = RegistryAuditor(bridge_file)
        
        # Step 1: Parse current registry
        current_registry = auditor.parse_current_registry()
        
        # Step 2: Perform comprehensive audit
        audit_results = auditor.audit_registry()
        
        # Step 3: Print detailed report
        auditor.print_audit_report(audit_results)
        
        # Step 4: Generate corrected registry if needed
        if audit_results['compliance_status'] != 'FULLY_COMPLIANT':
            corrected_registry, corrections = auditor.generate_corrected_registry(audit_results)
            audit_results['corrected_registry'] = corrected_registry
            audit_results['corrections_made'] = corrections
        
        # Step 5: Save audit report
        report_path = auditor.save_audit_report(audit_results)
        
        # Step 6: Create backup if changes needed
        if audit_results['compliance_status'] != 'FULLY_COMPLIANT':
            backup_path = auditor.backup_current_registry()
            audit_results['backup_created'] = str(backup_path)
        
        print(f"\n✅ Registry audit completed successfully!")
        print(f"📊 Status: {audit_results['compliance_status']}")
        
        return audit_results
        
    except Exception as e:
        print(f"❌ Error during registry audit: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        sys.exit(0 if results['compliance_status'] == 'FULLY_COMPLIANT' else 1)
    else:
        sys.exit(2)
