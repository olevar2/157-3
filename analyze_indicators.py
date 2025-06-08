"""
Analyze and consolidate indicator findings from multiple sources
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import re

class IndicatorAnalyzer:
    def __init__(self):
        self.root = Path(__file__).parent
        self.true_indicators = set()
        self.false_positives = set()
        self.consolidated_indicators = {}
        
        # Known indicator patterns
        self.valid_indicator_patterns = [
            'RSI', 'MACD', 'EMA', 'SMA', 'Bollinger', 'Stochastic',
            'ADX', 'ATR', 'CCI', 'ROC', 'Williams', 'OBV', 'MFI',
            'Ichimoku', 'Parabolic', 'Keltner', 'Donchian', 'Pivot',
            'Fibonacci', 'Gann', 'Elliott', 'Aroon', 'TRIX', 'TSI',
            'PPO', 'DPO', 'CMO', 'UO', 'Awesome', 'Chaikin', 'Vortex',
            'KAMA', 'HMA', 'DEMA', 'TEMA', 'WMA', 'VWAP', 'VWMA',
            'Accumulation', 'Distribution', 'Alligator', 'Balance',
            'Bands', 'Channel', 'Cloud', 'Convergence', 'Divergence',
            'Flow', 'Force', 'Index', 'Momentum', 'Oscillator',
            'Pattern', 'Range', 'Strength', 'Trend', 'Volume'
        ]
        
        # Known false positives to filter out
        self.false_positive_patterns = [
            'ABC', 'ArgumentIndex', 'MultiIndex', 'TimedeltaIndex',
            'DataFrame', 'Series', 'Error', 'Exception', 'Warning',
            'Test', 'Mock', 'Fixture', 'Helper', 'Util', 'Base',
            'Abstract', 'Interface', 'Mixin', 'Factory', 'Builder'
        ]
        
    def load_reports(self):
        """Load existing reports"""
        reports = {}
        
        # Load the main indicator audit report
        audit_report_path = self.root / 'indicator_audit_report.json'
        if audit_report_path.exists():
            with open(audit_report_path, 'r') as f:
                reports['audit'] = json.load(f)
                
        # Load the comprehensive report
        comprehensive_report_path = self.root / 'comprehensive_indicator_report.json'
        if comprehensive_report_path.exists():
            with open(comprehensive_report_path, 'r') as f:
                reports['comprehensive'] = json.load(f)
                
        return reports
    
    def filter_indicators(self, indicators: List[str]) -> List[str]:
        """Filter out false positives"""
        filtered = []
        
        for indicator in indicators:
            # Skip if it's a known false positive
            if any(fp in indicator for fp in self.false_positive_patterns):
                self.false_positives.add(indicator)
                continue
                
            # Keep if it matches known indicator patterns
            if any(pattern in indicator for pattern in self.valid_indicator_patterns):
                filtered.append(indicator)
                self.true_indicators.add(indicator)
            # Also keep if it ends with common indicator suffixes
            elif indicator.endswith(('Indicator', 'Oscillator', 'Index', 'Average')):
                filtered.append(indicator)
                self.true_indicators.add(indicator)
                
        return filtered
    
    def consolidate_findings(self):
        """Consolidate findings from all sources"""
        reports = self.load_reports()
        
        print("📊 Analyzing indicator findings...")
        
        # Process audit report
        if 'audit' in reports:
            audit_indicators = reports['audit'].get('all_indicators', {})
            print(f"\n✅ Audit report: {len(audit_indicators)} indicators")
            
            for name, info in audit_indicators.items():
                self.consolidated_indicators[name] = {
                    'name': name,
                    'category': info.get('category', 'unknown'),
                    'implemented': info.get('implemented', False),
                    'file': info.get('file', ''),
                    'source': 'audit'
                }
        
        # Process comprehensive report
        if 'comprehensive' in reports:
            comp_indicators = reports['comprehensive'].get('indicators', [])
            print(f"✅ Comprehensive report: {len(comp_indicators)} indicators (before filtering)")
            
            # Filter out false positives
            filtered = self.filter_indicators(comp_indicators)
            print(f"✅ After filtering: {len(filtered)} likely indicators")
            
            for indicator in filtered:
                if indicator not in self.consolidated_indicators:
                    self.consolidated_indicators[indicator] = {
                        'name': indicator,
                        'category': 'uncategorized',
                        'implemented': False,
                        'file': '',
                        'source': 'comprehensive'
                    }
        
        return self.consolidated_indicators
    
    def categorize_indicators(self):
        """Categorize uncategorized indicators"""
        for name, info in self.consolidated_indicators.items():
            if info['category'] == 'uncategorized' or info['category'] == 'unknown':
                # Try to categorize based on name
                name_lower = name.lower()
                
                if any(word in name_lower for word in ['rsi', 'macd', 'momentum', 'stoch', 'oscillator']):
                    info['category'] = 'momentum'
                elif any(word in name_lower for word in ['ma', 'average', 'trend', 'adx']):
                    info['category'] = 'trend'
                elif any(word in name_lower for word in ['volume', 'obv', 'mfi', 'flow']):
                    info['category'] = 'volume'
                elif any(word in name_lower for word in ['atr', 'volatility', 'range', 'band']):
                    info['category'] = 'volatility'
                elif any(word in name_lower for word in ['pattern', 'candlestick', 'formation']):
                    info['category'] = 'pattern'
                elif any(word in name_lower for word in ['fibonacci', 'fib']):
                    info['category'] = 'fibonacci'
                elif any(word in name_lower for word in ['gann']):
                    info['category'] = 'gann'
    
    def generate_consolidated_report(self):
        """Generate the final consolidated report"""
        # Categorize indicators
        self.categorize_indicators()
        
        # Group by category
        by_category = {}
        for name, info in self.consolidated_indicators.items():
            category = info['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(info)
        
        # Sort each category
        for category in by_category:
            by_category[category].sort(key=lambda x: x['name'])
        
        # Generate report
        report = {
            'total_indicators': len(self.consolidated_indicators),
            'implemented_count': sum(1 for info in self.consolidated_indicators.values() if info['implemented']),
            'by_category': {cat: len(indicators) for cat, indicators in by_category.items()},
            'indicators_by_category': by_category,
            'false_positives_filtered': len(self.false_positives),
            'all_indicators': self.consolidated_indicators
        }
        
        # Save report
        report_path = self.root / 'consolidated_indicator_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Consolidated report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("CONSOLIDATED INDICATOR ANALYSIS")
        print("="*60)
        
        print(f"\n✅ Total unique indicators: {len(self.consolidated_indicators)}")
        print(f"✅ Implemented indicators: {report['implemented_count']}")
        print(f"✅ False positives filtered: {len(self.false_positives)}")
        
        print("\n📊 Indicators by category:")
        for category, count in sorted(report['by_category'].items()):
            print(f"   {category}: {count} indicators")
        
        print("\n🎯 Sample indicators found:")
        # Show first 5 from each major category
        major_categories = ['momentum', 'trend', 'volume', 'volatility', 'pattern']
        for category in major_categories:
            if category in by_category and by_category[category]:
                print(f"\n   {category.upper()}:")
                for indicator in by_category[category][:5]:
                    status = "✅" if indicator['implemented'] else "❌"
                    print(f"      {status} {indicator['name']}")
                if len(by_category[category]) > 5:
                    print(f"      ... and {len(by_category[category]) - 5} more")
        
        return report

def main():
    analyzer = IndicatorAnalyzer()
    analyzer.consolidate_findings()
    analyzer.generate_consolidated_report()

if __name__ == "__main__":
    main()
