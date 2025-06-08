"""
Comprehensive indicator finder using multiple search strategies
"""

import os
import re
from pathlib import Path
import json
from typing import Set, Dict, List
from datetime import datetime

class ComprehensiveIndicatorFinder:
    def __init__(self):
        self.root = Path(__file__).parent
        self.all_indicators = set()
        self.indicator_locations = {}
        self.file_count = 0
        self.indicators_by_type = {
            'classes': set(),
            'mentions': set(),
            'imports': set()
        }
        
    def find_all(self):
        """Use multiple strategies to find indicators"""
        print("🔍 Starting comprehensive indicator search...")
        print(f"📍 Root directory: {self.root}")
        
        # Strategy 1: Search by file names
        self._search_by_filename()
        
        # Strategy 2: Search by content patterns
        self._search_by_content()
        
        # Strategy 3: Search import statements
        self._search_imports()
        
        # Strategy 4: Search test files for indicator usage
        self._search_test_files()
        
        # Strategy 5: Search documentation
        self._search_documentation()
        
        # Combine all findings
        self.all_indicators = (
            self.indicators_by_type['classes'] |
            self.indicators_by_type['mentions'] |
            self.indicators_by_type['imports']
        )
        
        return self.all_indicators, self.indicator_locations
    
    def _search_by_filename(self):
        """Find files that might contain indicators"""
        print("\n📁 Searching by filename patterns...")
        
        patterns = [
            '**/*indicator*.py', '**/*oscillator*.py', '**/*index*.py',
            '**/*average*.py', '**/*band*.py', '**/*pattern*.py',
            '**/engines/**/*.py', '**/indicators/**/*.py'
        ]
        
        files_found = 0
        for pattern in patterns:
            for file in self.root.glob(pattern):
                if '__pycache__' not in str(file) and file.is_file():
                    files_found += 1
                    self._scan_file(file)
        
        print(f"   📄 Scanned {files_found} files matching patterns")
    
    def _search_by_content(self):
        """Search file contents for indicator patterns"""
        print("\n📄 Searching by content patterns...")
        
        # Common indicator names and variations
        indicator_patterns = [
            # Momentum indicators
            (r'\b(RSI|RelativeStrengthIndex)\b', 'RSI'),
            (r'\b(MACD|MovingAverageConvergenceDivergence)\b', 'MACD'),
            (r'\b(Stochastic|StochasticOscillator)\b', 'Stochastic'),
            (r'\b(Williams|WilliamsR|WilliamsPercentR)\b', 'WilliamsR'),
            (r'\b(CCI|CommodityChannelIndex)\b', 'CCI'),
            (r'\b(ROC|RateOfChange)\b', 'ROC'),
            (r'\b(TSI|TrueStrengthIndex)\b', 'TSI'),
            (r'\b(UO|UltimateOscillator)\b', 'UltimateOscillator'),
            
            # Trend indicators
            (r'\b(SMA|SimpleMovingAverage)\b', 'SMA'),
            (r'\b(EMA|ExponentialMovingAverage)\b', 'EMA'),
            (r'\b(WMA|WeightedMovingAverage)\b', 'WMA'),
            (r'\b(TEMA|TripleExponentialMovingAverage)\b', 'TEMA'),
            (r'\b(DEMA|DoubleExponentialMovingAverage)\b', 'DEMA'),
            (r'\b(HMA|HullMovingAverage)\b', 'HMA'),
            (r'\b(KAMA|KaufmanAdaptiveMovingAverage)\b', 'KAMA'),
            (r'\b(ADX|AverageDirectionalIndex)\b', 'ADX'),
            (r'\b(Aroon|AroonIndicator)\b', 'Aroon'),
            
            # Volume indicators
            (r'\b(OBV|OnBalanceVolume)\b', 'OBV'),
            (r'\b(MFI|MoneyFlowIndex)\b', 'MFI'),
            (r'\b(VWAP|VolumeWeightedAveragePrice)\b', 'VWAP'),
            (r'\b(CMF|ChaikinMoneyFlow)\b', 'ChaikinMoneyFlow'),
            
            # Volatility indicators
            (r'\b(ATR|AverageTrueRange)\b', 'ATR'),
            (r'\b(Bollinger|BollingerBands)\b', 'BollingerBands'),
            (r'\b(Keltner|KeltnerChannel)\b', 'KeltnerChannel'),
            (r'\b(Donchian|DonchianChannel)\b', 'DonchianChannel'),
            
            # Others
            (r'\b(Fibonacci|FibonacciRetracement)\b', 'Fibonacci'),
            (r'\b(Gann|GannAngles)\b', 'Gann'),
            (r'\b(Ichimoku|IchimokuCloud)\b', 'Ichimoku'),
            (r'\b(ParabolicSAR|PSAR)\b', 'ParabolicSAR'),
        ]
        
        files_scanned = 0
        matches_found = 0
        
        for py_file in self.root.rglob("*.py"):
            if '__pycache__' in str(py_file) or not py_file.is_file():
                continue
            
            files_scanned += 1
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for pattern, indicator_name in indicator_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        matches_found += 1
                        self.indicators_by_type['mentions'].add(indicator_name)
                        if indicator_name not in self.indicator_locations:
                            self.indicator_locations[indicator_name] = []
                        self.indicator_locations[indicator_name].append(str(py_file))
                        
            except Exception as e:
                continue
        
        print(f"   📄 Scanned {files_scanned} Python files")
        print(f"   ✅ Found {matches_found} indicator mentions")
        print(f"   🎯 Unique indicators: {len(self.indicators_by_type['mentions'])}")
    
    def _search_imports(self):
        """Search for indicator imports"""
        print("\n📦 Searching import statements...")
        
        import_patterns = [
            re.compile(r'from\s+[\w.]+\s+import\s+([\w\s,]+)'),
            re.compile(r'import\s+([\w.]+)')
        ]
        
        imports_found = 0
        
        for py_file in self.root.rglob("*.py"):
            if '__pycache__' in str(py_file) or not py_file.is_file():
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for pattern in import_patterns:
                    matches = pattern.findall(content)
                    for match in matches:
                        if isinstance(match, str):
                            imports = [match]
                        else:
                            imports = [imp.strip() for imp in match.split(',')]
                        
                        for imp in imports:
                            if any(keyword in imp.lower() for keyword in ['indicator', 'oscillator', 'index', 'average']):
                                imports_found += 1
                                # Extract the class name
                                class_name = imp.split('.')[-1]
                                self.indicators_by_type['imports'].add(class_name)
                                
            except Exception:
                continue
        
        print(f"   📦 Found {imports_found} indicator-related imports")
        print(f"   🎯 Unique imported indicators: {len(self.indicators_by_type['imports'])}")
    
    def _search_test_files(self):
        """Search test files for indicator usage"""
        print("\n🧪 Searching test files...")
        
        test_files_found = 0
        for test_file in self.root.rglob("*test*.py"):
            if test_file.is_file():
                test_files_found += 1
                self._scan_file(test_file)
        
        print(f"   🧪 Scanned {test_files_found} test files")
    
    def _search_documentation(self):
        """Search documentation for indicator mentions"""
        print("\n📚 Searching documentation...")
        
        doc_patterns = ['*.md', '*.rst', '*.txt', '*.json']
        docs_found = 0
        
        for pattern in doc_patterns:
            for doc_file in self.root.rglob(pattern):
                if doc_file.is_file():
                    docs_found += 1
                    try:
                        with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Look for indicator lists or mentions
                        if 'indicator' in content.lower():
                            # Find capitalized words that might be indicators
                            words = re.findall(r'\b[A-Z][a-zA-Z]+(?:Indicator|Oscillator|Index|Average)?\b', content)
                            for word in words:
                                if len(word) > 2 and any(kw in word for kw in ['Indicator', 'Oscillator', 'Index', 'Average', 'RSI', 'MACD', 'EMA', 'SMA']):
                                    self.indicators_by_type['mentions'].add(word)
                                    
                    except Exception:
                        continue
        
        print(f"   📚 Scanned {docs_found} documentation files")
    
    def _scan_file(self, file_path: Path):
        """Scan a file for indicators"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Look for class definitions
            class_pattern = re.compile(r'class\s+(\w+)(?:\([^)]*\))?:')
            matches = class_pattern.findall(content)
            
            for match in matches:
                # Check if it's likely an indicator
                if any(keyword in match.lower() for keyword in ['indicator', 'oscillator', 'index', 'average', 'band', 'channel']):
                    self.indicators_by_type['classes'].add(match)
                    if match not in self.indicator_locations:
                        self.indicator_locations[match] = []
                    self.indicator_locations[match].append(str(file_path))
                    
        except Exception:
            pass
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = {
            'total_indicators_found': len(self.all_indicators),
            'indicators': sorted(list(self.all_indicators)),
            'locations': self.indicator_locations,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('comprehensive_indicator_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n✅ Found {len(self.all_indicators)} unique indicators total")
        print(f"   📊 Classes: {len(self.indicators_by_type['classes'])}")
        print(f"   📝 Mentions: {len(self.indicators_by_type['mentions'])}")
        print(f"   📦 Imports: {len(self.indicators_by_type['imports'])}")
        
        # Print summary
        print("\n🎯 Sample indicators found:")
        for indicator in sorted(list(self.all_indicators))[:20]:
            print(f"   - {indicator}")
            
        if len(self.all_indicators) > 20:
            print(f"   ... and {len(self.all_indicators) - 20} more")

if __name__ == "__main__":
    finder = ComprehensiveIndicatorFinder()
    finder.find_all()
    finder.generate_report()
