#!/usr/bin/env python3
"""
Platform3 Indicator Status Validator
Cross-validates documented indicators vs actually loadable indicators
"""

import os
import sys
import importlib
import inspect
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndicatorStatusValidator:
    def __init__(self):
        self.base_path = Path("engines")
        self.documented_indicators = {}
        self.loadable_indicators = {}
        self.missing_indicators = []
        self.extra_indicators = []
        
        # Documented indicators from indicator_implementation_priority.md
        self.expected_indicators = {
            # Fractal Geometry (15)
            "fractal": [
                "FractalDimensionCalculator",
                "MandelbrotFractal", 
                "FRAMA",
                "FractalChannel",
                "MFDFA",
                "FractalMarketHypothesis",
                "FractalEfficiencyRatio",
                "FractalBreakout",
                "FractalMomentumOscillator",
                "FractalVolumeAnalysis",
                "FractalCorrelationDimension",
                "FractalEnergyIndicator", 
                "FractalChaosOscillator",
                "FractalWaveCounter",
                "FractalMarketProfile"
            ],
            
            # Candlestick Patterns (25)
            "patterns": [
                "DojiStandard", "DojiDragonfly", "DojiGravestone", "DojiLongLegged",
                "Hammer", "HangingMan", "InvertedHammer", "ShootingStar",
                "Marubozu", "SpinningTop", "HighWaveCandle",
                "BullishEngulfing", "BearishEngulfing", "BullishHarami", "BearishHarami",
                "PiercingLine", "DarkCloudCover", "TweezerTops", "TweezerBottoms",
                "BeltHold", "KickingPattern", "MorningStar", "EveningStar",
                "ThreeWhiteSoldiers", "ThreeBlackCrows", "ThreeInsideUp", "ThreeInsideDown",
                "ThreeOutsideUp", "ThreeOutsideDown", "AbandonedBaby", 
                "ThreeLineStrike", "MatchingLow", "MatchingHigh"
            ],
            
            # Core Technical - Momentum (15)
            "momentum": [
                "RSI", "MACD", "StochasticOscillator", "WilliamsR", "CCI",
                "ROC", "TSI", "UltimateOscillator", "AwesomeOscillator",
                "PPO", "DPO", "CMO", "KST", "TRIX", "MomentumIndicator"
            ],
            
            # Core Technical - Trend (15)
            "trend": [
                "SMA", "EMA", "WMA", "TEMA", "DEMA", "HMA", "KAMA",
                "ADX", "AroonIndicator", "IchimokuCloud", "ParabolicSAR",
                "SuperTrend", "VWMA", "McGinleyDynamic", "ZeroLagEMA"
            ],
            
            # Core Technical - Volatility (10)
            "volatility": [
                "BollingerBands", "ATR", "KeltnerChannels", "DonchianChannels",
                "StandardDeviationChannels", "VolatilityIndex", "HistoricalVolatility",
                "ChaikinVolatility", "MassIndex", "RVI"
            ],
            
            # Volume & Market Structure (15)
            "volume": [
                "OBV", "MFI", "VWAP", "VolumeProfile", "ChaikinMoneyFlow",
                "AccumulationDistribution", "EaseOfMovement", "VolumePriceTrend",
                "NegativeVolumeIndex", "PositiveVolumeIndex", "VolumeRateOfChange",
                "PriceVolumeRank", "VolumeOscillator", "KlingerOscillator", "ForceIndex"
            ],
            
            # Advanced - Statistical (10)
            "statistical": [
                "LinearRegression", "StandardDeviation", "CorrelationCoefficient",
                "ZScore", "BetaCoefficient", "RSquared", "VarianceRatio",
                "SkewnessKurtosis", "Cointegration", "Autocorrelation"
            ],
            
            # Advanced - Fibonacci (5)
            "fibonacci": [
                "FibonacciRetracement", "FibonacciExtension", "FibonacciTimeZones",
                "FibonacciArc", "FibonacciFan"
            ],
            
            # Advanced - Gann (5)
            "gann": [
                "GannFanLines", "GannSquareOfNine", "GannTimeCycles",
                "PriceTimeRelationships", "GannGrid"
            ],
            
            # Bonus (5+)
            "bonus": [
                "ElliottWaveCounter", "HarmonicPatternDetector", "MarketProfile",
                "PivotPoints", "CustomAIComposite"
            ]
        }

    def scan_loadable_indicators(self):
        """Scan and attempt to load all indicators from engines directory"""
        logger.info("🔍 Scanning for loadable indicators...")
        
        # Add the project root to Python path
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        for category_path in self.base_path.iterdir():
            if not category_path.is_dir() or category_path.name.startswith('__'):
                continue
                
            category = category_path.name
            self.loadable_indicators[category] = []
            
            for file_path in category_path.rglob("*.py"):
                if file_path.name.startswith('__'):
                    continue
                    
                try:
                    # Convert file path to module path
                    relative_path = file_path.relative_to(project_root)
                    module_path = '.'.join(relative_path.with_suffix('').parts)
                    
                    # Try to import the module
                    module = importlib.import_module(module_path)
                    
                    # Find all classes that look like indicators
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if self._is_indicator_class(name, obj, module_path):
                            self.loadable_indicators[category].append({
                                'name': name,
                                'file': str(file_path),
                                'module': module_path
                            })
                            
                except Exception as e:
                    logger.debug(f"Could not load {file_path}: {e}")
                    continue
        
        total_loadable = sum(len(indicators) for indicators in self.loadable_indicators.values())
        logger.info(f"✅ Found {total_loadable} loadable indicators across {len(self.loadable_indicators)} categories")

    def _is_indicator_class(self, name: str, obj, module_path: str) -> bool:
        """Determine if a class is likely an indicator"""
        # Skip if class is not defined in this module
        if obj.__module__ != module_path:
            return False
            
        # Skip utility/base classes
        utility_patterns = [
            'base', 'util', 'helper', 'config', 'exception', 'error',
            'manager', 'logger', 'database', 'communication', 'framework'
        ]
        
        if any(pattern in name.lower() for pattern in utility_patterns):
            return False
            
        # Look for indicator patterns
        indicator_patterns = [
            'indicator', 'oscillator', 'average', 'index', 'ratio',
            'analysis', 'detector', 'calculator', 'engine', 'pattern',
            'band', 'channel', 'line', 'signal', 'wave', 'fractal',
            'fibonacci', 'gann', 'rsi', 'macd', 'stochastic', 'bollinger'
        ]
        
        return any(pattern in name.lower() for pattern in indicator_patterns)

    def cross_validate_indicators(self):
        """Cross-validate documented vs loadable indicators"""
        logger.info("🔍 Cross-validating documented vs loadable indicators...")
        
        # Flatten expected indicators for easier comparison
        all_expected = []
        for category, indicators in self.expected_indicators.items():
            all_expected.extend(indicators)
        
        # Flatten loadable indicators
        all_loadable = []
        for category, indicators in self.loadable_indicators.items():
            for indicator in indicators:
                all_loadable.append(indicator['name'])
        
        # Find missing (documented but not loadable)
        self.missing_indicators = []
        for expected in all_expected:
            found = False
            for loadable in all_loadable:
                if self._names_match(expected, loadable):
                    found = True
                    break
            if not found:
                self.missing_indicators.append(expected)
        
        # Find extra (loadable but not documented)
        self.extra_indicators = []
        for loadable in all_loadable:
            found = False
            for expected in all_expected:
                if self._names_match(expected, loadable):
                    found = True
                    break
            if not found:
                self.extra_indicators.append(loadable)

    def _names_match(self, expected: str, actual: str) -> bool:
        """Check if indicator names match (allowing for variations)"""
        # Normalize names - remove common suffixes/prefixes and make lowercase
        def normalize(name):
            name = name.lower()
            # Remove common suffixes
            suffixes = ['indicator', 'oscillator', 'analysis', 'detector', 'calculator', 'engine', 'pattern']
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            return name.strip('_')
        
        norm_expected = normalize(expected)
        norm_actual = normalize(actual)
        
        # Direct match
        if norm_expected == norm_actual:
            return True
            
        # Partial match for common abbreviations
        abbreviations = {
            'rsi': 'relativestrengthindex',
            'macd': 'movingaverageconvergencedivergence',
            'cci': 'commoditychannelindex',
            'atr': 'averagetruerange',
            'obv': 'onbalancevolume',
            'mfi': 'moneyflowindex',
            'vwap': 'volumeweightedaverageprice',
            'adx': 'averagedirectionalindex',
            'roc': 'rateofchange',
            'tsi': 'truestrengthindex',
            'ppo': 'percentagepriceoscillator',
            'dpo': 'detrendedpriceoscillator',
            'cmo': 'chandemomentumoscillator',
            'kst': 'knowsurething'
        }
        
        for abbr, full in abbreviations.items():
            if (norm_expected == abbr and abbr in norm_actual) or \
               (norm_actual == abbr and abbr in norm_expected) or \
               (norm_expected == full and abbr in norm_actual) or \
               (norm_actual == full and abbr in norm_expected):
                return True
        
        return False

    def generate_report(self):
        """Generate comprehensive validation report"""
        total_expected = sum(len(indicators) for indicators in self.expected_indicators.values())
        total_loadable = sum(len(indicators) for indicators in self.loadable_indicators.values())
        total_missing = len(self.missing_indicators)
        total_extra = len(self.extra_indicators)
        
        coverage_rate = ((total_expected - total_missing) / total_expected) * 100 if total_expected > 0 else 0
        
        print("\n" + "="*80)
        print("PLATFORM3 INDICATOR STATUS VALIDATION REPORT")
        print("="*80)
        print(f"Expected Indicators (Documented): {total_expected}")
        print(f"Loadable Indicators (Found): {total_loadable}")
        print(f"Missing Indicators: {total_missing}")
        print(f"Extra Indicators: {total_extra}")
        print(f"Coverage Rate: {coverage_rate:.1f}%")
        
        if self.missing_indicators:
            print(f"\n❌ MISSING INDICATORS ({len(self.missing_indicators)}):")
            print("-" * 40)
            for indicator in sorted(self.missing_indicators):
                print(f"  └── {indicator}")
        
        if self.extra_indicators:
            print(f"\n➕ EXTRA INDICATORS ({len(self.extra_indicators)}):")
            print("-" * 40)
            for indicator in sorted(self.extra_indicators):
                print(f"  └── {indicator}")
        
        print(f"\n📊 LOADABLE INDICATORS BY CATEGORY:")
        print("-" * 40)
        for category, indicators in sorted(self.loadable_indicators.items()):
            if indicators:
                print(f"{category:15} : {len(indicators):3d} indicators")
                for indicator in indicators[:3]:  # Show first 3
                    print(f"  └── {indicator['name']}")
                if len(indicators) > 3:
                    print(f"  └── ... and {len(indicators) - 3} more")
        
        print("\n" + "="*80)
        
        return {
            'total_expected': total_expected,
            'total_loadable': total_loadable,
            'missing_count': total_missing,
            'extra_count': total_extra,
            'coverage_rate': coverage_rate,
            'missing_indicators': self.missing_indicators,
            'extra_indicators': self.extra_indicators
        }

def main():
    print("🚀 Platform3 Indicator Status Validation Starting...")
    
    validator = IndicatorStatusValidator()
    
    # Scan for loadable indicators
    validator.scan_loadable_indicators()
    
    # Cross-validate with documented indicators
    validator.cross_validate_indicators()
    
    # Generate comprehensive report
    report = validator.generate_report()
    
    print(f"\n✅ Validation complete!")
    print(f"Coverage: {report['coverage_rate']:.1f}% ({report['total_expected'] - report['missing_count']}/{report['total_expected']} documented indicators found)")

if __name__ == "__main__":
    main()
