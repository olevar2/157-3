#!/usr/bin/env python3
"""
Comprehensive Indicator Loader for Platform3
Discovers and loads ALL 164 indicators across all categories with proper deduplication
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IndicatorInfo:
    """Information about a loaded indicator"""
    name: str
    category: str
    class_obj: Any
    file_path: str
    
class ComprehensiveIndicatorLoader:
    """Loads ALL indicators from Platform3 engines directory with deduplication"""
    
    def __init__(self):
        self.indicators: Dict[str, IndicatorInfo] = {}
        self.loaded_classes: Dict[str, str] = {}  # class_name -> full_path mapping for deduplication
        self.duplicate_classes: Dict[str, List[str]] = {}  # track duplicates
        self.utility_classes = {
            'IndicatorBase', 'IndicatorResult', 'IndicatorType', 'IndicatorConfig', 
            'IndicatorSignal', 'TechnicalIndicator', 'Platform3ErrorSystem',
            'IndicatorInterface', 'BaseIndicator', 'AbstractIndicator'
        }
        self.categories = [
            'volume', 'volatility', 'trend', 'statistical', 'sentiment', 
            'pattern', 'momentum', 'indicators', 'fractal', 'gann', 
            'elliott_wave', 'fibonacci', 'cycle', 'divergence', 
            'core_trend', 'core_momentum', 'advanced', 'ai_enhancement'
        ]
        self.loaded_count = 0
        self.error_count = 0
        self.duplicate_count = 0
        self.utility_count = 0
        
    def discover_all_indicators(self) -> Dict[str, IndicatorInfo]:
        """Discover and load ALL indicators from all categories"""
        logger.info("🔍 Starting comprehensive indicator discovery...")
        
        for category in self.categories:
            self._scan_category(category)
            
        logger.info(f"✅ Discovery complete: {self.loaded_count} unique indicators loaded, {self.duplicate_count} duplicates skipped, {self.utility_count} utility classes filtered, {self.error_count} errors")
        return self.indicators
    
    def _scan_category(self, category: str) -> None:
        """Scan a specific category directory for indicators"""
        category_path = f"engines/{category}"
        
        if not os.path.exists(category_path):
            logger.warning(f"⚠️  Category directory not found: {category_path}")
            return
            
        logger.info(f"📁 Scanning category: {category}")
        category_count = 0
        
        for file_name in os.listdir(category_path):
            if file_name.endswith('.py') and file_name != '__init__.py':
                try:
                    indicator_classes = self._load_indicator_file(category, file_name)
                    for class_name, class_obj in indicator_classes.items():
                        # Skip utility/base classes
                        if class_name in self.utility_classes:
                            self.utility_count += 1
                            logger.debug(f"🔧 Skipped utility class: {class_name}")
                            continue
                            
                        # Check for duplicates
                        full_path = f"{category_path}/{file_name}"
                        if class_name in self.loaded_classes:
                            existing_path = self.loaded_classes[class_name]
                            if class_name not in self.duplicate_classes:
                                self.duplicate_classes[class_name] = [existing_path]
                            self.duplicate_classes[class_name].append(full_path)
                            self.duplicate_count += 1
                            logger.warning(f"🔄 Duplicate indicator found: {class_name} in {full_path} (original: {existing_path})")
                            continue
                            
                        # Store unique indicator
                        self.loaded_classes[class_name] = full_path
                        indicator_info = IndicatorInfo(
                            name=class_name,
                            category=category,
                            class_obj=class_obj,
                            file_path=full_path
                        )
                        self.indicators[class_name] = indicator_info
                        category_count += 1
                        self.loaded_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error loading {category}/{file_name}: {str(e)}")
                    self.error_count += 1
                    
        logger.info(f"   └── {category}: {category_count} indicators loaded")
    
    def _load_indicator_file(self, category: str, file_name: str) -> Dict[str, Any]:
        """Load all indicator classes from a specific file"""
        module_name = f"engines.{category}.{file_name[:-3]}"  # Remove .py
        
        try:
            # Import the module
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
            
            module = sys.modules[module_name]
              # Find all classes in the module that look like indicators
            indicator_classes = {}
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (name.endswith('Indicator') or 
                    name.endswith('Oscillator') or 
                    name.endswith('Index') or
                    name.endswith('Analysis') or
                    name.endswith('Calculator') or
                    name.endswith('Detector') or
                    name.endswith('Pattern') or
                    name.endswith('Engine') or
                    name.endswith('System') or
                    'Indicator' in name or
                    # Core trading indicators with standard names
                    name in ['MovingAverages', 'ADX', 'Ichimoku', 'RSI', 'MACD', 'Stochastic',
                            'StochasticOscillator', 'WilliamsR', 'CCI', 'ROC', 'TSI', 'UltimateOscillator',
                            'AwesomeOscillator', 'PPO', 'DPO', 'CMO', 'KST', 'TRIX', 'Momentum',
                            'BollingerBands', 'ATR', 'Keltner', 'Donchian', 'SuperTrend', 'VWAP',
                            'OBV', 'MFI', 'AccumulationDistribution', 'VolumeProfile', 'Aroon'] or
                    # Pattern names
                    name.startswith('Doji') or name.startswith('Hammer') or name.startswith('Marubozu') or
                    name.startswith('Engulfing') or name.startswith('Harami') or name.startswith('Star') or
                    # Fractal and mathematical indicators
                    name.startswith('Fractal') or name.startswith('Fibonacci') or name.startswith('Gann')):
                    indicator_classes[name] = obj
                    
            return indicator_classes
            
        except Exception as e:
            raise Exception(f"Failed to load module {module_name}: {str(e)}")
    
    def get_indicators_by_category(self, category: str) -> List[IndicatorInfo]:
        """Get all indicators for a specific category"""
        return [info for info in self.indicators.values() if info.category == category]
    
    def get_indicator(self, name: str) -> Optional[IndicatorInfo]:
        """Get a specific indicator by name"""
        return self.indicators.get(name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loading statistics"""
        stats = {
            'total_loaded': self.loaded_count,
            'total_errors': self.error_count,
            'total_duplicates': self.duplicate_count,
            'total_utilities_filtered': self.utility_count,
            'success_rate': (self.loaded_count / (self.loaded_count + self.error_count)) * 100 if (self.loaded_count + self.error_count) > 0 else 0,
            'categories': {},
            'duplicates': dict(self.duplicate_classes)
        }
        
        for category in self.categories:
            category_indicators = self.get_indicators_by_category(category)
            stats['categories'][category] = len(category_indicators)
            
        return stats
    
    def print_comprehensive_report(self) -> None:
        """Print a detailed report of all loaded indicators"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE PLATFORM3 INDICATOR REPORT")
        print("="*80)
        print(f"Total Indicators Loaded: {stats['total_loaded']}")
        print(f"Loading Errors: {stats['total_errors']}")
        print(f"Duplicates Found: {stats['total_duplicates']}")
        print(f"Utility Classes Filtered: {stats['total_utilities_filtered']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print("\nIndicators by Category:")
        print("-" * 40)
        
        for category, count in stats['categories'].items():
            if count > 0:
                print(f"{category:20s}: {count:3d} indicators")
                category_indicators = self.get_indicators_by_category(category)
                for indicator in category_indicators[:3]:  # Show first 3
                    print(f"  └── {indicator.name}")
                if len(category_indicators) > 3:
                    print(f"  └── ... and {len(category_indicators) - 3} more")
        
        # Show duplicate report if any found
        if stats['total_duplicates'] > 0:
            print(f"\nDuplicate Indicators Found ({stats['total_duplicates']} total):")
            print("-" * 40)
            for class_name, paths in stats['duplicates'].items():
                print(f"{class_name}:")
                for path in paths:
                    print(f"  └── {path}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("🚀 Platform3 Comprehensive Indicator Discovery Starting...")
    
    loader = ComprehensiveIndicatorLoader()
    indicators = loader.discover_all_indicators()
    
    loader.print_comprehensive_report()
    
    return loader

if __name__ == "__main__":
    main()
