"""
Platform3 Indicator Registry
Central registry for all 115+ trading indicators
"""

from typing import Dict, List, Type, Optional
from .indicator_base import IndicatorBase as BaseIndicator
import importlib
import inspect

class IndicatorRegistry:
    """Central registry for all Platform3 indicators"""
    
    def __init__(self):
        self._indicators: Dict[str, Type[BaseIndicator]] = {}
        self._categories: Dict[str, List[str]] = {
            'momentum': [],
            'trend': [],
            'volume': [],
            'volatility': [],
            'pattern': [],
            'sentiment': [],
            'statistical': [],
            'gann': [],
            'fibonacci': [],
            'fractal': [],
            'elliott_wave': [],
            'cycle': [],
            'divergence': [],
            'ai_enhancement': [],
            'custom': [],
            'ml_enhanced': [],
            'hybrid': []
        }
        self._load_all_indicators()
    
    def _load_all_indicators(self):
        """Load all available indicators using the same approach as dynamic loader"""
        try:
            import os
            import sys
            import importlib
            import inspect
            from pathlib import Path
            
            # Add current directory to path
            sys.path.insert(0, os.path.abspath('.'))
            
            # Get the engines directory path
            engines_path = Path(os.path.dirname(__file__))
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(engines_path):
                # Skip validation, test, and pycache directories
                if any(skip in str(root) for skip in ['validation', 'test', '__pycache__', 'backup']):
                    continue
                    
                category = os.path.basename(root)
                
                # Ensure category exists in our categories dict
                if category not in self._categories:
                    self._categories[category] = []
                    
                for file in files:
                    if (file.endswith('.py') and 
                        not file.startswith('__') and
                        not file.endswith('_backup') and
                        'backup' not in file and
                        file not in ['indicator_base.py', 'indicator_registry.py']):
                        
                        try:
                            # Convert file path to module path
                            relative_path = os.path.relpath(os.path.join(root, file), '.')
                            module_path = relative_path.replace(os.sep, '.').replace('.py', '')
                            
                            # Try to import the module
                            module = importlib.import_module(module_path)
                            
                            # Look for indicator classes
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    name not in ['IndicatorBase', 'BaseIndicator', 'TechnicalIndicator'] and
                                    not name.startswith('_')):
                                    
                                    # Check if it might be an indicator class
                                    if any(base.__name__ in ['IndicatorBase', 'TechnicalIndicator'] 
                                           for base in obj.__mro__):
                                        
                                        indicator_name = f"{category}.{name}"
                                        self._indicators[indicator_name] = obj
                                        
                                        # Add to category
                                        if name not in self._categories[category]:
                                            self._categories[category].append(name)
                                        
                        except Exception as e:
                            # Silent fail for individual modules
                            pass
                            
        except Exception as e:
            print(f"Error loading indicators: {e}")
            # Fallback - load what we can manually
            self._load_ai_enhancement_indicators()

    def _load_ai_enhancement_indicators(self):
        """Load indicators from ai_enhancement which contains most of the indicators"""
        try:
            import engines.ai_enhancement as ai_enhancement
            
            # Get all classes from ai_enhancement
            for name, obj in inspect.getmembers(ai_enhancement, inspect.isclass):
                if (hasattr(obj, '__module__') and 
                    obj.__module__.startswith('engines.ai_enhancement') and
                    name not in ['BaseIndicator', 'TechnicalIndicator', 'IndicatorBase']):
                    
                    indicator_name = f"ai_enhancement.{name}"
                    self._indicators[indicator_name] = obj
                    
                    # Add to ai_enhancement category
                    if 'ai_enhancement' not in self._categories:
                        self._categories['ai_enhancement'] = []
                    if name not in self._categories['ai_enhancement']:
                        self._categories['ai_enhancement'].append(name)
                        
        except Exception as e:
            print(f"Error loading ai_enhancement indicators: {e}")

    def _load_indicator(self, category: str, module_name: str):
        """Load a single indicator module"""
        try:
            # Import the module
            module_path = f"engines.{category}.{module_name}"
            module = importlib.import_module(module_path)
            
            # Find indicator classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseIndicator) and 
                    obj != BaseIndicator):
                    
                    indicator_name = f"{category}.{name}"
                    self._indicators[indicator_name] = obj
                    self._categories[category].append(name)
                    
        except ImportError as e:
            print(f"Warning: Could not load {category}.{module_name}: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize indicator name for consistent access"""
        return name.lower()
    
    def get_indicator(self, name: str) -> Optional[Type[BaseIndicator]]:
        """Get indicator class by name"""
        # Try exact match first
        if name in self._indicators:
            return self._indicators[name]
        
        # Try normalized match
        normalized = self._normalize_name(name)
        for key, value in self._indicators.items():
            if key.lower() == normalized or key.split('.')[-1].lower() == normalized:
                return value
        
        return None
    
    def get_all_indicators(self) -> Dict[str, Type[BaseIndicator]]:
        """Get all registered indicators"""
        return self._indicators.copy()
    
    def get_indicators_by_category(self, category: str) -> List[str]:
        """Get all indicators in a specific category"""
        return self._categories.get(category, [])
    
    def get_indicator_count(self) -> int:
        """Get total number of registered indicators"""
        return len(self._indicators)
    
    def create_indicator(self, name: str, config: Optional[Dict] = None):
        """Create an instance of an indicator"""
        indicator_class = self.get_indicator(name)
        if indicator_class:
            return indicator_class(config or {})
        raise ValueError(f"Indicator '{name}' not found in registry")

# Global registry instance
indicator_registry = IndicatorRegistry()
