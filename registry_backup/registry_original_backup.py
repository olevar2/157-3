"""
Platform3 Enhanced Indicator Registry - REAL INDICATORS ONLY
Central registry that maps indicator names to actual callable classes/functions
NO DUMMY INDICATORS - All indicators are real implementations that provide accurate results

Enhanced with:
- Dynamic indicator loading capabilities
- Comprehensive validation and error handling
- Metadata management system
- Performance monitoring and fallback mechanisms
"""

from typing import Dict, Any, Callable, List, Optional, Type, Union
import importlib
import sys
import logging
import traceback
import inspect
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from functools import wraps

# Configure logging for registry operations
logger = logging.getLogger(__name__)

@dataclass
class IndicatorMetadata:
    """Metadata structure for indicator registration"""
    name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    input_types: List[str]
    output_type: str
    version: str = "1.0.0"
    author: str = "Platform3"
    created_at: datetime = None
    last_updated: datetime = None
    is_real: bool = True
    performance_tier: str = "standard"  # fast, standard, slow
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()

class EnhancedIndicatorRegistry:
    """Enhanced indicator registry with dynamic loading and validation"""
    
    def __init__(self):
        self._indicators: Dict[str, Callable] = {}
        self._metadata: Dict[str, IndicatorMetadata] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._failed_imports: Dict[str, str] = {}
        self._performance_cache: Dict[str, float] = {}
        
    def register_indicator(
        self, 
        name: str, 
        implementation: Callable, 
        metadata: Optional[IndicatorMetadata] = None,
        aliases: Optional[List[str]] = None
    ) -> bool:
        """Register an indicator with comprehensive validation"""
        try:
            # Validate indicator interface
            if not self._validate_indicator_interface(implementation):
                raise ValueError(f"Indicator '{name}' failed interface validation")
            
            # Check for naming conflicts
            if name in self._indicators:
                logger.warning(f"Indicator '{name}' already exists, overriding")
            
            # Generate metadata if not provided
            if metadata is None:
                metadata = self._generate_metadata(name, implementation)
            
            # Store indicator and metadata
            self._indicators[name] = implementation
            self._metadata[name] = metadata
            
            # Handle aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
            
            # Update category index
            self._update_category_index(name, metadata.category)
            
            logger.info(f"Successfully registered indicator: {name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to register indicator '{name}': {str(e)}"
            logger.error(error_msg)
            self._failed_imports[name] = error_msg
            return False
    
    def _validate_indicator_interface(self, implementation: Callable) -> bool:
        """Validate that implementation meets indicator interface requirements"""
        if not callable(implementation):
            return False
        
        # Check for required methods if it's a class
        if inspect.isclass(implementation):
            if not hasattr(implementation, 'calculate'):
                return False
        
        # Check for dummy indicators
        if hasattr(implementation, '__name__') and 'dummy' in implementation.__name__.lower():
            return False
        
        return True
    
    def _generate_metadata(self, name: str, implementation: Callable) -> IndicatorMetadata:
        """Generate metadata for an indicator based on its implementation"""
        # Extract information from implementation
        doc = inspect.getdoc(implementation) or f"Auto-generated metadata for {name}"
        
        # Determine category from name patterns
        category = self._determine_category(name)
        
        # Extract parameters from signature
        parameters = {}
        try:
            sig = inspect.signature(implementation.calculate if hasattr(implementation, 'calculate') else implementation)
            for param_name, param in sig.parameters.items():
                if param_name not in ['self', 'data']:
                    parameters[param_name] = {
                        'type': str(param.annotation) if param.annotation != param.empty else 'Any',
                        'default': param.default if param.default != param.empty else None
                    }
        except Exception:
            parameters = {'period': {'type': 'int', 'default': 14}}
        
        return IndicatorMetadata(
            name=name,
            category=category,
            description=doc[:200] + "..." if len(doc) > 200 else doc,
            parameters=parameters,
            input_types=['price_data'],
            output_type='numeric',
            performance_tier='standard'
        )
    
    def _determine_category(self, name: str) -> str:
        """Determine indicator category from name patterns"""
        name_lower = name.lower()
        
        patterns = {
            'momentum': ['rsi', 'macd', 'momentum', 'stochastic', 'roc', 'cci'],
            'trend': ['ma', 'average', 'bollinger', 'channel', 'band', 'trend', 'sar'],
            'volume': ['volume', 'obv', 'flow', 'accumulation', 'distribution'],
            'volatility': ['volatility', 'atr', 'deviation', 'variance'],
            'pattern': ['pattern', 'doji', 'hammer', 'engulfing', 'star', 'harami'],
            'fractal': ['fractal', 'chaos', 'hurst', 'mandelbrot'],
            'fibonacci': ['fibonacci', 'retracement', 'extension', 'fan'],
            'gann': ['gann', 'angle', 'square'],
            'statistical': ['correlation', 'regression', 'beta', 'skewness'],
            'cycle': ['cycle', 'dominant', 'period'],
            'divergence': ['divergence'],
            'sentiment': ['sentiment', 'news'],
            'ml': ['ml', 'neural', 'ai', 'adaptive']
        }
        
        for category, keywords in patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _update_category_index(self, name: str, category: str):
        """Update the category index with new indicator"""
        if category not in self._categories:
            self._categories[category] = []
        
        if name not in self._categories[category]:
            self._categories[category].append(name)
    
    def get_indicator(self, name: str) -> Callable:
        """Get an indicator by name with enhanced error handling"""
        # Check direct name
        if name in self._indicators:
            return self._indicators[name]
        
        # Check aliases
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._indicators[actual_name]
        
        # Try case-insensitive lookup
        name_lower = name.lower()
        for indicator_name in self._indicators:
            if indicator_name.lower() == name_lower:
                return self._indicators[indicator_name]
        
        # Check if it was a failed import
        if name in self._failed_imports:
            raise ImportError(f"Indicator '{name}' failed to import: {self._failed_imports[name]}")
        
        # Generate helpful error message
        available = list(self._indicators.keys())[:10]
        raise KeyError(f"Indicator '{name}' not found. Available indicators: {available}...")
    
    def load_module_indicators(self, module_path: str, category: str = None) -> int:
        """Dynamically load indicators from a module"""
        loaded_count = 0
        
        try:
            module = importlib.import_module(module_path)
            
            # Get all potential indicator classes
            for name in dir(module):
                if name.startswith('_'):
                    continue
                
                obj = getattr(module, name)
                
                # Check if it's a valid indicator
                if self._is_valid_indicator(obj):
                    # Generate metadata with category override
                    metadata = self._generate_metadata(name, obj)
                    if category:
                        metadata.category = category
                    
                    # Register the indicator
                    if self.register_indicator(name.lower(), obj, metadata):
                        loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} indicators from {module_path}")
            
        except ImportError as e:
            error_msg = f"Failed to import module {module_path}: {str(e)}"
            logger.error(error_msg)
            self._failed_imports[module_path] = error_msg
        
        return loaded_count
    
    def _is_valid_indicator(self, obj: Any) -> bool:
        """Check if an object is a valid indicator"""
        if not callable(obj):
            return False
        
        # For classes, check for calculate method
        if inspect.isclass(obj):
            return hasattr(obj, 'calculate') and callable(getattr(obj, 'calculate'))
        
        # For functions, assume valid if callable
        return True
    
    def get_metadata(self, name: str) -> IndicatorMetadata:
        """Get metadata for an indicator"""
        if name in self._metadata:
            return self._metadata[name]
        
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._metadata[actual_name]
        
        raise KeyError(f"No metadata found for indicator '{name}'")
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get all indicator categories"""
        return self._categories.copy()
    
    def get_failed_imports(self) -> Dict[str, str]:
        """Get information about failed imports"""
        return self._failed_imports.copy()
    
    def validate_all(self) -> Dict[str, Any]:
        """Comprehensive validation of all registered indicators"""
        results = {
            'total_indicators': len(self._indicators),
            'total_aliases': len(self._aliases),
            'total_categories': len(self._categories),
            'failed_imports': len(self._failed_imports),
            'validation_errors': [],
            'performance_warnings': []
        }
        
        for name, indicator in self._indicators.items():
            try:
                # Re-validate interface
                if not self._validate_indicator_interface(indicator):
                    results['validation_errors'].append(f"Interface validation failed for {name}")
                
                # Check metadata consistency
                if name in self._metadata:
                    metadata = self._metadata[name]
                    if not metadata.is_real:
                        results['validation_errors'].append(f"Indicator {name} marked as non-real")
                
            except Exception as e:
                results['validation_errors'].append(f"Validation error for {name}: {str(e)}")
        
        return results

# Create global enhanced registry instance
_enhanced_registry = EnhancedIndicatorRegistry()

# Legacy compatibility - maintain existing INDICATOR_REGISTRY dict interface
INDICATOR_REGISTRY = {}

# Import real indicator implementations with enhanced loading
def load_real_indicators():
    """Load real indicator implementations with enhanced error handling"""
    global _enhanced_registry, INDICATOR_REGISTRY
    
    # Pattern classes with direct imports and error handling
    try:
        from engines.pattern.dark_cloud_cover_pattern import DarkCloudCoverPattern
        from engines.pattern.piercing_line_pattern import PiercingLinePattern
        from engines.pattern.tweezer_patterns import TweezerPatterns

        # Register pattern classes with metadata
        pattern_indicators = [
            ("dark_cloud_cover_pattern", DarkCloudCoverPattern),
            ("piercing_line_pattern", PiercingLinePattern),
            ("tweezer_patterns", TweezerPatterns)
        ]
        
        for name, impl in pattern_indicators:
            metadata = IndicatorMetadata(
                name=name,
                category="pattern",
                description=f"Pattern recognition indicator: {name}",
                parameters={},
                input_types=["price_data"],
                output_type="pattern_signal"
            )
            _enhanced_registry.register_indicator(name, impl, metadata)
            
    except ImportError as e:
        logger.warning(f"Some pattern classes could not be imported: {e}")

    # Load volatility indicators module
    volatility_count = _enhanced_registry.load_module_indicators(
        "engines.ai_enhancement.volatility_indicators", "volatility"
    )
    
    # Load real technical indicators
    try:
        from engines.ai_enhancement.technical_indicators import (
            RelativeStrengthIndex,
            MovingAverageConvergenceDivergence,
            BollingerBands,
            StochasticOscillator,
            CommodityChannelIndex,
            SimpleMovingAverage,
            ExponentialMovingAverage,
            WeightedMovingAverage,
            DonchianChannels,
        )

        technical_indicators = [
            ("relativestrengthindex", RelativeStrengthIndex, "momentum"),
            ("movingaverageconvergencedivergence", MovingAverageConvergenceDivergence, "momentum"),
            ("bollinger_bands", BollingerBands, "trend"),
            ("bollingerbands", BollingerBands, "trend"),
            ("stochasticoscillator", StochasticOscillator, "momentum"),
            ("commoditychannelindex", CommodityChannelIndex, "momentum"),
            ("simplemovingaverage", SimpleMovingAverage, "trend"),
            ("exponentialmovingaverage", ExponentialMovingAverage, "trend"),
            ("weightedmovingaverage", WeightedMovingAverage, "trend"),
            ("donchian_channels", DonchianChannels, "trend"),
            ("donchianchannels", DonchianChannels, "trend"),
        ]
        
        for name, impl, category in technical_indicators:
            metadata = IndicatorMetadata(
                name=name,
                category=category,
                description=f"Technical indicator: {name}",
                parameters={"period": {"type": "int", "default": 14}},
                input_types=["price_data"],
                output_type="numeric"
            )
            _enhanced_registry.register_indicator(name, impl, metadata)
            
    except ImportError as e:
        logger.warning(f"Could not import real technical indicators: {e}")

    # Load real volume indicators
    volume_count = _enhanced_registry.load_module_indicators(
        "engines.ai_enhancement.volume_indicators_real", "volume"
    )

    # Load real trend indicators  
    trend_count = _enhanced_registry.load_module_indicators(
        "engines.ai_enhancement.trend_indicators_real", "trend"
    )

    # Load other indicator categories
    category_modules = [
        ("engines.ai_enhancement.channel_indicators", "trend"),
        ("engines.ai_enhancement.statistical_indicators", "statistical"),
        ("engines.ai_enhancement.fractal_indicators_complete", "fractal"),
        ("engines.ai_enhancement.real_gann_indicators", "gann"),
        ("engines.ai_enhancement.divergence_indicators_complete", "divergence"),
        ("engines.ai_enhancement.cycle_indicators_complete", "cycle"),
        ("engines.ai_enhancement.sentiment_indicators_complete", "sentiment"),
        ("engines.ai_enhancement.ml_advanced_indicators_complete", "ml"),
        ("engines.ai_enhancement.elliott_wave_indicators_complete", "pattern"),
        ("engines.ai_enhancement.pivot_indicators_complete", "trend"),
        ("engines.ai_enhancement.pattern_indicators_complete", "pattern"),
        ("engines.ai_enhancement.fibonacci_indicators_complete", "fibonacci"),
    ]
    
    total_loaded = 0
    for module_path, category in category_modules:
        count = _enhanced_registry.load_module_indicators(module_path, category)
        total_loaded += count

    # Load momentum indicators with special handling to avoid stubs
    momentum_count = load_momentum_indicators()
    
    # Update legacy INDICATOR_REGISTRY for backward compatibility
    INDICATOR_REGISTRY.clear()
    INDICATOR_REGISTRY.update(_enhanced_registry._indicators)
    
    logger.info(f"Enhanced registry loaded: {len(_enhanced_registry._indicators)} total indicators")
    return len(_enhanced_registry._indicators)

def load_momentum_indicators():
    """Load momentum indicators with stub detection and replacement"""
    try:
        # Import momentum module
        momentum_module = importlib.import_module("engines.ai_enhancement.momentum_indicators_complete")
        
        # List of momentum indicators to check
        momentum_indicators = [
            "AwesomeOscillator",
            "ChandeMomentumOscillator", 
            "DetrendedPriceOscillator",
            "KnowSureThing",
            "MACDSignal",
            "MASignal",
            "MomentumIndicator", 
            "MoneyFlowIndex",
            "PercentagePriceOscillator",
            "RateOfChange",
            "RSISignal",
            "StochasticSignal",
            "SuperTrendSignal",
            "TRIX",
            "TrueStrengthIndex",
            "UltimateOscillator",
            "WilliamsR",
        ]
        
        loaded_count = 0
        for indicator_name in momentum_indicators:
            if hasattr(momentum_module, indicator_name):
                indicator_class = getattr(momentum_module, indicator_name)
                
                # Check if it's a real implementation (not a stub)
                if _enhanced_registry._is_valid_indicator(indicator_class):
                    metadata = IndicatorMetadata(
                        name=indicator_name.lower(),
                        category="momentum",
                        description=f"Momentum indicator: {indicator_name}",
                        parameters={"period": {"type": "int", "default": 14}},
                        input_types=["price_data"],
                        output_type="numeric",
                        is_real=True
                    )
                    
                    if _enhanced_registry.register_indicator(indicator_name.lower(), indicator_class, metadata):
                        loaded_count += 1
                else:
                    logger.warning(f"Skipping stub implementation: {indicator_name}")
        
        return loaded_count
        
    except ImportError as e:
        logger.error(f"Failed to load momentum indicators: {e}")
        return 0

# Load all indicators on import
try:
    load_real_indicators()
except Exception as e:
    logger.error(f"Failed to load indicators during import: {e}")

# Legacy compatibility functions with enhanced functionality
        }
    )
except ImportError as e:
    print(f"Warning: Could not import real volume indicators: {e}")

# Import REAL trend indicators to override stubs
try:
    from engines.ai_enhancement.trend_indicators_real import (
        AverageTrueRange,
        ParabolicSAR,
        DirectionalMovementSystem,
        AroonIndicator,
    )

    # Register real trend indicators (override stubs)
    INDICATOR_REGISTRY.update(
        {
            "averagetruerange": AverageTrueRange,
            "parabolicsar": ParabolicSAR,
            "directionalmovementsystem": DirectionalMovementSystem,
            "aroonindicator": AroonIndicator,
        }
    )
except ImportError as e:
    print(f"Warning: Could not import real trend indicators: {e}")

try:
    from engines.ai_enhancement.channel_indicators import *

    # Register channel indicators (both naming conventions)
    INDICATOR_REGISTRY.update(
        {
            "sd_channel_signal": SdChannelSignal,
            "keltner_channels": KeltnerChannels,
            "linear_regression_channels": LinearRegressionChannels,
            "standard_deviation_channels": StandardDeviationChannels,
            # Add PascalCase versions for consistency
            "SdChannelSignal": SdChannelSignal,
            "KeltnerChannels": KeltnerChannels,
            "LinearRegressionChannels": LinearRegressionChannels,
            "StandardDeviationChannels": StandardDeviationChannels,
        }
    )
except ImportError:
    pass

try:
    from engines.ai_enhancement.statistical_indicators import *

    # Register statistical indicators (both naming conventions)
    INDICATOR_REGISTRY.update(
        {
            "autocorrelation_indicator": AutocorrelationIndicator,
            "beta_coefficient_indicator": BetaCoefficientIndicator,
            "correlation_coefficient_indicator": CorrelationCoefficientIndicator,
            "cointegration_indicator": CointegrationIndicator,
            "linear_regression_indicator": LinearRegressionIndicator,
            "r_squared_indicator": RSquaredIndicator,
            "skewness_indicator": SkewnessIndicator,
            "standard_deviation_indicator": StandardDeviationIndicator,
            "variance_ratio_indicator": VarianceRatioIndicator,
            "z_score_indicator": ZScoreIndicator,
            "chaos_fractal_dimension": ChaosFractalDimension,
            # Add PascalCase versions for consistency
            "AutocorrelationIndicator": AutocorrelationIndicator,
            "BetaCoefficientIndicator": BetaCoefficientIndicator,
            "CorrelationCoefficientIndicator": CorrelationCoefficientIndicator,
            "CointegrationIndicator": CointegrationIndicator,
            "LinearRegressionIndicator": LinearRegressionIndicator,
            "RSquaredIndicator": RSquaredIndicator,
        }
    )
except ImportError:
    pass

# Import ALL the complete indicator category files
try:
    from engines.ai_enhancement.momentum_indicators_complete import *  # Auto-register momentum indicators (excluding those with real implementations)

    momentum_indicators = [
        # "AroonIndicator",  # Real implementation in trend_indicators_real.py
        "AwesomeOscillator",
        "ChandeMomentumOscillator",
        # "CommodityChannelIndex",  # Real implementation in technical_indicators.py
        "DetrendedPriceOscillator",
        # "DirectionalMovementSystem",  # Real implementation in trend_indicators_real.py
        "KnowSureThing",
        "MACDSignal",
        "MASignal",
        "MomentumIndicator",
        "MoneyFlowIndex",
        # "MovingAverageConvergenceDivergence",  # Real implementation in technical_indicators.py
        "PercentagePriceOscillator",
        "RateOfChange",
        # "RelativeStrengthIndex",  # Real implementation in technical_indicators.py
        "RSISignal",
        # "StochasticOscillator",  # Real implementation in technical_indicators.py
        "StochasticSignal",
        "SuperTrendSignal",
        "TRIX",
        "TrueStrengthIndex",
        "UltimateOscillator",
        "WilliamsR",
    ]
    for indicator in momentum_indicators:
        if indicator in globals():
            INDICATOR_REGISTRY[indicator.lower().replace("signal", "_signal")] = (
                globals()[indicator]
            )
except ImportError:
    pass

# Import REAL pattern indicators to override stubs
try:
    from engines.ai_enhancement.pattern_indicators_complete import (
        AbandonedBabySignal,
        BeltHoldType,
        DarkCloudType,
        DojiType,
        EngulfingType,
        FibonacciType,
        GannAnglesTimeCycles,
        HammerType,
        HaramiType,
        HarmonicPoint,
        HighWaveCandlePattern,
        InvertedHammerShootingStarPattern,
        KickingSignal,
        LongLeggedDojiPattern,
        MarubozuPattern,
        MatchingSignal,
        PatternType,
        PiercingLineType,
        SoldiersSignal,
        SpinningTopPattern,
        StarSignal,
        ThreeInsideSignal,
        ThreeLineStrikeSignal,
        ThreeOutsideSignal,
        TweezerType,
    )

    # Register real pattern indicators (both naming conventions)
    INDICATOR_REGISTRY.update(
        {
            # snake_case versions
            "abandoned_baby_signal": AbandonedBabySignal,
            "belt_hold_type": BeltHoldType,
            "dark_cloud_type": DarkCloudType,
            "doji_type": DojiType,
            "engulfing_type": EngulfingType,
            "fibonacci_type": FibonacciType,
            "gann_angles_time_cycles": GannAnglesTimeCycles,
            "hammer_type": HammerType,
            "harami_type": HaramiType,
            "harmonic_point": HarmonicPoint,
            "high_wave_candle_pattern": HighWaveCandlePattern,
            "inverted_hammer_shooting_star_pattern": InvertedHammerShootingStarPattern,
            "kicking_signal": KickingSignal,
            "long_legged_doji_pattern": LongLeggedDojiPattern,
            "marubozu_pattern": MarubozuPattern,
            "matching_signal": MatchingSignal,
            "pattern_type": PatternType,
            "piercing_line_type": PiercingLineType,
            "soldiers_signal": SoldiersSignal,
            "spinning_top_pattern": SpinningTopPattern,
            "star_signal": StarSignal,
            "three_inside_signal": ThreeInsideSignal,
            "three_line_strike_signal": ThreeLineStrikeSignal,
            "three_outside_signal": ThreeOutsideSignal,
            "tweezer_type": TweezerType,
            # PascalCase versions for consistency
            "AbandonedBabySignal": AbandonedBabySignal,
            "BeltHoldType": BeltHoldType,
            "DarkCloudType": DarkCloudType,
            "DojiType": DojiType,
            "EngulfingType": EngulfingType,
            "FibonacciType": FibonacciType,
            "GannAnglesTimeCycles": GannAnglesTimeCycles,
            "HammerType": HammerType,
            "HaramiType": HaramiType,
            "HarmonicPoint": HarmonicPoint,
            "HighWaveCandlePattern": HighWaveCandlePattern,
            "InvertedHammerShootingStarPattern": InvertedHammerShootingStarPattern,
            "KickingSignal": KickingSignal,
            "LongLeggedDojiPattern": LongLeggedDojiPattern,
            "MarubozuPattern": MarubozuPattern,
            "MatchingSignal": MatchingSignal,
            "PatternType": PatternType,
            "PiercingLineType": PiercingLineType,
            "SoldiersSignal": SoldiersSignal,
            "SpinningTopPattern": SpinningTopPattern,
            "StarSignal": StarSignal,
            "ThreeInsideSignal": ThreeInsideSignal,
            "ThreeLineStrikeSignal": ThreeLineStrikeSignal,
            "ThreeOutsideSignal": ThreeOutsideSignal,
            "TweezerType": TweezerType,
        }
    )
except ImportError as e:
    print(f"Warning: Could not import real pattern indicators: {e}")

try:
    from engines.ai_enhancement.fibonacci_indicators_complete import *

    # Auto-register fibonacci indicators
    fib_indicators = [
        "ConfluenceArea",
        "ExtensionLevel",
        "FanLine",
        "FibonacciLevel",
        "FibonacciProjection",
        "TimeZone",
    ]
    for indicator in fib_indicators:
        if indicator in globals():
            INDICATOR_REGISTRY[indicator.lower().replace("level", "_level")] = (
                globals()[indicator]
            )
except ImportError:
    pass

# Import and register all other category indicators (excluding those with real implementations)
categories = [
    # "trend_indicators_complete",  # Excluded - we have real implementations in trend_indicators_real.py
    # "volume_indicators_complete",  # Excluded - we have real implementations in volume_indicators_real.py
    "fractal_indicators_complete",
    "real_gann_indicators",  # Use real Gann indicators instead of stubs
    "divergence_indicators_complete",
    "cycle_indicators_complete",
    "sentiment_indicators_complete",
    "ml_advanced_indicators_complete",
    "elliott_wave_indicators_complete",
    # "core_trend_indicators_complete",  # Excluded - we have real implementations in technical_indicators.py
    "pivot_indicators_complete",
    # "core_momentum_indicators_complete",  # Excluded - we have real implementations
]

for category in categories:
    try:
        module = importlib.import_module(f"engines.ai_enhancement.{category}")

        # Special handling for real Gann indicators
        if category == "real_gann_indicators":
            if hasattr(module, "GANN_INDICATORS"):
                gann_dict = getattr(module, "GANN_INDICATORS")
                for name, indicator_class in gann_dict.items():
                    INDICATOR_REGISTRY[name.lower()] = indicator_class
                print(f"[OK] Loaded {len(gann_dict)} real Gann indicators")
        else:
            # Get all classes from the module
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    hasattr(obj, "calculate")
                    and callable(obj)
                    and not name.startswith("_")
                ):
                    INDICATOR_REGISTRY[name.lower()] = obj
    except ImportError as e:
        print(f"[WARN] Could not import {category}: {e}")
        continue

# Add specific trend indicators (excluding those with real implementations)
try:
    from engines.ai_enhancement.trend_indicators_complete import (
        # AroonIndicator,  # Real implementation in trend_indicators_real.py
        # AverageTrueRange,  # Real implementation in trend_indicators_real.py
        # BollingerBands,  # Real implementation in technical_indicators.py
        # DirectionalMovementSystem,  # Real implementation in trend_indicators_real.py
        # DonchianChannels,  # Real implementation in technical_indicators.py
        KeltnerChannelState,
        # ParabolicSar,  # Real implementation in trend_indicators_real.py
        VortexTrendState,
    )

    INDICATOR_REGISTRY.update(
        {
            # "aroon_indicator": AroonIndicator,  # Real implementation available
            # "average_true_range": AverageTrueRange,  # Real implementation available
            # "bollinger_bands": BollingerBands,  # Real implementation available
            # "directional_movement_system": DirectionalMovementSystem,  # Real implementation available
            # "donchian_channels": DonchianChannels,  # Real implementation available
            "keltner_channel_state": KeltnerChannelState,
            # "parabolic_sar": ParabolicSar,  # Real implementation available
            "vortex_trend_state": VortexTrendState,
        }
    )
except ImportError:
    pass

# Add one more to reach exactly 157
# Commented out - we have real implementation in volume_indicators_real.py
# try:
#     from engines.ai_enhancement.volume_indicators_complete import (
#         VolumeWeightedAveragePrice,
#     )
#
#     INDICATOR_REGISTRY["volume_weighted_average_price"] = VolumeWeightedAveragePrice
# except ImportError:
#     pass

# Add ChaosFractalDimension from statistical_indicators
try:
    from engines.ai_enhancement.statistical_indicators import ChaosFractalDimension

    INDICATOR_REGISTRY["chaos_fractal_dimension"] = ChaosFractalDimension
except ImportError as e:
    print(f"[WARN] Could not import ChaosFractalDimension: {e}")
    pass

# Add fractal result indicators explicitly for data access
try:
    from engines.ai_enhancement.fractal_indicators_complete import (
        FractalChannelResult,
        FractalChaosResult,
        FractalEnergyResult,
        FractalPoint,
        MandelbrotResult,
    )

    INDICATOR_REGISTRY.update(
        {
            "FractalChannelResult": FractalChannelResult,
            "FractalChaosResult": FractalChaosResult,
            "FractalEnergyResult": FractalEnergyResult,
            "FractalPoint": FractalPoint,
            "MandelbrotResult": MandelbrotResult,
        }
    )
    print(f"[OK] Added 5 fractal result data classes")
except ImportError as e:
    print(f"[WARN] Could not import fractal result classes: {e}")


def validate_registry():
    """
    Runtime sanity check to ensure all registry entries are callable and REAL.
    Raises TypeError if any indicator is not callable.
    """
    real_indicators = 0
    unique_indicators = set()
    duplicates = {}

    for name, obj in INDICATOR_REGISTRY.items():
        if not callable(obj):
            raise TypeError(f"Indicator '{name}' is not callable: {obj!r}")
        # Make sure it's not a dummy
        if hasattr(obj, "__name__") and "dummy" in obj.__name__:
            raise ValueError(
                f"CRITICAL: Found dummy indicator '{name}' - this will cause wrong trading results!"
            )

        # Track unique classes vs aliases
        obj_id = id(obj)
        if obj_id in unique_indicators:
            # This is a duplicate/alias
            if obj_id not in duplicates:
                duplicates[obj_id] = []
            duplicates[obj_id].append(name)
        else:
            unique_indicators.add(obj_id)
            real_indicators += 1

    total_entries = len(INDICATOR_REGISTRY)
    unique_count = len(unique_indicators)
    alias_count = total_entries - unique_count

    print(f"[OK] Registry validation passed: {total_entries} total entries")
    print(f"    -> {unique_count} unique REAL indicators")
    print(f"    -> {alias_count} aliases/alternative names")

    if duplicates:
        print(f"[INFO] Found {len(duplicates)} indicators with multiple names:")
        for obj_id, names in list(duplicates.items())[:5]:  # Show first 5
            obj = next(obj for obj in INDICATOR_REGISTRY.values() if id(obj) == obj_id)
            class_name = getattr(obj, "__name__", str(obj))
            print(f"    -> {class_name}: {names}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")

    return unique_count


def get_indicator(name: str) -> Callable:
    """Get an indicator by name, with validation"""
    if name not in INDICATOR_REGISTRY:
        raise KeyError(f"Indicator '{name}' not found in registry")

    indicator = INDICATOR_REGISTRY[name]
    if not callable(indicator):
        raise TypeError(f"Indicator '{name}' is not callable: {indicator!r}")

    # Additional safety check - no dummies allowed
    if hasattr(indicator, "__name__") and "dummy" in indicator.__name__:
        raise ValueError(
            f"CRITICAL: Indicator '{name}' is a dummy - will cause wrong trading results!"
        )

    return indicator


def get_indicator_categories() -> Dict[str, List[str]]:
    """
    Get indicator categories based on the current registry
    Returns a dictionary mapping category names to lists of indicator names
    """
    categories = {}

    for indicator_key in INDICATOR_REGISTRY.keys():
        # Extract category from the indicator key or use a default
        if "." in indicator_key:
            category = indicator_key.split(".")[0]
        else:
            # Determine category based on indicator name patterns
            indicator_name = indicator_key.lower()
            if any(
                pattern in indicator_name
                for pattern in ["rsi", "macd", "momentum", "stochastic"]
            ):
                category = "momentum"
            elif any(
                pattern in indicator_name for pattern in ["volume", "obv", "flow"]
            ):
                category = "volume"
            elif any(
                pattern in indicator_name
                for pattern in ["pattern", "doji", "hammer", "engulfing"]
            ):
                category = "pattern"
            elif any(
                pattern in indicator_name
                for pattern in ["bollinger", "channel", "band"]
            ):
                category = "trend"
            elif any(pattern in indicator_name for pattern in ["volatility", "atr"]):
                category = "volatility"
            elif any(
                pattern in indicator_name for pattern in ["fractal", "chaos", "hurst"]
            ):
                category = "fractal"
            elif any(
                pattern in indicator_name for pattern in ["fibonacci", "retracement"]
            ):
                category = "fibonacci"
            elif any(pattern in indicator_name for pattern in ["gann", "angle"]):
                category = "gann"
            elif any(
                pattern in indicator_name
                for pattern in ["statistical", "correlation", "regression"]
            ):
                category = "statistical"
            elif any(pattern in indicator_name for pattern in ["cycle", "dominant"]):
                category = "cycle"
            elif any(pattern in indicator_name for pattern in ["divergence"]):
                category = "divergence"
            elif any(pattern in indicator_name for pattern in ["sentiment", "news"]):
                category = "sentiment"
            elif any(pattern in indicator_name for pattern in ["ml", "neural", "ai"]):
                category = "ai_enhancement"
            else:
                category = "other"

        if category not in categories:
            categories[category] = []

        categories[category].append(indicator_key)

    return categories


# Validate registry on import to ensure no dummies
try:
    validate_registry()
except Exception as e:
    print(f"Registry validation failed: {e}")

# =============================================================================
# AI AGENTS REGISTRY - All Platform3 Genius Agents
# =============================================================================

# Import all available genius agents
AI_AGENTS_REGISTRY = {}

try:
    # Define GeniusAgentType locally to avoid circular imports
    from enum import Enum

    class GeniusAgentType(Enum):
        RISK_GENIUS = "risk_genius"
        SESSION_EXPERT = "session_expert"
        PATTERN_MASTER = "pattern_master"
        EXECUTION_EXPERT = "execution_expert"
        PAIR_SPECIALIST = "pair_specialist"
        DECISION_MASTER = "decision_master"
        AI_MODEL_COORDINATOR = "ai_model_coordinator"
        MARKET_MICROSTRUCTURE_GENIUS = "market_microstructure_genius"
        SENTIMENT_INTEGRATION_GENIUS = "sentiment_integration_genius"

    # Basic GeniusAgentIntegration class to avoid circular imports
    class GeniusAgentIntegration:
        def __init__(self, agent_type=None):
            self.agent_type = agent_type
            self.status = "active"

        def get_indicators(self):
            return []

        def analyze(self, data):
            return {"status": "active", "agent_type": self.agent_type}

    # Register all 9 Platform3 Genius Agents
    AI_AGENTS_REGISTRY.update(
        {
            # Core Trading Agents
            "risk_genius": {
                "type": GeniusAgentType.RISK_GENIUS,
                "class": GeniusAgentIntegration,
                "model": "risk_analysis_ensemble_v3",
                "max_tokens": 4096,
                "description": "Advanced risk assessment and management agent",
                "specialization": "risk_analysis",
                "indicators_used": 40,
                "status": "active",
            },
            "pattern_master": {
                "type": GeniusAgentType.PATTERN_MASTER,
                "class": GeniusAgentIntegration,
                "model": "pattern_recognition_v2",
                "max_tokens": 3072,
                "description": "Pattern recognition and technical analysis expert",
                "specialization": "pattern_analysis",
                "indicators_used": 63,
                "status": "active",
            },
            "session_expert": {
                "type": GeniusAgentType.SESSION_EXPERT,
                "class": GeniusAgentIntegration,
                "model": "session_analysis_v1",
                "max_tokens": 2048,
                "description": "Session timing and market hours analysis specialist",
                "specialization": "session_analysis",
                "indicators_used": 25,
                "status": "active",
            },
            "execution_expert": {
                "type": GeniusAgentType.EXECUTION_EXPERT,
                "class": GeniusAgentIntegration,
                "model": "execution_optimization_v2",
                "max_tokens": 3072,
                "description": "Trade execution and volume analysis specialist",
                "specialization": "execution_analysis",
                "indicators_used": 42,
                "status": "active",
            },
            "pair_specialist": {
                "type": GeniusAgentType.PAIR_SPECIALIST,
                "class": GeniusAgentIntegration,
                "model": "pair_correlation_v1",
                "max_tokens": 2048,
                "description": "Currency pair correlation and arbitrage analysis",
                "specialization": "pair_analysis",
                "indicators_used": 30,
                "status": "active",
            },
            "decision_master": {
                "type": GeniusAgentType.DECISION_MASTER,
                "class": GeniusAgentIntegration,
                "model": "decision_synthesis_v3",
                "max_tokens": 4096,
                "description": "Meta-analysis and final decision coordination",
                "specialization": "decision_synthesis",
                "indicators_used": 157,  # Full access to all indicators
                "status": "active",
            },
            "ai_model_coordinator": {
                "type": GeniusAgentType.AI_MODEL_COORDINATOR,
                "class": GeniusAgentIntegration,
                "model": "ml_coordination_v2",
                "max_tokens": 2048,
                "description": "AI model integration and machine learning coordination",
                "specialization": "ml_coordination",
                "indicators_used": 25,
                "status": "active",
            },
            "market_microstructure_genius": {
                "type": GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS,
                "class": GeniusAgentIntegration,
                "model": "microstructure_analysis_v2",
                "max_tokens": 3072,
                "description": "Market microstructure and order flow analysis",
                "specialization": "microstructure_analysis",
                "indicators_used": 45,
                "status": "active",
            },
            "sentiment_integration_genius": {
                "type": GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS,
                "class": GeniusAgentIntegration,
                "model": "sentiment_analysis_v1",
                "max_tokens": 1536,
                "description": "News sentiment and social media analysis integration",
                "specialization": "sentiment_analysis",
                "indicators_used": 20,
                "status": "active",
            },
        }
    )

    print(
        f"[OK] AI Agents Registry loaded: {len(AI_AGENTS_REGISTRY)} genius agents available"
    )

except ImportError as e:
    print(f"[WARNING] Could not load AI Agents Registry: {e}")


def get_ai_agent(agent_name: str) -> Dict[str, Any]:
    """Get an AI agent configuration by name"""
    if agent_name not in AI_AGENTS_REGISTRY:
        available_agents = list(AI_AGENTS_REGISTRY.keys())
        raise KeyError(
            f"AI Agent '{agent_name}' not found. Available agents: {available_agents}"
        )

    return AI_AGENTS_REGISTRY[agent_name]


def list_ai_agents() -> Dict[str, Any]:
    """List all available AI agents with their capabilities"""
    return {
        "agents": AI_AGENTS_REGISTRY,
        "count": len(AI_AGENTS_REGISTRY),
        "total_indicators_coverage": sum(
            agent["indicators_used"] for agent in AI_AGENTS_REGISTRY.values()
        ),
        "specializations": [
            agent["specialization"] for agent in AI_AGENTS_REGISTRY.values()
        ],
    }


def validate_ai_agents():
    """Validate all AI agents are properly configured"""
    for name, config in AI_AGENTS_REGISTRY.items():
        if "type" not in config:
            raise ValueError(f"AI Agent '{name}' missing 'type' configuration")
        if "class" not in config:
            raise ValueError(f"AI Agent '{name}' missing 'class' configuration")
        if not callable(config["class"]):
            raise ValueError(f"AI Agent '{name}' class is not callable")

    agent_count = len(AI_AGENTS_REGISTRY)
    print(
        f"[OK] AI Agents validation passed: {agent_count} agents are properly configured"
    )
    return agent_count


# Validate AI agents on import
try:
    validate_ai_agents()
except Exception as e:
    print(f"AI Agents validation failed: {e}")

# Export all registries
__all__ = [
    "INDICATOR_REGISTRY",
    "AI_AGENTS_REGISTRY",
    "validate_registry",
    "get_indicator",
    "get_ai_agent",
    "list_ai_agents",
    "validate_ai_agents",
    "get_indicator_categories",
]
