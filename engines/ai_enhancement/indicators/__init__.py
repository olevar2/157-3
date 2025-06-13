"""
Platform3 Individual Indicators Package

This package contains all individual indicator implementations that follow the
StandardIndicatorInterface for trading-grade accuracy and reliability.

Directory Structure:
- base_indicator.py: StandardIndicatorInterface and core classes
- momentum/: Momentum indicators (RSI, MACD, Stochastic, etc.)
- trend/: Trend indicators (Moving Averages, Bollinger Bands, etc.)
- volume/: Volume indicators (OBV, CMF, etc.)
- volatility/: Volatility indicators (ATR, Bollinger Width, etc.)
- pattern/: Pattern recognition indicators
- fractal/: Fractal and chaos indicators
- fibonacci/: Fibonacci-based indicators
- gann/: Gann analysis indicators
- statistical/: Statistical indicators
- cycle/: Cycle analysis indicators
- ml/: Machine learning indicators
- sentiment/: Sentiment analysis indicators

Implementation Standards:
1. All indicators must inherit from StandardIndicatorInterface
2. Each indicator in its own file (e.g., rsi.py, macd.py)
3. Comprehensive trading-grade tests for each indicator
4. Performance benchmarks and validation
5. Consistent naming: snake_case for files, PascalCase for classes

Example indicator file structure:
```
indicators/ None
├── __init__.py
├── base_indicator.py
├── momentum/ None
│   ├── __init__.py
│   ├── rsi.py                    # RelativeStrengthIndex class
│   ├── macd.py                   # MACD class
│   └── stochastic.py             # StochasticOscillator class
├── trend/ None
│   ├── __init__.py
│   ├── sma.py                    # SimpleMovingAverage class
│   └── bollinger_bands.py        # BollingerBands class
└── ...
```

Quality Assurance:
- Each indicator validated for trading accuracy
- Performance benchmarks for real-time usage
- Comprehensive test coverage
- Error handling for edge cases
- Numerical precision validation
"""

from .base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
    TradingGradeValidator,
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Platform3"
__description__ = "Individual trading-grade indicator implementations"

# Export core interfaces
__all__ = [
    "StandardIndicatorInterface",
    "IndicatorMetadata",
    "IndicatorValidationError",
    "TradingGradeValidator",
]
