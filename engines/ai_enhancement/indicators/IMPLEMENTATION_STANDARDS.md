# Platform3 Individual Indicator Implementation Standards

## Overview
This document defines the mandatory standards for implementing individual indicators in Platform3. All indicators must meet these requirements to ensure trading-grade accuracy and system reliability.

## File Structure and Naming Conventions

### File Naming
- **Files**: Use snake_case (e.g., `relative_strength_index.py`, `bollinger_bands.py`)
- **Classes**: Use PascalCase (e.g., `RelativeStrengthIndex`, `BollingerBands`)
- **Methods**: Use snake_case (e.g., `calculate`, `validate_parameters`)

### Directory Organization
```
indicators/
├── __init__.py                   # Package initialization
├── base_indicator.py             # StandardIndicatorInterface
├── IMPLEMENTATION_STANDARDS.md   # This document
├── momentum/                     # Momentum indicators
├── trend/                        # Trend indicators  
├── volume/                       # Volume indicators
├── volatility/                   # Volatility indicators
├── pattern/                      # Pattern recognition
├── fractal/                      # Fractal analysis
├── fibonacci/                    # Fibonacci tools
├── gann/                         # Gann analysis
├── statistical/                  # Statistical indicators
├── cycle/                        # Cycle analysis
├── ml/                          # Machine learning
└── sentiment/                   # Sentiment analysis
```

## Implementation Requirements

### 0. PRE-IMPLEMENTATION: MANDATORY DUPLICATE CHECK
**BEFORE implementing ANY new indicator, you MUST run the duplicate-check system:**

```bash
# Step 1: Check for duplicates (MANDATORY)
python duplicate_check.py "Your Indicator Name"

# Step 2: Only proceed if result shows SAFE (not DUPLICATE)
# Step 3: If DUPLICATE found, investigate existing implementation
# Step 4: Document your duplicate-check results
```

**Why this is mandatory:**
- Platform3 has 77 discovered indicator files + 7 unique advanced indicators
- Many indicators already exist but may not be obvious
- Prevents wasted development effort on redundant functionality
- Ensures focus on the 73-83 genuinely missing indicators for 157 target

**Duplicate-check tools:**
- `duplicate_check.py`: Core duplicate detection system
- `indicator_dev_helper.py`: Interactive checking interface
- See `DUPLICATE_CHECK_USAGE_GUIDE.md` for complete instructions

### 1. Interface Compliance
All indicators MUST:
- Inherit from `StandardIndicatorInterface`
- Implement all abstract methods: `calculate()`, `validate_parameters()`, `get_metadata()`
- Follow the exact method signatures defined in the interface

### 2. Trading-Grade Accuracy
```python
class ExampleIndicator(StandardIndicatorInterface):
    """
    Example indicator implementation showing required standards.
    """
    
    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, period: int = 14, **kwargs):
        # Parameter validation BEFORE calling super
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError("Period must be a positive integer")
        
        super().__init__(period=period, **kwargs)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        MUST validate input and handle edge cases properly.
        """
        # Input validation
        self.validate_input_data(data)
        
        # Get parameters
        period = self.parameters['period']
        
        # Ensure sufficient data
        if len(data) < period:
            raise IndicatorValidationError(f"Insufficient data: need {period}, got {len(data)}")
        
        # Perform calculation with proper error handling
        try:
            result = self._perform_calculation(data, period)
            
            # Validate result quality
            if not TradingGradeValidator.validate_numerical_precision(result):
                raise IndicatorValidationError("Calculation failed numerical precision validation")
            
            self._last_calculation = result
            return result
            
        except Exception as e:
            logger.error(f"Calculation failed in {self.__class__.__name__}: {e}")
            raise
    
    def validate_parameters(self) -> bool:
        """
        MUST validate all parameters thoroughly.
        """
        period = self.parameters.get('period')
        
        if not isinstance(period, int):
            raise IndicatorValidationError("Period parameter must be an integer")
        
        if period < 1 or period > 1000:
            raise IndicatorValidationError("Period must be between 1 and 1000")
        
        return True
    
    def get_metadata(self) -> IndicatorMetadata:
        """
        MUST provide complete and accurate metadata.
        """
        return IndicatorMetadata(
            name=self.__class__.__name__,
            category=self.CATEGORY,
            description="Detailed description of what this indicator calculates",
            parameters={
                "period": {
                    "type": "int",
                    "default": 14,
                    "range": [1, 1000],
                    "description": "Look-back period for calculation"
                }
            },
            input_requirements=["close"],
            output_type="series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self.parameters['period'],
            performance_tier="fast"  # fast/standard/slow
        )
```

### 3. Required Methods Implementation

#### calculate()
- **Input**: `pd.DataFrame` with OHLCV data
- **Output**: `pd.Series` or `pd.DataFrame`
- **Must**: Validate input, handle edge cases, ensure numerical precision
- **Performance**: Optimize for real-time usage

#### validate_parameters()
- **Input**: None (uses self.parameters)
- **Output**: `bool` (True if valid)
- **Must**: Check all parameter types, ranges, and relationships
- **Raises**: `IndicatorValidationError` for invalid parameters

#### get_metadata()
- **Input**: None
- **Output**: `IndicatorMetadata` object
- **Must**: Provide complete specification including parameters, requirements, performance tier

### 4. Error Handling Standards
```python
# Parameter validation
if not isinstance(period, int) or period < 1:
    raise IndicatorValidationError("Period must be a positive integer")

# Data validation  
if len(data) < self.parameters['period']:
    raise IndicatorValidationError(f"Insufficient data points")

# Calculation errors
try:
    result = complex_calculation(data)
except ZeroDivisionError:
    raise IndicatorValidationError("Division by zero in calculation")
except Exception as e:
    logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
    raise
```

### 5. Performance Standards

#### Performance Tiers
- **Fast**: < 1ms per 1000 data points
- **Standard**: < 10ms per 1000 data points  
- **Slow**: < 100ms per 1000 data points

#### Optimization Guidelines
- Use vectorized pandas/numpy operations
- Avoid Python loops where possible
- Cache expensive calculations
- Use efficient algorithms (e.g., rolling windows)

### 6. Testing Requirements
Each indicator MUST include:
- Unit tests for basic functionality
- Edge case testing (empty data, insufficient data, NaN values)
- Accuracy validation against known benchmarks
- Performance benchmarking
- Parameter validation testing

### 7. Documentation Standards
```python
class IndicatorName(StandardIndicatorInterface):
    """
    Brief description of the indicator.
    
    Detailed explanation of:
    - What the indicator measures
    - How it's calculated
    - Trading interpretation
    - Usage guidelines
    
    Mathematical Formula:
    If applicable, include the mathematical formula in LaTeX or clear text.
    
    Parameters:
        param1 (type): Description and valid range
        param2 (type): Description and default value
    
    Example:
        >>> indicator = IndicatorName(period=14)
        >>> result = indicator.calculate(ohlcv_data)
        >>> print(result.tail())
    
    References:
        - Original paper/source if applicable
        - Industry standard definitions
    """
```

## Quality Assurance Checklist

Before submitting an indicator implementation, verify:

### ✅ Interface Compliance
- [ ] Inherits from `StandardIndicatorInterface`
- [ ] Implements all abstract methods
- [ ] Method signatures match interface exactly
- [ ] Class and file naming follows conventions

### ✅ Trading Accuracy
- [ ] Mathematical implementation is correct
- [ ] Handles edge cases properly (NaN, inf, empty data)
- [ ] Numerical precision validated
- [ ] Results match known benchmarks

### ✅ Performance
- [ ] Meets performance tier requirements
- [ ] Uses vectorized operations
- [ ] Benchmark tests pass
- [ ] Memory usage is reasonable

### ✅ Error Handling
- [ ] Parameter validation comprehensive
- [ ] Input data validation thorough
- [ ] Appropriate exceptions raised
- [ ] Logging for debugging

### ✅ Testing
- [ ] Unit tests cover basic functionality
- [ ] Edge cases tested
- [ ] Accuracy validation included
- [ ] Performance benchmarks pass

### ✅ Documentation
- [ ] Class docstring complete
- [ ] Method docstrings clear
- [ ] Parameter descriptions accurate
- [ ] Usage examples provided

## Migration from Grouped Files

When extracting indicators from existing grouped files:

1. **Preserve Logic**: Ensure exact mathematical implementation is maintained
2. **Update Interface**: Adapt to StandardIndicatorInterface requirements
3. **Add Validation**: Implement comprehensive parameter and data validation
4. **Create Tests**: Develop full test suite for the indicator
5. **Optimize Performance**: Refactor for real-time usage if needed
6. **Document Thoroughly**: Add complete documentation and examples

## Common Pitfalls to Avoid

1. **Insufficient Data Handling**: Always check for minimum data requirements
2. **Parameter Validation**: Don't skip thorough parameter checking
3. **Numerical Precision**: Use appropriate data types and handle edge cases
4. **Performance Issues**: Avoid inefficient calculations in loops
5. **Error Swallowing**: Don't catch exceptions without proper handling
6. **Incomplete Metadata**: Provide all required metadata information
7. **Testing Gaps**: Ensure comprehensive test coverage including edge cases

---

**Remember**: These standards ensure that all indicators are reliable, accurate, and suitable for making sound trading decisions. No shortcuts are acceptable when implementing trading-grade indicators.
