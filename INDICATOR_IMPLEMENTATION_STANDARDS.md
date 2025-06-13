# Platform3 Indicator Implementation Standards

## Quality Assurance Templates and Standards

### 1. **Indicator Implementation Template**

```python
class NewIndicatorName(BaseIndicator):
    """
    Brief description of what this indicator measures/calculates
    
    Parameters:
    -----------
    param1 : type
        Description of parameter
    param2 : type, optional (default=value)
        Description of optional parameter
        
    Mathematical Formula:
    --------------------
    [Provide the mathematical formula or algorithm description]
    
    Usage:
    ------
    >>> indicator = NewIndicatorName(period=14)
    >>> result = indicator.calculate(price_data)
    
    References:
    -----------
    [Academic papers, books, or standard definitions]
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize parameters with validation
        self.param1 = self._validate_param('param1', kwargs.get('param1'), required=True)
        self.param2 = self._validate_param('param2', kwargs.get('param2', default_value))
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator values
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pd.Series
            Calculated indicator values
            
        Raises:
        -------
        ValueError
            If input data is invalid or insufficient
        """
        # Input validation
        self._validate_input_data(data)
        
        # Check minimum data requirements
        if len(data) < self.minimum_periods:
            raise ValueError(f"Insufficient data: need at least {self.minimum_periods} periods")
        
        try:
            # Core calculation logic
            result = self._compute_indicator(data)
            
            # Post-processing and validation
            result = self._validate_output(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating {self.__class__.__name__}: {e}")
            raise
    
    def _compute_indicator(self, data: pd.DataFrame) -> pd.Series:
        """Core indicator calculation logic"""
        # Implement the actual mathematical calculation here
        # This method should contain the core algorithm
        raise NotImplementedError("Subclasses must implement _compute_indicator")
    
    @property
    def minimum_periods(self) -> int:
        """Minimum number of periods required for calculation"""
        return self.param1  # Usually related to the main period parameter
    
    def get_config(self) -> dict:
        """Return current indicator configuration"""
        return {
            'name': self.__class__.__name__,
            'param1': self.param1,
            'param2': self.param2,
            'minimum_periods': self.minimum_periods
        }
```

### 2. **Unit Test Template**

```python
import unittest
import pandas as pd
import numpy as np
from engines.ai_enhancement.indicator_name import NewIndicatorName

class TestNewIndicatorName(unittest.TestCase):
    """Comprehensive tests for NewIndicatorName indicator"""
    
    def setUp(self):
        """Set up test data and indicator instances"""
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': 1000000 + np.random.randint(-100000, 100000, 100)
        })
        
        # Ensure OHLC constraints
        self.sample_data['high'] = self.sample_data[['open', 'high', 'low', 'close']].max(axis=1)
        self.sample_data['low'] = self.sample_data[['open', 'high', 'low', 'close']].min(axis=1)
        
        self.indicator = NewIndicatorName(param1=14)
    
    def test_initialization(self):
        """Test indicator initialization with various parameters"""
        # Test default initialization
        indicator = NewIndicatorName(param1=14)
        self.assertEqual(indicator.param1, 14)
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            NewIndicatorName(param1=-1)  # Invalid parameter
    
    def test_calculate_basic(self):
        """Test basic calculation functionality"""
        result = self.indicator.calculate(self.sample_data)
        
        # Basic checks
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        self.assertFalse(result.isna().all())  # Should have some valid values
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        small_data = self.sample_data.head(5)  # Less than minimum required
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(small_data)
    
    def test_known_values(self):
        """Test against known/expected values"""
        # Create specific test case with known expected output
        known_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Calculate and verify against manually computed expected values
        result = self.indicator.calculate(known_data)
        # Add specific assertions based on expected values
        # self.assertAlmostEqual(result.iloc[-1], expected_value, places=6)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with constant values
        constant_data = self.sample_data.copy()
        constant_data[['open', 'high', 'low', 'close']] = 100
        
        result = self.indicator.calculate(constant_data)
        self.assertIsInstance(result, pd.Series)
        
        # Test with extreme values
        extreme_data = self.sample_data.copy()
        extreme_data.loc[0, 'high'] = extreme_data.loc[0, 'close'] * 10
        
        result = self.indicator.calculate(extreme_data)
        self.assertFalse(np.isinf(result).any())  # No infinite values
    
    def test_data_validation(self):
        """Test input data validation"""
        # Test missing columns
        incomplete_data = self.sample_data.drop(columns=['volume'])
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(incomplete_data)
        
        # Test with NaN values
        nan_data = self.sample_data.copy()
        nan_data.loc[10:15, 'close'] = np.nan
        
        result = self.indicator.calculate(nan_data)
        # Should handle NaN gracefully
        self.assertIsInstance(result, pd.Series)
    
    def test_performance(self):
        """Test performance with large datasets"""
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
        
        import time
        start_time = time.time()
        result = self.indicator.calculate(large_data)
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(elapsed_time, 1.0)  # Less than 1 second for 10k rows
        self.assertEqual(len(result), len(large_data))
    
    def test_parameter_sensitivity(self):
        """Test indicator sensitivity to parameter changes"""
        indicator1 = NewIndicatorName(param1=10)
        indicator2 = NewIndicatorName(param1=20)
        
        result1 = indicator1.calculate(self.sample_data)
        result2 = indicator2.calculate(self.sample_data)
        
        # Results should be different for different parameters
        self.assertFalse(result1.equals(result2))
    
    def test_config_export(self):
        """Test configuration export/import"""
        config = self.indicator.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('name', config)
        self.assertIn('param1', config)
        
        # Test recreation from config
        new_indicator = NewIndicatorName(**{k: v for k, v in config.items() if k != 'name'})
        self.assertEqual(new_indicator.param1, self.indicator.param1)

if __name__ == '__main__':
    unittest.main()
```

### 3. **Code Review Checklist**

#### **Implementation Quality:**
- [ ] Inherits from BaseIndicator correctly
- [ ] Follows naming conventions (PascalCase for classes)
- [ ] Proper docstring with parameters, returns, and examples
- [ ] Input validation implemented
- [ ] Error handling with meaningful error messages
- [ ] No hard-coded magic numbers
- [ ] Efficient algorithm implementation

#### **Mathematical Accuracy:**
- [ ] Formula correctly implemented
- [ ] Edge cases handled (division by zero, etc.)
- [ ] Proper handling of insufficient data
- [ ] Results validated against known values
- [ ] Numerical stability considered

#### **Testing Coverage:**
- [ ] Unit tests for all public methods
- [ ] Edge case testing
- [ ] Performance testing with large datasets
- [ ] Known value validation
- [ ] Error condition testing
- [ ] Parameter validation testing

#### **Documentation:**
- [ ] Clear mathematical formula provided
- [ ] Usage examples included
- [ ] Parameter descriptions accurate
- [ ] References to academic sources
- [ ] Integration with registry documented

#### **Integration:**
- [ ] Registered in registry.py
- [ ] Compatible with existing data formats
- [ ] No conflicts with existing indicators
- [ ] Proper logging implementation

### 4. **Performance Standards**

#### **Calculation Speed Requirements:**
- 1,000 data points: < 10ms
- 10,000 data points: < 100ms
- 100,000 data points: < 1s

#### **Memory Usage:**
- Should not hold unnecessary references to large datasets
- Temporary arrays should be properly cleaned up
- Memory usage should scale linearly with input size

#### **Numerical Accuracy:**
- Results accurate to at least 6 decimal places for known test cases
- No overflow/underflow issues with extreme values
- Proper handling of floating-point precision

### 5. **Documentation Standards**

#### **Required Documentation for Each Indicator:**
1. **Mathematical Definition**: Complete formula or algorithm
2. **Parameter Documentation**: All parameters with types and defaults
3. **Usage Examples**: At least 2 practical examples
4. **Interpretation Guide**: How to read and use the indicator
5. **References**: Academic or industry standard references
6. **Known Limitations**: What the indicator cannot do
7. **Related Indicators**: Similar or complementary indicators

#### **Code Comments:**
- Complex mathematical operations explained
- Non-obvious business logic documented
- Performance optimizations noted
- Edge case handling explained

---

## **Review and Approval Process**

1. **Self-Review**: Developer completes checklist
2. **Automated Testing**: All unit tests pass
3. **Performance Validation**: Meets speed requirements
4. **Mathematical Verification**: Results validated against known sources
5. **Integration Testing**: Works with existing system
6. **Documentation Review**: All required documentation complete
7. **Final Approval**: Senior developer approval required

---

*This template ensures consistent, high-quality indicator implementations across the Platform3 system.*
