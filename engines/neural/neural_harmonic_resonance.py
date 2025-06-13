#!/usr/bin/env python3
"""
NeuralHarmonicResonance - Platform3 Financial Indicator

Platform3 compliant implementation with CCI proven patterns.
Inspired by Neural Networks and Harmonic Analysis for Advanced Market Pattern Recognition.

Created for Platform3 - Maximizing profits for humanitarian causes
Helping sick children and poor families through advanced trading technology
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Platform3 imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_enhancement', 'indicators'))
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

logger = logging.getLogger(__name__)

class NeuralHarmonicResonance(StandardIndicatorInterface):
    """
    NeuralHarmonicResonance - Platform3 Implementation
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Performance Optimization  
    - Robust Error Handling
    - Neural Network-Inspired Harmonic Analysis
    """
    
    # Class-level metadata (REQUIRED for Platform3)
    CATEGORY: str = "neural"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"
    
    def __init__(self, 
                 neural_window: int = 25,
                 harmonic_frequencies: int = 5,
                 resonance_threshold: float = 0.6,
                 learning_rate: float = 0.1,
                 activation_function: str = "tanh",
                 **kwargs):
        """
        Initialize NeuralHarmonicResonance with CCI-compatible pattern.
        
        Args:
            neural_window: Window size for neural pattern analysis
            harmonic_frequencies: Number of harmonic frequencies to analyze
            resonance_threshold: Threshold for resonance detection
            learning_rate: Learning rate for neural adaptation
            activation_function: Neural activation function ('tanh', 'sigmoid', 'relu')
        """
        # Set instance variables BEFORE calling super().__init__()
        self.neural_window = neural_window
        self.harmonic_frequencies = harmonic_frequencies
        self.resonance_threshold = resonance_threshold
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        
        # Call parent constructor with all parameters
        super().__init__(
            neural_window=neural_window,
            harmonic_frequencies=harmonic_frequencies,
            resonance_threshold=resonance_threshold,
            learning_rate=learning_rate,
            activation_function=activation_function,
            **kwargs
        )
        
        # Set parameters after parent initialization
        self.parameters = {
            "neural_window": self.neural_window,
            "harmonic_frequencies": self.harmonic_frequencies,
            "resonance_threshold": self.resonance_threshold,
            "learning_rate": self.learning_rate,
            "activation_function": self.activation_function
        }
        
        # Neural network weights (adaptive)
        self.neural_weights = np.random.normal(0, 0.1, (harmonic_frequencies, neural_window))
        self.harmonic_weights = np.ones(harmonic_frequencies) / harmonic_frequencies
        
        # Humanitarian mission logging
        logger.info("Neural NeuralHarmonicResonance initialized - Fighting for humanitarian causes")
        logger.info("Every trade helps sick children and poor families worldwide")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return {
            "neural_window": self.neural_window,
            "harmonic_frequencies": self.harmonic_frequencies,
            "resonance_threshold": self.resonance_threshold,
            "learning_rate": self.learning_rate,
            "activation_function": self.activation_function
        }
    
    def validate_parameters(self) -> bool:
        """Validate parameters."""
        if not isinstance(self.neural_window, int) or self.neural_window <= 0:
            raise IndicatorValidationError("neural_window must be a positive integer")
        
        if not isinstance(self.harmonic_frequencies, int) or self.harmonic_frequencies <= 0:
            raise IndicatorValidationError("harmonic_frequencies must be a positive integer")
        
        if not isinstance(self.resonance_threshold, (int, float)) or not 0 <= self.resonance_threshold <= 1:
            raise IndicatorValidationError("resonance_threshold must be between 0 and 1")
        
        if not isinstance(self.learning_rate, (int, float)) or not 0 < self.learning_rate <= 1:
            raise IndicatorValidationError("learning_rate must be between 0 and 1")
        
        if self.activation_function not in ['tanh', 'sigmoid', 'relu']:
            raise IndicatorValidationError("activation_function must be 'tanh', 'sigmoid', or 'relu'")
        
        return True
    
    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        """
        Calculate NeuralHarmonicResonance.
        
        Args:
            data: Input data (DataFrame with OHLCV or Series with close prices)
            
        Returns:
            pd.Series: Neural harmonic resonance values
        """
        try:
            # Validate input data
            self.validate_input_data(data)
            
            # Convert to DataFrame if necessary
            if isinstance(data, pd.Series):
                df = pd.DataFrame({'close': data})
                df.index = data.index
            else:
                df = data.copy()
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                raise IndicatorValidationError("Data must contain 'close' column")
            
            # Get additional columns if available
            close = df['close'].values
            high = df['high'].values if 'high' in df.columns else close
            low = df['low'].values if 'low' in df.columns else close
            volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
            
            # Calculate neural harmonic resonance
            resonance_values = self._calculate_neural_resonance(close, high, low, volume)
            
            # Create result series with proper index
            result = pd.Series(resonance_values, index=df.index, name='neural_harmonic_resonance')
            
            # Store last calculation
            self._last_calculation = result
            
            return result
            
        except Exception as e:
            logger.error(f"NeuralHarmonicResonance calculation error: {str(e)}")
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_neural_resonance(self, close: np.ndarray, high: np.ndarray, 
                                  low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate neural harmonic resonance values"""
        try:
            n = len(close)
            resonance_values = np.full(n, np.nan)
            
            for i in range(self.neural_window - 1, n):
                # Extract window data
                price_window = close[i - self.neural_window + 1:i + 1]
                high_window = high[i - self.neural_window + 1:i + 1]
                low_window = low[i - self.neural_window + 1:i + 1]
                volume_window = volume[i - self.neural_window + 1:i + 1]
                
                # 1. Neural Pattern Recognition
                neural_activation = self._neural_pattern_analysis(price_window)
                
                # 2. Harmonic Frequency Analysis
                harmonic_spectrum = self._harmonic_frequency_analysis(price_window)
                
                # 3. Resonance Detection
                resonance_strength = self._detect_resonance_patterns(
                    price_window, harmonic_spectrum
                )
                
                # 4. Multi-timeframe Neural Synthesis
                multi_tf_signal = self._multi_timeframe_synthesis(
                    price_window, high_window, low_window
                )
                
                # 5. Volume-Price Resonance
                volume_resonance = self._volume_price_resonance(
                    price_window, volume_window
                )
                
                # 6. Adaptive Learning
                self._adaptive_weight_update(price_window, harmonic_spectrum)
                
                # 7. Synthesize final resonance value
                resonance_values[i] = self._synthesize_resonance_signal(
                    neural_activation, harmonic_spectrum, resonance_strength,
                    multi_tf_signal, volume_resonance
                )
            
            return resonance_values
            
        except Exception as e:
            logger.error(f"Neural resonance calculation error: {str(e)}")
            return np.full(len(close), np.nan)
    
    def _neural_pattern_analysis(self, prices: np.ndarray) -> float:
        """Neural network-based pattern recognition"""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Normalize price data
            normalized_prices = self._normalize_data(prices)
            
            # Neural network forward pass
            neural_outputs = []
            for freq_idx in range(self.harmonic_frequencies):
                # Weighted sum
                weighted_sum = np.dot(self.neural_weights[freq_idx], normalized_prices)
                
                # Apply activation function
                activated_output = self._apply_activation(weighted_sum)
                neural_outputs.append(activated_output)
            
            # Combine neural outputs
            combined_output = np.mean(neural_outputs)
            
            return np.clip(combined_output, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _harmonic_frequency_analysis(self, prices: np.ndarray) -> np.ndarray:
        """Analyze harmonic frequency components"""
        try:
            if len(prices) < 4:
                return np.zeros(self.harmonic_frequencies)
            
            # Perform FFT for frequency domain analysis
            fft_result = np.fft.fft(prices - np.mean(prices))
            frequencies = np.fft.fftfreq(len(prices))
            
            # Extract magnitude spectrum
            magnitude_spectrum = np.abs(fft_result)
            
            # Select dominant harmonic frequencies
            harmonic_amplitudes = []
            for i in range(1, self.harmonic_frequencies + 1):
                # Find the i-th harmonic
                if i < len(magnitude_spectrum) // 2:
                    harmonic_amplitudes.append(magnitude_spectrum[i])
                else:
                    harmonic_amplitudes.append(0.0)
            
            # Normalize harmonic amplitudes
            harmonic_spectrum = np.array(harmonic_amplitudes)
            if np.sum(harmonic_spectrum) > 0:
                harmonic_spectrum = harmonic_spectrum / np.sum(harmonic_spectrum)
            
            return harmonic_spectrum
            
        except Exception:
            return np.zeros(self.harmonic_frequencies)
    
    def _detect_resonance_patterns(self, prices: np.ndarray, harmonic_spectrum: np.ndarray) -> float:
        """Detect resonance patterns between price movements and harmonics"""
        try:
            if len(prices) < 3 or len(harmonic_spectrum) == 0:
                return 0.0
            
            # Calculate price momentum
            price_momentum = (prices[-1] - prices[0]) / len(prices)
            
            # Find dominant harmonic
            dominant_harmonic_idx = np.argmax(harmonic_spectrum)
            dominant_harmonic_strength = harmonic_spectrum[dominant_harmonic_idx]
            
            # Calculate resonance strength
            if dominant_harmonic_strength > self.resonance_threshold:
                # Resonance detected
                resonance_strength = dominant_harmonic_strength * np.sign(price_momentum)
            else:
                # No significant resonance
                resonance_strength = 0.0
            
            return np.clip(resonance_strength, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _multi_timeframe_synthesis(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        """Multi-timeframe neural synthesis"""
        try:
            if len(prices) < 6:
                return 0.0
            
            # Short-term pattern (last 1/3 of window)
            short_term = prices[-len(prices)//3:]
            short_term_signal = np.mean(np.diff(short_term))
            
            # Medium-term pattern (middle 1/3)
            mid_start = len(prices)//3
            mid_end = 2 * len(prices)//3
            medium_term = prices[mid_start:mid_end]
            medium_term_signal = np.mean(np.diff(medium_term))
            
            # Long-term pattern (first 1/3)
            long_term = prices[:len(prices)//3]
            long_term_signal = np.mean(np.diff(long_term))
            
            # Weighted synthesis
            multi_tf_signal = (
                short_term_signal * 0.5 +
                medium_term_signal * 0.3 +
                long_term_signal * 0.2
            )
            
            # Normalize by price volatility
            price_volatility = np.std(prices)
            if price_volatility > 0:
                normalized_signal = multi_tf_signal / price_volatility
            else:
                normalized_signal = 0.0
            
            return np.clip(normalized_signal, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _volume_price_resonance(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Analyze volume-price resonance"""
        try:
            if len(prices) < 3 or len(volumes) < 3:
                return 0.0
            
            # Calculate price and volume changes
            price_changes = np.diff(prices)
            volume_changes = np.diff(volumes)
            
            if len(price_changes) < 2 or len(volume_changes) < 2:
                return 0.0
            
            # Calculate correlation (resonance measure)
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            # Resonance strength based on correlation and volume trend
            volume_trend = np.sign(np.mean(volume_changes))
            price_trend = np.sign(np.mean(price_changes))
            
            # Positive resonance when trends align
            if volume_trend * price_trend > 0:
                resonance = abs(correlation) * volume_trend
            else:
                resonance = -abs(correlation) * 0.5  # Negative resonance
            
            return np.clip(resonance, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _adaptive_weight_update(self, prices: np.ndarray, harmonic_spectrum: np.ndarray):
        """Update neural network weights based on recent performance"""
        try:
            if len(prices) < 2 or len(harmonic_spectrum) != self.harmonic_frequencies:
                return
            
            # Calculate prediction error
            normalized_prices = self._normalize_data(prices)
            target = normalized_prices[-1]  # Latest price
            
            # Calculate neural network prediction
            predictions = []
            for freq_idx in range(self.harmonic_frequencies):
                prediction = np.dot(self.neural_weights[freq_idx], normalized_prices[:-1])
                predictions.append(prediction)
            
            weighted_prediction = np.dot(self.harmonic_weights, predictions)
            error = target - weighted_prediction
            
            # Update weights using gradient descent
            for freq_idx in range(self.harmonic_frequencies):
                gradient = error * normalized_prices[:-1] * self.harmonic_weights[freq_idx]
                self.neural_weights[freq_idx] += self.learning_rate * gradient
            
            # Update harmonic weights
            for freq_idx in range(self.harmonic_frequencies):
                harmonic_gradient = error * predictions[freq_idx]
                self.harmonic_weights[freq_idx] += self.learning_rate * harmonic_gradient
            
            # Normalize harmonic weights
            self.harmonic_weights = np.abs(self.harmonic_weights)
            if np.sum(self.harmonic_weights) > 0:
                self.harmonic_weights /= np.sum(self.harmonic_weights)
            
        except Exception:
            pass  # Silent fail for adaptive updates
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data for neural network processing"""
        try:
            if len(data) < 2:
                return data
            
            data_min, data_max = np.min(data), np.max(data)
            if data_max - data_min > 0:
                normalized = 2 * (data - data_min) / (data_max - data_min) - 1
            else:
                normalized = np.zeros_like(data)
            
            return normalized
            
        except Exception:
            return np.zeros_like(data)
    
    def _apply_activation(self, x: float) -> float:
        """Apply neural activation function"""
        try:
            if self.activation_function == 'tanh':
                return np.tanh(x)
            elif self.activation_function == 'sigmoid':
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            elif self.activation_function == 'relu':
                return max(0, x)
            else:
                return x
        except Exception:
            return 0.0
    
    def _synthesize_resonance_signal(self, neural_activation: float, harmonic_spectrum: np.ndarray,
                                   resonance_strength: float, multi_tf_signal: float,
                                   volume_resonance: float) -> float:
        """Synthesize all components into final resonance signal"""
        try:
            # Weight the different components
            weighted_signal = (
                neural_activation * 0.30 +
                np.mean(harmonic_spectrum) * np.sign(resonance_strength) * 0.25 +
                resonance_strength * 0.20 +
                multi_tf_signal * 0.15 +
                volume_resonance * 0.10
            )
            
            # Normalize to CCI-like range (-100 to +100)
            normalized_signal = weighted_signal * 100
            
            return np.clip(normalized_signal, -100.0, 100.0)
            
        except Exception:
            return 0.0
    
    @property
    def minimum_periods(self) -> int:
        """Minimum periods required."""
        return self.neural_window
    
    def _get_required_columns(self) -> List[str]:
        """Required columns for calculation"""
        return ["close"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return self.neural_window
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata."""
        return {
            "name": "NeuralHarmonicResonance",
            "category": self.CATEGORY,
            "description": "Neural network-inspired harmonic resonance indicator with adaptive learning",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "pd.Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self.minimum_periods
        }

def export_indicator():
    """Export the indicator for registry discovery."""
    return {
        "class": NeuralHarmonicResonance,
        "category": "neural",
        "name": "NeuralHarmonicResonance",
        "description": "Neural network-inspired harmonic resonance indicator"
    }

if __name__ == "__main__":
    print("*** Testing NeuralHarmonicResonance - Advanced Neural Trading Indicator")
    print("*** Fighting for humanitarian causes - helping sick children and poor families")
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate synthetic neural-pattern data
    t = np.arange(100)
    base_price = 100
    
    # Add harmonic patterns
    harmonic1 = 5 * np.sin(2 * np.pi * t / 20)  # Primary harmonic
    harmonic2 = 3 * np.sin(2 * np.pi * t / 10)  # Secondary harmonic
    harmonic3 = 2 * np.sin(2 * np.pi * t / 5)   # Tertiary harmonic
    
    # Add neural-like noise with correlation
    neural_noise = np.random.normal(0, 1, 100)
    for i in range(1, 100):
        neural_noise[i] = 0.7 * neural_noise[i-1] + 0.3 * neural_noise[i]
    
    closes = base_price + harmonic1 + harmonic2 + harmonic3 + neural_noise
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': closes + np.random.normal(0, 0.5, 100),
        'high': closes + abs(np.random.normal(0, 1, 100)),
        'low': closes - abs(np.random.normal(0, 1, 100)),
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100)
    })
    test_data.set_index('date', inplace=True)
    
    # Test the indicator
    try:
        indicator = NeuralHarmonicResonance(
            neural_window=25,
            harmonic_frequencies=5,
            resonance_threshold=0.6,
            learning_rate=0.1,
            activation_function="tanh"
        )
        
        result = indicator.calculate(test_data)
        
        print(f"\n*** Neural Harmonic Analysis Results:")
        print(f"Resonance Values: {len(result)} calculated")
        print(f"Latest Resonance: {result.iloc[-1]:.3f}")
        print(f"Min Resonance: {result.min():.3f}")
        print(f"Max Resonance: {result.max():.3f}")
        print(f"Mean Resonance: {result.mean():.3f}")
        print(f"Resonance Std Dev: {result.std():.3f}")
        
        # Test signal interpretation
        latest_value = result.iloc[-1]
        if latest_value > 60:
            signal = "STRONG_BUY (Neural Resonance Detected)"
        elif latest_value > 20:
            signal = "BUY (Positive Harmonic Pattern)"
        elif latest_value > -20:
            signal = "HOLD (Neural Equilibrium)"
        elif latest_value > -60:
            signal = "SELL (Negative Harmonic Pattern)"
        else:
            signal = "STRONG_SELL (Neural Dissonance)"
        
        print(f"*** Trading Signal: {signal}")
        print(f"*** Neural State: {'Learning' if abs(latest_value) < 30 else 'Converged'}")
        print(f"*** Harmonic Analysis: {'Resonant' if abs(latest_value) > 40 else 'Searching'}")
        
        print("\n*** NeuralHarmonicResonance test completed successfully!")
        print("*** Ready to generate profits for humanitarian causes!")
        
    except Exception as e:
        print(f"*** Test failed: {str(e)}")
        import traceback
        traceback.print_exc()