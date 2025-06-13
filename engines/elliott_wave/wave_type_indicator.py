"""
WaveType - Elliott Wave Type Classification Indicator for Platform3

This indicator classifies Elliott Wave types and provides detailed wave
characteristics including impulse types, corrective patterns, and wave
degree analysis for comprehensive Elliott Wave pattern recognition.

Version: 1.0.0
Category: Elliott Wave
Complexity: Advanced
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import logging

from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface


class WaveTypeIndicator(StandardIndicatorInterface):
    """
    Advanced Elliott Wave Type Classification Indicator
    
    Classifies Elliott Wave patterns into specific types:
    - Impulse wave types (leading, ending, extended)
    - Corrective wave types (zigzag, flat, triangle, complex)
    - Wave degree classification (primary, intermediate, minor)
    - Wave characteristics and quality assessment
    - Pattern completion probability
    
    This indicator provides sophisticated wave type identification
    for advanced Elliott Wave analysis and trading decisions.
    """
    
    # Class-level metadata
    INDICATOR_NAME = "WaveType"
    INDICATOR_VERSION = "1.0.0"
    INDICATOR_CATEGORY = "elliott_wave"
    INDICATOR_TYPE = "advanced"
    INDICATOR_COMPLEXITY = "advanced"
    
    # Wave type classifications
    IMPULSE_TYPES = ['leading_diagonal', 'ending_diagonal', 'standard_impulse', 'extended_impulse']
    CORRECTIVE_TYPES = ['zigzag', 'flat', 'triangle', 'complex_correction', 'irregular_correction']
    WAVE_DEGREES = ['grand_supercycle', 'supercycle', 'cycle', 'primary', 'intermediate', 'minor', 'minute', 'minuette', 'subminuette']
    
    def __init__(self, **kwargs):
        """
        Initialize WaveType indicator
        
        Args:
            parameters: Dictionary containing indicator parameters
                - period: Analysis period (default: 34)
                - min_wave_size: Minimum wave size threshold (default: 0.02)
                - fibonacci_tolerance: Fibonacci ratio tolerance (default: 0.05)
                - degree_multiplier: Wave degree size multiplier (default: 2.618)
                - pattern_confidence_threshold: Minimum confidence for pattern recognition (default: 0.7)
                - extension_ratio: Extension wave ratio threshold (default: 1.618)
                - diagonal_ratio: Diagonal pattern ratio threshold (default: 0.618)
        """
        super().__init__(**kwargs)
        
        # Get parameters with defaults
        self.period = int(self.parameters.get('period', 34))
        self.min_wave_size = float(self.parameters.get('min_wave_size', 0.02))
        self.fibonacci_tolerance = float(self.parameters.get('fibonacci_tolerance', 0.05))
        self.degree_multiplier = float(self.parameters.get('degree_multiplier', 2.618))
        self.pattern_confidence_threshold = float(self.parameters.get('pattern_confidence_threshold', 0.7))
        self.extension_ratio = float(self.parameters.get('extension_ratio', 1.618))
        self.diagonal_ratio = float(self.parameters.get('diagonal_ratio', 0.618))
        
        # Validation
        if self.period < 8:
            raise ValueError("Period must be at least 8")
        if self.min_wave_size <= 0:
            raise ValueError("Minimum wave size must be positive")
        if not 0 < self.pattern_confidence_threshold <= 1:
            raise ValueError("Pattern confidence threshold must be between 0 and 1")
            
        # Initialize state
        self.wave_patterns = []
        self.wave_types = []
        self.current_wave_type = None
        self.degree_analysis = {}
        
        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for WaveType calculation"""
        try:
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Need: {required_columns}")
                return False
                
            if len(data) < self.period:
                self.logger.warning(f"Insufficient data length: {len(data)} < {self.period}")
                return False
                
            if data[required_columns].isnull().any().any():
                self.logger.warning("Data contains NaN values")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _identify_wave_pivots(self, data: pd.DataFrame) -> List[Dict]:
        """Identify significant wave pivot points"""
        try:
            pivots = []
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values
            
            for i in range(3, len(data) - 3):
                # High pivot
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i-3] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2] and highs[i] > highs[i+3]):
                    
                    # Calculate significance
                    significance = self._calculate_pivot_significance(data, i, 'high')
                    
                    if significance > self.min_wave_size:
                        pivots.append({
                            'index': i,
                            'price': highs[i],
                            'type': 'high',
                            'volume': volumes[i],
                            'significance': significance,
                            'timestamp': data.index[i] if hasattr(data.index, 'to_list') else i
                        })
                
                # Low pivot
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i-3] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2] and lows[i] < lows[i+3]):
                    
                    # Calculate significance
                    significance = self._calculate_pivot_significance(data, i, 'low')
                    
                    if significance > self.min_wave_size:
                        pivots.append({
                            'index': i,
                            'price': lows[i],
                            'type': 'low',
                            'volume': volumes[i],
                            'significance': significance,
                            'timestamp': data.index[i] if hasattr(data.index, 'to_list') else i
                        })
            
            return sorted(pivots, key=lambda x: x['index'])
            
        except Exception as e:
            self.logger.error(f"Error identifying wave pivots: {str(e)}")
            return []
    
    def _calculate_pivot_significance(self, data: pd.DataFrame, index: int, pivot_type: str) -> float:
        """Calculate the significance of a pivot point"""
        try:
            if pivot_type == 'high':
                price = data['high'].iloc[index]
                # Compare with recent highs
                start_idx = max(0, index - self.period)
                end_idx = min(len(data), index + self.period)
                reference_highs = data['high'].iloc[start_idx:end_idx]
                max_ref = reference_highs.max()
                min_ref = reference_highs.min()
            else:
                price = data['low'].iloc[index]
                # Compare with recent lows
                start_idx = max(0, index - self.period)
                end_idx = min(len(data), index + self.period)
                reference_lows = data['low'].iloc[start_idx:end_idx]
                max_ref = reference_lows.max()
                min_ref = reference_lows.min()
            
            if max_ref == min_ref:
                return 0.0
            
            # Normalize significance
            if pivot_type == 'high':
                significance = (price - min_ref) / (max_ref - min_ref)
            else:
                significance = (max_ref - price) / (max_ref - min_ref)
            
            return significance
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot significance: {str(e)}")
            return 0.0
    
    def _classify_impulse_wave_type(self, wave_data: List[Dict]) -> Dict:
        """Classify impulse wave patterns"""
        try:
            if len(wave_data) < 5:
                return {'type': 'incomplete', 'confidence': 0.0}
            
            # Calculate wave properties
            wave_lengths = []
            wave_ratios = []
            
            for i in range(len(wave_data) - 1):
                length = abs(wave_data[i+1]['price'] - wave_data[i]['price'])
                wave_lengths.append(length)
            
            # Calculate ratios between waves
            for i in range(len(wave_lengths) - 1):
                if wave_lengths[i] > 0:
                    ratio = wave_lengths[i+1] / wave_lengths[i]
                    wave_ratios.append(ratio)
            
            # Classify based on wave relationships
            if len(wave_lengths) >= 5:
                # Check for extended waves
                max_wave_idx = wave_lengths.index(max(wave_lengths))
                extension_ratio = max(wave_lengths) / np.mean(wave_lengths)
                
                if extension_ratio > self.extension_ratio:
                    if max_wave_idx == 0:  # Wave 1 extended
                        return self._classify_extended_wave('wave_1_extended', wave_data, wave_lengths)
                    elif max_wave_idx == 2:  # Wave 3 extended
                        return self._classify_extended_wave('wave_3_extended', wave_data, wave_lengths)
                    elif max_wave_idx == 4:  # Wave 5 extended
                        return self._classify_extended_wave('wave_5_extended', wave_data, wave_lengths)
                
                # Check for diagonal patterns
                if self._is_diagonal_pattern(wave_data, wave_lengths):
                    return self._classify_diagonal_pattern(wave_data, wave_lengths)
                
                # Standard impulse wave
                return self._classify_standard_impulse(wave_data, wave_lengths)
            
            return {'type': 'incomplete', 'confidence': 0.0}
            
        except Exception as e:
            self.logger.error(f"Error classifying impulse wave type: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _classify_corrective_wave_type(self, wave_data: List[Dict]) -> Dict:
        """Classify corrective wave patterns"""
        try:
            if len(wave_data) < 3:
                return {'type': 'incomplete', 'confidence': 0.0}
            
            # Calculate wave properties
            wave_a = abs(wave_data[1]['price'] - wave_data[0]['price'])
            wave_b = abs(wave_data[2]['price'] - wave_data[1]['price']) if len(wave_data) > 2 else 0
            wave_c = abs(wave_data[3]['price'] - wave_data[2]['price']) if len(wave_data) > 3 else 0
            
            # Calculate retracement ratios
            if wave_a > 0:
                b_retracement = wave_b / wave_a
                c_retracement = wave_c / wave_a if wave_c > 0 else 0
            else:
                return {'type': 'invalid', 'confidence': 0.0}
            
            # Classify correction type
            if len(wave_data) >= 4:
                # Zigzag pattern (sharp correction)
                if 0.5 <= b_retracement <= 0.786 and 0.618 <= c_retracement <= 1.618:
                    return self._classify_zigzag_pattern(wave_data, b_retracement, c_retracement)
                
                # Flat pattern (sideways correction)
                elif 0.786 <= b_retracement <= 1.236 and 0.618 <= c_retracement <= 1.0:
                    return self._classify_flat_pattern(wave_data, b_retracement, c_retracement)
                
                # Triangle pattern
                elif self._is_triangle_pattern(wave_data):
                    return self._classify_triangle_pattern(wave_data)
                
                # Complex correction
                else:
                    return self._classify_complex_correction(wave_data, b_retracement, c_retracement)
            
            return {'type': 'simple_correction', 'confidence': 0.5}
            
        except Exception as e:
            self.logger.error(f"Error classifying corrective wave type: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _classify_extended_wave(self, extension_type: str, wave_data: List[Dict], wave_lengths: List[float]) -> Dict:
        """Classify extended impulse wave patterns"""
        try:
            confidence = 0.6  # Base confidence for extended waves
            
            # Calculate extension ratio
            max_length = max(wave_lengths)
            avg_length = np.mean([l for l in wave_lengths if l != max_length])
            
            if avg_length > 0:
                extension_ratio = max_length / avg_length
                
                # Higher confidence for stronger extensions
                if extension_ratio > 2.618:
                    confidence += 0.3
                elif extension_ratio > 1.618:
                    confidence += 0.2
            
            # Check Fibonacci relationships
            fibonacci_score = self._calculate_fibonacci_score(wave_lengths)
            confidence += fibonacci_score * 0.1
            
            return {
                'type': 'extended_impulse',
                'subtype': extension_type,
                'confidence': min(confidence, 1.0),
                'extension_ratio': extension_ratio if 'extension_ratio' in locals() else 0,
                'fibonacci_score': fibonacci_score
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying extended wave: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _is_diagonal_pattern(self, wave_data: List[Dict], wave_lengths: List[float]) -> bool:
        """Check if wave pattern is diagonal"""
        try:
            if len(wave_data) < 5:
                return False
            
            # Check for overlapping waves (characteristic of diagonals)
            for i in range(len(wave_data) - 2):
                current_level = wave_data[i]['price']
                next_level = wave_data[i+2]['price']
                
                # Check for overlap
                if wave_data[i]['type'] == wave_data[i+2]['type']:
                    if (wave_data[i]['type'] == 'high' and next_level < current_level) or \
                       (wave_data[i]['type'] == 'low' and next_level > current_level):
                        return True
            
            # Check wave length progression (contracting or expanding)
            if len(wave_lengths) >= 4:
                ratio1 = wave_lengths[1] / wave_lengths[0] if wave_lengths[0] > 0 else 0
                ratio2 = wave_lengths[3] / wave_lengths[2] if wave_lengths[2] > 0 else 0
                
                # Contracting diagonal
                if ratio1 < 1.0 and ratio2 < 1.0 and ratio2 < ratio1:
                    return True
                
                # Expanding diagonal
                if ratio1 > 1.0 and ratio2 > 1.0 and ratio2 > ratio1:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking diagonal pattern: {str(e)}")
            return False
    
    def _classify_diagonal_pattern(self, wave_data: List[Dict], wave_lengths: List[float]) -> Dict:
        """Classify diagonal wave patterns"""
        try:
            confidence = 0.7
            
            # Determine if leading or ending diagonal
            if len(wave_lengths) >= 4:
                ratio1 = wave_lengths[1] / wave_lengths[0] if wave_lengths[0] > 0 else 0
                ratio2 = wave_lengths[3] / wave_lengths[2] if wave_lengths[2] > 0 else 0
                
                if ratio1 < 1.0 and ratio2 < ratio1:
                    diagonal_type = 'ending_diagonal'
                    confidence += 0.1
                elif ratio1 > 1.0 and ratio2 > ratio1:
                    diagonal_type = 'leading_diagonal'
                    confidence += 0.1
                else:
                    diagonal_type = 'irregular_diagonal'
            else:
                diagonal_type = 'unknown_diagonal'
            
            return {
                'type': diagonal_type,
                'confidence': min(confidence, 1.0),
                'wave_overlap': True,
                'fibonacci_score': self._calculate_fibonacci_score(wave_lengths)
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying diagonal pattern: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _classify_standard_impulse(self, wave_data: List[Dict], wave_lengths: List[float]) -> Dict:
        """Classify standard impulse wave patterns"""
        try:
            confidence = 0.8
            
            # Check Elliott Wave rules
            rules_score = 0
            
            # Rule 1: Wave 3 is not the shortest
            if len(wave_lengths) >= 5:
                if wave_lengths[2] > min(wave_lengths[0], wave_lengths[4]):
                    rules_score += 1
                    confidence += 0.1
            
            # Rule 2: Wave 2 doesn't retrace beyond wave 1 start
            # (This would need price level analysis)
            
            # Rule 3: Wave 4 doesn't overlap wave 1
            # (This would need price level analysis)
            
            fibonacci_score = self._calculate_fibonacci_score(wave_lengths)
            confidence += fibonacci_score * 0.1
            
            return {
                'type': 'standard_impulse',
                'confidence': min(confidence, 1.0),
                'rules_compliance': rules_score,
                'fibonacci_score': fibonacci_score
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying standard impulse: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _classify_zigzag_pattern(self, wave_data: List[Dict], b_retracement: float, c_retracement: float) -> Dict:
        """Classify zigzag corrective patterns"""
        try:
            confidence = 0.7
            
            # Check ideal Fibonacci ratios
            ideal_b = 0.618
            ideal_c = 1.0
            
            b_accuracy = 1 - abs(b_retracement - ideal_b) / ideal_b
            c_accuracy = 1 - abs(c_retracement - ideal_c) / ideal_c
            
            confidence += (b_accuracy + c_accuracy) * 0.15
            
            # Determine zigzag subtype
            if b_retracement < 0.5:
                subtype = 'sharp_zigzag'
            elif b_retracement > 0.7:
                subtype = 'deep_zigzag'
            else:
                subtype = 'regular_zigzag'
            
            return {
                'type': 'zigzag',
                'subtype': subtype,
                'confidence': min(confidence, 1.0),
                'b_retracement': b_retracement,
                'c_retracement': c_retracement,
                'fibonacci_accuracy': (b_accuracy + c_accuracy) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying zigzag pattern: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _classify_flat_pattern(self, wave_data: List[Dict], b_retracement: float, c_retracement: float) -> Dict:
        """Classify flat corrective patterns"""
        try:
            confidence = 0.6
            
            # Check ideal Fibonacci ratios for flats
            if 0.9 <= b_retracement <= 1.05:
                subtype = 'regular_flat'
                confidence += 0.2
            elif b_retracement > 1.05:
                subtype = 'expanded_flat'
                confidence += 0.15
            else:
                subtype = 'contracting_flat'
                confidence += 0.1
            
            # C wave analysis
            if 0.9 <= c_retracement <= 1.1:
                confidence += 0.1
            
            return {
                'type': 'flat',
                'subtype': subtype,
                'confidence': min(confidence, 1.0),
                'b_retracement': b_retracement,
                'c_retracement': c_retracement
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying flat pattern: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _is_triangle_pattern(self, wave_data: List[Dict]) -> bool:
        """Check if pattern is a triangle"""
        try:
            if len(wave_data) < 5:
                return False
            
            # Triangle patterns have converging trendlines
            # Check for decreasing wave amplitude
            wave_amplitudes = []
            for i in range(len(wave_data) - 1):
                amplitude = abs(wave_data[i+1]['price'] - wave_data[i]['price'])
                wave_amplitudes.append(amplitude)
            
            if len(wave_amplitudes) >= 4:
                # Check if amplitudes are generally decreasing
                decreasing_count = 0
                for i in range(len(wave_amplitudes) - 1):
                    if wave_amplitudes[i+1] < wave_amplitudes[i]:
                        decreasing_count += 1
                
                return decreasing_count >= len(wave_amplitudes) // 2
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking triangle pattern: {str(e)}")
            return False
    
    def _classify_triangle_pattern(self, wave_data: List[Dict]) -> Dict:
        """Classify triangle corrective patterns"""
        try:
            confidence = 0.65
            
            # Analyze triangle characteristics
            wave_amplitudes = []
            for i in range(len(wave_data) - 1):
                amplitude = abs(wave_data[i+1]['price'] - wave_data[i]['price'])
                wave_amplitudes.append(amplitude)
            
            # Determine triangle type based on convergence pattern
            if len(wave_amplitudes) >= 4:
                first_half_avg = np.mean(wave_amplitudes[:len(wave_amplitudes)//2])
                second_half_avg = np.mean(wave_amplitudes[len(wave_amplitudes)//2:])
                
                convergence_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1
                
                if convergence_ratio < 0.7:
                    subtype = 'contracting_triangle'
                    confidence += 0.15
                elif convergence_ratio > 1.3:
                    subtype = 'expanding_triangle'
                    confidence += 0.1
                else:
                    subtype = 'symmetrical_triangle'
                    confidence += 0.2
            else:
                subtype = 'incomplete_triangle'
            
            return {
                'type': 'triangle',
                'subtype': subtype,
                'confidence': min(confidence, 1.0),
                'convergence_ratio': convergence_ratio if 'convergence_ratio' in locals() else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying triangle pattern: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _classify_complex_correction(self, wave_data: List[Dict], b_retracement: float, c_retracement: float) -> Dict:
        """Classify complex corrective patterns"""
        try:
            confidence = 0.4  # Lower confidence for complex patterns
            
            # Analyze complexity characteristics
            if len(wave_data) > 4:
                subtype = 'double_correction'
                confidence += 0.1
            elif b_retracement > 1.38 or c_retracement > 1.618:
                subtype = 'irregular_correction'
                confidence += 0.15
            else:
                subtype = 'running_correction'
                confidence += 0.05
            
            return {
                'type': 'complex_correction',
                'subtype': subtype,
                'confidence': min(confidence, 1.0),
                'b_retracement': b_retracement,
                'c_retracement': c_retracement,
                'complexity_score': len(wave_data) / 3  # Simple complexity measure
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying complex correction: {str(e)}")
            return {'type': 'error', 'confidence': 0.0}
    
    def _calculate_fibonacci_score(self, wave_lengths: List[float]) -> float:
        """Calculate how well wave lengths match Fibonacci ratios"""
        try:
            if len(wave_lengths) < 2:
                return 0.0
            
            fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
            total_score = 0.0
            comparison_count = 0
            
            for i in range(len(wave_lengths) - 1):
                if wave_lengths[i] > 0:
                    ratio = wave_lengths[i+1] / wave_lengths[i]
                    
                    # Find closest Fibonacci ratio
                    closest_fib = min(fibonacci_ratios, key=lambda x: abs(x - ratio))
                    accuracy = 1 - abs(ratio - closest_fib) / closest_fib
                    
                    if accuracy > 0.8:  # Only count good matches
                        total_score += accuracy
                        comparison_count += 1
            
            return total_score / comparison_count if comparison_count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci score: {str(e)}")
            return 0.0
    
    def _determine_wave_degree(self, wave_data: List[Dict]) -> Dict:
        """Determine the degree of the wave pattern"""
        try:
            if not wave_data:
                return {'degree': 'unknown', 'confidence': 0.0}
            
            # Calculate wave size (price range)
            price_range = max(w['price'] for w in wave_data) - min(w['price'] for w in wave_data)
            
            # Calculate time span
            time_span = max(w['index'] for w in wave_data) - min(w['index'] for w in wave_data)
            
            # Simple degree classification based on size and time
            # This is a simplified approach - real Elliott Wave degree analysis is much more complex
            
            if time_span > 252:  # More than a year of daily data
                if price_range > 0.5:  # Large price movement
                    degree = 'cycle'
                else:
                    degree = 'primary'
            elif time_span > 50:  # 2-3 months
                if price_range > 0.2:
                    degree = 'intermediate'
                else:
                    degree = 'minor'
            elif time_span > 10:  # 2-3 weeks
                degree = 'minute'
            else:
                degree = 'minuette'
            
            # Calculate confidence based on how clear the degree classification is
            confidence = min(0.7 + (time_span / 1000) * 0.3, 1.0)
            
            return {
                'degree': degree,
                'confidence': confidence,
                'time_span': time_span,
                'price_range': price_range
            }
            
        except Exception as e:
            self.logger.error(f"Error determining wave degree: {str(e)}")
            return {'degree': 'unknown', 'confidence': 0.0}
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate WaveType indicator
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing wave type classification
        """
        try:
            if not self.validate_data(data):
                return {}
            
            # Identify wave pivots
            pivots = self._identify_wave_pivots(data)
            
            if len(pivots) < 3:
                return {
                    'wave_type': 'insufficient_data',
                    'confidence': 0.0,
                    'pivots_found': len(pivots),
                    'analysis_complete': False
                }
            
            # Analyze impulse patterns (5-wave)
            impulse_analysis = {}
            if len(pivots) >= 5:
                impulse_waves = pivots[:5]  # Take first 5 pivots
                impulse_analysis = self._classify_impulse_wave_type(impulse_waves)
            
            # Analyze corrective patterns (3-wave)
            corrective_analysis = {}
            if len(pivots) >= 3:
                corrective_waves = pivots[:3]  # Take first 3 pivots
                corrective_analysis = self._classify_corrective_wave_type(corrective_waves)
            
            # Determine most likely wave type
            if impulse_analysis.get('confidence', 0) > corrective_analysis.get('confidence', 0):
                primary_wave_type = impulse_analysis
                primary_pattern = 'impulse'
            else:
                primary_wave_type = corrective_analysis
                primary_pattern = 'corrective'
            
            # Degree analysis
            degree_analysis = self._determine_wave_degree(pivots)
            self.degree_analysis = degree_analysis
            
            # Current wave state
            current_wave_type = primary_wave_type.get('type', 'unknown')
            overall_confidence = primary_wave_type.get('confidence', 0.0)
            
            # Pattern completion analysis
            completion_probability = self._calculate_completion_probability(pivots, primary_wave_type)
            
            result = {
                'wave_type': current_wave_type,
                'pattern_category': primary_pattern,
                'confidence': overall_confidence,
                'subtype': primary_wave_type.get('subtype', ''),
                'degree': degree_analysis.get('degree', 'unknown'),
                'degree_confidence': degree_analysis.get('confidence', 0.0),
                'impulse_analysis': impulse_analysis,
                'corrective_analysis': corrective_analysis,
                'completion_probability': completion_probability,
                'pivot_count': len(pivots),
                'fibonacci_score': primary_wave_type.get('fibonacci_score', 0.0),
                'analysis_complete': overall_confidence >= self.pattern_confidence_threshold,
                'pivots': pivots[-5:] if len(pivots) > 5 else pivots  # Return last 5 pivots
            }
            
            self.current_wave_type = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating WaveType: {str(e)}")
            return {}
    
    def _calculate_completion_probability(self, pivots: List[Dict], wave_analysis: Dict) -> float:
        """Calculate probability that current wave pattern is complete"""
        try:
            if not pivots or not wave_analysis:
                return 0.0
            
            wave_type = wave_analysis.get('type', '')
            confidence = wave_analysis.get('confidence', 0.0)
            
            base_probability = confidence * 0.7  # Base on pattern confidence
            
            # Adjust based on wave type
            if 'impulse' in wave_type:
                if len(pivots) >= 5:
                    base_probability += 0.3
                elif len(pivots) >= 3:
                    base_probability += 0.1
            elif 'corrective' in wave_type:
                if len(pivots) >= 3:
                    base_probability += 0.3
            
            # Adjust based on Fibonacci relationships
            fibonacci_score = wave_analysis.get('fibonacci_score', 0.0)
            base_probability += fibonacci_score * 0.1
            
            return min(base_probability, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating completion probability: {str(e)}")
            return 0.0
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata"""
        return {
            'name': self.INDICATOR_NAME,
            'version': self.INDICATOR_VERSION,
            'category': self.INDICATOR_CATEGORY,
            'type': self.INDICATOR_TYPE,
            'complexity': self.INDICATOR_COMPLEXITY,
            'parameters': {
                'period': self.period,
                'min_wave_size': self.min_wave_size,
                'fibonacci_tolerance': self.fibonacci_tolerance,
                'degree_multiplier': self.degree_multiplier,
                'pattern_confidence_threshold': self.pattern_confidence_threshold,
                'extension_ratio': self.extension_ratio,
                'diagonal_ratio': self.diagonal_ratio
            },
            'data_requirements': ['high', 'low', 'close', 'volume'],
            'output_format': 'wave_type_classification',
            'supported_wave_types': self.IMPULSE_TYPES + self.CORRECTIVE_TYPES,
            'supported_degrees': self.WAVE_DEGREES
        }
    def validate_parameters(self) -> bool:
        """Validate parameters"""
        # Add specific validation logic as needed
        return True



def export() -> Dict[str, Any]:
    """
    Export function for the WaveType indicator.
    
    This function is used by the indicator registry to discover and load the indicator.
    
    Returns:
        Dictionary containing indicator information for registry
    """
    return {
        'class': WaveTypeIndicator,
        'name': 'WaveType',
        'category': 'elliott_wave',
        'version': '1.0.0',
        'description': 'Advanced Elliott Wave type classification with pattern recognition',
        'complexity': 'advanced',
        'parameters': {
            'period': {'type': 'int', 'default': 34, 'min': 8, 'max': 200},
            'min_wave_size': {'type': 'float', 'default': 0.02, 'min': 0.001, 'max': 0.1},
            'fibonacci_tolerance': {'type': 'float', 'default': 0.05, 'min': 0.01, 'max': 0.2},
            'degree_multiplier': {'type': 'float', 'default': 2.618, 'min': 1.5, 'max': 5.0},
            'pattern_confidence_threshold': {'type': 'float', 'default': 0.7, 'min': 0.1, 'max': 1.0},
            'extension_ratio': {'type': 'float', 'default': 1.618, 'min': 1.2, 'max': 3.0},
            'diagonal_ratio': {'type': 'float', 'default': 0.618, 'min': 0.3, 'max': 1.0}
        },
        'data_requirements': ['high', 'low', 'close', 'volume'],
        'output_type': 'wave_classification'
    }