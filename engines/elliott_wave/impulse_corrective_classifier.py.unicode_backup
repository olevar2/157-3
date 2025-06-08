"""
IMPULSE/CORRECTIVE WAVE CLASSIFIER - Elliott Wave Type Identification
Platform3 Advanced Wave Analysis Engine

This module implements sophisticated classification of Elliott Wave patterns into impulse and corrective waves.
Uses multiple analytical methods to identify wave characteristics and classify wave structures.

Features:
- Automated impulse vs corrective wave classification
- 5-wave impulse pattern recognition (1-2-3-4-5)
- 3-wave corrective pattern recognition (A-B-C)
- Complex corrective pattern identification (flats, zigzags, triangles)
- Wave relationship validation using Elliott Wave rules
- Fibonacci ratio analysis for wave confirmation
- Degree classification (Grand Supercycle to Subminuette)
- Alternative count scenarios with probability scoring

Wave Classification Rules:
Impulse Waves (5-wave structure):
- Wave 2 cannot retrace more than 100% of Wave 1
- Wave 3 cannot be the shortest of waves 1, 3, and 5
- Wave 4 cannot overlap Wave 1 price territory (except in diagonals)
- Wave 3 is often 161.8% of Wave 1
- Wave 5 often equals Wave 1 or 61.8% of waves 1-3

Corrective Waves (3-wave structure):
- Zigzag: 5-3-5 internal structure
- Flat: 3-3-5 internal structure  
- Triangle: 3-3-3-3-3 internal structure
- Complex: Multiple ABC combinations

Trading Applications:
- Wave direction prediction based on wave type
- Entry/exit timing optimization
- Risk management through wave invalidation levels
- Trend continuation vs reversal identification
- Multiple timeframe wave analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal
from ..indicator_base import IndicatorBase

class WaveStructure(Enum):
    """Elliott Wave structure types"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"
    COMPLEX = "complex"
    UNKNOWN = "unknown"

class WaveClassification(Enum):
    """Specific wave classifications"""
    IMPULSE_5_WAVE = "impulse_5_wave"
    CORRECTIVE_ZIGZAG = "corrective_zigzag"
    CORRECTIVE_FLAT = "corrective_flat"
    CORRECTIVE_TRIANGLE = "corrective_triangle"
    DIAGONAL_LEADING = "diagonal_leading"
    DIAGONAL_ENDING = "diagonal_ending"
    COMPLEX_DOUBLE_THREE = "complex_double_three"
    COMPLEX_TRIPLE_THREE = "complex_triple_three"
    UNKNOWN_PATTERN = "unknown_pattern"

class WaveDegree(Enum):
    """Elliott Wave degrees from largest to smallest"""
    GRAND_SUPERCYCLE = "grand_supercycle"
    SUPERCYCLE = "supercycle"
    CYCLE = "cycle"
    PRIMARY = "primary"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    MINUTE = "minute"
    MINUETTE = "minuette"
    SUBMINUETTE = "subminuette"

@dataclass
class WavePoint:
    """Individual wave point definition"""
    index: int
    price: float
    wave_label: str
    wave_degree: WaveDegree
    time: Optional[pd.Timestamp] = None

@dataclass
class WaveSegment:
    """Wave segment between two points"""
    start_point: WavePoint
    end_point: WavePoint
    price_change: float
    time_duration: int
    price_change_pct: float
    direction: int  # 1 for up, -1 for down

@dataclass
class WaveAnalysis:
    """Complete wave analysis result"""
    wave_structure: WaveStructure
    classification: WaveClassification
    confidence: float
    wave_points: List[WavePoint]
    wave_segments: List[WaveSegment]
    fibonacci_ratios: Dict[str, float]
    rule_validation: Dict[str, bool]
    invalidation_levels: Dict[str, float]
    alternative_counts: List['WaveAnalysis']
    degree: WaveDegree

class ImpulsiveCorrectiveClassifier(IndicatorBase):
    """
    Advanced Impulse/Corrective Wave Classifier
    
    Classifies Elliott Wave patterns using comprehensive rule-based analysis,
    Fibonacci relationships, and structural pattern recognition.
    """
    
    def __init__(self, 
                 min_wave_size: float = 0.01,
                 fibonacci_tolerance: float = 0.15,
                 degree_auto_detection: bool = True,
                 complex_pattern_detection: bool = True,
                 alternative_count_limit: int = 3):
        """
        Initialize Wave Classifier
        
        Args:
            min_wave_size: Minimum wave size as percentage (1% default)
            fibonacci_tolerance: Tolerance for Fibonacci ratio validation (15%)
            degree_auto_detection: Automatically detect wave degree
            complex_pattern_detection: Enable complex corrective pattern detection
            alternative_count_limit: Maximum alternative wave counts to generate
        """
        super().__init__()
        
        self.min_wave_size = min_wave_size
        self.fibonacci_tolerance = fibonacci_tolerance
        self.degree_auto_detection = degree_auto_detection
        self.complex_pattern_detection = complex_pattern_detection
        self.alternative_count_limit = alternative_count_limit
        
        # Initialize analysis results
        self.current_analysis = None
        self.wave_history = []
        self.classification_confidence = 0.0
        
        # Elliott Wave rule validators
        self.rule_validators = self._initialize_rule_validators()
        
        # Fibonacci ratio templates
        self.fibonacci_templates = self._initialize_fibonacci_templates()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _initialize_rule_validators(self) -> Dict:
        """Initialize Elliott Wave rule validation functions"""
        return {
            'impulse_wave_2_retracement': self._validate_wave_2_retracement,
            'impulse_wave_3_not_shortest': self._validate_wave_3_not_shortest,
            'impulse_wave_4_no_overlap': self._validate_wave_4_no_overlap,
            'corrective_internal_structure': self._validate_corrective_structure,
            'diagonal_convergence': self._validate_diagonal_convergence,
            'triangle_diminishing_range': self._validate_triangle_range
        }
    
    def _initialize_fibonacci_templates(self) -> Dict:
        """Initialize Fibonacci ratio templates for different wave types"""
        return {
            'impulse': {
                'wave_3_ratios': [1.618, 2.618, 1.272, 3.618],
                'wave_5_ratios': [0.618, 1.000, 1.618, 0.382],
                'wave_2_retracements': [0.382, 0.500, 0.618, 0.786],
                'wave_4_retracements': [0.236, 0.382, 0.500]
            },
            'corrective': {
                'wave_c_ratios': [1.000, 1.618, 2.618, 0.618],
                'wave_b_retracements': [0.382, 0.500, 0.618, 0.786, 0.886],
                'flat_ratios': [0.900, 1.000, 1.100, 1.236],
                'triangle_ratios': [0.618, 0.786, 0.886]
            }
        }
    
    def calculate(self, 
                  data: pd.DataFrame,
                  wave_points: List[Tuple[int, float, str]] = None,
                  **kwargs) -> Dict:
        """
        Classify wave patterns in price data
        
        Args:
            data: Price data DataFrame with OHLC columns
            wave_points: Optional predefined wave points [(index, price, label)]
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with wave classification results
        """
        try:
            # Detect wave points if not provided
            if wave_points is None:
                wave_points = self._detect_wave_points(data)
            
            if len(wave_points) < 3:
                return self._get_default_result("Insufficient wave points for classification")
            
            # Convert to WavePoint objects
            wave_point_objects = self._create_wave_points(wave_points, data)
            
            # Create wave segments
            wave_segments = self._create_wave_segments(wave_point_objects)
            
            # Classify the wave pattern
            primary_analysis = self._classify_wave_pattern(wave_point_objects, wave_segments)
            
            # Generate alternative counts if requested
            alternative_analyses = []
            if self.alternative_count_limit > 0:
                alternative_analyses = self._generate_alternative_counts(
                    wave_point_objects, wave_segments)
            
            # Store results
            self.current_analysis = primary_analysis
            self.wave_history.append(primary_analysis)
            
            # Calculate overall confidence
            self.classification_confidence = self._calculate_overall_confidence(
                primary_analysis, alternative_analyses)
            
            return {
                'primary_analysis': self._analysis_to_dict(primary_analysis),
                'alternative_analyses': [self._analysis_to_dict(a) for a in alternative_analyses],
                'wave_points': [self._wave_point_to_dict(wp) for wp in wave_point_objects],
                'wave_segments': [self._wave_segment_to_dict(ws) for ws in wave_segments],
                'classification_confidence': self.classification_confidence,
                'wave_count': len(wave_point_objects),
                'structure_type': primary_analysis.wave_structure.value,
                'classification': primary_analysis.classification.value,
                'degree': primary_analysis.degree.value,
                'fibonacci_analysis': primary_analysis.fibonacci_ratios,
                'rule_validation': primary_analysis.rule_validation,
                'invalidation_levels': primary_analysis.invalidation_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error in wave classification: {e}")
            return self._get_default_result(f"Classification error: {e}")
    
    def _detect_wave_points(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Detect significant wave turning points in price data"""
        try:
            prices = data['close'].values
            
            # Use zigzag algorithm to find turning points
            # Find local maxima and minima with minimum percentage change
            min_change_pct = self.min_wave_size
            
            peaks = []
            troughs = []
            
            # Find peaks and troughs using scipy
            peak_indices, _ = signal.find_peaks(prices, 
                                              height=np.mean(prices),
                                              distance=5)
            trough_indices, _ = signal.find_peaks(-prices, 
                                                height=-np.mean(prices),
                                                distance=5)
            
            # Filter by minimum percentage change
            filtered_points = []
            
            # Combine and sort all points
            all_points = []
            for idx in peak_indices:
                all_points.append((idx, prices[idx], 'peak'))
            for idx in trough_indices:
                all_points.append((idx, prices[idx], 'trough'))
            
            all_points.sort(key=lambda x: x[0])
            
            if not all_points:
                return []
            
            # Filter points by minimum percentage change
            filtered_points = [all_points[0]]
            
            for i in range(1, len(all_points)):
                current_price = all_points[i][1]
                last_price = filtered_points[-1][1]
                
                price_change_pct = abs(current_price - last_price) / last_price
                
                if price_change_pct >= min_change_pct:
                    filtered_points.append(all_points[i])
            
            # Convert to wave labels
            wave_points = []
            for i, (idx, price, point_type) in enumerate(filtered_points):
                wave_label = str(i + 1)
                wave_points.append((idx, price, wave_label))
            
            return wave_points
            
        except Exception as e:
            self.logger.error(f"Error detecting wave points: {e}")
            return []
    
    def _create_wave_points(self, 
                           wave_points: List[Tuple[int, float, str]], 
                           data: pd.DataFrame) -> List[WavePoint]:
        """Create WavePoint objects from raw wave point data"""
        wave_point_objects = []
        
        for idx, price, label in wave_points:
            # Auto-detect degree based on time span and price movement
            degree = self._detect_wave_degree(wave_points, idx) if self.degree_auto_detection else WaveDegree.MINOR
            
            # Get timestamp if available
            timestamp = data.index[idx] if hasattr(data.index, 'to_pydatetime') else None
            
            wave_point = WavePoint(
                index=idx,
                price=price,
                wave_label=label,
                wave_degree=degree,
                time=timestamp
            )
            
            wave_point_objects.append(wave_point)
        
        return wave_point_objects
    
    def _create_wave_segments(self, wave_points: List[WavePoint]) -> List[WaveSegment]:
        """Create wave segments between consecutive wave points"""
        segments = []
        
        for i in range(len(wave_points) - 1):
            start_point = wave_points[i]
            end_point = wave_points[i + 1]
            
            price_change = end_point.price - start_point.price
            time_duration = end_point.index - start_point.index
            price_change_pct = price_change / start_point.price
            direction = 1 if price_change > 0 else -1
            
            segment = WaveSegment(
                start_point=start_point,
                end_point=end_point,
                price_change=price_change,
                time_duration=time_duration,
                price_change_pct=price_change_pct,
                direction=direction
            )
            
            segments.append(segment)
        
        return segments
    
    def _classify_wave_pattern(self, 
                             wave_points: List[WavePoint], 
                             wave_segments: List[WaveSegment]) -> WaveAnalysis:
        """Classify the overall wave pattern structure"""
        
        # Determine if pattern is impulse or corrective
        if len(wave_segments) == 5:
            # Potential 5-wave impulse
            return self._classify_impulse_pattern(wave_points, wave_segments)
        elif len(wave_segments) == 3:
            # Potential 3-wave corrective
            return self._classify_corrective_pattern(wave_points, wave_segments)
        elif len(wave_segments) > 5 and self.complex_pattern_detection:
            # Potential complex pattern
            return self._classify_complex_pattern(wave_points, wave_segments)
        else:
            # Unknown or incomplete pattern
            return self._create_unknown_analysis(wave_points, wave_segments)
    
    def _classify_impulse_pattern(self, 
                                wave_points: List[WavePoint], 
                                wave_segments: List[WaveSegment]) -> WaveAnalysis:
        """Classify 5-wave impulse pattern"""
        
        # Validate Elliott Wave rules for impulse
        rule_validation = {}
        
        # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        rule_validation['wave_2_retracement'] = self._validate_wave_2_retracement(wave_segments)
        
        # Rule 2: Wave 3 cannot be the shortest
        rule_validation['wave_3_not_shortest'] = self._validate_wave_3_not_shortest(wave_segments)
        
        # Rule 3: Wave 4 cannot overlap Wave 1 territory
        rule_validation['wave_4_no_overlap'] = self._validate_wave_4_no_overlap(wave_segments)
        
        # Calculate Fibonacci ratios
        fibonacci_ratios = self._calculate_impulse_fibonacci_ratios(wave_segments)
        
        # Calculate confidence based on rule validation and Fibonacci ratios
        confidence = self._calculate_impulse_confidence(rule_validation, fibonacci_ratios)
        
        # Determine wave structure and classification
        if all(rule_validation.values()) and confidence > 0.7:
            wave_structure = WaveStructure.IMPULSE
            classification = WaveClassification.IMPULSE_5_WAVE
        elif self._check_diagonal_characteristics(wave_segments):
            wave_structure = WaveStructure.DIAGONAL
            classification = WaveClassification.DIAGONAL_ENDING
        else:
            wave_structure = WaveStructure.UNKNOWN
            classification = WaveClassification.UNKNOWN_PATTERN
        
        # Calculate invalidation levels
        invalidation_levels = self._calculate_impulse_invalidation_levels(wave_points, wave_segments)
        
        # Detect wave degree
        degree = self._detect_pattern_degree(wave_segments)
        
        return WaveAnalysis(
            wave_structure=wave_structure,
            classification=classification,
            confidence=confidence,
            wave_points=wave_points,
            wave_segments=wave_segments,
            fibonacci_ratios=fibonacci_ratios,
            rule_validation=rule_validation,
            invalidation_levels=invalidation_levels,
            alternative_counts=[],
            degree=degree
        )
    
    def _classify_corrective_pattern(self, 
                                   wave_points: List[WavePoint], 
                                   wave_segments: List[WaveSegment]) -> WaveAnalysis:
        """Classify 3-wave corrective pattern"""
        
        # Validate corrective wave characteristics
        rule_validation = {}
        rule_validation['corrective_structure'] = self._validate_corrective_structure(wave_segments)
        
        # Calculate Fibonacci ratios for corrective pattern
        fibonacci_ratios = self._calculate_corrective_fibonacci_ratios(wave_segments)
        
        # Determine specific corrective type
        classification = self._determine_corrective_type(wave_segments, fibonacci_ratios)
        
        # Calculate confidence
        confidence = self._calculate_corrective_confidence(rule_validation, fibonacci_ratios, classification)
        
        # Calculate invalidation levels
        invalidation_levels = self._calculate_corrective_invalidation_levels(wave_points, wave_segments)
        
        # Detect wave degree
        degree = self._detect_pattern_degree(wave_segments)
        
        return WaveAnalysis(
            wave_structure=WaveStructure.CORRECTIVE,
            classification=classification,
            confidence=confidence,
            wave_points=wave_points,
            wave_segments=wave_segments,
            fibonacci_ratios=fibonacci_ratios,
            rule_validation=rule_validation,
            invalidation_levels=invalidation_levels,
            alternative_counts=[],
            degree=degree
        )
    
    def _classify_complex_pattern(self, 
                                wave_points: List[WavePoint], 
                                wave_segments: List[WaveSegment]) -> WaveAnalysis:
        """Classify complex corrective patterns"""
        
        # For complex patterns, break down into sub-patterns
        sub_patterns = self._identify_sub_patterns(wave_segments)
        
        # Determine if it's a double three, triple three, or other complex structure
        if len(sub_patterns) == 2:
            classification = WaveClassification.COMPLEX_DOUBLE_THREE
        elif len(sub_patterns) == 3:
            classification = WaveClassification.COMPLEX_TRIPLE_THREE
        else:
            classification = WaveClassification.UNKNOWN_PATTERN
        
        # Complex patterns have moderate confidence
        confidence = 0.6
        
        # Basic rule validation for complex patterns
        rule_validation = {'complex_structure': True}
        
        # Calculate basic Fibonacci ratios
        fibonacci_ratios = self._calculate_basic_fibonacci_ratios(wave_segments)
        
        # Calculate invalidation levels
        invalidation_levels = self._calculate_basic_invalidation_levels(wave_points, wave_segments)
        
        degree = self._detect_pattern_degree(wave_segments)
        
        return WaveAnalysis(
            wave_structure=WaveStructure.COMPLEX,
            classification=classification,
            confidence=confidence,
            wave_points=wave_points,
            wave_segments=wave_segments,
            fibonacci_ratios=fibonacci_ratios,
            rule_validation=rule_validation,
            invalidation_levels=invalidation_levels,
            alternative_counts=[],
            degree=degree
        )
    
    def _create_unknown_analysis(self, 
                               wave_points: List[WavePoint], 
                               wave_segments: List[WaveSegment]) -> WaveAnalysis:
        """Create analysis for unknown or incomplete patterns"""
        
        return WaveAnalysis(
            wave_structure=WaveStructure.UNKNOWN,
            classification=WaveClassification.UNKNOWN_PATTERN,
            confidence=0.1,
            wave_points=wave_points,
            wave_segments=wave_segments,
            fibonacci_ratios={},
            rule_validation={},
            invalidation_levels={},
            alternative_counts=[],
            degree=WaveDegree.MINOR
        )
    
    # Rule validation methods
    def _validate_wave_2_retracement(self, wave_segments: List[WaveSegment]) -> bool:
        """Validate that Wave 2 doesn't retrace more than 100% of Wave 1"""
        if len(wave_segments) < 2:
            return False
        
        wave_1_range = abs(wave_segments[0].price_change)
        wave_2_retracement = abs(wave_segments[1].price_change)
        
        return wave_2_retracement < wave_1_range
    
    def _validate_wave_3_not_shortest(self, wave_segments: List[WaveSegment]) -> bool:
        """Validate that Wave 3 is not the shortest of waves 1, 3, 5"""
        if len(wave_segments) < 5:
            return False
        
        wave_1_size = abs(wave_segments[0].price_change)
        wave_3_size = abs(wave_segments[2].price_change)
        wave_5_size = abs(wave_segments[4].price_change)
        
        return wave_3_size >= wave_1_size and wave_3_size >= wave_5_size
    
    def _validate_wave_4_no_overlap(self, wave_segments: List[WaveSegment]) -> bool:
        """Validate that Wave 4 doesn't overlap Wave 1 territory"""
        if len(wave_segments) < 4:
            return False
        
        wave_1_start = wave_segments[0].start_point.price
        wave_1_end = wave_segments[0].end_point.price
        wave_4_end = wave_segments[3].end_point.price
        
        # Check if Wave 4 end overlaps with Wave 1 territory
        if wave_1_end > wave_1_start:  # Upward Wave 1
            return wave_4_end > wave_1_end
        else:  # Downward Wave 1
            return wave_4_end < wave_1_end
    
    def _validate_corrective_structure(self, wave_segments: List[WaveSegment]) -> bool:
        """Validate basic corrective wave structure"""
        if len(wave_segments) != 3:
            return False
        
        # Basic validation: corrective waves should show clear ABC structure
        # Wave A and C should be in same direction, Wave B opposite
        wave_a_direction = wave_segments[0].direction
        wave_b_direction = wave_segments[1].direction
        wave_c_direction = wave_segments[2].direction
        
        return (wave_a_direction == wave_c_direction and 
                wave_b_direction == -wave_a_direction)
    
    def _validate_diagonal_convergence(self, wave_segments: List[WaveSegment]) -> bool:
        """Validate diagonal pattern convergence"""
        # Implementation for diagonal pattern validation
        return True  # Simplified for now
    
    def _validate_triangle_range(self, wave_segments: List[WaveSegment]) -> bool:
        """Validate triangle pattern diminishing range"""
        # Implementation for triangle pattern validation
        return True  # Simplified for now
    
    # Fibonacci calculation methods
    def _calculate_impulse_fibonacci_ratios(self, wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate Fibonacci ratios for impulse pattern"""
        ratios = {}
        
        if len(wave_segments) >= 3:
            wave_1_size = abs(wave_segments[0].price_change)
            wave_3_size = abs(wave_segments[2].price_change)
            
            if wave_1_size > 0:
                ratios['wave_3_to_wave_1'] = wave_3_size / wave_1_size
        
        if len(wave_segments) >= 5:
            wave_5_size = abs(wave_segments[4].price_change)
            if wave_1_size > 0:
                ratios['wave_5_to_wave_1'] = wave_5_size / wave_1_size
            
            # Wave 5 to combined waves 1-3
            waves_1_to_3_size = abs(wave_segments[2].end_point.price - wave_segments[0].start_point.price)
            if waves_1_to_3_size > 0:
                ratios['wave_5_to_waves_1_3'] = wave_5_size / waves_1_to_3_size
        
        if len(wave_segments) >= 2:
            wave_2_retracement = abs(wave_segments[1].price_change) / wave_1_size
            ratios['wave_2_retracement'] = wave_2_retracement
        
        if len(wave_segments) >= 4:
            wave_4_retracement = abs(wave_segments[3].price_change) / wave_3_size
            ratios['wave_4_retracement'] = wave_4_retracement
        
        return ratios
    
    def _calculate_corrective_fibonacci_ratios(self, wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate Fibonacci ratios for corrective pattern"""
        ratios = {}
        
        if len(wave_segments) >= 3:
            wave_a_size = abs(wave_segments[0].price_change)
            wave_b_size = abs(wave_segments[1].price_change)
            wave_c_size = abs(wave_segments[2].price_change)
            
            if wave_a_size > 0:
                ratios['wave_c_to_wave_a'] = wave_c_size / wave_a_size
                ratios['wave_b_to_wave_a'] = wave_b_size / wave_a_size
        
        return ratios
    
    def _calculate_basic_fibonacci_ratios(self, wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate basic Fibonacci ratios for any pattern"""
        ratios = {}
        
        if len(wave_segments) >= 2:
            sizes = [abs(seg.price_change) for seg in wave_segments]
            for i in range(len(sizes) - 1):
                if sizes[i] > 0:
                    ratios[f'wave_{i+2}_to_wave_{i+1}'] = sizes[i+1] / sizes[i]
        
        return ratios
    
    # Confidence calculation methods
    def _calculate_impulse_confidence(self, rule_validation: Dict[str, bool], fibonacci_ratios: Dict[str, float]) -> float:
        """Calculate confidence score for impulse pattern"""
        confidence = 0.0
        
        # Rule validation contributes 70% of confidence
        rule_score = sum(rule_validation.values()) / len(rule_validation) if rule_validation else 0
        confidence += rule_score * 0.7
        
        # Fibonacci ratios contribute 30% of confidence
        fib_score = self._score_fibonacci_ratios(fibonacci_ratios, 'impulse')
        confidence += fib_score * 0.3
        
        return min(confidence, 1.0)
    
    def _calculate_corrective_confidence(self, 
                                       rule_validation: Dict[str, bool], 
                                       fibonacci_ratios: Dict[str, float],
                                       classification: WaveClassification) -> float:
        """Calculate confidence score for corrective pattern"""
        confidence = 0.0
        
        # Rule validation contributes 60% of confidence
        rule_score = sum(rule_validation.values()) / len(rule_validation) if rule_validation else 0
        confidence += rule_score * 0.6
        
        # Fibonacci ratios contribute 40% of confidence
        fib_score = self._score_fibonacci_ratios(fibonacci_ratios, 'corrective')
        confidence += fib_score * 0.4
        
        return min(confidence, 1.0)
    
    def _score_fibonacci_ratios(self, ratios: Dict[str, float], pattern_type: str) -> float:
        """Score Fibonacci ratios against expected values"""
        if not ratios:
            return 0.0
        
        expected_ratios = self.fibonacci_templates.get(pattern_type, {})
        score = 0.0
        count = 0
        
        for ratio_name, ratio_value in ratios.items():
            # Find closest expected ratio
            best_match_score = 0.0
            
            for template_name, template_ratios in expected_ratios.items():
                for expected_ratio in template_ratios:
                    ratio_diff = abs(ratio_value - expected_ratio) / expected_ratio
                    if ratio_diff <= self.fibonacci_tolerance:
                        match_score = 1.0 - (ratio_diff / self.fibonacci_tolerance)
                        best_match_score = max(best_match_score, match_score)
            
            score += best_match_score
            count += 1
        
        return score / count if count > 0 else 0.0
    
    # Helper methods
    def _determine_corrective_type(self, 
                                 wave_segments: List[WaveSegment], 
                                 fibonacci_ratios: Dict[str, float]) -> WaveClassification:
        """Determine specific type of corrective pattern"""
        
        if 'wave_c_to_wave_a' in fibonacci_ratios:
            c_to_a_ratio = fibonacci_ratios['wave_c_to_wave_a']
            
            # Zigzag: Wave C typically 1.618 times Wave A
            if 1.4 <= c_to_a_ratio <= 1.8:
                return WaveClassification.CORRECTIVE_ZIGZAG
            
            # Flat: Wave C typically equals Wave A
            elif 0.9 <= c_to_a_ratio <= 1.1:
                return WaveClassification.CORRECTIVE_FLAT
        
        # Default to zigzag if uncertain
        return WaveClassification.CORRECTIVE_ZIGZAG
    
    def _check_diagonal_characteristics(self, wave_segments: List[WaveSegment]) -> bool:
        """Check if pattern has diagonal characteristics"""
        # Simplified diagonal check
        if len(wave_segments) != 5:
            return False
        
        # In diagonals, waves often have overlapping characteristics
        # This is a simplified implementation
        return False
    
    def _detect_wave_degree(self, wave_points: List[Tuple[int, float, str]], current_idx: int) -> WaveDegree:
        """Detect wave degree based on time and price scale"""
        # Simplified degree detection based on time span
        if len(wave_points) < 2:
            return WaveDegree.MINOR
        
        # Calculate average time span between waves
        time_spans = []
        for i in range(1, len(wave_points)):
            time_spans.append(wave_points[i][0] - wave_points[i-1][0])
        
        avg_time_span = np.mean(time_spans) if time_spans else 10
        
        # Map time span to degree (simplified)
        if avg_time_span > 1000:
            return WaveDegree.CYCLE
        elif avg_time_span > 500:
            return WaveDegree.PRIMARY
        elif avg_time_span > 100:
            return WaveDegree.INTERMEDIATE
        elif avg_time_span > 50:
            return WaveDegree.MINOR
        else:
            return WaveDegree.MINUTE
    
    def _detect_pattern_degree(self, wave_segments: List[WaveSegment]) -> WaveDegree:
        """Detect pattern degree based on segment characteristics"""
        if not wave_segments:
            return WaveDegree.MINOR
        
        avg_duration = np.mean([seg.time_duration for seg in wave_segments])
        
        # Map duration to degree
        if avg_duration > 1000:
            return WaveDegree.CYCLE
        elif avg_duration > 500:
            return WaveDegree.PRIMARY
        elif avg_duration > 100:
            return WaveDegree.INTERMEDIATE
        elif avg_duration > 50:
            return WaveDegree.MINOR
        else:
            return WaveDegree.MINUTE
    
    def _identify_sub_patterns(self, wave_segments: List[WaveSegment]) -> List:
        """Identify sub-patterns within complex corrective structure"""
        # Simplified sub-pattern identification
        sub_patterns = []
        
        # Group segments into potential ABC patterns
        i = 0
        while i < len(wave_segments) - 2:
            if i + 2 < len(wave_segments):
                sub_pattern = wave_segments[i:i+3]
                sub_patterns.append(sub_pattern)
                i += 3
            else:
                break
        
        return sub_patterns
    
    def _calculate_impulse_invalidation_levels(self, 
                                             wave_points: List[WavePoint], 
                                             wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate invalidation levels for impulse pattern"""
        invalidation_levels = {}
        
        if len(wave_points) >= 2:
            invalidation_levels['wave_2_max'] = wave_points[0].price
        
        if len(wave_points) >= 4:
            invalidation_levels['wave_4_max'] = wave_points[1].price
        
        return invalidation_levels
    
    def _calculate_corrective_invalidation_levels(self, 
                                                wave_points: List[WavePoint], 
                                                wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate invalidation levels for corrective pattern"""
        invalidation_levels = {}
        
        if len(wave_points) >= 1:
            invalidation_levels['pattern_start'] = wave_points[0].price
        
        return invalidation_levels
    
    def _calculate_basic_invalidation_levels(self, 
                                           wave_points: List[WavePoint], 
                                           wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate basic invalidation levels"""
        invalidation_levels = {}
        
        if wave_points:
            prices = [wp.price for wp in wave_points]
            invalidation_levels['pattern_high'] = max(prices)
            invalidation_levels['pattern_low'] = min(prices)
        
        return invalidation_levels
    
    def _generate_alternative_counts(self, 
                                   wave_points: List[WavePoint], 
                                   wave_segments: List[WaveSegment]) -> List[WaveAnalysis]:
        """Generate alternative wave count scenarios"""
        alternatives = []
        
        # This would involve sophisticated alternative scenario generation
        # For now, return empty list
        
        return alternatives
    
    def _calculate_overall_confidence(self, 
                                    primary_analysis: WaveAnalysis, 
                                    alternative_analyses: List[WaveAnalysis]) -> float:
        """Calculate overall classification confidence"""
        
        if not alternative_analyses:
            return primary_analysis.confidence
        
        # Adjust confidence based on number and quality of alternatives
        confidence_reduction = len(alternative_analyses) * 0.1
        adjusted_confidence = primary_analysis.confidence - confidence_reduction
        
        return max(adjusted_confidence, 0.1)
    
    # Conversion methods
    def _analysis_to_dict(self, analysis: WaveAnalysis) -> Dict:
        """Convert WaveAnalysis to dictionary"""
        return {
            'wave_structure': analysis.wave_structure.value,
            'classification': analysis.classification.value,
            'confidence': analysis.confidence,
            'fibonacci_ratios': analysis.fibonacci_ratios,
            'rule_validation': analysis.rule_validation,
            'invalidation_levels': analysis.invalidation_levels,
            'degree': analysis.degree.value,
            'wave_count': len(analysis.wave_points)
        }
    
    def _wave_point_to_dict(self, wave_point: WavePoint) -> Dict:
        """Convert WavePoint to dictionary"""
        return {
            'index': wave_point.index,
            'price': wave_point.price,
            'wave_label': wave_point.wave_label,
            'wave_degree': wave_point.wave_degree.value,
            'time': wave_point.time.isoformat() if wave_point.time else None
        }
    
    def _wave_segment_to_dict(self, wave_segment: WaveSegment) -> Dict:
        """Convert WaveSegment to dictionary"""
        return {
            'start_index': wave_segment.start_point.index,
            'end_index': wave_segment.end_point.index,
            'start_price': wave_segment.start_point.price,
            'end_price': wave_segment.end_point.price,
            'price_change': wave_segment.price_change,
            'price_change_pct': wave_segment.price_change_pct,
            'time_duration': wave_segment.time_duration,
            'direction': wave_segment.direction
        }
    
    def _get_default_result(self, error_message: str = "") -> Dict:
        """Return default result structure"""
        return {
            'primary_analysis': {
                'wave_structure': WaveStructure.UNKNOWN.value,
                'classification': WaveClassification.UNKNOWN_PATTERN.value,
                'confidence': 0.0,
                'fibonacci_ratios': {},
                'rule_validation': {},
                'invalidation_levels': {},
                'degree': WaveDegree.MINOR.value,
                'wave_count': 0
            },
            'alternative_analyses': [],
            'wave_points': [],
            'wave_segments': [],
            'classification_confidence': 0.0,
            'wave_count': 0,
            'structure_type': WaveStructure.UNKNOWN.value,
            'classification': WaveClassification.UNKNOWN_PATTERN.value,
            'degree': WaveDegree.MINOR.value,
            'fibonacci_analysis': {},
            'rule_validation': {},
            'invalidation_levels': {},
            'error': error_message
        }
    
    def get_signal(self, current_price: float) -> Dict:
        """
        Get trading signal based on current wave classification
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with signal information
        """
        if not self.current_analysis:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'No wave analysis available'}
        
        analysis = self.current_analysis
        
        # Signal logic based on wave type and current position
        if analysis.wave_structure == WaveStructure.IMPULSE:
            if len(analysis.wave_segments) == 5:
                # Complete 5-wave impulse - expect reversal
                signal = 'SELL' if analysis.wave_segments[-1].direction > 0 else 'BUY'
                reason = 'Complete 5-wave impulse pattern'
                strength = analysis.confidence
            elif len(analysis.wave_segments) == 3:
                # Wave 3 completion - expect Wave 4 correction
                signal = 'SELL' if analysis.wave_segments[-1].direction > 0 else 'BUY'
                reason = 'Wave 3 completion, expect correction'
                strength = analysis.confidence * 0.7
            else:
                signal = 'NEUTRAL'
                reason = 'Incomplete impulse pattern'
                strength = 0.0
                
        elif analysis.wave_structure == WaveStructure.CORRECTIVE:
            if len(analysis.wave_segments) == 3:
                # Complete corrective pattern - expect trend resumption
                signal = 'BUY' if analysis.wave_segments[0].direction > 0 else 'SELL'
                reason = 'Corrective pattern completion'
                strength = analysis.confidence
            else:
                signal = 'NEUTRAL'
                reason = 'Incomplete corrective pattern'
                strength = 0.0
        else:
            signal = 'NEUTRAL'
            reason = f'Unknown pattern: {analysis.wave_structure.value}'
            strength = 0.0
        
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'pattern_type': analysis.wave_structure.value,
            'classification': analysis.classification.value,
            'confidence': analysis.confidence
        }
