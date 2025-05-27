"""
Pattern Recognition Accuracy Validation Module
Comprehensive testing and validation of pattern recognition algorithms

Features:
- Pattern detection accuracy testing
- False positive/negative analysis
- Pattern classification validation
- Performance benchmarking
- Historical pattern validation
- Real-time pattern verification
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns to validate"""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE = "triangle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE = "wedge"
    CHANNEL = "channel"
    SUPPORT_RESISTANCE = "support_resistance"
    TREND_LINE = "trend_line"

class ValidationResult(Enum):
    """Pattern validation results"""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"

@dataclass
class PatternDetection:
    """Pattern detection record"""
    detection_id: str
    pattern_type: PatternType
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    confidence: float
    key_points: List[Tuple[datetime, float]]
    predicted_direction: str  # 'bullish', 'bearish', 'neutral'
    predicted_target: Optional[float]
    detected_by: str  # Algorithm/model name
    timestamp: datetime

@dataclass
class ValidationCase:
    """Pattern validation test case"""
    case_id: str
    pattern_type: PatternType
    symbol: str
    timeframe: str
    historical_data: pd.DataFrame
    expected_patterns: List[PatternDetection]
    validation_period: Tuple[datetime, datetime]
    ground_truth_source: str

@dataclass
class ValidationMetrics:
    """Pattern validation metrics"""
    pattern_type: PatternType
    algorithm_name: str
    total_detections: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    confidence_threshold: float
    avg_detection_time: float
    validation_timestamp: datetime

class PatternRecognitionValidator:
    """
    Comprehensive pattern recognition validation system
    """
    
    def __init__(self):
        self.validation_cases: Dict[str, ValidationCase] = {}
        self.detection_results: Dict[str, List[PatternDetection]] = {}
        self.validation_metrics: Dict[str, ValidationMetrics] = {}
        self.performance_benchmarks = {}
        
        # Validation configuration
        self.config = {
            'confidence_thresholds': [0.5, 0.6, 0.7, 0.8, 0.9],
            'time_tolerance': timedelta(hours=4),  # Tolerance for pattern timing
            'price_tolerance': 0.001,  # 10 pips tolerance for key points
            'min_pattern_duration': timedelta(hours=2),
            'max_pattern_duration': timedelta(days=7),
            'validation_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'validation_timeframes': ['M15', 'H1', 'H4', 'D1']
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_validations': 0,
            'patterns_validated': 0,
            'algorithms_tested': 0,
            'average_accuracy': 0.0,
            'best_performing_algorithm': None,
            'worst_performing_algorithm': None,
            'validation_time_ms': 0.0
        }
        
        logger.info("PatternRecognitionValidator initialized")

    async def create_validation_case(self, case_id: str, pattern_type: PatternType, 
                                   symbol: str, timeframe: str, 
                                   historical_data: pd.DataFrame,
                                   expected_patterns: List[PatternDetection]) -> bool:
        """Create a new validation test case"""
        try:
            validation_period = (
                historical_data.index.min(),
                historical_data.index.max()
            )
            
            validation_case = ValidationCase(
                case_id=case_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                historical_data=historical_data,
                expected_patterns=expected_patterns,
                validation_period=validation_period,
                ground_truth_source="manual_analysis"
            )
            
            self.validation_cases[case_id] = validation_case
            
            logger.info(f"✅ Created validation case {case_id} for {pattern_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create validation case {case_id}: {e}")
            return False

    async def validate_pattern_algorithm(self, algorithm_name: str, 
                                       pattern_detection_function,
                                       case_ids: Optional[List[str]] = None) -> Dict[str, ValidationMetrics]:
        """Validate a pattern recognition algorithm against test cases"""
        try:
            start_time = datetime.now()
            results = {}
            
            # Use all cases if none specified
            if case_ids is None:
                case_ids = list(self.validation_cases.keys())
            
            for case_id in case_ids:
                if case_id not in self.validation_cases:
                    logger.warning(f"Validation case {case_id} not found")
                    continue
                
                case = self.validation_cases[case_id]
                
                # Run pattern detection algorithm
                detected_patterns = await self._run_pattern_detection(
                    pattern_detection_function, case
                )
                
                # Store detection results
                self.detection_results[f"{algorithm_name}_{case_id}"] = detected_patterns
                
                # Calculate validation metrics
                metrics = await self._calculate_validation_metrics(
                    algorithm_name, case, detected_patterns
                )
                
                results[case_id] = metrics
                self.validation_metrics[f"{algorithm_name}_{case_id}"] = metrics
            
            # Calculate overall performance
            overall_metrics = await self._calculate_overall_metrics(algorithm_name, results)
            
            # Update performance stats
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['total_validations'] += 1
            self.performance_stats['validation_time_ms'] = validation_time
            
            logger.info(f"✅ Validated algorithm {algorithm_name} - Overall accuracy: {overall_metrics.get('accuracy', 0):.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to validate algorithm {algorithm_name}: {e}")
            return {}

    async def _run_pattern_detection(self, detection_function, case: ValidationCase) -> List[PatternDetection]:
        """Run pattern detection algorithm on validation case"""
        try:
            # Call the pattern detection function with case data
            detected_patterns = await detection_function(
                case.historical_data,
                case.pattern_type,
                case.symbol,
                case.timeframe
            )
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error running pattern detection: {e}")
            return []

    async def _calculate_validation_metrics(self, algorithm_name: str, 
                                          case: ValidationCase,
                                          detected_patterns: List[PatternDetection]) -> ValidationMetrics:
        """Calculate validation metrics for detected patterns"""
        try:
            expected_patterns = case.expected_patterns
            
            # Match detected patterns with expected patterns
            matches = await self._match_patterns(detected_patterns, expected_patterns)
            
            # Calculate confusion matrix elements
            true_positives = len([m for m in matches if m['match_type'] == ValidationResult.TRUE_POSITIVE])
            false_positives = len([m for m in matches if m['match_type'] == ValidationResult.FALSE_POSITIVE])
            false_negatives = len([m for m in matches if m['match_type'] == ValidationResult.FALSE_NEGATIVE])
            
            # True negatives are harder to calculate for pattern detection
            # Approximate based on time periods without patterns
            total_possible_detections = len(detected_patterns) + len(expected_patterns)
            true_negatives = max(0, total_possible_detections - true_positives - false_positives - false_negatives)
            
            # Calculate metrics
            total_detections = len(detected_patterns)
            accuracy = (true_positives + true_negatives) / max(1, total_possible_detections)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1 = 2 * (precision * recall) / max(1, precision + recall)
            specificity = true_negatives / max(1, true_negatives + false_positives)
            
            # Calculate average detection time
            detection_times = [
                (p.timestamp - p.start_time).total_seconds() * 1000 
                for p in detected_patterns
            ]
            avg_detection_time = statistics.mean(detection_times) if detection_times else 0
            
            return ValidationMetrics(
                pattern_type=case.pattern_type,
                algorithm_name=algorithm_name,
                total_detections=total_detections,
                true_positives=true_positives,
                false_positives=false_positives,
                true_negatives=true_negatives,
                false_negatives=false_negatives,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                specificity=specificity,
                confidence_threshold=0.7,  # Default threshold
                avg_detection_time=avg_detection_time,
                validation_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating validation metrics: {e}")
            raise

    async def _match_patterns(self, detected: List[PatternDetection], 
                            expected: List[PatternDetection]) -> List[Dict]:
        """Match detected patterns with expected patterns"""
        try:
            matches = []
            used_expected = set()
            
            # Match detected patterns with expected patterns
            for detected_pattern in detected:
                best_match = None
                best_score = 0
                
                for i, expected_pattern in enumerate(expected):
                    if i in used_expected:
                        continue
                    
                    # Calculate match score based on time and location overlap
                    score = await self._calculate_pattern_similarity(detected_pattern, expected_pattern)
                    
                    if score > best_score and score > 0.5:  # Minimum similarity threshold
                        best_score = score
                        best_match = i
                
                if best_match is not None:
                    matches.append({
                        'detected': detected_pattern,
                        'expected': expected[best_match],
                        'match_type': ValidationResult.TRUE_POSITIVE,
                        'similarity_score': best_score
                    })
                    used_expected.add(best_match)
                else:
                    matches.append({
                        'detected': detected_pattern,
                        'expected': None,
                        'match_type': ValidationResult.FALSE_POSITIVE,
                        'similarity_score': 0
                    })
            
            # Add false negatives (expected patterns not detected)
            for i, expected_pattern in enumerate(expected):
                if i not in used_expected:
                    matches.append({
                        'detected': None,
                        'expected': expected_pattern,
                        'match_type': ValidationResult.FALSE_NEGATIVE,
                        'similarity_score': 0
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error matching patterns: {e}")
            return []

    async def _calculate_pattern_similarity(self, detected: PatternDetection, 
                                          expected: PatternDetection) -> float:
        """Calculate similarity score between two patterns"""
        try:
            # Time overlap score
            time_overlap = self._calculate_time_overlap(
                (detected.start_time, detected.end_time),
                (expected.start_time, expected.end_time)
            )
            
            # Key points similarity
            points_similarity = self._calculate_key_points_similarity(
                detected.key_points, expected.key_points
            )
            
            # Direction agreement
            direction_match = 1.0 if detected.predicted_direction == expected.predicted_direction else 0.0
            
            # Confidence factor
            confidence_factor = detected.confidence
            
            # Weighted similarity score
            similarity = (
                time_overlap * 0.3 +
                points_similarity * 0.4 +
                direction_match * 0.2 +
                confidence_factor * 0.1
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    def _calculate_time_overlap(self, period1: Tuple[datetime, datetime], 
                              period2: Tuple[datetime, datetime]) -> float:
        """Calculate time overlap between two periods"""
        try:
            start1, end1 = period1
            start2, end2 = period2
            
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start >= overlap_end:
                return 0.0
            
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            total_duration = max(
                (end1 - start1).total_seconds(),
                (end2 - start2).total_seconds()
            )
            
            return overlap_duration / max(1, total_duration)
            
        except Exception as e:
            logger.error(f"Error calculating time overlap: {e}")
            return 0.0

    def _calculate_key_points_similarity(self, points1: List[Tuple[datetime, float]], 
                                       points2: List[Tuple[datetime, float]]) -> float:
        """Calculate similarity between key points of patterns"""
        try:
            if not points1 or not points2:
                return 0.0
            
            # Simple approach: compare closest points
            similarities = []
            
            for time1, price1 in points1:
                best_similarity = 0.0
                
                for time2, price2 in points2:
                    # Time similarity
                    time_diff = abs((time1 - time2).total_seconds())
                    time_sim = max(0, 1 - time_diff / (24 * 3600))  # 24 hour max difference
                    
                    # Price similarity
                    price_diff = abs(price1 - price2) / max(price1, price2)
                    price_sim = max(0, 1 - price_diff / 0.01)  # 1% max difference
                    
                    similarity = (time_sim + price_sim) / 2
                    best_similarity = max(best_similarity, similarity)
                
                similarities.append(best_similarity)
            
            return statistics.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating key points similarity: {e}")
            return 0.0

    async def _calculate_overall_metrics(self, algorithm_name: str, 
                                       results: Dict[str, ValidationMetrics]) -> Dict:
        """Calculate overall performance metrics across all test cases"""
        try:
            if not results:
                return {}
            
            metrics_list = list(results.values())
            
            overall = {
                'algorithm_name': algorithm_name,
                'total_cases': len(metrics_list),
                'accuracy': statistics.mean([m.accuracy for m in metrics_list]),
                'precision': statistics.mean([m.precision for m in metrics_list]),
                'recall': statistics.mean([m.recall for m in metrics_list]),
                'f1_score': statistics.mean([m.f1_score for m in metrics_list]),
                'avg_detection_time': statistics.mean([m.avg_detection_time for m in metrics_list]),
                'total_detections': sum([m.total_detections for m in metrics_list]),
                'total_true_positives': sum([m.true_positives for m in metrics_list]),
                'total_false_positives': sum([m.false_positives for m in metrics_list]),
                'total_false_negatives': sum([m.false_negatives for m in metrics_list])
            }
            
            return overall
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return {}

    def get_validation_report(self, algorithm_name: Optional[str] = None) -> Dict:
        """Generate comprehensive validation report"""
        try:
            if algorithm_name:
                # Report for specific algorithm
                relevant_metrics = {
                    k: v for k, v in self.validation_metrics.items() 
                    if k.startswith(algorithm_name)
                }
            else:
                # Report for all algorithms
                relevant_metrics = self.validation_metrics
            
            report = {
                'validation_summary': {
                    'total_algorithms_tested': len(set(m.algorithm_name for m in relevant_metrics.values())),
                    'total_test_cases': len(relevant_metrics),
                    'validation_timestamp': datetime.now().isoformat()
                },
                'performance_metrics': {},
                'detailed_results': [],
                'performance_stats': self.performance_stats
            }
            
            # Group by algorithm
            by_algorithm = {}
            for key, metrics in relevant_metrics.items():
                alg_name = metrics.algorithm_name
                if alg_name not in by_algorithm:
                    by_algorithm[alg_name] = []
                by_algorithm[alg_name].append(metrics)
            
            # Calculate performance for each algorithm
            for alg_name, metrics_list in by_algorithm.items():
                report['performance_metrics'][alg_name] = {
                    'accuracy': statistics.mean([m.accuracy for m in metrics_list]),
                    'precision': statistics.mean([m.precision for m in metrics_list]),
                    'recall': statistics.mean([m.recall for m in metrics_list]),
                    'f1_score': statistics.mean([m.f1_score for m in metrics_list]),
                    'avg_detection_time': statistics.mean([m.avg_detection_time for m in metrics_list]),
                    'total_test_cases': len(metrics_list)
                }
            
            # Add detailed results
            for metrics in relevant_metrics.values():
                report['detailed_results'].append({
                    'algorithm_name': metrics.algorithm_name,
                    'pattern_type': metrics.pattern_type.value,
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'true_positives': metrics.true_positives,
                    'false_positives': metrics.false_positives,
                    'false_negatives': metrics.false_negatives,
                    'validation_timestamp': metrics.validation_timestamp.isoformat()
                })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return {'error': str(e)}
