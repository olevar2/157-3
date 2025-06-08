"""
Advanced Overfitting Detection System for Trading Models

This module provides comprehensive overfitting detection using multiple statistical
methods and machine learning techniques to ensure model reliability in live trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from sklearn.model_selection import TimeSeriesSplit, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging
import json
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class OverfittingMetrics:
    """Comprehensive overfitting detection metrics"""
    validation_degradation: float
    complexity_penalty: float
    temporal_stability: float
    cross_validation_variance: float
    learning_curve_slope: float
    feature_importance_stability: float
    prediction_confidence_ratio: float
    sharpe_ratio_stability: float
    maximum_drawdown_consistency: float
    out_of_sample_correlation: float
    information_coefficient_decay: float
    regime_robustness_score: float
    overall_overfitting_score: float
    confidence_level: str
    recommendation: str


@dataclass
class ModelPerformanceData:
    """Structure for model performance data"""
    in_sample_returns: List[float]
    out_sample_returns: List[float]
    validation_returns: List[float]
    feature_importance: Dict[str, float]
    predictions: List[float]
    actual_values: List[float]
    timestamps: List[datetime]
    model_complexity: int
    training_duration: float


class OverfitDetector:
    """
    Advanced Overfitting Detection System
    
    Implements multiple detection methods:
    - Cross-validation degradation analysis
    - Temporal stability testing
    - Feature importance consistency
    - Learning curve analysis
    - Statistical significance testing
    - Regime change robustness
    """
    
    def __init__(self, 
                 validation_threshold: float = 0.15,
                 confidence_level: float = 0.95,
                 min_sample_size: int = 100,
                 max_complexity_ratio: float = 0.1):
        """
        Initialize OverfitDetector
        
        Args:
            validation_threshold: Maximum acceptable validation degradation
            confidence_level: Statistical confidence level for tests
            min_sample_size: Minimum samples required for reliable testing
            max_complexity_ratio: Maximum model complexity relative to sample size
        """
        self.validation_threshold = validation_threshold
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.max_complexity_ratio = max_complexity_ratio
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Detection history
        self.detection_history: List[OverfittingMetrics] = []
        
        # Thresholds for different metrics
        self.thresholds = {
            'validation_degradation': validation_threshold,
            'complexity_penalty': 0.2,
            'temporal_stability': 0.7,
            'cv_variance': 0.3,
            'learning_curve_slope': -0.1,
            'feature_stability': 0.6,
            'confidence_ratio': 0.8,
            'sharpe_stability': 0.5,
            'drawdown_consistency': 0.7,
            'oos_correlation': 0.5,
            'ic_decay': 0.3,
            'regime_robustness': 0.6
        }
        
        self.logger.info("OverfitDetector initialized with advanced detection methods")

    def detect_overfitting(self, 
                          model_performance: Union[Dict, ModelPerformanceData],
                          model_object: Optional[Any] = None,
                          detailed_analysis: bool = True) -> OverfittingMetrics:
        """
        Comprehensive overfitting detection analysis
        
        Args:
            model_performance: Performance data for analysis
            model_object: Trained model object for complexity analysis
            detailed_analysis: Whether to perform full detailed analysis
            
        Returns:
            OverfittingMetrics: Comprehensive overfitting assessment
        """
        try:
            # Convert input to standardized format
            if isinstance(model_performance, dict):
                perf_data = self._dict_to_performance_data(model_performance)
            else:
                perf_data = model_performance
                
            self.logger.info("Starting comprehensive overfitting detection")
            
            # Validate input data
            self._validate_performance_data(perf_data)
            
            # Run all detection methods
            metrics = {}
            
            # 1. Validation degradation analysis
            metrics['validation_degradation'] = self._calculate_validation_degradation(perf_data)
            
            # 2. Model complexity penalty
            metrics['complexity_penalty'] = self._calculate_complexity_penalty(perf_data, model_object)
            
            # 3. Temporal stability analysis
            metrics['temporal_stability'] = self._analyze_temporal_stability(perf_data)
            
            # 4. Cross-validation variance
            metrics['cross_validation_variance'] = self._calculate_cv_variance(perf_data)
            
            if detailed_analysis:
                # 5. Learning curve analysis
                metrics['learning_curve_slope'] = self._analyze_learning_curve(perf_data)
                
                # 6. Feature importance stability
                metrics['feature_importance_stability'] = self._analyze_feature_stability(perf_data)
                
                # 7. Prediction confidence analysis
                metrics['prediction_confidence_ratio'] = self._analyze_prediction_confidence(perf_data)
                
                # 8. Financial metrics stability
                metrics['sharpe_ratio_stability'] = self._analyze_sharpe_stability(perf_data)
                metrics['maximum_drawdown_consistency'] = self._analyze_drawdown_consistency(perf_data)
                
                # 9. Out-of-sample correlation
                metrics['out_of_sample_correlation'] = self._calculate_oos_correlation(perf_data)
                
                # 10. Information coefficient decay
                metrics['information_coefficient_decay'] = self._analyze_ic_decay(perf_data)
                
                # 11. Regime robustness
                metrics['regime_robustness_score'] = self._analyze_regime_robustness(perf_data)
            
            # Calculate overall overfitting score
            overall_score = self._calculate_overall_score(metrics)
            confidence_level, recommendation = self._determine_recommendation(overall_score, metrics)
            
            # Create comprehensive results
            overfitting_metrics = OverfittingMetrics(
                validation_degradation=metrics.get('validation_degradation', 0.0),
                complexity_penalty=metrics.get('complexity_penalty', 0.0),
                temporal_stability=metrics.get('temporal_stability', 1.0),
                cross_validation_variance=metrics.get('cross_validation_variance', 0.0),
                learning_curve_slope=metrics.get('learning_curve_slope', 0.0),
                feature_importance_stability=metrics.get('feature_importance_stability', 1.0),
                prediction_confidence_ratio=metrics.get('prediction_confidence_ratio', 1.0),
                sharpe_ratio_stability=metrics.get('sharpe_ratio_stability', 1.0),
                maximum_drawdown_consistency=metrics.get('maximum_drawdown_consistency', 1.0),
                out_of_sample_correlation=metrics.get('out_of_sample_correlation', 1.0),
                information_coefficient_decay=metrics.get('information_coefficient_decay', 0.0),
                regime_robustness_score=metrics.get('regime_robustness_score', 1.0),
                overall_overfitting_score=overall_score,
                confidence_level=confidence_level,
                recommendation=recommendation
            )
            
            # Store in history
            self.detection_history.append(overfitting_metrics)
            
            self.logger.info(f"Overfitting detection completed. Overall score: {overall_score:.3f}")
            return overfitting_metrics
            
        except Exception as e:
            self.logger.error(f"Error in overfitting detection: {str(e)}")
            raise

    def _calculate_validation_degradation(self, perf_data: ModelPerformanceData) -> float:
        """Calculate validation performance degradation"""
        try:
            if not perf_data.in_sample_returns or not perf_data.out_sample_returns:
                return 0.0
                
            is_sharpe = self._calculate_sharpe_ratio(perf_data.in_sample_returns)
            oos_sharpe = self._calculate_sharpe_ratio(perf_data.out_sample_returns)
            
            if is_sharpe <= 0:
                return 1.0
                
            degradation = max(0, (is_sharpe - oos_sharpe) / is_sharpe)
            return min(1.0, degradation)
            
        except Exception:
            return 0.5

    def _calculate_complexity_penalty(self, 
                                    perf_data: ModelPerformanceData, 
                                    model_object: Optional[Any]) -> float:
        """Calculate model complexity penalty"""
        try:
            sample_size = len(perf_data.in_sample_returns)
            complexity = perf_data.model_complexity
            
            if sample_size < self.min_sample_size:
                return 1.0
                
            complexity_ratio = complexity / sample_size
            penalty = min(1.0, complexity_ratio / self.max_complexity_ratio)
            
            return penalty
            
        except Exception:
            return 0.3

    def _analyze_temporal_stability(self, perf_data: ModelPerformanceData) -> float:
        """Analyze temporal stability of predictions"""
        try:
            if len(perf_data.predictions) < 50:
                return 0.5
                
            # Split into temporal segments
            n_segments = 5
            segment_size = len(perf_data.predictions) // n_segments
            segment_correlations = []
            
            for i in range(n_segments - 1):
                start1 = i * segment_size
                end1 = (i + 1) * segment_size
                start2 = (i + 1) * segment_size
                end2 = (i + 2) * segment_size
                
                seg1_pred = perf_data.predictions[start1:end1]
                seg1_actual = perf_data.actual_values[start1:end1]
                seg2_pred = perf_data.predictions[start2:end2]
                seg2_actual = perf_data.actual_values[start2:end2]
                
                corr1 = np.corrcoef(seg1_pred, seg1_actual)[0, 1]
                corr2 = np.corrcoef(seg2_pred, seg2_actual)[0, 1]
                
                if not (np.isnan(corr1) or np.isnan(corr2)):
                    segment_correlations.append(abs(corr1 - corr2))
            
            if not segment_correlations:
                return 0.5
                
            stability = 1.0 - np.mean(segment_correlations)
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.5

    def _calculate_cv_variance(self, perf_data: ModelPerformanceData) -> float:
        """Calculate cross-validation variance"""
        try:
            if len(perf_data.validation_returns) < 30:
                return 0.3
                
            # Simulate time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            X = np.array(perf_data.predictions).reshape(-1, 1)
            y = np.array(perf_data.actual_values)
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                if len(val_idx) > 5:
                    val_score = np.corrcoef(y[val_idx], X[val_idx].flatten())[0, 1]
                    if not np.isnan(val_score):
                        cv_scores.append(val_score)
            
            if len(cv_scores) < 3:
                return 0.3
                
            variance = np.std(cv_scores)
            return min(1.0, variance * 3)  # Scale variance
            
        except Exception:
            return 0.3

    def _analyze_learning_curve(self, perf_data: ModelPerformanceData) -> float:
        """Analyze learning curve slope"""
        try:
            if len(perf_data.in_sample_returns) < 50:
                return 0.0
                
            # Create learning curve segments
            n_points = 10
            segment_size = len(perf_data.in_sample_returns) // n_points
            learning_scores = []
            
            for i in range(1, n_points + 1):
                end_idx = i * segment_size
                segment_returns = perf_data.in_sample_returns[:end_idx]
                score = self._calculate_sharpe_ratio(segment_returns)
                learning_scores.append(score)
            
            # Calculate slope of learning curve
            x = np.arange(len(learning_scores))
            slope, _, _, _, _ = stats.linregress(x, learning_scores)
            
            # Negative slope indicates potential overfitting
            return max(-1.0, min(0.0, slope))
            
        except Exception:
            return 0.0

    def _analyze_feature_stability(self, perf_data: ModelPerformanceData) -> float:
        """Analyze feature importance stability"""
        try:
            if not perf_data.feature_importance:
                return 0.5
                
            # Simulate feature importance changes over time
            # In practice, this would compare feature importance across time periods
            importance_values = list(perf_data.feature_importance.values())
            
            if len(importance_values) < 3:
                return 0.5
                
            # Calculate coefficient of variation
            cv = np.std(importance_values) / (np.mean(importance_values) + 1e-8)
            stability = 1.0 / (1.0 + cv)
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.5

    def _analyze_prediction_confidence(self, perf_data: ModelPerformanceData) -> float:
        """Analyze prediction confidence ratio"""
        try:
            predictions = np.array(perf_data.predictions)
            actual = np.array(perf_data.actual_values)
            
            if len(predictions) < 10:
                return 0.5
                
            # Calculate prediction errors
            errors = np.abs(predictions - actual)
            
            # High confidence should correlate with low errors
            # Simple confidence proxy: consistency of predictions
            pred_consistency = 1.0 - (np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8))
            error_consistency = 1.0 - (np.std(errors) / (np.mean(errors) + 1e-8))
            
            confidence_ratio = (pred_consistency + error_consistency) / 2
            return max(0.0, min(1.0, confidence_ratio))
            
        except Exception:
            return 0.5

    def _analyze_sharpe_stability(self, perf_data: ModelPerformanceData) -> float:
        """Analyze Sharpe ratio stability across periods"""
        try:
            returns = perf_data.out_sample_returns
            if len(returns) < 50:
                return 0.5
                
            # Calculate rolling Sharpe ratios
            window = len(returns) // 5
            rolling_sharpes = []
            
            for i in range(len(returns) - window + 1):
                window_returns = returns[i:i + window]
                sharpe = self._calculate_sharpe_ratio(window_returns)
                rolling_sharpes.append(sharpe)
            
            if len(rolling_sharpes) < 3:
                return 0.5
                
            # Calculate stability as inverse of coefficient of variation
            mean_sharpe = np.mean(rolling_sharpes)
            std_sharpe = np.std(rolling_sharpes)
            
            if abs(mean_sharpe) < 1e-8:
                return 0.0
                
            stability = 1.0 / (1.0 + abs(std_sharpe / mean_sharpe))
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.5

    def _analyze_drawdown_consistency(self, perf_data: ModelPerformanceData) -> float:
        """Analyze maximum drawdown consistency"""
        try:
            returns = perf_data.out_sample_returns
            if len(returns) < 30:
                return 0.5
                
            # Calculate rolling maximum drawdowns
            n_periods = 5
            period_size = len(returns) // n_periods
            drawdowns = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = min((i + 1) * period_size, len(returns))
                period_returns = returns[start_idx:end_idx]
                
                cumulative = np.cumprod(1 + np.array(period_returns))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
                drawdowns.append(abs(max_drawdown))
            
            if len(drawdowns) < 2:
                return 0.5
                
            # Consistency is inverse of coefficient of variation
            mean_dd = np.mean(drawdowns)
            std_dd = np.std(drawdowns)
            
            if mean_dd < 1e-8:
                return 1.0
                
            consistency = 1.0 / (1.0 + std_dd / mean_dd)
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5

    def _calculate_oos_correlation(self, perf_data: ModelPerformanceData) -> float:
        """Calculate out-of-sample correlation"""
        try:
            if (len(perf_data.predictions) < 10 or 
                len(perf_data.actual_values) < 10):
                return 0.0
                
            # Split data for out-of-sample analysis
            split_point = len(perf_data.predictions) // 2
            
            oos_predictions = perf_data.predictions[split_point:]
            oos_actual = perf_data.actual_values[split_point:]
            
            correlation = np.corrcoef(oos_predictions, oos_actual)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
                
            return max(0.0, correlation)
            
        except Exception:
            return 0.0

    def _analyze_ic_decay(self, perf_data: ModelPerformanceData) -> float:
        """Analyze information coefficient decay"""
        try:
            if len(perf_data.predictions) < 50:
                return 0.3
                
            # Calculate rolling information coefficients
            window = 20
            ics = []
            
            for i in range(len(perf_data.predictions) - window + 1):
                pred_window = perf_data.predictions[i:i + window]
                actual_window = perf_data.actual_values[i:i + window]
                
                ic = np.corrcoef(pred_window, actual_window)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
            
            if len(ics) < 10:
                return 0.3
                
            # Calculate decay trend
            x = np.arange(len(ics))
            slope, _, _, _, _ = stats.linregress(x, ics)
            
            # Negative slope indicates decay (potential overfitting)
            decay = max(0.0, -slope)
            return min(1.0, decay * 10)  # Scale decay
            
        except Exception:
            return 0.3

    def _analyze_regime_robustness(self, perf_data: ModelPerformanceData) -> float:
        """Analyze robustness across market regimes"""
        try:
            returns = perf_data.out_sample_returns
            if len(returns) < 60:
                return 0.5
                
            # Identify market regimes using volatility
            volatility = pd.Series(returns).rolling(window=20).std()
            high_vol_threshold = volatility.quantile(0.7)
            low_vol_threshold = volatility.quantile(0.3)
            
            high_vol_mask = volatility > high_vol_threshold
            low_vol_mask = volatility < low_vol_threshold
            
            high_vol_returns = [r for i, r in enumerate(returns) if high_vol_mask.iloc[i]]
            low_vol_returns = [r for i, r in enumerate(returns) if low_vol_mask.iloc[i]]
            
            if len(high_vol_returns) < 10 or len(low_vol_returns) < 10:
                return 0.5
                
            # Compare performance across regimes
            high_vol_sharpe = self._calculate_sharpe_ratio(high_vol_returns)
            low_vol_sharpe = self._calculate_sharpe_ratio(low_vol_returns)
            
            if abs(high_vol_sharpe) < 1e-8 and abs(low_vol_sharpe) < 1e-8:
                return 0.5
                
            # Robustness is similarity of performance across regimes
            if abs(high_vol_sharpe) < 1e-8 or abs(low_vol_sharpe) < 1e-8:
                robustness = 0.3
            else:
                ratio = min(high_vol_sharpe, low_vol_sharpe) / max(high_vol_sharpe, low_vol_sharpe)
                robustness = max(0.0, ratio)
                
            return min(1.0, robustness)
            
        except Exception:
            return 0.5

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall overfitting score"""
        # Weights for different metrics
        weights = {
            'validation_degradation': 0.25,
            'complexity_penalty': 0.15,
            'temporal_stability': 0.15,
            'cross_validation_variance': 0.15,
            'learning_curve_slope': 0.10,
            'feature_importance_stability': 0.05,
            'prediction_confidence_ratio': 0.05,
            'sharpe_ratio_stability': 0.05,
            'maximum_drawdown_consistency': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                # Convert stability metrics to overfitting indicators
                if metric in ['temporal_stability', 'feature_importance_stability', 
                             'prediction_confidence_ratio', 'sharpe_ratio_stability',
                             'maximum_drawdown_consistency']:
                    overfitting_indicator = 1.0 - value
                else:
                    overfitting_indicator = value
                    
                score += weights[metric] * overfitting_indicator
                total_weight += weights[metric]
        
        if total_weight > 0:
            score = score / total_weight
        else:
            score = 0.5
            
        return max(0.0, min(1.0, score))

    def _determine_recommendation(self, 
                                overall_score: float, 
                                metrics: Dict[str, float]) -> Tuple[str, str]:
        """Determine confidence level and recommendation"""
        if overall_score < 0.2:
            confidence = "HIGH"
            recommendation = "Model shows low overfitting risk. Suitable for live trading."
        elif overall_score < 0.4:
            confidence = "MODERATE"
            recommendation = "Model shows moderate overfitting risk. Monitor closely in live trading."
        elif overall_score < 0.6:
            confidence = "LOW"
            recommendation = "Model shows significant overfitting risk. Consider retraining with regularization."
        else:
            confidence = "VERY_LOW"
            recommendation = "Model shows high overfitting risk. Retrain with reduced complexity and more data."
            
        return confidence, recommendation

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
                
            returns_array = np.array(returns)
            excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
                
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except Exception:
            return 0.0

    def _dict_to_performance_data(self, data_dict: Dict) -> ModelPerformanceData:
        """Convert dictionary to ModelPerformanceData"""
        return ModelPerformanceData(
            in_sample_returns=data_dict.get('in_sample_returns', []),
            out_sample_returns=data_dict.get('out_sample_returns', []),
            validation_returns=data_dict.get('validation_returns', []),
            feature_importance=data_dict.get('feature_importance', {}),
            predictions=data_dict.get('predictions', []),
            actual_values=data_dict.get('actual_values', []),
            timestamps=data_dict.get('timestamps', []),
            model_complexity=data_dict.get('model_complexity', 10),
            training_duration=data_dict.get('training_duration', 0.0)
        )

    def _validate_performance_data(self, perf_data: ModelPerformanceData):
        """Validate performance data"""
        if len(perf_data.predictions) != len(perf_data.actual_values):
            raise ValueError("Predictions and actual values must have same length")
            
        if len(perf_data.in_sample_returns) < 10:
            self.logger.warning("In-sample returns has fewer than 10 observations")
            
        if len(perf_data.out_sample_returns) < 10:
            self.logger.warning("Out-of-sample returns has fewer than 10 observations")

    def generate_detailed_report(self, 
                               metrics: OverfittingMetrics,
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate detailed overfitting analysis report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "overall_assessment": {
                    "overfitting_score": metrics.overall_overfitting_score,
                    "confidence_level": metrics.confidence_level,
                    "recommendation": metrics.recommendation
                },
                "detailed_metrics": asdict(metrics),
                "interpretation": self._interpret_metrics(metrics),
                "improvement_suggestions": self._generate_improvement_suggestions(metrics)
            }
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}

    def _interpret_metrics(self, metrics: OverfittingMetrics) -> Dict[str, str]:
        """Interpret individual metrics"""
        interpretations = {}
        
        if metrics.validation_degradation > 0.3:
            interpretations["validation_degradation"] = "High validation degradation indicates overfitting"
        elif metrics.validation_degradation > 0.15:
            interpretations["validation_degradation"] = "Moderate validation degradation - monitor closely"
        else:
            interpretations["validation_degradation"] = "Low validation degradation - good generalization"
            
        if metrics.temporal_stability < 0.5:
            interpretations["temporal_stability"] = "Low temporal stability - model may not adapt well to changing conditions"
        else:
            interpretations["temporal_stability"] = "Good temporal stability"
            
        return interpretations

    def _generate_improvement_suggestions(self, metrics: OverfittingMetrics) -> List[str]:
        """Generate improvement suggestions based on metrics"""
        suggestions = []
        
        if metrics.validation_degradation > 0.2:
            suggestions.append("Consider reducing model complexity or adding regularization")
            
        if metrics.complexity_penalty > 0.3:
            suggestions.append("Model complexity is high relative to sample size - consider feature selection")
            
        if metrics.temporal_stability < 0.6:
            suggestions.append("Improve temporal stability by using rolling window training or adaptive features")
            
        if metrics.cross_validation_variance > 0.4:
            suggestions.append("High cross-validation variance - consider ensemble methods or more stable features")
            
        return suggestions

    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Get historical overfitting detection results"""
        return [asdict(metrics) for metrics in self.detection_history]

    def clear_history(self):
        """Clear detection history"""
        self.detection_history.clear()
        self.logger.info("Detection history cleared")


# Example usage and testing
if __name__ == "__main__":
    # Example model performance data
    sample_data = ModelPerformanceData(
        in_sample_returns=[0.001, 0.002, -0.001, 0.003, 0.001] * 50,
        out_sample_returns=[0.0005, 0.001, -0.0015, 0.002, 0.0005] * 30,
        validation_returns=[0.0008, 0.0012, -0.0012, 0.0025, 0.0008] * 20,
        feature_importance={'feature_1': 0.3, 'feature_2': 0.25, 'feature_3': 0.2, 'feature_4': 0.25},
        predictions=[0.1, 0.2, -0.1, 0.3, 0.15] * 40,
        actual_values=[0.08, 0.18, -0.12, 0.28, 0.12] * 40,
        timestamps=[datetime.now() + timedelta(days=i) for i in range(200)],
        model_complexity=50,
        training_duration=120.5
    )
    
    # Initialize detector
    detector = OverfitDetector(validation_threshold=0.15)
    
    # Run detection
    results = detector.detect_overfitting(sample_data, detailed_analysis=True)
    
    print(f"Overall Overfitting Score: {results.overall_overfitting_score:.3f}")
    print(f"Confidence Level: {results.confidence_level}")
    print(f"Recommendation: {results.recommendation}")
    
    # Generate detailed report
    report = detector.generate_detailed_report(results)
    print(f"\nDetailed analysis completed with {len(report)} sections")
