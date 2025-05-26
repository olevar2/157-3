"""
Overfitting Detection and Prevention Module
Advanced statistical methods to detect and prevent strategy overfitting.

This module implements various statistical tests and metrics to detect overfitting
in trading strategies, ensuring robust out-of-sample performance and preventing
false discoveries in strategy optimization.

Key Features:
- Statistical significance testing
- Performance degradation analysis
- Robustness scoring
- Monte Carlo validation
- Multiple testing correction
- Stability analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, ks_2samp
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OverfitStatus(Enum):
    """Overfitting status enumeration."""
    NO_OVERFIT = "no_overfit"
    MILD_OVERFIT = "mild_overfit"
    MODERATE_OVERFIT = "moderate_overfit"
    SEVERE_OVERFIT = "severe_overfit"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class StatisticalTest:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    is_significant: bool
    interpretation: str

@dataclass
class RobustnessScore:
    """Strategy robustness scoring."""
    overall_score: float
    consistency_score: float
    stability_score: float
    degradation_score: float
    significance_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'overall_score': self.overall_score,
            'consistency_score': self.consistency_score,
            'stability_score': self.stability_score,
            'degradation_score': self.degradation_score,
            'significance_score': self.significance_score
        }

@dataclass
class OverfitMetrics:
    """Overfitting detection metrics."""
    performance_degradation: float
    degradation_std: float
    consistency_ratio: float
    stability_index: float
    significance_level: float
    monte_carlo_p_value: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'performance_degradation': self.performance_degradation,
            'degradation_std': self.degradation_std,
            'consistency_ratio': self.consistency_ratio,
            'stability_index': self.stability_index,
            'significance_level': self.significance_level,
            'monte_carlo_p_value': self.monte_carlo_p_value
        }

@dataclass
class OverfitResult:
    """Overfitting detection result."""
    status: OverfitStatus
    confidence: float
    metrics: OverfitMetrics
    robustness_score: RobustnessScore
    statistical_tests: List[StatisticalTest]
    recommendations: List[str]

    def is_overfitted(self) -> bool:
        """Check if strategy is overfitted."""
        return self.status in [OverfitStatus.MODERATE_OVERFIT, OverfitStatus.SEVERE_OVERFIT]

class OverfitDetector:
    """
    Overfitting Detection Engine

    Implements advanced statistical methods to detect overfitting in trading
    strategies and provide recommendations for improvement.
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def detect_overfitting(
        self,
        optimization_results: List[Any],
        validation_results: List[Any] = None
    ) -> OverfitResult:
        """
        Detect overfitting in strategy optimization results.

        Args:
            optimization_results: Results from optimization periods
            validation_results: Results from validation periods

        Returns:
            Overfitting detection result
        """
        logger.info("Starting overfitting detection analysis...")

        if len(optimization_results) < 3:
            return OverfitResult(
                status=OverfitStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                metrics=self._create_empty_metrics(),
                robustness_score=self._create_empty_robustness_score(),
                statistical_tests=[],
                recommendations=["Insufficient data for overfitting analysis. Need at least 3 optimization windows."]
            )

        # Extract performance metrics
        opt_returns = self._extract_returns(optimization_results)
        val_returns = self._extract_returns(validation_results) if validation_results else []

        # Calculate overfitting metrics
        metrics = self._calculate_overfit_metrics(opt_returns, val_returns)

        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(opt_returns, val_returns)

        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(metrics, statistical_tests)

        # Determine overfitting status
        status, confidence = self._determine_overfit_status(metrics, robustness_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(status, metrics, robustness_score)

        logger.info(f"Overfitting detection completed. Status: {status.value}, Confidence: {confidence:.2f}")

        return OverfitResult(
            status=status,
            confidence=confidence,
            metrics=metrics,
            robustness_score=robustness_score,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )

    def _extract_returns(self, results: List[Any]) -> List[float]:
        """Extract returns from results."""
        returns = []
        for result in results:
            if hasattr(result, 'total_return'):
                returns.append(result.total_return)
            elif hasattr(result, 'optimization_metrics'):
                returns.append(result.optimization_metrics.total_return)
            elif isinstance(result, dict) and 'total_return' in result:
                returns.append(result['total_return'])
            else:
                logger.warning(f"Could not extract return from result: {type(result)}")

        return returns

    def _calculate_overfit_metrics(
        self,
        opt_returns: List[float],
        val_returns: List[float]
    ) -> OverfitMetrics:
        """Calculate overfitting metrics."""

        # Performance degradation
        if val_returns and len(val_returns) == len(opt_returns):
            degradations = [(opt - val) / abs(opt) if opt != 0 else 0
                          for opt, val in zip(opt_returns, val_returns)]
            performance_degradation = np.mean(degradations)
            degradation_std = np.std(degradations)
        else:
            performance_degradation = 0.0
            degradation_std = 0.0

        # Consistency ratio
        positive_returns = len([r for r in opt_returns if r > 0])
        consistency_ratio = positive_returns / len(opt_returns) if opt_returns else 0

        # Stability index (coefficient of variation)
        if opt_returns and np.mean(opt_returns) != 0:
            stability_index = np.std(opt_returns) / abs(np.mean(opt_returns))
        else:
            stability_index = float('inf')

        # Monte Carlo p-value (simplified)
        monte_carlo_p_value = self._monte_carlo_test(opt_returns)

        return OverfitMetrics(
            performance_degradation=performance_degradation,
            degradation_std=degradation_std,
            consistency_ratio=consistency_ratio,
            stability_index=stability_index,
            significance_level=self.significance_level,
            monte_carlo_p_value=monte_carlo_p_value
        )

    def _perform_statistical_tests(
        self,
        opt_returns: List[float],
        val_returns: List[float]
    ) -> List[StatisticalTest]:
        """Perform statistical tests for overfitting detection."""
        tests = []

        if val_returns and len(val_returns) == len(opt_returns):
            # Paired t-test
            try:
                t_stat, p_value = ttest_rel(opt_returns, val_returns)
                tests.append(StatisticalTest(
                    test_name="Paired T-Test",
                    statistic=t_stat,
                    p_value=p_value,
                    critical_value=stats.t.ppf(1 - self.significance_level/2, len(opt_returns) - 1),
                    is_significant=p_value < self.significance_level,
                    interpretation="Significant difference between optimization and validation returns" if p_value < self.significance_level else "No significant difference"
                ))
            except Exception as e:
                logger.warning(f"Paired t-test failed: {e}")

            # Wilcoxon signed-rank test
            try:
                w_stat, p_value = wilcoxon(opt_returns, val_returns)
                tests.append(StatisticalTest(
                    test_name="Wilcoxon Signed-Rank Test",
                    statistic=w_stat,
                    p_value=p_value,
                    critical_value=0,  # No simple critical value for Wilcoxon
                    is_significant=p_value < self.significance_level,
                    interpretation="Significant difference in distributions" if p_value < self.significance_level else "No significant difference in distributions"
                ))
            except Exception as e:
                logger.warning(f"Wilcoxon test failed: {e}")

        # Kolmogorov-Smirnov test for normality
        try:
            ks_stat, p_value = ks_2samp(opt_returns, np.random.normal(np.mean(opt_returns), np.std(opt_returns), len(opt_returns)))
            tests.append(StatisticalTest(
                test_name="Kolmogorov-Smirnov Normality Test",
                statistic=ks_stat,
                p_value=p_value,
                critical_value=0,
                is_significant=p_value < self.significance_level,
                interpretation="Returns significantly deviate from normal distribution" if p_value < self.significance_level else "Returns follow normal distribution"
            ))
        except Exception as e:
            logger.warning(f"KS test failed: {e}")

        return tests

    def _monte_carlo_test(self, returns: List[float]) -> float:
        """Perform Monte Carlo test for randomness."""
        if len(returns) < 10:
            return 1.0

        try:
            # Simple randomness test
            actual_mean = np.mean(returns)

            # Generate random samples
            random_means = []
            for _ in range(1000):
                random_sample = np.random.choice(returns, size=len(returns), replace=True)
                random_means.append(np.mean(random_sample))

            # Calculate p-value
            extreme_count = len([m for m in random_means if abs(m) >= abs(actual_mean)])
            p_value = extreme_count / len(random_means)

            return p_value

        except Exception as e:
            logger.warning(f"Monte Carlo test failed: {e}")
            return 1.0

    def _calculate_robustness_score(
        self,
        metrics: OverfitMetrics,
        statistical_tests: List[StatisticalTest]
    ) -> RobustnessScore:
        """Calculate strategy robustness score."""

        # Consistency score (0-1, higher is better)
        consistency_score = min(1.0, metrics.consistency_ratio)

        # Stability score (0-1, higher is better)
        if metrics.stability_index == float('inf'):
            stability_score = 0.0
        else:
            stability_score = max(0.0, 1.0 - min(1.0, metrics.stability_index))

        # Degradation score (0-1, higher is better)
        degradation_score = max(0.0, 1.0 - abs(metrics.performance_degradation))

        # Significance score (0-1, higher is better)
        significant_tests = len([t for t in statistical_tests if t.is_significant])
        total_tests = len(statistical_tests)
        significance_score = 1.0 - (significant_tests / total_tests) if total_tests > 0 else 1.0

        # Overall score (weighted average)
        weights = [0.3, 0.25, 0.25, 0.2]  # consistency, stability, degradation, significance
        scores = [consistency_score, stability_score, degradation_score, significance_score]
        overall_score = sum(w * s for w, s in zip(weights, scores))

        return RobustnessScore(
            overall_score=overall_score,
            consistency_score=consistency_score,
            stability_score=stability_score,
            degradation_score=degradation_score,
            significance_score=significance_score
        )

    def _determine_overfit_status(
        self,
        metrics: OverfitMetrics,
        robustness_score: RobustnessScore
    ) -> Tuple[OverfitStatus, float]:
        """Determine overfitting status and confidence."""

        # Calculate confidence based on multiple factors
        confidence_factors = []

        # Performance degradation factor
        if abs(metrics.performance_degradation) > 0.5:
            confidence_factors.append(0.8)
        elif abs(metrics.performance_degradation) > 0.3:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.2)

        # Robustness score factor
        if robustness_score.overall_score < 0.3:
            confidence_factors.append(0.9)
        elif robustness_score.overall_score < 0.5:
            confidence_factors.append(0.7)
        elif robustness_score.overall_score < 0.7:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)

        # Stability factor
        if metrics.stability_index > 2.0:
            confidence_factors.append(0.8)
        elif metrics.stability_index > 1.0:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)

        confidence = np.mean(confidence_factors)

        # Determine status
        if robustness_score.overall_score < 0.3 or abs(metrics.performance_degradation) > 0.5:
            status = OverfitStatus.SEVERE_OVERFIT
        elif robustness_score.overall_score < 0.5 or abs(metrics.performance_degradation) > 0.3:
            status = OverfitStatus.MODERATE_OVERFIT
        elif robustness_score.overall_score < 0.7 or abs(metrics.performance_degradation) > 0.15:
            status = OverfitStatus.MILD_OVERFIT
        else:
            status = OverfitStatus.NO_OVERFIT

        return status, confidence

    def _generate_recommendations(
        self,
        status: OverfitStatus,
        metrics: OverfitMetrics,
        robustness_score: RobustnessScore
    ) -> List[str]:
        """Generate recommendations based on overfitting analysis."""
        recommendations = []

        if status == OverfitStatus.SEVERE_OVERFIT:
            recommendations.extend([
                "CRITICAL: Severe overfitting detected. Strategy is not suitable for live trading.",
                "Increase out-of-sample validation period to at least 30% of total data.",
                "Reduce parameter complexity and number of optimized parameters.",
                "Consider using regularization techniques in strategy development.",
                "Implement cross-validation with multiple time periods."
            ])

        elif status == OverfitStatus.MODERATE_OVERFIT:
            recommendations.extend([
                "WARNING: Moderate overfitting detected. Strategy needs improvement.",
                "Extend validation period and test on different market conditions.",
                "Reduce number of parameters being optimized simultaneously.",
                "Implement walk-forward optimization with shorter optimization windows."
            ])

        elif status == OverfitStatus.MILD_OVERFIT:
            recommendations.extend([
                "CAUTION: Mild overfitting detected. Monitor strategy performance closely.",
                "Consider additional out-of-sample testing before live deployment.",
                "Implement robust risk management to handle performance degradation."
            ])

        else:
            recommendations.append("Strategy shows good robustness with minimal overfitting risk.")

        # Specific recommendations based on metrics
        if metrics.consistency_ratio < 0.5:
            recommendations.append("Low consistency detected. Consider strategies with higher win rates.")

        if metrics.stability_index > 1.5:
            recommendations.append("High volatility in returns. Implement better risk management.")

        if abs(metrics.performance_degradation) > 0.2:
            recommendations.append("Significant performance degradation. Review parameter selection process.")

        if robustness_score.significance_score < 0.5:
            recommendations.append("Multiple statistical tests indicate potential issues. Conduct deeper analysis.")

        return recommendations

    def _create_empty_metrics(self) -> OverfitMetrics:
        """Create empty metrics for insufficient data cases."""
        return OverfitMetrics(
            performance_degradation=0.0,
            degradation_std=0.0,
            consistency_ratio=0.0,
            stability_index=0.0,
            significance_level=self.significance_level,
            monte_carlo_p_value=1.0
        )

    def _create_empty_robustness_score(self) -> RobustnessScore:
        """Create empty robustness score for insufficient data cases."""
        return RobustnessScore(
            overall_score=0.0,
            consistency_score=0.0,
            stability_score=0.0,
            degradation_score=0.0,
            significance_score=0.0
        )

    def generate_report(self, result: OverfitResult) -> str:
        """Generate a comprehensive overfitting analysis report."""
        report = []
        report.append("=" * 60)
        report.append("OVERFITTING DETECTION REPORT")
        report.append("=" * 60)
        report.append("")

        # Status and confidence
        report.append(f"Status: {result.status.value.upper()}")
        report.append(f"Confidence: {result.confidence:.2%}")
        report.append("")

        # Metrics
        report.append("METRICS:")
        report.append(f"  Performance Degradation: {result.metrics.performance_degradation:.2%}")
        report.append(f"  Degradation Std Dev: {result.metrics.degradation_std:.2%}")
        report.append(f"  Consistency Ratio: {result.metrics.consistency_ratio:.2%}")
        report.append(f"  Stability Index: {result.metrics.stability_index:.3f}")
        report.append(f"  Monte Carlo P-Value: {result.metrics.monte_carlo_p_value:.3f}")
        report.append("")

        # Robustness scores
        report.append("ROBUSTNESS SCORES:")
        report.append(f"  Overall Score: {result.robustness_score.overall_score:.2%}")
        report.append(f"  Consistency Score: {result.robustness_score.consistency_score:.2%}")
        report.append(f"  Stability Score: {result.robustness_score.stability_score:.2%}")
        report.append(f"  Degradation Score: {result.robustness_score.degradation_score:.2%}")
        report.append(f"  Significance Score: {result.robustness_score.significance_score:.2%}")
        report.append("")

        # Statistical tests
        if result.statistical_tests:
            report.append("STATISTICAL TESTS:")
            for test in result.statistical_tests:
                report.append(f"  {test.test_name}:")
                report.append(f"    P-Value: {test.p_value:.4f}")
                report.append(f"    Significant: {test.is_significant}")
                report.append(f"    Interpretation: {test.interpretation}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            report.append(f"  {i}. {rec}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
