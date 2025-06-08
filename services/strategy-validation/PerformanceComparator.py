#!/usr/bin/env python3
"""
Advanced Strategy Performance Comparison System for Platform3

This module provides comprehensive multi-strategy performance comparison including:
- Statistical significance testing for strategy comparisons
- Risk-adjusted performance metrics comparison
- Drawdown and recovery analysis
- Market regime-specific performance evaluation
- Multi-timeframe strategy comparison
- Ensemble strategy optimization
- Performance attribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """Container for comprehensive strategy performance metrics"""
    strategy_id: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    drawdown_duration: int
    win_rate: float
    profit_factor: float
    trades_count: int
    avg_trade_duration: float
    var_95: float
    expected_shortfall: float
    information_ratio: float
    alpha: float
    beta: float
    tracking_error: float
    omega_ratio: float
    tail_ratio: float

@dataclass
class ComparisonResult:
    """Container for strategy comparison results"""
    primary_strategy: str
    comparison_strategy: str
    performance_difference: Dict[str, float]
    statistical_significance: Dict[str, Tuple[float, float]]  # (statistic, p_value)
    risk_adjusted_ranking: Dict[str, int]
    market_regime_performance: Dict[str, Dict[str, float]]
    recommendation: str
    confidence_level: float

class PerformanceComparator:
    """
    Advanced strategy performance comparison system with statistical validation
    """
    
    def __init__(self, 
                 benchmark_strategy: Optional[str] = None,
                 significance_level: float = 0.05,
                 min_trades_threshold: int = 30):
        """
        Initialize the Performance Comparator
        
        Args:
            benchmark_strategy: Reference strategy for comparison
            significance_level: Statistical significance threshold (default 0.05)
            min_trades_threshold: Minimum trades required for valid comparison
        """
        self.benchmark_strategy = benchmark_strategy
        self.significance_level = significance_level
        self.min_trades_threshold = min_trades_threshold
        self.strategies_data = {}
        self.comparison_cache = {}
        self.market_regimes = {}
        
        logger.info(f"PerformanceComparator initialized with benchmark: {benchmark_strategy}")
    
    def load_strategy_data(self, 
                          strategy_id: str, 
                          trades_df: pd.DataFrame,
                          portfolio_values: pd.DataFrame,
                          benchmark_returns: Optional[pd.Series] = None) -> None:
        """
        Load strategy performance data for comparison
        
        Args:
            strategy_id: Unique identifier for the strategy
            trades_df: DataFrame with columns ['entry_time', 'exit_time', 'pnl', 'size', 'symbol']
            portfolio_values: DataFrame with datetime index and 'value' column
            benchmark_returns: Optional benchmark returns for alpha/beta calculation
        """
        try:
            # Validate required columns
            required_trade_cols = ['entry_time', 'exit_time', 'pnl', 'size']
            missing_cols = [col for col in required_trade_cols if col not in trades_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in trades_df: {missing_cols}")
            
            # Calculate returns
            returns = portfolio_values['value'].pct_change().dropna()
            
            # Store strategy data
            self.strategies_data[strategy_id] = {
                'trades': trades_df.copy(),
                'portfolio_values': portfolio_values.copy(),
                'returns': returns,
                'benchmark_returns': benchmark_returns
            }
            
            logger.info(f"Loaded data for strategy {strategy_id}: {len(trades_df)} trades, {len(returns)} return periods")
            
        except Exception as e:
            logger.error(f"Error loading strategy data for {strategy_id}: {str(e)}")
            raise
    
    def calculate_strategy_metrics(self, strategy_id: str) -> StrategyMetrics:
        """
        Calculate comprehensive performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            StrategyMetrics object with all calculated metrics
        """
        try:
            if strategy_id not in self.strategies_data:
                raise ValueError(f"Strategy {strategy_id} data not loaded")
            
            data = self.strategies_data[strategy_id]
            trades = data['trades']
            returns = data['returns']
            portfolio_values = data['portfolio_values']
            benchmark_returns = data.get('benchmark_returns')
            
            # Basic performance metrics
            total_return = (portfolio_values['value'].iloc[-1] / portfolio_values['value'].iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Downside deviation for Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Drawdown duration
            drawdown_periods = self._calculate_drawdown_duration(drawdowns)
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade-based metrics
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Trade duration
            trades['duration'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 3600
            avg_trade_duration = trades['duration'].mean()
            
            # Risk metrics
            var_95 = np.percentile(returns, 5)
            expected_shortfall = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            # Benchmark-relative metrics
            alpha, beta, information_ratio, tracking_error = 0, 1, 0, 0
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                aligned_returns, aligned_benchmark = self._align_returns(returns, benchmark_returns)
                if len(aligned_returns) > 10:  # Minimum data points for reliable regression
                    beta, alpha = np.polyfit(aligned_benchmark, aligned_returns, 1)
                    residuals = aligned_returns - (alpha + beta * aligned_benchmark)
                    tracking_error = residuals.std() * np.sqrt(252)
                    information_ratio = (aligned_returns.mean() - aligned_benchmark.mean()) * 252 / tracking_error if tracking_error > 0 else 0
            
            # Advanced risk metrics
            omega_ratio = self._calculate_omega_ratio(returns, risk_free_rate / 252)
            tail_ratio = self._calculate_tail_ratio(returns)
            
            return StrategyMetrics(
                strategy_id=strategy_id,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                drawdown_duration=max_drawdown_duration,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trades_count=len(trades),
                avg_trade_duration=avg_trade_duration,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                information_ratio=information_ratio,
                alpha=alpha,
                beta=beta,
                tracking_error=tracking_error,
                omega_ratio=omega_ratio,
                tail_ratio=tail_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics for strategy {strategy_id}: {str(e)}")
            raise
    
    def compare_strategies(self, 
                          strategy_a_id: str, 
                          strategy_b_id: str,
                          detailed_analysis: bool = True) -> ComparisonResult:
        """
        Comprehensive comparison between two strategies
        
        Args:
            strategy_a_id: First strategy identifier
            strategy_b_id: Second strategy identifier
            detailed_analysis: Whether to perform detailed statistical tests
            
        Returns:
            ComparisonResult with comprehensive comparison analysis
        """
        try:
            # Validate strategies exist
            if strategy_a_id not in self.strategies_data:
                raise ValueError(f"Strategy {strategy_a_id} data not loaded")
            if strategy_b_id not in self.strategies_data:
                raise ValueError(f"Strategy {strategy_b_id} data not loaded")
            
            # Calculate metrics for both strategies
            metrics_a = self.calculate_strategy_metrics(strategy_a_id)
            metrics_b = self.calculate_strategy_metrics(strategy_b_id)
            
            # Performance differences
            performance_diff = {
                'total_return_diff': metrics_a.total_return - metrics_b.total_return,
                'sharpe_diff': metrics_a.sharpe_ratio - metrics_b.sharpe_ratio,
                'sortino_diff': metrics_a.sortino_ratio - metrics_b.sortino_ratio,
                'max_drawdown_diff': metrics_a.max_drawdown - metrics_b.max_drawdown,
                'win_rate_diff': metrics_a.win_rate - metrics_b.win_rate,
                'profit_factor_diff': metrics_a.profit_factor - metrics_b.profit_factor,
                'volatility_diff': metrics_a.volatility - metrics_b.volatility
            }
            
            # Statistical significance tests
            statistical_tests = {}
            if detailed_analysis:
                statistical_tests = self._perform_statistical_tests(strategy_a_id, strategy_b_id)
            
            # Risk-adjusted ranking
            risk_adjusted_ranking = self._calculate_risk_adjusted_ranking(metrics_a, metrics_b)
            
            # Market regime analysis
            market_regime_performance = {}
            if detailed_analysis:
                market_regime_performance = self._analyze_market_regime_performance(strategy_a_id, strategy_b_id)
            
            # Generate recommendation
            recommendation, confidence = self._generate_recommendation(metrics_a, metrics_b, statistical_tests)
            
            result = ComparisonResult(
                primary_strategy=strategy_a_id,
                comparison_strategy=strategy_b_id,
                performance_difference=performance_diff,
                statistical_significance=statistical_tests,
                risk_adjusted_ranking=risk_adjusted_ranking,
                market_regime_performance=market_regime_performance,
                recommendation=recommendation,
                confidence_level=confidence
            )
            
            # Cache result
            cache_key = f"{strategy_a_id}_vs_{strategy_b_id}"
            self.comparison_cache[cache_key] = result
            
            logger.info(f"Completed comparison: {strategy_a_id} vs {strategy_b_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing strategies {strategy_a_id} vs {strategy_b_id}: {str(e)}")
            raise
    
    def rank_strategies(self, 
                       strategy_ids: List[str],
                       ranking_criteria: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Rank multiple strategies based on various criteria
        
        Args:
            strategy_ids: List of strategy identifiers to rank
            ranking_criteria: List of metrics to use for ranking
            
        Returns:
            Dictionary with ranking results and detailed analysis
        """
        try:
            if ranking_criteria is None:
                ranking_criteria = ['sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 'information_ratio']
            
            # Calculate metrics for all strategies
            all_metrics = {}
            for strategy_id in strategy_ids:
                if strategy_id in self.strategies_data:
                    all_metrics[strategy_id] = self.calculate_strategy_metrics(strategy_id)
                else:
                    logger.warning(f"Strategy {strategy_id} data not loaded, skipping from ranking")
            
            if len(all_metrics) < 2:
                raise ValueError("Need at least 2 strategies for ranking")
            
            # Create ranking for each criterion
            rankings = {}
            for criterion in ranking_criteria:
                if hasattr(list(all_metrics.values())[0], criterion):
                    criterion_values = [(strategy_id, getattr(metrics, criterion)) 
                                      for strategy_id, metrics in all_metrics.items()]
                    # Sort in descending order (higher is better for most metrics)
                    if criterion in ['max_drawdown', 'volatility', 'var_95', 'expected_shortfall']:
                        criterion_values.sort(key=lambda x: x[1])  # Lower is better
                    else:
                        criterion_values.sort(key=lambda x: x[1], reverse=True)  # Higher is better
                    
                    rankings[criterion] = {
                        'ranking': [(i+1, strategy_id, value) for i, (strategy_id, value) in enumerate(criterion_values)],
                        'best_strategy': criterion_values[0][0],
                        'best_value': criterion_values[0][1]
                    }
            
            # Composite ranking using weighted average
            composite_scores = {}
            weights = {criterion: 1.0 for criterion in ranking_criteria}  # Equal weights
            
            for strategy_id in all_metrics.keys():
                score = 0
                for criterion in ranking_criteria:
                    if criterion in rankings:
                        rank = next(rank for rank, sid, _ in rankings[criterion]['ranking'] if sid == strategy_id)
                        # Convert rank to score (lower rank = higher score)
                        score += weights[criterion] * (len(all_metrics) - rank + 1) / len(all_metrics)
                
                composite_scores[strategy_id] = score / len(ranking_criteria)
            
            # Sort by composite score
            composite_ranking = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'individual_rankings': rankings,
                'composite_ranking': composite_ranking,
                'composite_scores': composite_scores,
                'ranking_criteria': ranking_criteria,
                'total_strategies': len(all_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error ranking strategies: {str(e)}")
            raise
    
    def _perform_statistical_tests(self, strategy_a_id: str, strategy_b_id: str) -> Dict[str, Tuple[float, float]]:
        """Perform statistical significance tests between strategies"""
        try:
            returns_a = self.strategies_data[strategy_a_id]['returns']
            returns_b = self.strategies_data[strategy_b_id]['returns']
            
            # Align returns to same time periods
            aligned_a, aligned_b = self._align_returns(returns_a, returns_b)
            
            tests = {}
            
            # T-test for mean returns difference
            if len(aligned_a) > 10 and len(aligned_b) > 10:
                t_stat, t_pvalue = stats.ttest_ind(aligned_a, aligned_b)
                tests['mean_returns_ttest'] = (t_stat, t_pvalue)
            
            # Mann-Whitney U test (non-parametric)
            if len(aligned_a) > 5 and len(aligned_b) > 5:
                u_stat, u_pvalue = stats.mannwhitneyu(aligned_a, aligned_b, alternative='two-sided')
                tests['mann_whitney_u'] = (u_stat, u_pvalue)
            
            # Kolmogorov-Smirnov test for distribution differences
            if len(aligned_a) > 5 and len(aligned_b) > 5:
                ks_stat, ks_pvalue = stats.ks_2samp(aligned_a, aligned_b)
                tests['kolmogorov_smirnov'] = (ks_stat, ks_pvalue)
            
            return tests
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {str(e)}")
            return {}
    
    def _align_returns(self, returns_a: pd.Series, returns_b: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two return series to same time periods"""
        try:
            # Find common time periods
            common_index = returns_a.index.intersection(returns_b.index)
            
            if len(common_index) == 0:
                logger.warning("No common time periods found between return series")
                return pd.Series(), pd.Series()
            
            aligned_a = returns_a.loc[common_index]
            aligned_b = returns_b.loc[common_index]
            
            return aligned_a, aligned_b
            
        except Exception as e:
            logger.error(f"Error aligning returns: {str(e)}")
            return pd.Series(), pd.Series()
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> List[int]:
        """Calculate drawdown durations in periods"""
        try:
            durations = []
            in_drawdown = False
            duration = 0
            
            for dd in drawdowns:
                if dd < 0:  # In drawdown
                    if not in_drawdown:
                        in_drawdown = True
                        duration = 1
                    else:
                        duration += 1
                else:  # Not in drawdown
                    if in_drawdown:
                        durations.append(duration)
                        in_drawdown = False
                        duration = 0
            
            # Handle case where series ends in drawdown
            if in_drawdown and duration > 0:
                durations.append(duration)
            
            return durations
            
        except Exception as e:
            logger.error(f"Error calculating drawdown durations: {str(e)}")
            return []
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio"""
        try:
            if len(returns) == 0:
                return 0
            
            excess_returns = returns - threshold
            positive_returns = excess_returns[excess_returns > 0].sum()
            negative_returns = abs(excess_returns[excess_returns < 0].sum())
            
            if negative_returns == 0:
                return float('inf') if positive_returns > 0 else 1
            
            return positive_returns / negative_returns
            
        except Exception as e:
            logger.error(f"Error calculating Omega ratio: {str(e)}")
            return 0
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        try:
            if len(returns) < 20:  # Need sufficient data
                return 1
            
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            
            if p5 == 0:
                return float('inf') if p95 > 0 else 1
            
            return abs(p95 / p5)
            
        except Exception as e:
            logger.error(f"Error calculating tail ratio: {str(e)}")
            return 1
    
    def _calculate_risk_adjusted_ranking(self, metrics_a: StrategyMetrics, metrics_b: StrategyMetrics) -> Dict[str, int]:
        """Calculate risk-adjusted rankings"""
        try:
            ranking = {}
            
            # Compare key risk-adjusted metrics
            metrics_comparison = [
                ('sharpe_ratio', metrics_a.sharpe_ratio, metrics_b.sharpe_ratio),
                ('sortino_ratio', metrics_a.sortino_ratio, metrics_b.sortino_ratio),
                ('calmar_ratio', metrics_a.calmar_ratio, metrics_b.calmar_ratio),
                ('information_ratio', metrics_a.information_ratio, metrics_b.information_ratio),
                ('omega_ratio', metrics_a.omega_ratio, metrics_b.omega_ratio)
            ]
            
            for metric_name, value_a, value_b in metrics_comparison:
                if value_a > value_b:
                    ranking[f"{metric_name}_{metrics_a.strategy_id}"] = 1
                    ranking[f"{metric_name}_{metrics_b.strategy_id}"] = 2
                elif value_b > value_a:
                    ranking[f"{metric_name}_{metrics_a.strategy_id}"] = 2
                    ranking[f"{metric_name}_{metrics_b.strategy_id}"] = 1
                else:
                    ranking[f"{metric_name}_{metrics_a.strategy_id}"] = 1
                    ranking[f"{metric_name}_{metrics_b.strategy_id}"] = 1
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted ranking: {str(e)}")
            return {}
    
    def _analyze_market_regime_performance(self, strategy_a_id: str, strategy_b_id: str) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different market regimes"""
        try:
            returns_a = self.strategies_data[strategy_a_id]['returns']
            returns_b = self.strategies_data[strategy_b_id]['returns']
            
            # Simple regime classification based on volatility
            rolling_vol = returns_a.rolling(20).std()
            vol_threshold_high = rolling_vol.quantile(0.7)
            vol_threshold_low = rolling_vol.quantile(0.3)
            
            regimes = pd.Series('normal', index=returns_a.index)
            regimes[rolling_vol > vol_threshold_high] = 'high_volatility'
            regimes[rolling_vol < vol_threshold_low] = 'low_volatility'
            
            regime_performance = {}
            
            for regime in ['low_volatility', 'normal', 'high_volatility']:
                regime_mask = regimes == regime
                
                if regime_mask.sum() > 5:  # Need sufficient data
                    regime_returns_a = returns_a[regime_mask]
                    regime_returns_b = returns_b[regime_mask]
                    
                    regime_performance[regime] = {
                        f'{strategy_a_id}_mean_return': regime_returns_a.mean() * 252,
                        f'{strategy_b_id}_mean_return': regime_returns_b.mean() * 252,
                        f'{strategy_a_id}_volatility': regime_returns_a.std() * np.sqrt(252),
                        f'{strategy_b_id}_volatility': regime_returns_b.std() * np.sqrt(252),
                        f'{strategy_a_id}_sharpe': (regime_returns_a.mean() * 252) / (regime_returns_a.std() * np.sqrt(252)) if regime_returns_a.std() > 0 else 0,
                        f'{strategy_b_id}_sharpe': (regime_returns_b.mean() * 252) / (regime_returns_b.std() * np.sqrt(252)) if regime_returns_b.std() > 0 else 0
                    }
            
            return regime_performance
            
        except Exception as e:
            logger.error(f"Error analyzing market regime performance: {str(e)}")
            return {}
    
    def _generate_recommendation(self, 
                               metrics_a: StrategyMetrics, 
                               metrics_b: StrategyMetrics,
                               statistical_tests: Dict[str, Tuple[float, float]]) -> Tuple[str, float]:
        """Generate recommendation based on comprehensive analysis"""
        try:
            # Score each strategy across multiple dimensions
            score_a = 0
            score_b = 0
            total_criteria = 0
            
            # Risk-adjusted return criteria
            criteria = [
                ('sharpe_ratio', metrics_a.sharpe_ratio, metrics_b.sharpe_ratio, 3),  # High weight
                ('sortino_ratio', metrics_a.sortino_ratio, metrics_b.sortino_ratio, 2),
                ('calmar_ratio', metrics_a.calmar_ratio, metrics_b.calmar_ratio, 2),
                ('information_ratio', metrics_a.information_ratio, metrics_b.information_ratio, 1),
                ('max_drawdown', -metrics_a.max_drawdown, -metrics_b.max_drawdown, 2),  # Negative because lower is better
                ('win_rate', metrics_a.win_rate, metrics_b.win_rate, 1),
                ('profit_factor', min(metrics_a.profit_factor, 10), min(metrics_b.profit_factor, 10), 1)  # Cap to avoid outliers
            ]
            
            for criterion_name, value_a, value_b, weight in criteria:
                if not (np.isnan(value_a) or np.isnan(value_b) or np.isinf(value_a) or np.isinf(value_b)):
                    if value_a > value_b:
                        score_a += weight
                    elif value_b > value_a:
                        score_b += weight
                    # Tie adds nothing to either score
                    total_criteria += weight
            
            # Statistical significance adjustment
            significance_bonus = 0
            if 'mean_returns_ttest' in statistical_tests:
                _, p_value = statistical_tests['mean_returns_ttest']
                if p_value < self.significance_level:
                    significance_bonus = 1
            
            # Calculate confidence level
            if total_criteria > 0:
                confidence_a = (score_a + significance_bonus) / (total_criteria + 1)
                confidence_b = (score_b + significance_bonus) / (total_criteria + 1)
            else:
                confidence_a = confidence_b = 0.5
            
            # Generate recommendation
            if abs(score_a - score_b) <= 1:  # Close performance
                recommendation = f"Performance is comparable between {metrics_a.strategy_id} and {metrics_b.strategy_id}. Consider portfolio allocation to both strategies."
                confidence = 0.6
            elif score_a > score_b:
                margin = score_a - score_b
                if margin >= 3:
                    recommendation = f"Strong recommendation for {metrics_a.strategy_id} over {metrics_b.strategy_id} based on superior risk-adjusted performance."
                    confidence = min(0.9, 0.7 + margin * 0.05)
                else:
                    recommendation = f"Moderate preference for {metrics_a.strategy_id} over {metrics_b.strategy_id}."
                    confidence = min(0.8, 0.6 + margin * 0.05)
            else:
                margin = score_b - score_a
                if margin >= 3:
                    recommendation = f"Strong recommendation for {metrics_b.strategy_id} over {metrics_a.strategy_id} based on superior risk-adjusted performance."
                    confidence = min(0.9, 0.7 + margin * 0.05)
                else:
                    recommendation = f"Moderate preference for {metrics_b.strategy_id} over {metrics_a.strategy_id}."
                    confidence = min(0.8, 0.6 + margin * 0.05)
            
            return recommendation, confidence
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return "Unable to generate recommendation due to analysis error.", 0.0
    
    def generate_comparison_report(self, comparison_result: ComparisonResult) -> str:
        """
        Generate a comprehensive comparison report
        
        Args:
            comparison_result: Result from compare_strategies method
            
        Returns:
            Formatted string report
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("STRATEGY PERFORMANCE COMPARISON REPORT")
            report.append("=" * 80)
            report.append(f"Primary Strategy: {comparison_result.primary_strategy}")
            report.append(f"Comparison Strategy: {comparison_result.comparison_strategy}")
            report.append(f"Confidence Level: {comparison_result.confidence_level:.2%}")
            report.append("")
            
            # Performance differences
            report.append("PERFORMANCE DIFFERENCES:")
            report.append("-" * 40)
            for metric, diff in comparison_result.performance_difference.items():
                report.append(f"{metric:<25}: {diff:+.4f}")
            report.append("")
            
            # Statistical significance
            if comparison_result.statistical_significance:
                report.append("STATISTICAL SIGNIFICANCE TESTS:")
                report.append("-" * 40)
                for test_name, (statistic, p_value) in comparison_result.statistical_significance.items():
                    significance = "Significant" if p_value < self.significance_level else "Not Significant"
                    report.append(f"{test_name:<25}: p-value = {p_value:.4f} ({significance})")
                report.append("")
            
            # Market regime performance
            if comparison_result.market_regime_performance:
                report.append("MARKET REGIME PERFORMANCE:")
                report.append("-" * 40)
                for regime, metrics in comparison_result.market_regime_performance.items():
                    report.append(f"{regime.title()} Market:")
                    for metric, value in metrics.items():
                        report.append(f"  {metric:<30}: {value:.4f}")
                report.append("")
            
            # Recommendation
            report.append("RECOMMENDATION:")
            report.append("-" * 40)
            report.append(comparison_result.recommendation)
            report.append("")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def export_results(self, 
                      comparison_result: ComparisonResult,
                      filename: str) -> None:
        """
        Export comparison results to JSON file
        
        Args:
            comparison_result: Result from compare_strategies method
            filename: Output filename
        """
        try:
            # Convert dataclass to dictionary for JSON serialization
            result_dict = {
                'primary_strategy': comparison_result.primary_strategy,
                'comparison_strategy': comparison_result.comparison_strategy,
                'performance_difference': comparison_result.performance_difference,
                'statistical_significance': {k: list(v) for k, v in comparison_result.statistical_significance.items()},
                'risk_adjusted_ranking': comparison_result.risk_adjusted_ranking,
                'market_regime_performance': comparison_result.market_regime_performance,
                'recommendation': comparison_result.recommendation,
                'confidence_level': comparison_result.confidence_level,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"Comparison results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize comparator
    comparator = PerformanceComparator(significance_level=0.05)
    
    # Example with synthetic data
    try:
        # Generate sample data for demonstration
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Strategy A: Higher return, higher volatility
        returns_a = np.random.normal(0.0008, 0.02, 252)  # ~20% annual return, 32% volatility
        portfolio_a = pd.DataFrame({'value': (1 + pd.Series(returns_a, index=dates)).cumprod() * 100000}, index=dates)
        trades_a = pd.DataFrame({
            'entry_time': dates[::10],
            'exit_time': dates[5::10],
            'pnl': np.random.normal(100, 500, 25),
            'size': np.random.uniform(0.1, 1.0, 25)
        })
        
        # Strategy B: Lower return, lower volatility
        returns_b = np.random.normal(0.0004, 0.015, 252)  # ~10% annual return, 24% volatility
        portfolio_b = pd.DataFrame({'value': (1 + pd.Series(returns_b, index=dates)).cumprod() * 100000}, index=dates)
        trades_b = pd.DataFrame({
            'entry_time': dates[::8],
            'exit_time': dates[4::8],
            'pnl': np.random.normal(50, 200, 31),
            'size': np.random.uniform(0.1, 1.0, 31)
        })
        
        # Load data
        comparator.load_strategy_data('strategy_a', trades_a, portfolio_a)
        comparator.load_strategy_data('strategy_b', trades_b, portfolio_b)
        
        # Compare strategies
        comparison = comparator.compare_strategies('strategy_a', 'strategy_b')
        
        # Generate and print report
        report = comparator.generate_comparison_report(comparison)
        print(report)
        
        # Rank strategies
        ranking = comparator.rank_strategies(['strategy_a', 'strategy_b'])
        print("\nSTRATEGY RANKING:")
        print(f"Best performing strategy: {ranking['composite_ranking'][0][0]}")
        print(f"Composite score: {ranking['composite_ranking'][0][1]:.3f}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        print(f"Example execution failed: {str(e)}")