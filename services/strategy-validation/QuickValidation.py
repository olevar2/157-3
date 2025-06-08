"""
Quick Validation Service

Rapid strategy validation framework for fast performance assessment and initial screening
of trading strategies before full backtesting.

Features:
- Rapid statistical validation using sample data
- Quick performance metrics calculation
- Multi-timeframe validation
- Risk assessment and red flag detection
- Benchmark comparison
- Strategy fitness scoring
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ValidationResult(Enum):
    """Quick validation results"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INSUFFICIENT_DATA = "insufficient_data"


class ValidationCategory(Enum):
    """Validation test categories"""
    PERFORMANCE = "performance"
    RISK = "risk"
    STATISTICAL = "statistical"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class QuickValidationReport:
    """Quick validation comprehensive report"""
    strategy_id: str
    validation_time: datetime
    overall_result: ValidationResult
    fitness_score: float  # 0-100
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    
    # Performance metrics
    quick_return: float
    quick_sharpe: float
    quick_sortino: float
    quick_max_dd: float
    quick_win_rate: float
    
    # Risk metrics
    risk_score: float
    volatility: float
    var_95: float
    expected_shortfall: float
    
    # Statistical tests
    normality_test: bool
    stationarity_test: bool
    autocorrelation_test: bool
    
    # Detailed results
    test_results: Dict[str, Dict] = None
    recommendations: List[str] = None
    red_flags: List[str] = None


class QuickValidation:
    """
    Advanced quick validation system for trading strategies
    
    Provides rapid assessment of strategy viability through:
    - Statistical performance analysis
    - Risk assessment and red flag detection
    - Benchmark comparison
    - Multi-timeframe validation
    - Strategy fitness scoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize QuickValidation
        
        Args:
            config: Configuration dictionary with validation settings
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Validation settings
        self.min_sample_size = self.config.get('min_sample_size', 50)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.benchmark_return = self.config.get('benchmark_return', 0.08)  # 8% annual
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual
        
        # Performance thresholds
        self.min_sharpe = self.config.get('min_sharpe', 1.0)
        self.max_drawdown = self.config.get('max_drawdown', 0.20)  # 20%
        self.min_win_rate = self.config.get('min_win_rate', 0.40)  # 40%
        
        # Statistical test settings
        self.alpha = 1 - self.confidence_level
        
        # Thread pool for parallel validation
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("QuickValidation initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('QuickValidation')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_strategy(self, strategy_id: str, returns_data: pd.Series = None, 
                         trades_data: pd.DataFrame = None, 
                         price_data: pd.DataFrame = None) -> QuickValidationReport:
        """
        Perform comprehensive quick validation
        
        Args:
            strategy_id: Strategy identifier
            returns_data: Strategy returns time series
            trades_data: Individual trade data
            price_data: Price/equity curve data
            
        Returns:
            QuickValidationReport: Comprehensive validation results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting quick validation for strategy: {strategy_id}")
            
            # Prepare data
            if returns_data is None and trades_data is not None:
                returns_data = self._calculate_returns_from_trades(trades_data)
            elif returns_data is None and price_data is not None:
                returns_data = price_data.pct_change().dropna()
            
            if returns_data is None or len(returns_data) < self.min_sample_size:
                return self._create_insufficient_data_report(strategy_id)
            
            # Initialize test results storage
            test_results = {}
            red_flags = []
            recommendations = []
            
            # Parallel validation execution
            futures = []
            
            # Performance validation
            futures.append(self.executor.submit(self._validate_performance, returns_data))
            
            # Risk validation
            futures.append(self.executor.submit(self._validate_risk, returns_data))
            
            # Statistical validation
            futures.append(self.executor.submit(self._validate_statistics, returns_data))
            
            # Technical validation
            if trades_data is not None:
                futures.append(self.executor.submit(self._validate_technical, trades_data))
            
            # Collect results
            for future in as_completed(futures):
                category_results = future.result()
                test_results.update(category_results)
            
            # Generate fitness score
            fitness_score = self._calculate_fitness_score(test_results)
            
            # Count test results
            passed_tests = sum(1 for result in test_results.values() 
                             if result.get('result') == ValidationResult.PASS)
            failed_tests = sum(1 for result in test_results.values() 
                             if result.get('result') == ValidationResult.FAIL)
            warning_tests = sum(1 for result in test_results.values() 
                              if result.get('result') == ValidationResult.WARNING)
            total_tests = len(test_results)
            
            # Determine overall result
            overall_result = self._determine_overall_result(fitness_score, failed_tests, total_tests)
            
            # Generate recommendations and red flags
            recommendations, red_flags = self._generate_recommendations(test_results, returns_data)
            
            # Quick performance metrics
            quick_metrics = self._calculate_quick_metrics(returns_data)
            
            # Create validation report
            report = QuickValidationReport(
                strategy_id=strategy_id,
                validation_time=datetime.now(),
                overall_result=overall_result,
                fitness_score=fitness_score,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                warning_tests=warning_tests,
                quick_return=quick_metrics['annual_return'],
                quick_sharpe=quick_metrics['sharpe_ratio'],
                quick_sortino=quick_metrics['sortino_ratio'],
                quick_max_dd=quick_metrics['max_drawdown'],
                quick_win_rate=quick_metrics['win_rate'],
                risk_score=quick_metrics['risk_score'],
                volatility=quick_metrics['volatility'],
                var_95=quick_metrics['var_95'],
                expected_shortfall=quick_metrics['expected_shortfall'],
                normality_test=test_results.get('normality_test', {}).get('result') == ValidationResult.PASS,
                stationarity_test=test_results.get('stationarity_test', {}).get('result') == ValidationResult.PASS,
                autocorrelation_test=test_results.get('autocorrelation_test', {}).get('result') == ValidationResult.PASS,
                test_results=test_results,
                recommendations=recommendations,
                red_flags=red_flags
            )
            
            validation_time = time.time() - start_time
            self.logger.info(f"Quick validation completed for {strategy_id} in {validation_time:.2f}s")
            self.logger.info(f"Fitness Score: {fitness_score:.1f}/100, Result: {overall_result.value}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in quick validation for {strategy_id}: {e}")
            return self._create_error_report(strategy_id, str(e))
    
    def _calculate_returns_from_trades(self, trades_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from trade data"""
        if 'pnl' not in trades_data.columns:
            raise ValueError("Trades data must contain 'pnl' column")
        
        # Sort by timestamp if available
        if 'timestamp' in trades_data.columns:
            trades_data = trades_data.sort_values('timestamp')
        
        # Calculate cumulative equity and returns
        cumulative_pnl = trades_data['pnl'].cumsum()
        starting_capital = 100000  # Default starting capital
        equity_curve = starting_capital + cumulative_pnl
        
        returns = equity_curve.pct_change().dropna()
        return returns
    
    def _validate_performance(self, returns: pd.Series) -> Dict[str, Dict]:
        """Validate performance metrics"""
        results = {}
        
        try:
            # Annual return
            annual_return = returns.mean() * 252
            results['annual_return'] = {
                'value': annual_return,
                'result': ValidationResult.PASS if annual_return > 0 else ValidationResult.FAIL,
                'threshold': 0,
                'category': ValidationCategory.PERFORMANCE
            }
            
            # Sharpe ratio
            sharpe_ratio = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252))
            results['sharpe_ratio'] = {
                'value': sharpe_ratio,
                'result': ValidationResult.PASS if sharpe_ratio >= self.min_sharpe else ValidationResult.FAIL,
                'threshold': self.min_sharpe,
                'category': ValidationCategory.PERFORMANCE
            }
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
            sortino_ratio = (returns.mean() * 252 - self.risk_free_rate) / downside_std
            results['sortino_ratio'] = {
                'value': sortino_ratio,
                'result': ValidationResult.PASS if sortino_ratio >= self.min_sharpe else ValidationResult.WARNING,
                'threshold': self.min_sharpe,
                'category': ValidationCategory.PERFORMANCE
            }
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            results['max_drawdown'] = {
                'value': max_drawdown,
                'result': ValidationResult.PASS if max_drawdown <= self.max_drawdown else ValidationResult.FAIL,
                'threshold': self.max_drawdown,
                'category': ValidationCategory.PERFORMANCE
            }
            
            # Win rate (percentage of positive returns)
            win_rate = (returns > 0).mean()
            results['win_rate'] = {
                'value': win_rate,
                'result': ValidationResult.PASS if win_rate >= self.min_win_rate else ValidationResult.WARNING,
                'threshold': self.min_win_rate,
                'category': ValidationCategory.PERFORMANCE
            }
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            results['calmar_ratio'] = {
                'value': calmar_ratio,
                'result': ValidationResult.PASS if calmar_ratio >= 1.0 else ValidationResult.WARNING,
                'threshold': 1.0,
                'category': ValidationCategory.PERFORMANCE
            }
            
        except Exception as e:
            self.logger.error(f"Error in performance validation: {e}")
        
        return results
    
    def _validate_risk(self, returns: pd.Series) -> Dict[str, Dict]:
        """Validate risk metrics"""
        results = {}
        
        try:
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            results['volatility'] = {
                'value': volatility,
                'result': ValidationResult.PASS if volatility <= 0.30 else ValidationResult.WARNING,
                'threshold': 0.30,
                'category': ValidationCategory.RISK
            }
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5)
            results['var_95'] = {
                'value': var_95,
                'result': ValidationResult.PASS if var_95 >= -0.05 else ValidationResult.WARNING,
                'threshold': -0.05,
                'category': ValidationCategory.RISK
            }
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns[returns <= var_95]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_95
            results['expected_shortfall'] = {
                'value': expected_shortfall,
                'result': ValidationResult.PASS if expected_shortfall >= -0.08 else ValidationResult.WARNING,
                'threshold': -0.08,
                'category': ValidationCategory.RISK
            }
            
            # Skewness
            skewness = returns.skew()
            results['skewness'] = {
                'value': skewness,
                'result': ValidationResult.PASS if skewness >= -0.5 else ValidationResult.WARNING,
                'threshold': -0.5,
                'category': ValidationCategory.RISK
            }
            
            # Kurtosis
            kurtosis = returns.kurtosis()
            results['kurtosis'] = {
                'value': kurtosis,
                'result': ValidationResult.PASS if kurtosis <= 5.0 else ValidationResult.WARNING,
                'threshold': 5.0,
                'category': ValidationCategory.RISK
            }
            
            # Tail ratio
            tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 1
            results['tail_ratio'] = {
                'value': tail_ratio,
                'result': ValidationResult.PASS if 0.5 <= tail_ratio <= 2.0 else ValidationResult.WARNING,
                'threshold': (0.5, 2.0),
                'category': ValidationCategory.RISK
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk validation: {e}")
        
        return results
    
    def _validate_statistics(self, returns: pd.Series) -> Dict[str, Dict]:
        """Validate statistical properties"""
        results = {}
        
        try:
            # Normality test (Jarque-Bera)
            _, normality_p = stats.jarque_bera(returns)
            results['normality_test'] = {
                'value': normality_p,
                'result': ValidationResult.PASS if normality_p > self.alpha else ValidationResult.WARNING,
                'threshold': self.alpha,
                'category': ValidationCategory.STATISTICAL
            }
            
            # Stationarity test (Augmented Dickey-Fuller)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_stat, adf_p, _, _, _, _ = adfuller(returns)
                results['stationarity_test'] = {
                    'value': adf_p,
                    'result': ValidationResult.PASS if adf_p <= self.alpha else ValidationResult.WARNING,
                    'threshold': self.alpha,
                    'category': ValidationCategory.STATISTICAL
                }
            except ImportError:
                self.logger.warning("statsmodels not available for stationarity test")
            
            # Autocorrelation test (Ljung-Box)
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat, lb_p = acorr_ljungbox(returns, lags=10, return_df=False)
                results['autocorrelation_test'] = {
                    'value': np.min(lb_p) if isinstance(lb_p, np.ndarray) else lb_p,
                    'result': ValidationResult.PASS if (np.min(lb_p) if isinstance(lb_p, np.ndarray) else lb_p) > self.alpha else ValidationResult.WARNING,
                    'threshold': self.alpha,
                    'category': ValidationCategory.STATISTICAL
                }
            except ImportError:
                self.logger.warning("statsmodels not available for autocorrelation test")
            
            # Outlier detection
            q1 = returns.quantile(0.25)
            q3 = returns.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((returns < lower_bound) | (returns > upper_bound)).sum()
            outlier_ratio = outliers / len(returns)
            
            results['outlier_test'] = {
                'value': outlier_ratio,
                'result': ValidationResult.PASS if outlier_ratio <= 0.05 else ValidationResult.WARNING,
                'threshold': 0.05,
                'category': ValidationCategory.STATISTICAL
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical validation: {e}")
        
        return results
    
    def _validate_technical(self, trades_data: pd.DataFrame) -> Dict[str, Dict]:
        """Validate technical trading metrics"""
        results = {}
        
        try:
            # Average trade duration
            if 'entry_time' in trades_data.columns and 'exit_time' in trades_data.columns:
                durations = pd.to_datetime(trades_data['exit_time']) - pd.to_datetime(trades_data['entry_time'])
                avg_duration = durations.mean().total_seconds() / 3600  # hours
                results['avg_trade_duration'] = {
                    'value': avg_duration,
                    'result': ValidationResult.PASS if 1 <= avg_duration <= 168 else ValidationResult.WARNING,  # 1 hour to 1 week
                    'threshold': (1, 168),
                    'category': ValidationCategory.TECHNICAL
                }
            
            # Profit factor
            if 'pnl' in trades_data.columns:
                winning_trades = trades_data[trades_data['pnl'] > 0]['pnl'].sum()
                losing_trades = abs(trades_data[trades_data['pnl'] < 0]['pnl'].sum())
                profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')
                
                results['profit_factor'] = {
                    'value': profit_factor,
                    'result': ValidationResult.PASS if profit_factor >= 1.2 else ValidationResult.FAIL,
                    'threshold': 1.2,
                    'category': ValidationCategory.TECHNICAL
                }
            
            # Maximum consecutive losses
            if 'pnl' in trades_data.columns:
                consecutive_losses = 0
                max_consecutive_losses = 0
                
                for pnl in trades_data['pnl']:
                    if pnl < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
                
                results['max_consecutive_losses'] = {
                    'value': max_consecutive_losses,
                    'result': ValidationResult.PASS if max_consecutive_losses <= 5 else ValidationResult.WARNING,
                    'threshold': 5,
                    'category': ValidationCategory.TECHNICAL
                }
            
        except Exception as e:
            self.logger.error(f"Error in technical validation: {e}")
        
        return results
    
    def _calculate_fitness_score(self, test_results: Dict[str, Dict]) -> float:
        """Calculate overall fitness score (0-100)"""
        if not test_results:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # Weight assignments for different test categories
        weights = {
            ValidationCategory.PERFORMANCE: 40,
            ValidationCategory.RISK: 30,
            ValidationCategory.STATISTICAL: 20,
            ValidationCategory.TECHNICAL: 10
        }
        
        category_scores = defaultdict(list)
        
        # Group results by category and calculate scores
        for test_name, result in test_results.items():
            category = result.get('category', ValidationCategory.TECHNICAL)
            
            if result.get('result') == ValidationResult.PASS:
                category_scores[category].append(100)
            elif result.get('result') == ValidationResult.WARNING:
                category_scores[category].append(70)
            elif result.get('result') == ValidationResult.FAIL:
                category_scores[category].append(30)
            else:
                category_scores[category].append(50)
        
        # Calculate weighted average
        for category, scores in category_scores.items():
            if scores:
                category_score = np.mean(scores)
                weight = weights.get(category, 10)
                score += category_score * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _determine_overall_result(self, fitness_score: float, failed_tests: int, total_tests: int) -> ValidationResult:
        """Determine overall validation result"""
        failure_rate = failed_tests / total_tests if total_tests > 0 else 0
        
        if fitness_score >= 80 and failure_rate <= 0.1:
            return ValidationResult.PASS
        elif fitness_score >= 60 and failure_rate <= 0.2:
            return ValidationResult.WARNING
        else:
            return ValidationResult.FAIL
    
    def _generate_recommendations(self, test_results: Dict[str, Dict], returns: pd.Series) -> Tuple[List[str], List[str]]:
        """Generate recommendations and red flags"""
        recommendations = []
        red_flags = []
        
        # Analyze failed tests
        failed_tests = [name for name, result in test_results.items() 
                       if result.get('result') == ValidationResult.FAIL]
        
        # Performance recommendations
        if 'sharpe_ratio' in failed_tests:
            recommendations.append("Consider improving risk-adjusted returns by optimizing position sizing or entry/exit rules")
        
        if 'max_drawdown' in failed_tests:
            red_flags.append("Excessive drawdown detected - implement stricter risk management")
        
        if 'win_rate' in failed_tests:
            recommendations.append("Low win rate - consider refining strategy logic or adding filters")
        
        # Risk recommendations
        if 'var_95' in failed_tests:
            red_flags.append("High Value at Risk - strategy may be too aggressive")
        
        if 'volatility' in failed_tests:
            recommendations.append("High volatility detected - consider reducing position sizes")
        
        # Statistical recommendations
        if 'normality_test' in failed_tests:
            recommendations.append("Returns not normally distributed - consider robust risk metrics")
        
        if 'stationarity_test' in failed_tests:
            recommendations.append("Non-stationary returns detected - strategy may not be robust over time")
        
        # Technical recommendations
        if 'profit_factor' in failed_tests:
            red_flags.append("Profit factor below 1.2 - strategy may not be profitable after costs")
        
        return recommendations, red_flags
    
    def _calculate_quick_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate quick performance metrics"""
        metrics = {}
        
        # Basic performance
        metrics['annual_return'] = returns.mean() * 252
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / metrics['volatility']
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else metrics['volatility']
        metrics['sortino_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / downside_std
        
        # Drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        metrics['max_drawdown'] = abs(drawdown.min())
        
        # Win rate
        metrics['win_rate'] = (returns > 0).mean()
        
        # Risk metrics
        metrics['var_95'] = np.percentile(returns, 5)
        tail_returns = returns[returns <= metrics['var_95']]
        metrics['expected_shortfall'] = tail_returns.mean() if len(tail_returns) > 0 else metrics['var_95']
        
        # Risk score (0-100, higher = riskier)
        risk_score = 0
        if metrics['max_drawdown'] > 0.15: risk_score += 25
        if metrics['volatility'] > 0.25: risk_score += 25
        if metrics['var_95'] < -0.03: risk_score += 25
        if metrics['sharpe_ratio'] < 1.0: risk_score += 25
        metrics['risk_score'] = risk_score
        
        return metrics
    
    def _create_insufficient_data_report(self, strategy_id: str) -> QuickValidationReport:
        """Create report for insufficient data"""
        return QuickValidationReport(
            strategy_id=strategy_id,
            validation_time=datetime.now(),
            overall_result=ValidationResult.INSUFFICIENT_DATA,
            fitness_score=0.0,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            warning_tests=0,
            quick_return=0.0,
            quick_sharpe=0.0,
            quick_sortino=0.0,
            quick_max_dd=0.0,
            quick_win_rate=0.0,
            risk_score=100.0,
            volatility=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            normality_test=False,
            stationarity_test=False,
            autocorrelation_test=False,
            test_results={},
            recommendations=["Collect more trading data before validation"],
            red_flags=["Insufficient data for reliable validation"]
        )
    
    def _create_error_report(self, strategy_id: str, error_msg: str) -> QuickValidationReport:
        """Create report for validation errors"""
        return QuickValidationReport(
            strategy_id=strategy_id,
            validation_time=datetime.now(),
            overall_result=ValidationResult.FAIL,
            fitness_score=0.0,
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            warning_tests=0,
            quick_return=0.0,
            quick_sharpe=0.0,
            quick_sortino=0.0,
            quick_max_dd=0.0,
            quick_win_rate=0.0,
            risk_score=100.0,
            volatility=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            normality_test=False,
            stationarity_test=False,
            autocorrelation_test=False,
            test_results={},
            recommendations=["Fix validation errors and retry"],
            red_flags=[f"Validation error: {error_msg}"]
        )
    
    def validate_multiple_strategies(self, strategies_data: Dict[str, Dict]) -> Dict[str, QuickValidationReport]:
        """Validate multiple strategies in parallel"""
        reports = {}
        
        futures = {}
        for strategy_id, data in strategies_data.items():
            future = self.executor.submit(
                self.validate_strategy,
                strategy_id,
                data.get('returns'),
                data.get('trades'),
                data.get('prices')
            )
            futures[strategy_id] = future
        
        for strategy_id, future in futures.items():
            try:
                reports[strategy_id] = future.result(timeout=30)
            except Exception as e:
                self.logger.error(f"Error validating {strategy_id}: {e}")
                reports[strategy_id] = self._create_error_report(strategy_id, str(e))
        
        return reports
    
    def generate_validation_summary(self, reports: Dict[str, QuickValidationReport]) -> Dict[str, Any]:
        """Generate summary of multiple validation reports"""
        summary = {
            'total_strategies': len(reports),
            'passed': sum(1 for r in reports.values() if r.overall_result == ValidationResult.PASS),
            'failed': sum(1 for r in reports.values() if r.overall_result == ValidationResult.FAIL),
            'warnings': sum(1 for r in reports.values() if r.overall_result == ValidationResult.WARNING),
            'insufficient_data': sum(1 for r in reports.values() if r.overall_result == ValidationResult.INSUFFICIENT_DATA),
            'average_fitness_score': np.mean([r.fitness_score for r in reports.values()]),
            'best_strategy': max(reports.items(), key=lambda x: x[1].fitness_score)[0] if reports else None,
            'worst_strategy': min(reports.items(), key=lambda x: x[1].fitness_score)[0] if reports else None
        }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize validator
    validator = QuickValidation({
        'min_sample_size': 30,
        'min_sharpe': 1.0,
        'max_drawdown': 0.15
    })
    
    # Generate sample strategy returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))  # Daily returns
    
    # Validate strategy
    report = validator.validate_strategy('test_strategy', returns_data=returns)
    
    print(f"Validation Results for {report.strategy_id}:")
    print(f"Overall Result: {report.overall_result.value}")
    print(f"Fitness Score: {report.fitness_score:.1f}/100")
    print(f"Tests: {report.passed_tests} passed, {report.failed_tests} failed, {report.warning_tests} warnings")
    print(f"Quick Sharpe: {report.quick_sharpe:.2f}")
    print(f"Max Drawdown: {report.quick_max_dd:.2%}")
    
    if report.red_flags:
        print("\nRed Flags:")
        for flag in report.red_flags:
            print(f"  - {flag}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # Test with trade data
    trades_data = pd.DataFrame({
        'pnl': np.random.normal(10, 50, 50),
        'entry_time': pd.date_range('2024-01-01', periods=50, freq='D'),
        'exit_time': pd.date_range('2024-01-01 12:00:00', periods=50, freq='D')
    })
    
    report2 = validator.validate_strategy('test_strategy_2', trades_data=trades_data)
    print(f"\nValidation Results for {report2.strategy_id}:")
    print(f"Fitness Score: {report2.fitness_score:.1f}/100")
    print(f"Profit Factor: {report2.test_results.get('profit_factor', {}).get('value', 'N/A')}")


from collections import defaultdict
