"""
Advanced Monte Carlo Simulation Framework for Trading Strategies

This module provides comprehensive Monte Carlo simulation capabilities for:
- Strategy performance analysis
- Risk assessment and VaR calculations
- Bootstrap analysis
- Scenario generation
- Portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class SimulationParameters:
    """Monte Carlo simulation parameters"""
    n_simulations: int
    simulation_length: int
    random_seed: Optional[int] = None
    confidence_levels: List[float] = None
    bootstrap_samples: int = 1000
    scenario_analysis: bool = True
    correlation_modeling: bool = True
    fat_tail_modeling: bool = True


@dataclass
class MarketScenario:
    """Market scenario definition"""
    name: str
    probability: float
    return_adjustment: float
    volatility_adjustment: float
    correlation_adjustment: float
    duration_periods: int


@dataclass
class SimulationResults:
    """Monte Carlo simulation results"""
    simulation_summary: Dict[str, float]
    return_distribution: List[float]
    drawdown_distribution: List[float]
    var_estimates: Dict[str, float]
    expected_shortfall: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    scenario_analysis: Dict[str, Any]
    bootstrap_results: Dict[str, Any]
    risk_metrics: Dict[str, float]
    performance_percentiles: Dict[str, float]
    stress_test_results: Dict[str, Any]
    convergence_analysis: Dict[str, Any]


class MonteCarloSimulator:
    """
    Advanced Monte Carlo Simulation Framework
    
    Features:
    - Multiple simulation methods
    - Bootstrap analysis
    - Scenario modeling
    - Risk estimation (VaR, ES, etc.)
    - Stress testing
    - Correlation modeling
    - Fat-tail distributions
    """
    
    def __init__(self,
                 random_seed: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize MonteCarloSimulator
        
        Args:
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress information
        """
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Simulation history
        self.simulation_history: List[SimulationResults] = []
        
        # Default confidence levels
        self.default_confidence_levels = [0.90, 0.95, 0.99]
        
        # Supported distributions
        self.supported_distributions = [
            'normal', 'student_t', 'skewed_normal', 'empirical'
        ]
        
        self.logger.info("MonteCarloSimulator initialized with advanced simulation capabilities")

    def run_simulation(self, 
                      strategy_function: Callable,
                      historical_data: pd.DataFrame,
                      parameters: SimulationParameters,
                      market_scenarios: Optional[List[MarketScenario]] = None,
                      distribution_type: str = 'normal') -> SimulationResults:
        """
        Run comprehensive Monte Carlo simulation
        
        Args:
            strategy_function: Strategy function to simulate
            historical_data: Historical market data for parameter estimation
            parameters: Simulation parameters
            market_scenarios: Optional market scenarios for stress testing
            distribution_type: Return distribution type
            
        Returns:
            SimulationResults: Comprehensive simulation results
        """
        try:
            self.logger.info(f"Starting Monte Carlo simulation with {parameters.n_simulations} runs")
            
            # Validate inputs
            self._validate_inputs(strategy_function, historical_data, parameters)
            
            # Estimate market parameters from historical data
            market_params = self._estimate_market_parameters(
                historical_data, distribution_type
            )
            
            # Generate market scenarios
            if market_scenarios is None:
                market_scenarios = self._generate_default_scenarios()
            
            # Run primary simulations
            simulation_results = self._run_primary_simulations(
                strategy_function, market_params, parameters, distribution_type
            )
            
            # Bootstrap analysis
            bootstrap_results = self._run_bootstrap_analysis(
                historical_data, strategy_function, parameters
            )
            
            # Scenario analysis
            scenario_results = {}
            if parameters.scenario_analysis and market_scenarios:
                scenario_results = self._run_scenario_analysis(
                    strategy_function, market_params, parameters, market_scenarios
                )
            
            # Risk metrics calculation
            risk_metrics = self._calculate_risk_metrics(simulation_results)
            
            # VaR and Expected Shortfall
            var_estimates = self._calculate_var_estimates(
                simulation_results, parameters.confidence_levels or self.default_confidence_levels
            )
            expected_shortfall = self._calculate_expected_shortfall(
                simulation_results, parameters.confidence_levels or self.default_confidence_levels
            )
            
            # Performance percentiles
            performance_percentiles = self._calculate_performance_percentiles(simulation_results)
            
            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                simulation_results, parameters.confidence_levels or self.default_confidence_levels
            )
            
            # Stress testing
            stress_test_results = self._run_stress_tests(
                strategy_function, market_params, parameters
            )
            
            # Convergence analysis
            convergence_analysis = self._analyze_convergence(simulation_results)
            
            # Create comprehensive results
            results = SimulationResults(
                simulation_summary=self._create_simulation_summary(simulation_results),
                return_distribution=[result['total_return'] for result in simulation_results],
                drawdown_distribution=[result['max_drawdown'] for result in simulation_results],
                var_estimates=var_estimates,
                expected_shortfall=expected_shortfall,
                confidence_intervals=confidence_intervals,
                scenario_analysis=scenario_results,
                bootstrap_results=bootstrap_results,
                risk_metrics=risk_metrics,
                performance_percentiles=performance_percentiles,
                stress_test_results=stress_test_results,
                convergence_analysis=convergence_analysis
            )
            
            # Store in history
            self.simulation_history.append(results)
            
            self.logger.info("Monte Carlo simulation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise

    def _estimate_market_parameters(self, 
                                   data: pd.DataFrame, 
                                   distribution_type: str) -> Dict[str, Any]:
        """Estimate market parameters from historical data"""
        try:
            # Calculate returns
            if 'returns' not in data.columns:
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                else:
                    returns = data.iloc[:, 0].pct_change().dropna()
            else:
                returns = data['returns'].dropna()
            
            params = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'distribution_type': distribution_type
            }
            
            # Estimate distribution-specific parameters
            if distribution_type == 'student_t':
                # Fit t-distribution
                df, loc, scale = stats.t.fit(returns)
                params['df'] = df
                params['loc'] = loc
                params['scale'] = scale
                
            elif distribution_type == 'skewed_normal':
                # Fit skewed normal distribution
                a, loc, scale = stats.skewnorm.fit(returns)
                params['skew_param'] = a
                params['loc'] = loc
                params['scale'] = scale
                
            elif distribution_type == 'empirical':
                # Use empirical distribution
                params['empirical_returns'] = returns.values
            
            # Correlation analysis for multi-asset
            if len(data.columns) > 1:
                returns_matrix = data.pct_change().dropna()
                params['correlation_matrix'] = returns_matrix.corr().values
                params['covariance_matrix'] = returns_matrix.cov().values
            
            return params
            
        except Exception as e:
            self.logger.warning(f"Error estimating market parameters: {str(e)}")
            return {
                'mean': 0.0005,
                'std': 0.02,
                'skewness': 0.0,
                'kurtosis': 3.0,
                'distribution_type': 'normal'
            }

    def _run_primary_simulations(self,
                               strategy_function: Callable,
                               market_params: Dict[str, Any],
                               parameters: SimulationParameters,
                               distribution_type: str) -> List[Dict[str, float]]:
        """Run primary Monte Carlo simulations"""
        try:
            simulation_results = []
            
            # Set random seed for reproducibility
            if parameters.random_seed:
                np.random.seed(parameters.random_seed)
            
            # Parallel or sequential execution
            if self.n_jobs == 1:
                # Sequential execution
                for i in range(parameters.n_simulations):
                    if i % 1000 == 0 and self.verbose:
                        self.logger.info(f"Running simulation {i+1}/{parameters.n_simulations}")
                    
                    result = self._run_single_simulation(
                        strategy_function, market_params, parameters, distribution_type
                    )
                    simulation_results.append(result)
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                    futures = [
                        executor.submit(
                            self._run_single_simulation,
                            strategy_function, market_params, parameters, distribution_type
                        )
                        for _ in range(parameters.n_simulations)
                    ]
                    
                    for i, future in enumerate(as_completed(futures)):
                        if i % 1000 == 0 and self.verbose:
                            self.logger.info(f"Completed {i+1}/{parameters.n_simulations} simulations")
                        
                        try:
                            result = future.result()
                            simulation_results.append(result)
                        except Exception as e:
                            self.logger.warning(f"Simulation failed: {str(e)}")
                            simulation_results.append(self._get_default_simulation_result())
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Error in primary simulations: {str(e)}")
            return []

    def _run_single_simulation(self,
                             strategy_function: Callable,
                             market_params: Dict[str, Any],
                             parameters: SimulationParameters,
                             distribution_type: str) -> Dict[str, float]:
        """Run a single Monte Carlo simulation"""
        try:
            # Generate synthetic market data
            synthetic_data = self._generate_synthetic_data(
                market_params, parameters.simulation_length, distribution_type
            )
            
            # Run strategy on synthetic data
            strategy_result = strategy_function(synthetic_data)
            
            # Extract returns
            if isinstance(strategy_result, dict):
                returns = strategy_result.get('returns', [])
            elif isinstance(strategy_result, (list, np.ndarray)):
                returns = strategy_result
            else:
                returns = []
            
            if not returns:
                return self._get_default_simulation_result()
            
            # Calculate performance metrics
            performance = self._calculate_simulation_performance(returns)
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Single simulation failed: {str(e)}")
            return self._get_default_simulation_result()

    def _generate_synthetic_data(self,
                               market_params: Dict[str, Any],
                               length: int,
                               distribution_type: str) -> pd.DataFrame:
        """Generate synthetic market data"""
        try:
            if distribution_type == 'normal':
                returns = np.random.normal(
                    market_params['mean'],
                    market_params['std'],
                    length
                )
                
            elif distribution_type == 'student_t':
                returns = stats.t.rvs(
                    df=market_params.get('df', 3),
                    loc=market_params.get('loc', market_params['mean']),
                    scale=market_params.get('scale', market_params['std']),
                    size=length
                )
                
            elif distribution_type == 'skewed_normal':
                returns = stats.skewnorm.rvs(
                    a=market_params.get('skew_param', 0),
                    loc=market_params.get('loc', market_params['mean']),
                    scale=market_params.get('scale', market_params['std']),
                    size=length
                )
                
            elif distribution_type == 'empirical':
                empirical_returns = market_params.get('empirical_returns', [])
                if len(empirical_returns) > 0:
                    returns = np.random.choice(empirical_returns, size=length, replace=True)
                else:
                    returns = np.random.normal(market_params['mean'], market_params['std'], length)
            
            else:
                returns = np.random.normal(market_params['mean'], market_params['std'], length)
            
            # Generate price series
            initial_price = 100.0
            prices = initial_price * np.cumprod(1 + returns)
            
            # Create DataFrame
            dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
            synthetic_data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'returns': returns,
                'volume': np.random.lognormal(15, 0.5, length)  # Synthetic volume
            })
            
            return synthetic_data
            
        except Exception as e:
            self.logger.warning(f"Error generating synthetic data: {str(e)}")
            # Return minimal valid data
            dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
            returns = np.random.normal(0.0005, 0.02, length)
            prices = 100.0 * np.cumprod(1 + returns)
            
            return pd.DataFrame({
                'date': dates,
                'close': prices,
                'returns': returns
            })

    def _calculate_simulation_performance(self, returns: List[float]) -> Dict[str, float]:
        """Calculate performance metrics for a single simulation"""
        try:
            returns_array = np.array(returns)
            
            if len(returns_array) < 2:
                return self._get_default_simulation_result()
            
            # Basic performance metrics
            total_return = np.prod(1 + returns_array) - 1
            annual_return = np.prod(1 + returns_array) ** (252 / len(returns_array)) - 1
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Downside metrics
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Additional metrics
            win_rate = np.sum(returns_array > 0) / len(returns_array)
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns_array, 5)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'var_95': var_95,
                'skewness': stats.skew(returns_array),
                'kurtosis': stats.kurtosis(returns_array)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating simulation performance: {str(e)}")
            return self._get_default_simulation_result()

    def _run_bootstrap_analysis(self,
                              historical_data: pd.DataFrame,
                              strategy_function: Callable,
                              parameters: SimulationParameters) -> Dict[str, Any]:
        """Run bootstrap analysis on historical data"""
        try:
            self.logger.info("Running bootstrap analysis")
            
            # Extract returns from historical data
            if 'returns' not in historical_data.columns:
                if 'close' in historical_data.columns:
                    historical_returns = historical_data['close'].pct_change().dropna()
                else:
                    historical_returns = historical_data.iloc[:, 0].pct_change().dropna()
            else:
                historical_returns = historical_data['returns'].dropna()
            
            bootstrap_results = []
            n_bootstrap = parameters.bootstrap_samples
            
            for i in range(n_bootstrap):
                # Bootstrap sampling
                boot_returns = np.random.choice(
                    historical_returns.values,
                    size=len(historical_returns),
                    replace=True
                )
                
                # Create bootstrap dataset
                boot_prices = 100 * np.cumprod(1 + boot_returns)
                boot_data = pd.DataFrame({
                    'close': boot_prices,
                    'returns': boot_returns
                })
                
                # Run strategy on bootstrap data
                try:
                    strategy_result = strategy_function(boot_data)
                    if isinstance(strategy_result, dict):
                        returns = strategy_result.get('returns', [])
                    else:
                        returns = strategy_result
                    
                    if returns:
                        performance = self._calculate_simulation_performance(returns)
                        bootstrap_results.append(performance)
                        
                except Exception:
                    continue
            
            if not bootstrap_results:
                return {}
            
            # Aggregate bootstrap results
            bootstrap_summary = {}
            for metric in bootstrap_results[0].keys():
                values = [result[metric] for result in bootstrap_results]
                bootstrap_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'percentile_5': np.percentile(values, 5),
                    'percentile_25': np.percentile(values, 25),
                    'percentile_50': np.percentile(values, 50),
                    'percentile_75': np.percentile(values, 75),
                    'percentile_95': np.percentile(values, 95)
                }
            
            return {
                'bootstrap_summary': bootstrap_summary,
                'n_successful_bootstraps': len(bootstrap_results)
            }
            
        except Exception as e:
            self.logger.warning(f"Error in bootstrap analysis: {str(e)}")
            return {}

    def _run_scenario_analysis(self,
                             strategy_function: Callable,
                             market_params: Dict[str, Any],
                             parameters: SimulationParameters,
                             scenarios: List[MarketScenario]) -> Dict[str, Any]:
        """Run scenario analysis"""
        try:
            self.logger.info("Running scenario analysis")
            
            scenario_results = {}
            
            for scenario in scenarios:
                self.logger.info(f"Analyzing scenario: {scenario.name}")
                
                # Adjust market parameters for scenario
                adjusted_params = market_params.copy()
                adjusted_params['mean'] *= (1 + scenario.return_adjustment)
                adjusted_params['std'] *= (1 + scenario.volatility_adjustment)
                
                # Run simulations for this scenario
                scenario_simulations = []
                n_scenario_sims = min(1000, parameters.n_simulations // len(scenarios))
                
                for _ in range(n_scenario_sims):
                    result = self._run_single_simulation(
                        strategy_function, adjusted_params, parameters, 'normal'
                    )
                    scenario_simulations.append(result)
                
                # Aggregate scenario results
                if scenario_simulations:
                    scenario_summary = {}
                    for metric in scenario_simulations[0].keys():
                        values = [result[metric] for result in scenario_simulations]
                        scenario_summary[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'worst_case': np.min(values),
                            'best_case': np.max(values)
                        }
                    
                    scenario_results[scenario.name] = {
                        'probability': scenario.probability,
                        'summary': scenario_summary,
                        'n_simulations': len(scenario_simulations)
                    }
            
            return scenario_results
            
        except Exception as e:
            self.logger.warning(f"Error in scenario analysis: {str(e)}")
            return {}

    def _calculate_risk_metrics(self, simulation_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            if not simulation_results:
                return {}
            
            returns = [result['total_return'] for result in simulation_results]
            max_drawdowns = [result['max_drawdown'] for result in simulation_results]
            sharpe_ratios = [result['sharpe_ratio'] for result in simulation_results]
            
            risk_metrics = {
                'probability_of_loss': np.sum(np.array(returns) < 0) / len(returns),
                'probability_of_ruin': np.sum(np.array(returns) < -0.5) / len(returns),
                'expected_return': np.mean(returns),
                'return_volatility': np.std(returns),
                'worst_case_return': np.min(returns),
                'best_case_return': np.max(returns),
                'median_return': np.median(returns),
                'expected_max_drawdown': np.mean(max_drawdowns),
                'worst_case_drawdown': np.min(max_drawdowns),
                'median_sharpe_ratio': np.median(sharpe_ratios),
                'probability_positive_sharpe': np.sum(np.array(sharpe_ratios) > 0) / len(sharpe_ratios)
            }
            
            return risk_metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_var_estimates(self,
                               simulation_results: List[Dict[str, float]],
                               confidence_levels: List[float]) -> Dict[str, float]:
        """Calculate Value at Risk estimates"""
        try:
            returns = [result['total_return'] for result in simulation_results]
            
            var_estimates = {}
            for confidence in confidence_levels:
                percentile = (1 - confidence) * 100
                var_value = np.percentile(returns, percentile)
                var_estimates[f'VaR_{int(confidence*100)}'] = var_value
            
            return var_estimates
            
        except Exception:
            return {}

    def _calculate_expected_shortfall(self,
                                    simulation_results: List[Dict[str, float]],
                                    confidence_levels: List[float]) -> Dict[str, float]:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            returns = np.array([result['total_return'] for result in simulation_results])
            
            es_estimates = {}
            for confidence in confidence_levels:
                percentile = (1 - confidence) * 100
                var_threshold = np.percentile(returns, percentile)
                tail_returns = returns[returns <= var_threshold]
                
                if len(tail_returns) > 0:
                    es_value = np.mean(tail_returns)
                else:
                    es_value = var_threshold
                
                es_estimates[f'ES_{int(confidence*100)}'] = es_value
            
            return es_estimates
            
        except Exception:
            return {}

    def _calculate_performance_percentiles(self, 
                                         simulation_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate performance percentiles"""
        try:
            returns = [result['total_return'] for result in simulation_results]
            
            percentiles = {
                'percentile_1': np.percentile(returns, 1),
                'percentile_5': np.percentile(returns, 5),
                'percentile_10': np.percentile(returns, 10),
                'percentile_25': np.percentile(returns, 25),
                'percentile_50': np.percentile(returns, 50),
                'percentile_75': np.percentile(returns, 75),
                'percentile_90': np.percentile(returns, 90),
                'percentile_95': np.percentile(returns, 95),
                'percentile_99': np.percentile(returns, 99)
            }
            
            return percentiles
            
        except Exception:
            return {}

    def _calculate_confidence_intervals(self,
                                      simulation_results: List[Dict[str, float]],
                                      confidence_levels: List[float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        try:
            confidence_intervals = {}
            
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
            
            for metric in metrics:
                if metric in simulation_results[0]:
                    values = [result[metric] for result in simulation_results]
                    
                    for confidence in confidence_levels:
                        alpha = 1 - confidence
                        lower = np.percentile(values, (alpha/2) * 100)
                        upper = np.percentile(values, (1 - alpha/2) * 100)
                        
                        confidence_intervals[f'{metric}_CI_{int(confidence*100)}'] = (lower, upper)
            
            return confidence_intervals
            
        except Exception:
            return {}

    def _run_stress_tests(self,
                        strategy_function: Callable,
                        market_params: Dict[str, Any],
                        parameters: SimulationParameters) -> Dict[str, Any]:
        """Run stress tests"""
        try:
            self.logger.info("Running stress tests")
            
            stress_scenarios = [
                {'name': 'market_crash', 'return_adjustment': -0.3, 'volatility_adjustment': 2.0},
                {'name': 'high_volatility', 'return_adjustment': 0.0, 'volatility_adjustment': 3.0},
                {'name': 'low_return', 'return_adjustment': -0.8, 'volatility_adjustment': 0.5},
                {'name': 'extreme_negative', 'return_adjustment': -0.5, 'volatility_adjustment': 1.5}
            ]
            
            stress_results = {}
            
            for scenario in stress_scenarios:
                # Adjust parameters
                adjusted_params = market_params.copy()
                adjusted_params['mean'] *= (1 + scenario['return_adjustment'])
                adjusted_params['std'] *= (1 + scenario['volatility_adjustment'])
                
                # Run limited simulations
                n_stress_sims = min(500, parameters.n_simulations)
                stress_simulations = []
                
                for _ in range(n_stress_sims):
                    result = self._run_single_simulation(
                        strategy_function, adjusted_params, parameters, 'normal'
                    )
                    stress_simulations.append(result)
                
                if stress_simulations:
                    returns = [result['total_return'] for result in stress_simulations]
                    drawdowns = [result['max_drawdown'] for result in stress_simulations]
                    
                    stress_results[scenario['name']] = {
                        'worst_case_return': np.min(returns),
                        'median_return': np.median(returns),
                        'worst_case_drawdown': np.min(drawdowns),
                        'probability_of_loss': np.sum(np.array(returns) < 0) / len(returns)
                    }
            
            return stress_results
            
        except Exception as e:
            self.logger.warning(f"Error in stress tests: {str(e)}")
            return {}

    def _analyze_convergence(self, simulation_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze simulation convergence"""
        try:
            if len(simulation_results) < 100:
                return {}
            
            returns = [result['total_return'] for result in simulation_results]
            
            # Calculate rolling means
            window_sizes = [100, 500, 1000, min(5000, len(returns))]
            convergence_data = {}
            
            for window in window_sizes:
                if window <= len(returns):
                    rolling_means = []
                    for i in range(window, len(returns) + 1, window//10):
                        subset_mean = np.mean(returns[:i])
                        rolling_means.append(subset_mean)
                    
                    convergence_data[f'window_{window}'] = {
                        'rolling_means': rolling_means,
                        'final_mean': rolling_means[-1] if rolling_means else 0,
                        'convergence_rate': self._calculate_convergence_rate(rolling_means)
                    }
            
            return convergence_data
            
        except Exception:
            return {}

    def _calculate_convergence_rate(self, rolling_means: List[float]) -> float:
        """Calculate convergence rate"""
        try:
            if len(rolling_means) < 3:
                return 0.0
            
            # Calculate rate of change in rolling means
            changes = np.diff(rolling_means)
            recent_changes = changes[-min(5, len(changes)):]
            
            # Convergence is indicated by decreasing absolute changes
            convergence_rate = 1.0 / (1.0 + np.mean(np.abs(recent_changes)))
            
            return convergence_rate
            
        except Exception:
            return 0.0

    def _create_simulation_summary(self, simulation_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Create simulation summary statistics"""
        try:
            if not simulation_results:
                return {}
            
            # Aggregate all metrics
            summary = {}
            for metric in simulation_results[0].keys():
                values = [result[metric] for result in simulation_results]
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_median'] = np.median(values)
            
            summary['n_simulations'] = len(simulation_results)
            
            return summary
            
        except Exception:
            return {}

    def _generate_default_scenarios(self) -> List[MarketScenario]:
        """Generate default market scenarios"""
        return [
            MarketScenario("bull_market", 0.25, 0.5, -0.2, 0.1, 252),
            MarketScenario("bear_market", 0.25, -0.3, 0.5, -0.1, 126),
            MarketScenario("normal_market", 0.4, 0.0, 0.0, 0.0, 252),
            MarketScenario("high_volatility", 0.1, 0.0, 1.0, 0.2, 63)
        ]

    def _get_default_simulation_result(self) -> Dict[str, float]:
        """Get default simulation result"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.02,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.5,
            'var_95': 0.0,
            'skewness': 0.0,
            'kurtosis': 3.0
        }

    def _validate_inputs(self,
                        strategy_function: Callable,
                        historical_data: pd.DataFrame,
                        parameters: SimulationParameters):
        """Validate simulation inputs"""
        if not callable(strategy_function):
            raise ValueError("Strategy function must be callable")
        
        if historical_data.empty:
            raise ValueError("Historical data cannot be empty")
        
        if parameters.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        
        if parameters.simulation_length <= 0:
            raise ValueError("Simulation length must be positive")

    def generate_simulation_report(self,
                                 results: SimulationResults,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive simulation report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "simulation_overview": {
                    "n_simulations": results.simulation_summary.get('n_simulations', 0),
                    "expected_return": results.simulation_summary.get('total_return_mean', 0),
                    "return_volatility": results.simulation_summary.get('total_return_std', 0),
                    "probability_of_loss": results.risk_metrics.get('probability_of_loss', 0)
                },
                "risk_analysis": {
                    "var_estimates": results.var_estimates,
                    "expected_shortfall": results.expected_shortfall,
                    "risk_metrics": results.risk_metrics,
                    "stress_test_results": results.stress_test_results
                },
                "performance_analysis": {
                    "performance_percentiles": results.performance_percentiles,
                    "confidence_intervals": results.confidence_intervals,
                    "bootstrap_results": results.bootstrap_results
                },
                "scenario_analysis": results.scenario_analysis,
                "convergence_analysis": results.convergence_analysis
            }
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}

    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """Get simulation history"""
        return [asdict(result) for result in self.simulation_history]

    def clear_history(self):
        """Clear simulation history"""
        self.simulation_history.clear()
        self.logger.info("Simulation history cleared")


# Example usage and testing
if __name__ == "__main__":
    # Sample strategy function
    def sample_strategy(data):
        """Sample buy-and-hold strategy"""
        try:
            if 'returns' in data.columns:
                returns = data['returns'].tolist()
            elif 'close' in data.columns:
                returns = data['close'].pct_change().dropna().tolist()
            else:
                returns = []
            
            return {'returns': returns}
            
        except Exception:
            return {'returns': []}
    
    # Sample historical data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)
    
    historical_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'returns': returns
    })
    
    # Simulation parameters
    sim_params = SimulationParameters(
        n_simulations=1000,
        simulation_length=252,  # 1 year
        random_seed=42,
        confidence_levels=[0.90, 0.95, 0.99],
        bootstrap_samples=500,
        scenario_analysis=True,
        correlation_modeling=False,
        fat_tail_modeling=True
    )
    
    # Initialize simulator
    simulator = MonteCarloSimulator(
        random_seed=42,
        n_jobs=1,  # Use 1 for this example
        verbose=True
    )
    
    # Run simulation
    results = simulator.run_simulation(
        strategy_function=sample_strategy,
        historical_data=historical_data,
        parameters=sim_params,
        distribution_type='normal'
    )
    
    print(f"Expected Return: {results.simulation_summary.get('total_return_mean', 0):.4f}")
    print(f"Return Volatility: {results.simulation_summary.get('total_return_std', 0):.4f}")
    print(f"Probability of Loss: {results.risk_metrics.get('probability_of_loss', 0):.3f}")
    print(f"VaR 95%: {results.var_estimates.get('VaR_95', 0):.4f}")
    print(f"Expected Shortfall 95%: {results.expected_shortfall.get('ES_95', 0):.4f}")
    
    # Generate report
    report = simulator.generate_simulation_report(results)
    print(f"\nSimulation completed with {len(report)} analysis sections")
