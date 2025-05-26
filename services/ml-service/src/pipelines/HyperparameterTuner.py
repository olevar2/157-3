"""
Hyperparameter Tuner for ML Model Optimization

This module provides comprehensive hyperparameter optimization capabilities
for machine learning models used in trading applications. It includes various
optimization strategies and automated tuning workflows.

Key Features:
- Grid search optimization
- Random search optimization
- Bayesian optimization
- Genetic algorithm optimization
- Multi-objective optimization
- Automated hyperparameter tuning
- Performance tracking and analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import asyncio
import itertools
import random
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
try:
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import optuna
    from scipy.optimize import differential_evolution
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("Optimization libraries not available. Using mock implementations.")

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Optimization methods for hyperparameter tuning."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    HYPERBAND = "hyperband"

class SearchSpace(Enum):
    """Types of search spaces."""
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    FLOAT = "float"
    LOG_UNIFORM = "log_uniform"
    UNIFORM = "uniform"

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN
    n_trials: int = 100
    n_jobs: int = 1
    cv_folds: int = 5
    scoring_metric: str = "neg_mean_squared_error"
    direction: str = "minimize"  # minimize or maximize
    timeout: Optional[int] = None  # seconds
    early_stopping_rounds: Optional[int] = 10
    random_state: int = 42
    verbose: bool = True

@dataclass
class TuningResult:
    """Result from hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    all_trials: List[Dict[str, Any]]
    optimization_history: List[float]
    feature_importance: Optional[Dict[str, float]]
    tuning_time: float
    n_trials_completed: int

class HyperparameterTuner:
    """
    Comprehensive Hyperparameter Tuning Pipeline
    
    Provides various optimization strategies for hyperparameter tuning
    with support for different search spaces and objectives.
    """
    
    def __init__(self, config: TuningConfig = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            config: Tuning configuration
        """
        self.config = config or TuningConfig()
        self.search_space = {}
        self.objective_function = None
        self.study = None
        self.best_params = None
        self.tuning_history = []
        
        # Set random seed
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        logger.info(f"HyperparameterTuner initialized with method: {self.config.optimization_method.value}")
    
    def define_search_space(self, search_space: Dict[str, Dict[str, Any]]):
        """
        Define the hyperparameter search space.
        
        Args:
            search_space: Dictionary defining parameter ranges and types
                Example: {
                    'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
                    'hidden_units': {'type': 'categorical', 'choices': [64, 128, 256]},
                    'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5}
                }
        """
        self.search_space = search_space
        logger.info(f"Defined search space with {len(search_space)} parameters")
    
    def set_objective_function(self, objective_func: Callable):
        """
        Set the objective function for optimization.
        
        Args:
            objective_func: Function that takes parameters and returns score
        """
        self.objective_function = objective_func
        logger.info("Objective function set")
    
    async def optimize(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> TuningResult:
        """
        Optimize hyperparameters using the specified method.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            
        Returns:
            Tuning result with best parameters and performance
        """
        start_time = datetime.now()
        logger.info(f"Starting hyperparameter optimization with {self.config.optimization_method.value}...")
        
        if not self.search_space:
            raise ValueError("Search space must be defined before optimization")
        
        if not self.objective_function:
            raise ValueError("Objective function must be set before optimization")
        
        # Choose optimization method
        if self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
            result = await self._grid_search_optimization(X, y, validation_data)
        elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            result = await self._random_search_optimization(X, y, validation_data)
        elif self.config.optimization_method == OptimizationMethod.BAYESIAN:
            result = await self._bayesian_optimization(X, y, validation_data)
        elif self.config.optimization_method == OptimizationMethod.GENETIC_ALGORITHM:
            result = await self._genetic_algorithm_optimization(X, y, validation_data)
        else:
            # Default to Bayesian
            result = await self._bayesian_optimization(X, y, validation_data)
        
        tuning_time = (datetime.now() - start_time).total_seconds()
        result.tuning_time = tuning_time
        
        self.best_params = result.best_params
        
        logger.info(f"Hyperparameter optimization completed in {tuning_time:.2f}s. "
                   f"Best score: {result.best_score:.6f}")
        
        return result
    
    async def _grid_search_optimization(self, 
                                       X: pd.DataFrame, 
                                       y: pd.Series,
                                       validation_data: Optional[Tuple[pd.DataFrame, pd.Series]]) -> TuningResult:
        """Perform grid search optimization."""
        if not OPTIMIZATION_AVAILABLE:
            return await self._mock_optimization()
        
        # Convert search space to sklearn format
        param_grid = {}
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'categorical':
                param_grid[param_name] = param_config['choices']
            elif param_config['type'] in ['float', 'integer']:
                # Create grid for continuous parameters
                low, high = param_config['low'], param_config['high']
                n_points = param_config.get('n_points', 10)
                
                if param_config['type'] == 'integer':
                    param_grid[param_name] = list(range(int(low), int(high) + 1, max(1, (int(high) - int(low)) // n_points)))
                else:
                    if param_config.get('log', False):
                        param_grid[param_name] = np.logspace(np.log10(low), np.log10(high), n_points).tolist()
                    else:
                        param_grid[param_name] = np.linspace(low, high, n_points).tolist()
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Limit number of combinations if too many
        if len(param_combinations) > self.config.n_trials:
            param_combinations = random.sample(param_combinations, self.config.n_trials)
        
        # Evaluate each combination
        all_trials = []
        scores = []
        
        for i, params in enumerate(param_combinations):
            try:
                score = await self._evaluate_parameters(params, X, y, validation_data)
                all_trials.append({'trial': i, 'params': params, 'score': score})
                scores.append(score)
                
                if self.config.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(param_combinations)} trials")
                    
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
                continue
        
        if not scores:
            raise ValueError("All trials failed")
        
        # Find best parameters
        if self.config.direction == "minimize":
            best_idx = np.argmin(scores)
        else:
            best_idx = np.argmax(scores)
        
        best_params = param_combinations[best_idx]
        best_score = scores[best_idx]
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_trial=best_idx,
            all_trials=all_trials,
            optimization_history=scores,
            feature_importance=None,
            tuning_time=0.0,  # Will be set by caller
            n_trials_completed=len(scores)
        )
    
    async def _random_search_optimization(self, 
                                         X: pd.DataFrame, 
                                         y: pd.Series,
                                         validation_data: Optional[Tuple[pd.DataFrame, pd.Series]]) -> TuningResult:
        """Perform random search optimization."""
        if not OPTIMIZATION_AVAILABLE:
            return await self._mock_optimization()
        
        all_trials = []
        scores = []
        
        for i in range(self.config.n_trials):
            # Sample random parameters
            params = self._sample_random_parameters()
            
            try:
                score = await self._evaluate_parameters(params, X, y, validation_data)
                all_trials.append({'trial': i, 'params': params, 'score': score})
                scores.append(score)
                
                if self.config.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{self.config.n_trials} trials")
                    
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
                continue
        
        if not scores:
            raise ValueError("All trials failed")
        
        # Find best parameters
        if self.config.direction == "minimize":
            best_idx = np.argmin(scores)
        else:
            best_idx = np.argmax(scores)
        
        best_params = all_trials[best_idx]['params']
        best_score = scores[best_idx]
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_trial=best_idx,
            all_trials=all_trials,
            optimization_history=scores,
            feature_importance=None,
            tuning_time=0.0,
            n_trials_completed=len(scores)
        )
    
    async def _bayesian_optimization(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    validation_data: Optional[Tuple[pd.DataFrame, pd.Series]]) -> TuningResult:
        """Perform Bayesian optimization using Optuna."""
        if not OPTIMIZATION_AVAILABLE:
            return await self._mock_optimization()
        
        try:
            import optuna
            
            # Create study
            direction = "minimize" if self.config.direction == "minimize" else "maximize"
            self.study = optuna.create_study(direction=direction)
            
            # Define objective function for Optuna
            def objective(trial):
                params = {}
                for param_name, param_config in self.search_space.items():
                    if param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                    elif param_config['type'] == 'integer':
                        params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                    elif param_config['type'] == 'float':
                        if param_config.get('log', False):
                            params[param_name] = trial.suggest_loguniform(param_name, param_config['low'], param_config['high'])
                        else:
                            params[param_name] = trial.suggest_uniform(param_name, param_config['low'], param_config['high'])
                
                # Evaluate parameters
                try:
                    score = asyncio.run(self._evaluate_parameters(params, X, y, validation_data))
                    return score
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return float('inf') if direction == "minimize" else float('-inf')
            
            # Optimize
            self.study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
            
            # Extract results
            all_trials = []
            scores = []
            
            for i, trial in enumerate(self.study.trials):
                all_trials.append({
                    'trial': i,
                    'params': trial.params,
                    'score': trial.value if trial.value is not None else float('inf')
                })
                scores.append(trial.value if trial.value is not None else float('inf'))
            
            best_params = self.study.best_params
            best_score = self.study.best_value
            best_trial = self.study.best_trial.number
            
            return TuningResult(
                best_params=best_params,
                best_score=best_score,
                best_trial=best_trial,
                all_trials=all_trials,
                optimization_history=scores,
                feature_importance=None,
                tuning_time=0.0,
                n_trials_completed=len(scores)
            )
            
        except ImportError:
            logger.warning("Optuna not available. Using random search fallback.")
            return await self._random_search_optimization(X, y, validation_data)
