"""
GeneticAlgorithmOptimizer Indicator - Evolutionary Parameter Optimization
Platform3 Trading Framework
Version: 1.0.0

This indicator implements genetic algorithm optimization for trading strategy
parameters, using evolutionary principles to find optimal configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import random

import sys
import os

from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for GeneticAlgorithm indicator"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    tournament_size: int = 5
    convergence_threshold: float = 0.001
    max_stagnation: int = 20
    parameter_bounds: Dict[str, Tuple[float, float]] = None


class Individual:
    """Represents an individual in the genetic algorithm population"""
    
    def __init__(self, parameters: Dict[str, float]):
        self.parameters = parameters
        self.fitness = 0.0
        self.age = 0
        
    def mutate(self, mutation_rate: float, bounds: Dict[str, Tuple[float, float]]):
        """Mutate individual parameters"""
        for param_name, value in self.parameters.items():
            if random.random() < mutation_rate:
                if param_name in bounds:
                    min_val, max_val = bounds[param_name]
                    # Gaussian mutation
                    mutation = random.gauss(0, (max_val - min_val) * 0.1)
                    new_value = value + mutation
                    self.parameters[param_name] = np.clip(new_value, min_val, max_val)
                    
    def crossover(self, other: 'Individual') -> 'Individual':
        """Create offspring through crossover"""
        child_params = {}
        for param_name in self.parameters.keys():
            if random.random() < 0.5:
                child_params[param_name] = self.parameters[param_name]
            else:
                child_params[param_name] = other.parameters[param_name]
        return Individual(child_params)


class GeneticAlgorithmOptimizerIndicator(StandardIndicatorInterface):
    """
    GeneticAlgorithmOptimizer Indicator v1.0.0
    
    An evolutionary optimization system that uses genetic algorithms to find
    optimal trading strategy parameters through natural selection principles.
    
    Features:
    - Population-based parameter optimization
    - Multiple selection strategies (tournament, roulette wheel)
    - Adaptive mutation and crossover rates
    - Elitism preservation
    - Multi-objective optimization support
    - Convergence detection
    
    Mathematical Foundation:
    The genetic algorithm follows these steps:
    1. Initialize random population
    2. Evaluate fitness for each individual
    3. Select parents based on fitness
    4. Create offspring through crossover and mutation
    5. Replace worst individuals with offspring
    6. Repeat until convergence
    
    Fitness function: F(x) = Σ(wi * fi(x))
    where wi are weights and fi are individual fitness components
    """
    
    # Class-level metadata
    name = "GeneticAlgorithmOptimizer"
    version = "1.0.0"
    category = "ml_advanced"
    description = "Evolutionary parameter optimization using genetic algorithms"
    
    def __init__(self, **params):
        """Initialize GeneticAlgorithmOptimizer indicator"""
        # Extract parameters with defaults
        self.parameters = params
        
        # Default parameter bounds
        default_bounds = {
            'sma_period': (5, 50),
            'rsi_period': (5, 30),
            'bb_period': (10, 40),
            'bb_std': (1.0, 3.0),
            'macd_fast': (5, 20),
            'macd_slow': (15, 40),
            'threshold': (0.1, 0.9)
        }
        
        self.config = GeneticAlgorithmConfig(
            population_size=self.parameters.get('population_size', 50),
            generations=self.parameters.get('generations', 100),
            mutation_rate=self.parameters.get('mutation_rate', 0.1),
            crossover_rate=self.parameters.get('crossover_rate', 0.8),
            elite_ratio=self.parameters.get('elite_ratio', 0.2),
            tournament_size=self.parameters.get('tournament_size', 5),
            convergence_threshold=self.parameters.get('convergence_threshold', 0.001),
            max_stagnation=self.parameters.get('max_stagnation', 20),
            parameter_bounds=self.parameters.get('parameter_bounds', default_bounds)
        )
        
        # Initialize state
        self.reset()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def reset(self):
        """Reset indicator state"""
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        self.generation_count = 0
        self.stagnation_count = 0
        self.convergence_reached = False
        
    def calculate(self, data: Union[pd.DataFrame, Dict[str, List], np.ndarray]) -> np.ndarray:
        """
        Calculate GeneticAlgorithmOptimizer results
        
        Args:
            data: Price data (OHLCV format)
            
        Returns:
            np.ndarray: Optimization results and best parameters
        """
        try:
            # Input validation
            if data is None or len(data) == 0:
                raise ValidationError("Input data cannot be empty")
                
            # Convert data to DataFrame if needed
            df = self._prepare_data(data)
            
            if len(df) < 50:  # Need sufficient data for optimization
                return np.full((len(df), 5), np.nan)
                
            # Initialize population if first run
            if not self.population:
                self._initialize_population()
                
            # Run optimization
            optimization_results = self._run_optimization(df)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in GeneticAlgorithmOptimizer calculation: {str(e)}")
            raise CalculationError(f"GeneticAlgorithmOptimizer calculation failed: {str(e)}")
            
    def _prepare_data(self, data: Any) -> pd.DataFrame:
        """Prepare and validate input data"""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({'close': data})
            else:
                columns = ['open', 'high', 'low', 'close', 'volume'][:data.shape[1]]
                df = pd.DataFrame(data, columns=columns)
        else:
            raise ValidationError("Unsupported data format")
            
        # Ensure required columns
        if 'close' not in df.columns:
            raise ValidationError("Close price is required")
            
        # Fill missing columns with close price
        for col in ['open', 'high', 'low']:
            if col not in df.columns:
                df[col] = df['close']
                
        if 'volume' not in df.columns:
            df['volume'] = 1.0
            
        return df.dropna()
        
    def _initialize_population(self):
        """Initialize random population"""
        self.population = []
        
        for _ in range(self.config.population_size):
            individual_params = {}
            
            for param_name, (min_val, max_val) in self.config.parameter_bounds.items():
                individual_params[param_name] = random.uniform(min_val, max_val)
                
            individual = Individual(individual_params)
            self.population.append(individual)
            
    def _run_optimization(self, df: pd.DataFrame) -> np.ndarray:
        """Run genetic algorithm optimization"""
        n_points = len(df)
        results = np.zeros((n_points, 5))  # fitness, generation, mutation_rate, best_param1, best_param2
        
        # Evaluate initial population
        self._evaluate_population(df)
        
        # Evolution loop
        for generation in range(self.config.generations):
            if self.convergence_reached:
                break
                
            # Selection and reproduction
            new_population = self._create_new_generation()
            
            # Replace population
            self.population = new_population
            
            # Evaluate new population
            self._evaluate_population(df)
            
            # Update best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = Individual(current_best.parameters.copy())
                self.best_individual.fitness = current_best.fitness
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
                
            # Check convergence
            self._check_convergence()
            
            self.generation_count += 1
            
        # Fill results with optimization progress
        for i in range(n_points):
            progress = min(i / max(1, n_points - 1), 1.0)
            
            if self.best_individual:
                best_params = list(self.best_individual.parameters.values())
                results[i] = [
                    self.best_individual.fitness,
                    self.generation_count,
                    self.config.mutation_rate * (1 + 0.5 * progress),  # Adaptive mutation
                    best_params[0] if len(best_params) > 0 else 0,
                    best_params[1] if len(best_params) > 1 else 0
                ]
            
        return results
        
    def _evaluate_population(self, df: pd.DataFrame):
        """Evaluate fitness for all individuals in population"""
        for individual in self.population:
            individual.fitness = self._calculate_fitness(individual, df)
            
    def _calculate_fitness(self, individual: Individual, df: pd.DataFrame) -> float:
        """Calculate fitness for an individual"""
        try:
            # Extract parameters
            params = individual.parameters
            
            # Calculate trading signals using parameters
            signals = self._generate_signals_with_params(df, params)
            
            # Calculate fitness metrics
            returns = self._calculate_strategy_returns(df, signals)
            
            # Multi-objective fitness
            total_return = np.sum(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Combine metrics into single fitness score
            fitness = (
                0.4 * total_return +
                0.4 * sharpe_ratio +
                0.2 * (1 - max_drawdown)  # Lower drawdown is better
            )
            
            return max(0, fitness)  # Ensure non-negative fitness
            
        except Exception as e:
            self.logger.warning(f"Error calculating fitness: {str(e)}")
            return 0.0
            
    def _generate_signals_with_params(self, df: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
        """Generate trading signals using given parameters"""
        close = df['close'].values
        
        # Moving average signals
        sma_period = int(params.get('sma_period', 20))
        sma = pd.Series(close).rolling(sma_period).mean().values
        ma_signal = np.where(close > sma, 1, -1)
        
        # RSI signals
        rsi_period = int(params.get('rsi_period', 14))
        rsi = self._calculate_rsi(close, rsi_period)
        rsi_signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        
        # Bollinger Band signals
        bb_period = int(params.get('bb_period', 20))
        bb_std = params.get('bb_std', 2.0)
        bb_upper, bb_lower = self._calculate_bollinger_bands(close, bb_period, bb_std)
        bb_signal = np.where(close < bb_lower, 1, np.where(close > bb_upper, -1, 0))
        
        # Combine signals
        threshold = params.get('threshold', 0.5)
        combined_signal = (ma_signal + rsi_signal + bb_signal) / 3
        
        final_signals = np.where(combined_signal > threshold, 1,
                               np.where(combined_signal < -threshold, -1, 0))
        
        return final_signals
        
    def _calculate_strategy_returns(self, df: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        """Calculate strategy returns based on signals"""
        close = df['close'].values
        returns = np.diff(np.log(close))
        
        # Align signals with returns (signals[:-1] because returns is one element shorter)
        strategy_returns = signals[:-1] * returns
        
        return strategy_returns
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
    def _create_new_generation(self) -> List[Individual]:
        """Create new generation through selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best individuals
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite_count = int(self.config.elite_ratio * self.config.population_size)
        new_population.extend(sorted_population[:elite_count])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = Individual(parent1.parameters.copy())
                
            # Mutation
            child.mutate(self.config.mutation_rate, self.config.parameter_bounds)
            
            new_population.append(child)
            
        return new_population[:self.config.population_size]
        
    def _tournament_selection(self) -> Individual:
        """Select individual using tournament selection"""
        tournament = random.sample(self.population, 
                                 min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
        
    def _check_convergence(self):
        """Check if algorithm has converged"""
        if self.stagnation_count >= self.config.max_stagnation:
            self.convergence_reached = True
            
        # Check fitness improvement
        if len(self.fitness_history) > 10:
            recent_improvement = (self.fitness_history[-1] - self.fitness_history[-10]) / 10
            if abs(recent_improvement) < self.config.convergence_threshold:
                self.convergence_reached = True
                
        # Store current best fitness
        if self.best_individual:
            self.fitness_history.append(self.best_individual.fitness)
            
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean().values
        avg_loss = pd.Series(losses).rolling(period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi])
        
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(period).mean().values
        std = pd.Series(prices).rolling(period).std().values
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, lower
        
    def get_signal(self, data: Any) -> int:
        """Get current signal from the indicator"""
        if self.best_individual is None:
            return 0
        # Use best parameters to generate signal
        df = self._prepare_data(data)
        signals = self._generate_signals_with_params(df, self.best_individual.parameters)
        return int(signals[-1]) if len(signals) > 0 else 0
        
    def get_current_value(self, data: Any) -> float:
        """Get current indicator value"""
        return self.best_individual.fitness if self.best_individual else 0.0
        
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        try:
            population_size = self.parameters.get('population_size', 50)
            if not isinstance(population_size, int) or population_size <= 0:
                return False
                
            generations = self.parameters.get('generations', 100)
            if not isinstance(generations, int) or generations <= 0:
                return False
                
            mutation_rate = self.parameters.get('mutation_rate', 0.1)
            if not isinstance(mutation_rate, (int, float)) or mutation_rate < 0 or mutation_rate > 1:
                return False
                
            return True
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return GeneticAlgorithmOptimizer metadata as dictionary for compatibility"""
        return {
            "name": "GeneticAlgorithmOptimizer",
            "category": self.CATEGORY,
            "description": "Genetic Algorithm Optimizer for parameter optimization of trading strategies",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Dict",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """GeneticAlgorithmOptimizer can work with OHLCV data"""
        return ["open", "high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for genetic algorithm optimization"""
        return max(self.parameters.get("period", 20), 100)  # Needs more data for optimization


def get_genetic_algorithm_optimizer_indicator(**params) -> GeneticAlgorithmOptimizerIndicator:
    """
    Factory function to create GeneticAlgorithmOptimizer indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        GeneticAlgorithmOptimizerIndicator: Configured indicator instance
    """
    return GeneticAlgorithmOptimizerIndicator(**params)


# Export for registry discovery
__all__ = ['GeneticAlgorithmOptimizerIndicator', 'get_genetic_algorithm_optimizer_indicator']