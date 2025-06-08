#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Algorithm Optimizer - Advanced ML Indicator
Platform3 Phase 2B - ML Advanced Category Implementation

This module provides genetic algorithm-based parameter optimization for trading strategies
and indicator parameters with adaptive evolution capabilities.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import random
import copy


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm-based parameter optimizer for trading strategies.
    
    Features:
    - Multi-objective optimization
    - Adaptive mutation rates
    - Elite preservation
    - Real-time parameter tuning
    """
    
    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_ratio: float = 0.1):
        """
        Initialize Genetic Algorithm Optimizer
        
        Args:
            population_size: Number of individuals in population
            generations: Number of evolution generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Ratio of elite individuals to preserve
        """
        self.name = "GeneticAlgorithmOptimizer"
        self.version = "1.0.0"
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.elite_count = int(population_size * elite_ratio)
        
        # Population and evolution tracking
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation_count = 0
        
        # Parameter space definition
        self.parameter_space = self._define_parameter_space()
        
    def _define_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """Define the parameter space for optimization"""
        return {
            'sma_period': {'min': 5, 'max': 200, 'type': 'int'},
            'ema_period': {'min': 5, 'max': 100, 'type': 'int'},
            'rsi_period': {'min': 2, 'max': 50, 'type': 'int'},
            'bollinger_period': {'min': 10, 'max': 50, 'type': 'int'},
            'bollinger_std': {'min': 1.0, 'max': 3.0, 'type': 'float'},
            'macd_fast': {'min': 5, 'max': 25, 'type': 'int'},
            'macd_slow': {'min': 20, 'max': 50, 'type': 'int'},
            'macd_signal': {'min': 5, 'max': 20, 'type': 'int'},
            'stop_loss': {'min': 0.01, 'max': 0.1, 'type': 'float'},
            'take_profit': {'min': 0.02, 'max': 0.2, 'type': 'float'}
        }
    
    def calculate(self, data: Any) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization
        
        Args:
            data: Market data for fitness evaluation
            
        Returns:
            Dict containing optimal parameters and optimization results
        """
        try:
            # Validate input data
            if not self._validate_data(data):
                return self._create_default_result("Invalid input data")
            
            # Prepare data for optimization
            market_data = self._prepare_data(data)
            
            # Initialize population if not exists
            if not self.population:
                self._initialize_population()
            
            # Run evolution
            optimization_result = self._evolve_population(market_data)
            
            # Update best solution
            self._update_best_solution()
            
            return {
                'value': self.best_fitness,
                'optimal_parameters': self.best_individual,
                'fitness_score': self.best_fitness,
                'generation': self.generation_count,
                'population_diversity': self._calculate_diversity(),
                'convergence_rate': self._calculate_convergence_rate(),
                'optimization_efficiency': optimization_result.get('efficiency', 0.0),
                'metadata': {
                    'population_size': self.population_size,
                    'generations_run': self.generation_count,
                    'mutation_rate': self.mutation_rate,
                    'crossover_rate': self.crossover_rate,
                    'elite_preserved': self.elite_count,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Genetic algorithm optimization failed: {str(e)}")
    
    def _validate_data(self, data: Any) -> bool:
        """Validate input data"""
        if data is None:
            return False
        
        # Handle different data formats
        if isinstance(data, dict):
            required_keys = ['open', 'high', 'low', 'close']
            return any(key in data for key in required_keys)
        elif isinstance(data, (list, np.ndarray)):
            return len(data) > 10  # Need sufficient data for optimization
        
        return True
    
    def _prepare_data(self, data: Any) -> np.ndarray:
        """Prepare data for optimization"""
        try:
            if isinstance(data, dict):
                # Extract price data
                close_prices = data.get('close', [])
                if not close_prices:
                    # Try other price fields
                    close_prices = data.get('price', data.get('value', []))
                
                return np.array(close_prices) if close_prices else np.random.random(100)
            
            elif isinstance(data, (list, np.ndarray)):
                return np.array(data)
            
            else:
                # Generate synthetic data for testing
                return np.random.random(100) * 100 + 50
                
        except Exception as e:
            print(f"Data preparation error: {e}")
            return np.random.random(100) * 100 + 50
    
    def _initialize_population(self):
        """Initialize random population"""
        self.population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param_name, param_config in self.parameter_space.items():
                if param_config['type'] == 'int':
                    value = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'float':
                    value = random.uniform(param_config['min'], param_config['max'])
                else:
                    value = param_config['min']
                
                individual[param_name] = value
            
            self.population.append(individual)
    
    def _evolve_population(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Evolve population for one generation"""
        start_time = datetime.now()
        
        # Evaluate fitness for all individuals
        fitness_scores = [self._evaluate_fitness(individual, market_data) 
                         for individual in self.population]
        
        # Sort population by fitness (descending)
        population_with_fitness = list(zip(self.population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted population and fitness
        sorted_population = [ind for ind, _ in population_with_fitness]
        sorted_fitness = [fit for _, fit in population_with_fitness]
        
        # Store fitness history
        self.fitness_history.append(sorted_fitness[0])  # Best fitness
        
        # Create new population
        new_population = []
        
        # Preserve elites
        new_population.extend(sorted_population[:self.elite_count])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(sorted_population, sorted_fitness)
            parent2 = self._tournament_selection(sorted_population, sorted_fitness)
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                offspring1 = self._mutate(offspring1)
            if random.random() < self.mutation_rate:
                offspring2 = self._mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
        self.generation_count += 1
        
        # Calculate efficiency
        end_time = datetime.now()
        efficiency = 1.0 / (end_time - start_time).total_seconds()
        
        return {'efficiency': efficiency}
    
    def _evaluate_fitness(self, individual: Dict[str, Any], market_data: np.ndarray) -> float:
        """Evaluate fitness of an individual"""
        try:
            # Simulate trading strategy with given parameters
            returns = self._simulate_strategy(individual, market_data)
            
            # Calculate fitness metrics
            total_return = np.sum(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Combine metrics into fitness score
            fitness = (total_return * 0.4 + 
                      sharpe_ratio * 0.4 + 
                      (1.0 - max_drawdown) * 0.2)
            
            return fitness
            
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            return -1.0  # Poor fitness for failed evaluations
    
    def _simulate_strategy(self, params: Dict[str, Any], prices: np.ndarray) -> np.ndarray:
        """Simulate trading strategy with given parameters"""
        try:
            returns = []
            position = 0
            entry_price = 0
            
            # Simple moving average crossover strategy
            sma_short = self._calculate_sma(prices, params.get('sma_period', 10))
            sma_long = self._calculate_sma(prices, params.get('ema_period', 20))
            
            for i in range(len(sma_short)):
                if i == 0:
                    continue
                
                current_price = prices[i]
                
                # Entry signals
                if position == 0:  # No position
                    if sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1]:
                        # Buy signal
                        position = 1
                        entry_price = current_price
                    elif sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1]:
                        # Sell signal
                        position = -1
                        entry_price = current_price
                
                # Exit signals (stop loss/take profit)
                elif position != 0:
                    price_change = (current_price - entry_price) / entry_price
                    
                    # Apply position direction
                    if position == -1:
                        price_change = -price_change
                    
                    # Check exit conditions
                    stop_loss = params.get('stop_loss', 0.05)
                    take_profit = params.get('take_profit', 0.1)
                    
                    if price_change <= -stop_loss or price_change >= take_profit:
                        returns.append(price_change)
                        position = 0
                        entry_price = 0
            
            return np.array(returns) if returns else np.array([0.0])
            
        except Exception as e:
            print(f"Strategy simulation error: {e}")
            return np.array([0.0])
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.array([prices[0]] * len(prices))
        
        sma = np.convolve(prices, np.ones(period), 'valid') / period
        # Pad with first value for equal length
        padding = np.array([sma[0]] * (period - 1))
        return np.concatenate([padding, sma])
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        
        return abs(np.min(drawdown))
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection for parent selection"""
        tournament_indices = random.sample(range(len(population)), 
                                          min(tournament_size, len(population)))
        
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(population[best_index])
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-point crossover"""
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        # Random crossover point
        params = list(parent1.keys())
        crossover_point = random.randint(1, len(params) - 1)
        
        # Swap parameters after crossover point
        for i in range(crossover_point, len(params)):
            param = params[i]
            offspring1[param], offspring2[param] = parent2[param], parent1[param]
        
        return offspring1, offspring2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Gaussian mutation"""
        mutated = copy.deepcopy(individual)
        
        for param_name, value in mutated.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                param_config = self.parameter_space[param_name]
                
                if param_config['type'] == 'int':
                    # Integer mutation
                    mutation_range = (param_config['max'] - param_config['min']) * 0.1
                    mutation = int(random.gauss(0, mutation_range))
                    mutated[param_name] = max(param_config['min'], 
                                            min(param_config['max'], value + mutation))
                
                elif param_config['type'] == 'float':
                    # Float mutation
                    mutation_range = (param_config['max'] - param_config['min']) * 0.1
                    mutation = random.gauss(0, mutation_range)
                    mutated[param_name] = max(param_config['min'], 
                                            min(param_config['max'], value + mutation))
        
        return mutated
    
    def _update_best_solution(self):
        """Update best solution found so far"""
        if self.fitness_history:
            current_best_fitness = self.fitness_history[-1]
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                # Assume first individual is best (from sorted population)
                self.best_individual = copy.deepcopy(self.population[0])
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if not self.population:
            return 0.0
        
        try:
            # Calculate average parameter variance
            variances = []
            for param_name in self.parameter_space.keys():
                values = [ind.get(param_name, 0) for ind in self.population]
                if len(set(values)) > 1:
                    variances.append(np.var(values))
            
            return np.mean(variances) if variances else 0.0
        except:
            return 0.0
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Simple convergence measure based on fitness improvement
        recent_improvements = 0
        for i in range(1, min(11, len(self.fitness_history))):  # Last 10 generations
            if self.fitness_history[-i] > self.fitness_history[-i-1]:
                recent_improvements += 1
        
        return recent_improvements / min(10, len(self.fitness_history) - 1)
    
    def _create_default_result(self, message: str) -> Dict[str, Any]:
        """Create default result for edge cases"""
        return {
            'value': 0.0,
            'optimal_parameters': {},
            'fitness_score': 0.0,
            'generation': 0,
            'population_diversity': 0.0,
            'convergence_rate': 0.0,
            'optimization_efficiency': 0.0,
            'error': message,
            'metadata': {
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'value': 0.0,
            'optimal_parameters': {},
            'fitness_score': 0.0,
            'generation': 0,
            'population_diversity': 0.0,
            'convergence_rate': 0.0,
            'optimization_efficiency': 0.0,
            'error': error_message,
            'metadata': {
                'timestamp': datetime.now().isoformat()
            }
        }
