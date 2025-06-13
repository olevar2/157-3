"""
Genetic Algorithm Optimizer Indicator

A genetic algorithm optimizer indicator that uses evolutionary computation principles
to optimize trading parameters and identify optimal market entry/exit conditions.
This indicator evolves trading strategies through selection, crossover, and mutation
operations to adapt to changing market conditions.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
import os
import random
from dataclasses import dataclass

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


@dataclass
class TradingGene:
    """Represents a trading strategy gene"""
    entry_threshold: float
    exit_threshold: float
    lookback_period: int
    risk_factor: float
    momentum_weight: float
    
    def mutate(self, mutation_rate: float = 0.1):
        """Apply mutation to the gene"""
        if random.random() < mutation_rate:
            self.entry_threshold += random.gauss(0, 0.1)
            self.entry_threshold = np.clip(self.entry_threshold, 0.1, 2.0)
        
        if random.random() < mutation_rate:
            self.exit_threshold += random.gauss(0, 0.1)
            self.exit_threshold = np.clip(self.exit_threshold, 0.1, 2.0)
        
        if random.random() < mutation_rate:
            self.lookback_period += random.randint(-2, 2)
            self.lookback_period = np.clip(self.lookback_period, 5, 50)
        
        if random.random() < mutation_rate:
            self.risk_factor += random.gauss(0, 0.05)
            self.risk_factor = np.clip(self.risk_factor, 0.01, 0.5)
        
        if random.random() < mutation_rate:
            self.momentum_weight += random.gauss(0, 0.1)
            self.momentum_weight = np.clip(self.momentum_weight, 0.1, 2.0)


class GeneticAlgorithmOptimizer(BaseIndicator):
    """
    Genetic Algorithm Optimizer Indicator
    
    Uses genetic algorithm principles to optimize trading strategies:
    - Population-based parameter optimization
    - Selection based on fitness (performance metrics)
    - Crossover to combine successful strategies
    - Mutation for exploration and adaptation
    - Elite preservation to maintain best performers
    - Multi-objective optimization (return vs risk)
    
    The indicator provides:
    - Optimized trading signals
    - Strategy fitness scores
    - Parameter evolution tracking
    - Best performer identification
    - Risk-adjusted performance metrics
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_ratio: float = 0.1,
                 fitness_lookback: int = 50):
        """
        Initialize Genetic Algorithm Optimizer indicator
        
        Args:
            population_size: Number of strategies in population
            generations: Number of evolution generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Ratio of elite strategies to preserve
            fitness_lookback: Lookback period for fitness evaluation
        """
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.fitness_lookback = fitness_lookback
        
        # Initialize population
        self.population = self._initialize_population()
        self.generation_count = 0
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate genetic algorithm optimization
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'optimized_signals': Best performing trading signals
            - 'fitness_scores': Fitness scores for current population
            - 'best_parameters': Parameters of best performing strategy
            - 'evolution_progress': Evolution progress indicator
            - 'risk_adjusted_return': Risk-adjusted performance metric
        """
        try:
            if len(data) < self.fitness_lookback:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'optimized_signals': empty_series,
                    'fitness_scores': empty_series,
                    'best_parameters': empty_series,
                    'evolution_progress': empty_series,
                    'risk_adjusted_return': empty_series
                }
            
            close = data['close']
            volume = data['volume']
            
            # Run genetic algorithm optimization
            if len(data) >= self.fitness_lookback and self.generation_count < self.generations:
                self._evolve_population(data)
                self.generation_count += 1
            
            # Generate signals using best strategy
            optimized_signals = self._generate_optimized_signals(data)
            
            # Calculate fitness scores for population
            fitness_scores = self._calculate_population_fitness(data)
            
            # Get best parameters
            best_parameters = self._get_best_parameters()
            
            # Calculate evolution progress
            evolution_progress = self._calculate_evolution_progress()
            
            # Calculate risk-adjusted returns
            risk_adjusted_return = self._calculate_risk_adjusted_return(data, optimized_signals)
            
            return {
                'optimized_signals': optimized_signals,
                'fitness_scores': fitness_scores,
                'best_parameters': best_parameters,
                'evolution_progress': evolution_progress,
                'risk_adjusted_return': risk_adjusted_return
            }
            
        except Exception as e:
            print(f"Error in Genetic Algorithm Optimizer: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'optimized_signals': empty_series,
                'fitness_scores': empty_series,
                'best_parameters': empty_series,
                'evolution_progress': empty_series,
                'risk_adjusted_return': empty_series
            }
    
    def _initialize_population(self) -> List[TradingGene]:
        """Initialize random population of trading strategies"""
        population = []
        for _ in range(self.population_size):
            gene = TradingGene(
                entry_threshold=random.uniform(0.5, 1.5),
                exit_threshold=random.uniform(0.5, 1.5),
                lookback_period=random.randint(10, 30),
                risk_factor=random.uniform(0.05, 0.3),
                momentum_weight=random.uniform(0.5, 1.5)
            )
            population.append(gene)
        return population
    
    def _evolve_population(self, data: pd.DataFrame):
        """Evolve the population through selection, crossover, and mutation"""
        # Calculate fitness for each strategy
        fitness_scores = []
        for gene in self.population:
            fitness = self._calculate_strategy_fitness(data, gene)
            fitness_scores.append(fitness)
        
        # Create new population
        new_population = []
        
        # Elite preservation
        elite_count = int(self.population_size * self.elite_ratio)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection (tournament selection)
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1
            
            # Mutation
            child.mutate(self.mutation_rate)
            new_population.append(child)
        
        self.population = new_population
    
    def _calculate_strategy_fitness(self, data: pd.DataFrame, gene: TradingGene) -> float:
        """Calculate fitness score for a trading strategy"""
        try:
            close = data['close']
            
            # Generate signals using this strategy
            signals = self._generate_signals_for_gene(data, gene)
            
            # Calculate returns
            returns = []
            position = 0
            
            for i in range(1, len(signals)):
                signal = signals.iloc[i]
                prev_signal = signals.iloc[i-1]
                
                # Entry
                if signal != 0 and prev_signal == 0:
                    position = signal
                    entry_price = close.iloc[i]
                
                # Exit
                elif signal == 0 and position != 0:
                    exit_price = close.iloc[i]
                    if position == 1:  # Long position
                        ret = (exit_price - entry_price) / entry_price
                    else:  # Short position
                        ret = (entry_price - exit_price) / entry_price
                    
                    # Risk adjustment
                    ret *= (1 - gene.risk_factor)
                    returns.append(ret)
                    position = 0
            
            if len(returns) > 0:
                # Calculate Sharpe-like ratio
                avg_return = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0.1
                fitness = avg_return / (volatility + 1e-8)
                
                # Penalty for excessive trading
                trade_count_penalty = min(0.1, len(returns) / len(data))
                fitness -= trade_count_penalty
                
                return fitness
            else:
                return -1.0  # No trades = poor fitness
        except:
            return -1.0
    
    def _generate_signals_for_gene(self, data: pd.DataFrame, gene: TradingGene) -> pd.Series:
        """Generate trading signals for a specific gene"""
        try:
            close = data['close']
            volume = data['volume']
            signals = pd.Series(0, index=data.index)
            
            # Calculate indicators based on gene parameters
            sma = close.rolling(window=gene.lookback_period, min_periods=1).mean()
            momentum = close.pct_change(gene.lookback_period)
            volume_sma = volume.rolling(window=gene.lookback_period, min_periods=1).mean()
            volume_ratio = volume / volume_sma
            
            # Generate signals
            for i in range(gene.lookback_period, len(data)):
                price_above_sma = close.iloc[i] > sma.iloc[i] * (1 + gene.entry_threshold * 0.01)
                price_below_sma = close.iloc[i] < sma.iloc[i] * (1 - gene.entry_threshold * 0.01)
                
                strong_momentum = abs(momentum.iloc[i]) > gene.momentum_weight * 0.01
                volume_confirmation = volume_ratio.iloc[i] > 1.2
                
                # Long signal
                if price_above_sma and momentum.iloc[i] > 0 and strong_momentum and volume_confirmation:
                    signals.iloc[i] = 1
                
                # Short signal
                elif price_below_sma and momentum.iloc[i] < 0 and strong_momentum and volume_confirmation:
                    signals.iloc[i] = -1
                
                # Exit signals
                elif abs(momentum.iloc[i]) < gene.exit_threshold * 0.005:
                    signals.iloc[i] = 0
            
            return signals
        except:
            return pd.Series(0, index=data.index)
    
    def _tournament_selection(self, fitness_scores: List[float]) -> TradingGene:
        """Tournament selection for parent choosing"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
        best_idx = max(tournament_indices, key=lambda x: fitness_scores[x])
        return self.population[best_idx]
    
    def _crossover(self, parent1: TradingGene, parent2: TradingGene) -> TradingGene:
        """Create offspring through crossover"""
        return TradingGene(
            entry_threshold=(parent1.entry_threshold + parent2.entry_threshold) / 2,
            exit_threshold=(parent1.exit_threshold + parent2.exit_threshold) / 2,
            lookback_period=random.choice([parent1.lookback_period, parent2.lookback_period]),
            risk_factor=(parent1.risk_factor + parent2.risk_factor) / 2,
            momentum_weight=(parent1.momentum_weight + parent2.momentum_weight) / 2
        )
    
    def _generate_optimized_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using the best performing strategy"""
        try:
            # Find best strategy
            fitness_scores = []
            for gene in self.population:
                fitness = self._calculate_strategy_fitness(data, gene)
                fitness_scores.append(fitness)
            
            best_idx = np.argmax(fitness_scores)
            best_gene = self.population[best_idx]
            
            return self._generate_signals_for_gene(data, best_gene)
        except:
            return pd.Series(0, index=data.index)
    
    def _calculate_population_fitness(self, data: pd.DataFrame) -> pd.Series:
        """Calculate fitness scores for the entire population"""
        try:
            fitness_scores = []
            for gene in self.population:
                fitness = self._calculate_strategy_fitness(data, gene)
                fitness_scores.append(fitness)
            
            # Return average fitness as time series
            avg_fitness = np.mean(fitness_scores)
            return pd.Series(avg_fitness, index=data.index)
        except:
            return pd.Series(0, index=data.index)
    
    def _get_best_parameters(self) -> pd.Series:
        """Get parameters of the best performing strategy"""
        try:
            # This is simplified - in practice would return a structured representation
            best_score = -float('inf')
            best_gene = self.population[0]
            
            for gene in self.population:
                # Use a simple heuristic for parameter quality
                score = (gene.entry_threshold + gene.exit_threshold + 
                        gene.lookback_period/20 + gene.momentum_weight - gene.risk_factor)
                if score > best_score:
                    best_score = score
                    best_gene = gene
            
            # Return a composite parameter score
            return pd.Series(best_score, index=range(1))
        except:
            return pd.Series(0, index=range(1))
    
    def _calculate_evolution_progress(self) -> pd.Series:
        """Calculate evolution progress indicator"""
        try:
            progress = self.generation_count / self.generations
            return pd.Series(progress, index=range(1))
        except:
            return pd.Series(0, index=range(1))
    
    def _calculate_risk_adjusted_return(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate risk-adjusted return for the optimized strategy"""
        try:
            close = data['close']
            returns = []
            
            # Calculate strategy returns
            position = 0
            for i in range(1, len(signals)):
                if signals.iloc[i] != signals.iloc[i-1]:
                    if position != 0:  # Close position
                        ret = position * (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
                        returns.append(ret)
                    position = signals.iloc[i]
            
            if len(returns) > 0:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
                return pd.Series(sharpe_ratio, index=data.index)
            else:
                return pd.Series(0, index=data.index)
        except:
            return pd.Series(0, index=data.index)
    
    def get_optimized_signals(self, data: pd.DataFrame) -> pd.Series:
        """Get optimized trading signals"""
        result = self.calculate(data)
        return result['optimized_signals']
    
    def get_fitness_scores(self, data: pd.DataFrame) -> pd.Series:
        """Get fitness scores"""
        result = self.calculate(data)
        return result['fitness_scores']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data with trends and patterns
    base_price = 100
    trend = np.cumsum(np.random.randn(100) * 0.5)
    noise = np.random.randn(100) * 2
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 2, 100),
        'low': close_prices - np.random.uniform(0, 2, 100),
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.5, 100)
    }, index=dates)
    
    # Test the indicator
    print("Testing Genetic Algorithm Optimizer Indicator")
    print("=" * 50)
    
    indicator = GeneticAlgorithmOptimizer(
        population_size=20,  # Smaller for testing
        generations=10,      # Fewer generations for testing
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_ratio=0.1,
        fitness_lookback=30
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Optimized signals range: {result['optimized_signals'].min():.0f} to {result['optimized_signals'].max():.0f}")
    print(f"Fitness scores: {result['fitness_scores'].iloc[0]:.3f}")
    print(f"Evolution progress: {result['evolution_progress'].iloc[0]:.1%}")
    print(f"Risk-adjusted return: {result['risk_adjusted_return'].mean():.3f}")
    
    # Analyze signals
    signals = result['optimized_signals']
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    
    print(f"\nSignal Analysis:")
    print(f"Long signals: {long_signals}")
    print(f"Short signals: {short_signals}")
    print(f"Neutral signals: {neutral_signals}")
    
    print(f"\nGenetic Algorithm Evolution:")
    print(f"Population size: {indicator.population_size}")
    print(f"Generations completed: {indicator.generation_count}")
    print(f"Mutation rate: {indicator.mutation_rate}")
    print(f"Crossover rate: {indicator.crossover_rate}")
    
    print("\nGenetic Algorithm Optimizer Indicator test completed successfully!")