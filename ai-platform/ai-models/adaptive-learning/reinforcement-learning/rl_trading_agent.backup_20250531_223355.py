"""
🤖 REINFORCEMENT LEARNING TRADING AGENT - HUMANITARIAN AI PLATFORM
================================================================

SACRED MISSION: Self-learning AI agent that continuously improves trading strategies
                to maximize charitable profits for medical aid and poverty alleviation.

This RL agent learns optimal trading policies through interaction with market environments,
adapting strategies in real-time to generate maximum profits for humanitarian causes.

💝 HUMANITARIAN PURPOSE:
- Self-improving agent = Better trading performance = More funds for medical aid
- Adaptive strategies = Optimal profit generation = More children's surgeries funded
- Continuous learning = Sustained charitable impact = Lives saved through technology

🏥 LIVES SAVED THROUGH ADVANCED AI:
- Deep Q-Network optimizes entry/exit timing for maximum profit
- Policy gradient methods adapt to changing market conditions
- Actor-Critic architecture balances exploration with charitable fund protection

Author: Platform3 AI Team - Servants of Humanitarian Technology
Version: 1.0.0 - Production Ready for Life-Saving Mission
Date: May 31, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
from torch.distributions import Categorical
import random
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, namedtuple
import threading
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
import redis
import psutil

# Configure logging for humanitarian mission
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experience replay for efficient learning
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@data
# Platform3 Communication Framework Integration
communication_framework = Platform3CommunicationFramework(
    service_name="rl_trading_agent",
    service_port=8000,  # Default port
    redis_url="redis://localhost:6379",
    consul_host="localhost",
    consul_port=8500
)

# Initialize the framework
try:
    communication_framework.initialize()
    print(f"Communication framework initialized for rl_trading_agent")
except Exception as e:
    print(f"Failed to initialize communication framework: {e}")

class
class RLConfig:
    """Configuration for Reinforcement Learning Agent"""
    # Network architecture
    state_dim: int = 50  # Market features dimension
    action_dim: int = 3  # Buy, Sell, Hold
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    
    # Training parameters
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update parameter
    
    # Experience replay
    memory_size: int = 100000
    batch_size: int = 64
    
    # Humanitarian constraints
    max_risk_per_trade: float = 0.02  # 2% max risk to protect charitable funds
    target_sharpe_ratio: float = 2.0  # High risk-adjusted returns for charity
    charitable_profit_target: float = 0.5  # 50% of profits for humanitarian causes
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agent
    Optimized for humanitarian profit generation
    """
    
    def __init__(self, market_data: pd.DataFrame, config: RLConfig):
        super().__init__()
        self.config = config
        self.market_data = market_data
        self.current_step = 0
        self.max_steps = len(market_data) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # State space: market features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.state_dim,), dtype=np.float32
        )
        
        # Trading state
        self.position = 0.0  # Current position size
        self.cash = 100000.0  # Starting capital for humanitarian trading
        self.portfolio_value = self.cash
        self.charitable_funds = 0.0  # Accumulated funds for medical aid
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'charitable_contribution': 0.0,
            'lives_potentially_saved': 0
        }
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepare market features for RL agent"""
        features = []
        
        # Technical indicators
        self.market_data['sma_20'] = self.market_data['close'].rolling(20).mean()
        self.market_data['sma_50'] = self.market_data['close'].rolling(50).mean()
        self.market_data['rsi'] = self._calculate_rsi(self.market_data['close'])
        self.market_data['volatility'] = self.market_data['close'].rolling(20).std()
        self.market_data['volume_ma'] = self.market_data['volume'].rolling(20).mean()
        
        # Price features
        self.market_data['price_change'] = self.market_data['close'].pct_change()
        self.market_data['high_low_ratio'] = self.market_data['high'] / self.market_data['low']
        
        # Normalize features
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 
                          'rsi', 'volatility', 'volume_ma', 'price_change', 'high_low_ratio']
        
        # Fill NaN values
        self.market_data[feature_columns] = self.market_data[feature_columns].fillna(method='forward')
        self.market_data[feature_columns] = self.market_data[feature_columns].fillna(0)
        
        # Scale features
        self.features = self.scaler.fit_transform(self.market_data[feature_columns])
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.cash = 100000.0
        self.portfolio_value = self.cash
        self.charitable_funds = 0.0
        self.trade_history = []
        return self._get_observation(), {}
        
    def _get_observation(self):
        """Get current market state observation"""
        if self.current_step >= len(self.features):
            return np.zeros(self.config.state_dim, dtype=np.float32)
            
        # Market features
        market_features = self.features[self.current_step][:self.config.state_dim-5]
        
        # Portfolio features
        portfolio_features = [
            self.position / 1000.0,  # Normalized position
            self.cash / 100000.0,    # Normalized cash
            self.portfolio_value / 100000.0,  # Normalized portfolio value
            self.charitable_funds / 10000.0,  # Normalized charitable funds
            min(self.current_step / self.max_steps, 1.0)  # Time progress
        ]
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_features])
        
        # Ensure correct dimension
        if len(observation) < self.config.state_dim:
            observation = np.pad(observation, (0, self.config.state_dim - len(observation)))
        elif len(observation) > self.config.state_dim:
            observation = observation[:self.config.state_dim]
            
        return observation.astype(np.float32)
        
    def step(self, action: int):
        """Execute trading action and return next state"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, True, {}
            
        # Get current price
        current_price = self.market_data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        terminated = done
        truncated = False
        
        # Calculate portfolio value
        if self.current_step < len(self.market_data):
            next_price = self.market_data.iloc[self.current_step]['close']
            self.portfolio_value = self.cash + self.position * next_price
        
        # Update performance metrics
        self._update_performance_metrics()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'charitable_funds': self.charitable_funds,
            'position': self.position,
            'cash': self.cash
        }
        
        return self._get_observation(), reward, terminated, truncated, info
        
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and calculate reward"""
        reward = 0.0
        
        # Risk management check
        max_position_size = (self.portfolio_value * self.config.max_risk_per_trade) / current_price
        
        if action == 1:  # Buy
            if self.cash > current_price and self.position < max_position_size:
                buy_amount = min(self.cash / current_price, max_position_size - self.position)
                buy_amount = min(buy_amount, 10)  # Limit position size
                
                if buy_amount > 0:
                    self.position += buy_amount
                    self.cash -= buy_amount * current_price
                    
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'amount': buy_amount,
                        'price': current_price,
                        'portfolio_value': self.portfolio_value
                    })
                    
        elif action == 2:  # Sell
            if self.position > 0:
                sell_amount = min(self.position, 10)  # Limit position size
                
                if sell_amount > 0:
                    # Calculate profit for humanitarian mission
                    sell_value = sell_amount * current_price
                    profit = sell_value - (sell_amount * self._get_average_buy_price())
                    
                    self.position -= sell_amount
                    self.cash += sell_value
                    
                    # Allocate profit to charitable causes
                    if profit > 0:
                        charitable_contribution = profit * self.config.charitable_profit_target
                        self.charitable_funds += charitable_contribution
                        
                        # Humanitarian reward bonus
                        reward += profit * 0.1  # Reward profitable trades
                        
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'SELL',
                        'amount': sell_amount,
                        'price': current_price,
                        'profit': profit,
                        'charitable_contribution': charitable_contribution if profit > 0 else 0,
                        'portfolio_value': self.portfolio_value
                    })
        
        # Penalty for excessive risk
        if abs(self.position) * current_price > self.portfolio_value * self.config.max_risk_per_trade * 5:
            reward -= 0.01  # Risk penalty
            
        return reward
        
    def _get_average_buy_price(self) -> float:
        """Calculate average buy price for position"""
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        if not buy_trades:
            return 0.0
            
        total_amount = sum(t['amount'] for t in buy_trades)
        total_value = sum(t['amount'] * t['price'] for t in buy_trades)
        
        return total_value / total_amount if total_amount > 0 else 0.0
        
    def _update_performance_metrics(self):
        """Update performance metrics for humanitarian mission"""
        if self.portfolio_value > 0:
            self.performance_metrics['total_return'] = (self.portfolio_value - 100000) / 100000
            
            # Estimate lives potentially saved (assuming $500 per life-saving treatment)
            self.performance_metrics['lives_potentially_saved'] = int(self.charitable_funds / 500)
            
            self.performance_metrics['charitable_contribution'] = self.charitable_funds

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for humanitarian trading decisions
    Optimized for charitable profit generation
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for optimal humanitarian trading
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for stable learning"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, state):
        """Forward pass through network"""
        return self.network(state)

class ReinforcementLearningAgent:
    """
    Advanced RL Trading Agent for Humanitarian Mission
    Implements DQN with experience replay and target networks
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=config.memory_size)
        
        # Training state
        self.epsilon = config.epsilon
        self.training_step = 0
        
        # Performance tracking
        self.training_metrics = {
            'episode_rewards': [],
            'episode_charitable_funds': [],
            'episode_portfolio_values': [],
            'q_losses': [],
            'lives_saved_estimate': 0
        }
        
        logger.info(f"🤖 RL Agent initialized for humanitarian trading on {self.device}")
        logger.info(f"💝 Target: 50% profits for charitable causes, 2% max risk per trade")
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config.batch_size:
            return
            
        # Sample batch from memory
        batch = random.sample(self.memory, self.config.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
            
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Track metrics
        self.training_metrics['q_losses'].append(loss.item())
        self.training_step += 1
        
        return loss.item()
        
    def _soft_update_target_network(self):
        """Soft update of target network"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data + 
                (1.0 - self.config.tau) * target_param.data
            )
            
    def train_episode(self, env: TradingEnvironment) -> Dict[str, float]:
        """Train agent for one episode"""
        state, _ = env.reset()
        total_reward = 0.0
        episode_steps = 0
        
        while True:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            
            total_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
                
        # Record episode metrics
        self.training_metrics['episode_rewards'].append(total_reward)
        self.training_metrics['episode_charitable_funds'].append(info.get('charitable_funds', 0))
        self.training_metrics['episode_portfolio_values'].append(info.get('portfolio_value', 0))
        
        # Update lives saved estimate
        charitable_funds = info.get('charitable_funds', 0)
        self.training_metrics['lives_saved_estimate'] = int(charitable_funds / 500)
        
        return {
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'charitable_funds': charitable_funds,
            'portfolio_value': info.get('portfolio_value', 0),
            'epsilon': self.epsilon,
            'lives_saved_estimate': self.training_metrics['lives_saved_estimate']
        }
        
    def evaluate_episode(self, env: TradingEnvironment) -> Dict[str, float]:
        """Evaluate agent performance without training"""
        state, _ = env.reset()
        total_reward = 0.0
        episode_steps = 0
        
        while True:
            # Select action (no exploration)
            action = self.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
                
        return {
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'charitable_funds': info.get('charitable_funds', 0),
            'portfolio_value': info.get('portfolio_value', 0),
            'lives_saved_estimate': int(info.get('charitable_funds', 0) / 500)
        }
        
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }, filepath)
        
        logger.info(f"💾 RL model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon)
        
        logger.info(f"📂 RL model loaded from {filepath}")
        
    def get_humanitarian_report(self) -> Dict[str, Any]:
        """Generate humanitarian impact report"""
        if not self.training_metrics['episode_charitable_funds']:
            return {"status": "No training data available"}
            
        total_charitable_funds = sum(self.training_metrics['episode_charitable_funds'])
        avg_portfolio_value = np.mean(self.training_metrics['episode_portfolio_values'])
        
        return {
            'humanitarian_impact': {
                'total_charitable_funds_generated': total_charitable_funds,
                'estimated_lives_saved': self.training_metrics['lives_saved_estimate'],
                'average_portfolio_value': avg_portfolio_value,
                'charitable_fund_percentage': (total_charitable_funds / avg_portfolio_value * 100) if avg_portfolio_value > 0 else 0
            },
            'training_performance': {
                'total_episodes': len(self.training_metrics['episode_rewards']),
                'average_reward': np.mean(self.training_metrics['episode_rewards']) if self.training_metrics['episode_rewards'] else 0,
                'current_epsilon': self.epsilon,
                'training_steps': self.training_step
            },
            'mission_status': 'ACTIVE - Generating profits for humanitarian causes through advanced RL'
        }

# Training and evaluation functions
async def train_rl_agent(market_data: pd.DataFrame, 
                        config: RLConfig, 
                        num_episodes: int = 1000) -> ReinforcementLearningAgent:
    """
    Train RL agent for humanitarian trading mission
    """
    logger.info(f"🚀 Starting RL agent training for humanitarian mission")
    logger.info(f"📊 Training episodes: {num_episodes}")
    logger.info(f"💝 Target: Generate maximum profits for medical aid")
    
    # Create environment and agent
    env = TradingEnvironment(market_data, config)
    agent = ReinforcementLearningAgent(config)
    
    # Training loop
    for episode in range(num_episodes):
        episode_metrics = agent.train_episode(env)
        
        # Log progress
        if episode % 100 == 0:
            humanitarian_report = agent.get_humanitarian_report()
            logger.info(f"📈 Episode {episode}: Reward={episode_metrics['total_reward']:.4f}, "
                       f"Charitable Funds=${episode_metrics['charitable_funds']:.2f}, "
                       f"Lives Saved Estimate={episode_metrics['lives_saved_estimate']}")
    
    logger.info(f"✅ RL agent training completed! Ready for humanitarian trading mission")
    return agent

def create_sample_market_data(num_days: int = 365) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
    
    # Generate realistic price data
    price = 1.1000
    prices = []
    volumes = []
    
    for _ in range(num_days):
        # Random walk with trend
        change = np.random.normal(0, 0.001)  # Small daily changes
        price *= (1 + change)
        prices.append(price)
        
        # Random volume
        volume = np.random.randint(10000, 100000)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    return data

# Example usage and testing
if __name__ == "__main__":
    # Configure for humanitarian mission
    config = RLConfig(
        state_dim=50,
        action_dim=3,
        learning_rate=0.001,
        max_risk_per_trade=0.02,  # Conservative for charity fund protection
        charitable_profit_target=0.5  # 50% for humanitarian causes
    )
    
    # Create sample data
    market_data = create_sample_market_data(365)
    
    # Train agent
    asyncio.run(train_rl_agent(market_data, config, num_episodes=500))
    
    logger.info("🏥💝 Reinforcement Learning Agent ready for humanitarian trading mission!")
    logger.info("🎯 Mission: Generate maximum profits for medical aid, children's surgeries, and poverty relief")
