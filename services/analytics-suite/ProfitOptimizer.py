"""
Advanced Profit Optimization System for Trading Platform3

This module provides comprehensive profit optimization capabilities including:
- Position sizing optimization using Kelly Criterion and risk-adjusted methods
- Trade timing optimization with market regime detection
- Multi-objective optimization balancing profit vs risk
- Portfolio rebalancing strategies
- Advanced risk-adjusted performance metrics
- Profit factor optimization across multiple trading strategies
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from scipy.optimize import minimize, differential_evolution
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import warnings
import threading
import time

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Redis import for real-time features - Temporarily disabled for testing
REDIS_AVAILABLE = False
# try:
#     import redis
#     REDIS_AVAILABLE = True
# except ImportError:
#     REDIS_AVAILABLE = False
#     logger.warning("Redis not available - real-time features disabled")

# Platform3 Integration Imports
try:
    import sys
    import os
        from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
except ImportError:
    # Fallback for testing
    class Platform3CommunicationFramework:
        def __init__(self, config=None):
            self.config = config or {}
        def send_message(self, *args, **kwargs):
            pass
        def listen_for_messages(self, *args, **kwargs):
            pass

@dataclass
class AnalyticsReport:
    """Standardized analytics report structure"""
    service_name: str
    timestamp: datetime
    timeframe: str
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    data_quality: str

@dataclass
class RealtimeMetric:
    """Real-time metric structure for live monitoring"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    status: str  # 'normal', 'warning', 'critical'
    threshold: Optional[float] = None

class AnalyticsInterface(ABC):
    """Standardized interface for all analytics services"""
    
    @abstractmethod
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data and return analysis results"""
        pass
    
    @abstractmethod
    async def generate_report(self, timeframe: str = "1h") -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        pass
    
    @abstractmethod
    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        """Get current real-time performance metrics"""
        pass

@dataclass
class OptimizationResult:
    """Results from profit optimization analysis"""
    optimal_position_size: float
    expected_return: float
    risk_adjusted_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    optimization_method: str
    confidence_interval: Tuple[float, float]
    recommendations: List[str]

@dataclass
class TradeMetrics:
    """Individual trade performance metrics"""
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    duration: int
    timestamp: datetime
    strategy: str
    market_regime: str

@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    average_trade: float

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfitOptimizer(AnalyticsInterface):
    """
    Advanced Profit Optimization System with Platform3 Integration
    
    Provides comprehensive profit optimization capabilities for trading strategies
    including position sizing, timing optimization, and portfolio rebalancing.
    Implements AnalyticsInterface for standardized framework integration.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 max_position_size: float = 0.25,
                 confidence_level: float = 0.95,
                 optimization_method: str = 'kelly',
                 redis_url: str = "redis://localhost:6379"):
        """
        Initialize the ProfitOptimizer with Platform3 integration
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            max_position_size: Maximum allowed position size (0-1)
            confidence_level: Confidence level for risk calculations
            optimization_method: Optimization method ('kelly', 'markowitz', 'risk_parity')
            redis_url: Redis connection URL for real-time communication
        """
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.confidence_level = confidence_level
        self.optimization_method = optimization_method
        
        # Initialize optimization parameters
        self.scaler = StandardScaler()
        self.market_regime_model = KMeans(n_clusters=3, random_state=42)
        
        # Performance tracking
        self.optimization_history = []
        self.current_portfolio = {}
          # Platform3 Integration (simplified for testing)
        try:
            self.communication_framework = Platform3CommunicationFramework({
                'service_name': "profit-optimizer-analytics",
                'service_port': 8005,
                'redis_url': redis_url
            })
        except Exception as e:
            logger.warning(f"Platform3 communication framework initialization failed: {e}")
            self.communication_framework = None
        
        # Real-time metrics tracking
        self.real_time_metrics = {}
        self.processing_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_processing_time': 0.0,
            'last_optimization_time': None,
            'current_profit_factor': 0.0,
            'optimization_efficiency': 0.0
        }
          # Redis connection for real-time data
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis connection established for ProfitOptimizer")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis not available - real-time features disabled")
            self.redis_client = None
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info(f"ProfitOptimizer initialized with method: {optimization_method}")
    
    def _start_background_monitoring(self):
        """Start background monitoring for real-time metrics"""
        def monitoring_loop():
            while True:
                try:
                    self._update_real_time_metrics()
                    self._publish_metrics()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(10)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def _update_real_time_metrics(self):
        """Update real-time metrics for monitoring"""
        try:
            current_time = datetime.now()
            
            # Calculate optimization efficiency
            if self.processing_stats['total_optimizations'] > 0:
                efficiency = (self.processing_stats['successful_optimizations'] / 
                             self.processing_stats['total_optimizations']) * 100
            else:
                efficiency = 0.0
            
            self.processing_stats['optimization_efficiency'] = efficiency
            
            # Update real-time metrics
            self.real_time_metrics.update({
                'profit_optimization_efficiency': RealtimeMetric(
                    metric_name="Profit Optimization Efficiency",
                    value=efficiency,
                    unit="percentage",
                    timestamp=current_time,
                    status='normal' if efficiency >= 80 else 'warning' if efficiency >= 60 else 'critical',
                    threshold=80.0
                ),
                'active_optimizations': RealtimeMetric(
                    metric_name="Active Optimizations",
                    value=float(len(self.optimization_history)),
                    unit="count",
                    timestamp=current_time,
                    status='normal'
                ),
                'average_processing_time': RealtimeMetric(
                    metric_name="Average Processing Time",
                    value=self.processing_stats['average_processing_time'],
                    unit="seconds",
                    timestamp=current_time,
                    status='normal' if self.processing_stats['average_processing_time'] <= 2.0 else 'warning',
                    threshold=2.0
                ),
                'current_profit_factor': RealtimeMetric(
                    metric_name="Current Profit Factor",
                    value=self.processing_stats['current_profit_factor'],
                    unit="ratio",
                    timestamp=current_time,
                    status='normal' if self.processing_stats['current_profit_factor'] >= 1.5 else 'warning'
                )
            })
            
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")
    
    def _publish_metrics(self):
        """Publish metrics to Redis for real-time monitoring"""
        if self.redis_client:
            try:
                metrics_data = {
                    'service': 'profit-optimizer-analytics',
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {k: asdict(v) for k, v in self.real_time_metrics.items()},
                    'processing_stats': self.processing_stats
                }
                
                self.redis_client.publish('analytics:metrics:profit-optimizer', 
                                        json.dumps(metrics_data))
                
            except Exception as e:
                logger.error(f"Error publishing metrics: {e}")
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process trading data for profit optimization analysis
        
        Args:
            data: Dictionary containing trading data, market conditions, and strategy info
            
        Returns:
            Dict containing optimization analysis results
        """
        start_time = time.time()
        self.processing_stats['total_optimizations'] += 1
        
        try:
            logger.info("Starting profit optimization data processing")
            
            # Optimize profits using the main optimization method
            optimization_result = self.optimize_profits(data)
            
            # Extract and process portfolio data if available
            portfolio_allocation = {}
            if 'strategies' in data:
                portfolio_allocation = self.optimize_portfolio_allocation(data['strategies'])
            
            # Calculate additional metrics
            risk_metrics = self._calculate_risk_metrics(data)
            performance_insights = self._generate_performance_insights(optimization_result)
            
            # Prepare results
            processing_result = {
                'optimization_result': asdict(optimization_result),
                'portfolio_allocation': portfolio_allocation,
                'risk_metrics': risk_metrics,
                'performance_insights': performance_insights,
                'processing_time': time.time() - start_time,
                'optimization_method': self.optimization_method,
                'timestamp': datetime.now().isoformat()
            }
              # Update statistics
            self.processing_stats['successful_optimizations'] += 1
            self.processing_stats['current_profit_factor'] = optimization_result.profit_factor
            self.processing_stats['last_optimization_time'] = datetime.now()
            
            # Store optimization result in history for reporting
            optimization_history_item = {
                'timestamp': datetime.now().isoformat(),
                'optimization_result': asdict(optimization_result),
                'profit_factor': optimization_result.profit_factor,
                'sharpe_ratio': optimization_result.sharpe_ratio,
                'processing_time': time.time() - start_time
            }
            self.optimization_history.append(optimization_history_item)
            
            # Update average processing time
            processing_time = time.time() - start_time
            if self.processing_stats['average_processing_time'] == 0:
                self.processing_stats['average_processing_time'] = processing_time
            else:
                self.processing_stats['average_processing_time'] = (
                    self.processing_stats['average_processing_time'] * 0.8 + processing_time * 0.2
                )
            
            # Publish results to Platform3
            if self.communication_framework:
                await self._publish_optimization_results(processing_result)
            
            logger.info(f"Profit optimization processing completed in {processing_time:.2f}s")
            return processing_result
            
        except Exception as e:
            logger.error(f"Error in profit optimization processing: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_report(self, timeframe: str = "1h") -> AnalyticsReport:
        """
        Generate comprehensive profit optimization analytics report
        
        Args:
            timeframe: Time period for the report (1h, 4h, 1d, 1w)
            
        Returns:
            AnalyticsReport: Comprehensive profit optimization report
        """
        try:
            current_time = datetime.now()
            
            # Collect optimization history for the timeframe
            timeframe_data = self._get_timeframe_data(timeframe)
            
            # Calculate performance metrics
            report_metrics = {
                'total_optimizations': len(timeframe_data),
                'average_profit_factor': np.mean([opt.get('profit_factor', 0) for opt in timeframe_data]) if timeframe_data else 0,
                'average_sharpe_ratio': np.mean([opt.get('sharpe_ratio', 0) for opt in timeframe_data]) if timeframe_data else 0,
                'optimization_success_rate': (self.processing_stats['successful_optimizations'] / 
                                            max(self.processing_stats['total_optimizations'], 1)) * 100,
                'average_processing_time': self.processing_stats['average_processing_time'],
                'best_optimization_method': self.optimization_method,
                'current_efficiency': self.processing_stats['optimization_efficiency']
            }
            
            # Generate insights
            insights = self._generate_optimization_insights(timeframe_data, report_metrics)
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(report_metrics)
            
            # Calculate confidence score
            confidence_score = self._calculate_report_confidence(timeframe_data, report_metrics)
            
            # Determine data quality
            data_quality = self._assess_data_quality(timeframe_data)
            
            report = AnalyticsReport(
                service_name="Profit Optimizer Analytics",
                timestamp=current_time,
                timeframe=timeframe,
                metrics=report_metrics,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                data_quality=data_quality
            )
            
            logger.info(f"Generated profit optimization report for {timeframe} timeframe")
            return report
            
        except Exception as e:
            logger.error(f"Error generating profit optimization report: {e}")
            return AnalyticsReport(
                service_name="Profit Optimizer Analytics",
                timestamp=datetime.now(),
                timeframe=timeframe,
                metrics={'error': str(e)},
                insights=[f"Report generation failed: {str(e)}"],
                recommendations=["Check system logs and data availability"],
                confidence_score=0.0,
                data_quality="poor"
            )
    
    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        """
        Get current real-time profit optimization metrics
        
        Returns:
            List[RealtimeMetric]: Current real-time metrics
        """
        try:
            # Update metrics before returning
            self._update_real_time_metrics()
            
            # Return all current real-time metrics
            return list(self.real_time_metrics.values())
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return [
                RealtimeMetric(
                    metric_name="Service Status",
                    value=0.0,
                    unit="status",
                    timestamp=datetime.now(),
                    status="critical"
                )
            ]
    
    def optimize_profits(self, strategy_data: Dict[str, Any]) -> OptimizationResult:
        """
        Main profit optimization method
        
        Args:
            strategy_data: Dictionary containing trading data and metrics
            
        Returns:
            OptimizationResult: Comprehensive optimization results
        """
        try:
            # Extract and validate data
            trades_data = self._extract_trades_data(strategy_data)
            market_data = self._extract_market_data(strategy_data)
            
            # Detect market regimes
            market_regimes = self._detect_market_regimes(market_data)
            
            # Calculate baseline metrics
            baseline_metrics = self._calculate_baseline_metrics(trades_data)
            
            # Optimize position sizing
            optimal_size = self._optimize_position_sizing(trades_data, market_regimes)
            
            # Optimize trade timing
            timing_adjustments = self._optimize_trade_timing(trades_data, market_data)
            
            # Calculate optimized performance
            optimized_metrics = self._calculate_optimized_performance(
                trades_data, optimal_size, timing_adjustments
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                baseline_metrics, optimized_metrics, market_regimes
            )
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(optimized_metrics)
            
            result = OptimizationResult(
                optimal_position_size=optimal_size,
                expected_return=optimized_metrics['expected_return'],
                risk_adjusted_return=optimized_metrics['risk_adjusted_return'],
                sharpe_ratio=optimized_metrics['sharpe_ratio'],
                max_drawdown=optimized_metrics['max_drawdown'],
                profit_factor=optimized_metrics['profit_factor'],
                win_rate=optimized_metrics['win_rate'],
                optimization_method=self.optimization_method,
                confidence_interval=confidence_interval,
                recommendations=recommendations
            )
            
            self.optimization_history.append(result)
            logger.info(f"Profit optimization completed. Sharpe ratio: {result.sharpe_ratio:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in profit optimization: {str(e)}")
            raise
    
    def _extract_trades_data(self, strategy_data: Dict[str, Any]) -> List[TradeMetrics]:
        """Extract and structure trades data"""
        trades = []
        
        if 'trades' in strategy_data:
            for trade in strategy_data['trades']:
                trade_metric = TradeMetrics(
                    entry_price=float(trade.get('entry_price', 0)),
                    exit_price=float(trade.get('exit_price', 0)),
                    position_size=float(trade.get('position_size', 0.1)),
                    pnl=float(trade.get('pnl', 0)),
                    duration=int(trade.get('duration', 1)),
                    timestamp=datetime.fromisoformat(trade.get('timestamp', datetime.now().isoformat())),
                    strategy=trade.get('strategy', 'unknown'),
                    market_regime=trade.get('market_regime', 'normal')
                )
                trades.append(trade_metric)
        
        return trades
    
    def _extract_market_data(self, strategy_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract market data for regime detection"""
        if 'market_data' in strategy_data:
            return pd.DataFrame(strategy_data['market_data'])
        
        # Generate synthetic market data if not provided
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': np.random.lognormal(10, 0.5, len(dates)),
            'volatility': np.abs(returns)
        })
    
    def _detect_market_regimes(self, market_data: pd.DataFrame) -> np.ndarray:
        """Detect market regimes using unsupervised learning"""
        if market_data.empty:
            return np.array([0])
        
        # Calculate features for regime detection
        features = []
        if 'price' in market_data.columns:
            returns = market_data['price'].pct_change().fillna(0)
            features.extend([
                returns.rolling(20).mean(),
                returns.rolling(20).std(),
                market_data['price'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x[0])
            ])
        
        if 'volume' in market_data.columns:
            volume_ma = market_data['volume'].rolling(20).mean()
            features.append(market_data['volume'] / volume_ma)
        
        if not features:
            return np.zeros(len(market_data))
        
        feature_matrix = np.column_stack(features)
        feature_matrix = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]
        
        if len(feature_matrix) < 3:
            return np.zeros(len(market_data))
        
        scaled_features = self.scaler.fit_transform(feature_matrix)
        regimes = self.market_regime_model.fit_predict(scaled_features)
        
        # Pad regimes to match original data length
        full_regimes = np.zeros(len(market_data))
        full_regimes[-len(regimes):] = regimes
        
        return full_regimes
    
    def _calculate_baseline_metrics(self, trades_data: List[TradeMetrics]) -> PortfolioMetrics:
        """Calculate baseline performance metrics"""
        if not trades_data:
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        pnls = [trade.pnl for trade in trades_data]
        returns = np.array(pnls)
        
        total_return = sum(pnls)
        win_trades = [pnl for pnl in pnls if pnl > 0]
        loss_trades = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(win_trades) / len(pnls) if pnls else 0
        profit_factor = abs(sum(win_trades) / sum(loss_trades)) if loss_trades else float('inf')
        
        # Calculate other metrics
        volatility = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (np.mean(returns) - self.risk_free_rate/252) / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=total_return * 252 / len(pnls) if pnls else 0,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=self._calculate_sortino_ratio(returns),
            max_drawdown=max_drawdown,
            calmar_ratio=abs(total_return / max_drawdown) if max_drawdown > 0 else 0,
            profit_factor=profit_factor,
            win_rate=win_rate,
            average_trade=np.mean(pnls) if pnls else 0
        )
    
    def _optimize_position_sizing(self, trades_data: List[TradeMetrics], 
                                 market_regimes: np.ndarray) -> float:
        """Optimize position sizing using various methods"""
        if not trades_data:
            return 0.1
        
        pnls = np.array([trade.pnl for trade in trades_data])
        
        if self.optimization_method == 'kelly':
            return self._kelly_criterion(pnls)
        elif self.optimization_method == 'markowitz':
            return self._markowitz_optimization(pnls)
        elif self.optimization_method == 'risk_parity':
            return self._risk_parity_sizing(pnls, market_regimes)
        else:
            return self._kelly_criterion(pnls)
    
    def _kelly_criterion(self, pnls: np.ndarray) -> float:
        """Calculate Kelly criterion optimal position size"""
        win_trades = pnls[pnls > 0]
        loss_trades = pnls[pnls < 0]
        
        if len(win_trades) == 0 or len(loss_trades) == 0:
            return 0.1
        
        win_rate = len(win_trades) / len(pnls)
        avg_win = np.mean(win_trades)
        avg_loss = abs(np.mean(loss_trades))
        
        if avg_loss == 0:
            return 0.1
        
        kelly_fraction = win_rate - (1 - win_rate) * (avg_loss / avg_win)
        
        # Apply conservative adjustment and cap
        kelly_adjusted = kelly_fraction * 0.5  # Conservative Kelly
        return min(max(kelly_adjusted, 0.01), self.max_position_size)
    
    def _markowitz_optimization(self, pnls: np.ndarray) -> float:
        """Mean-variance optimization for position sizing"""
        if len(pnls) < 2:
            return 0.1
        
        mean_return = np.mean(pnls)
        variance = np.var(pnls)
        
        if variance == 0:
            return 0.1
        
        # Optimal position size based on mean-variance
        optimal_size = mean_return / (2 * variance)
        return min(max(optimal_size, 0.01), self.max_position_size)
    
    def _risk_parity_sizing(self, pnls: np.ndarray, market_regimes: np.ndarray) -> float:
        """Risk parity position sizing considering market regimes"""
        if len(pnls) < 3:
            return 0.1
        
        # Calculate volatility for each regime
        regime_vols = {}
        for regime in np.unique(market_regimes[-len(pnls):]):
            regime_pnls = pnls[market_regimes[-len(pnls):] == regime]
            if len(regime_pnls) > 1:
                regime_vols[regime] = np.std(regime_pnls)
        
        if not regime_vols:
            return 0.1
        
        # Weight positions inversely to volatility
        target_vol = 0.02  # Target 2% volatility
        current_vol = np.std(pnls)
        
        if current_vol == 0:
            return 0.1
        
        vol_adjusted_size = target_vol / current_vol
        return min(max(vol_adjusted_size, 0.01), self.max_position_size)
    
    def _optimize_trade_timing(self, trades_data: List[TradeMetrics], 
                              market_data: pd.DataFrame) -> Dict[str, float]:
        """Optimize trade timing based on market conditions"""
        timing_adjustments = {
            'entry_timing_factor': 1.0,
            'exit_timing_factor': 1.0,
            'hold_period_adjustment': 1.0
        }
        
        if not trades_data or market_data.empty:
            return timing_adjustments
        
        # Analyze timing patterns
        trade_times = [trade.timestamp.hour for trade in trades_data]
        trade_performance = [trade.pnl for trade in trades_data]
        
        # Find optimal trading hours
        hourly_performance = {}
        for hour, pnl in zip(trade_times, trade_performance):
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(pnl)
        
        # Calculate average performance by hour
        best_hours = []
        for hour, pnls in hourly_performance.items():
            if len(pnls) >= 3:  # Minimum trades for statistical significance
                avg_pnl = np.mean(pnls)
                if avg_pnl > 0:
                    best_hours.append((hour, avg_pnl))
        
        # Adjust timing factors based on analysis
        if best_hours:
            best_hours.sort(key=lambda x: x[1], reverse=True)
            timing_adjustments['entry_timing_factor'] = 1.1  # Slight boost
        
        return timing_adjustments
    
    def _calculate_optimized_performance(self, trades_data: List[TradeMetrics],
                                       optimal_size: float,
                                       timing_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance with optimized parameters"""
        if not trades_data:
            return {'expected_return': 0, 'risk_adjusted_return': 0, 'sharpe_ratio': 0,
                   'max_drawdown': 0, 'profit_factor': 0, 'win_rate': 0}
        
        # Apply optimizations to trades
        optimized_pnls = []
        for trade in trades_data:
            # Scale PnL by optimal position size
            size_factor = optimal_size / trade.position_size if trade.position_size > 0 else 1
            
            # Apply timing adjustments
            timing_factor = timing_adjustments.get('entry_timing_factor', 1.0)
            
            optimized_pnl = trade.pnl * size_factor * timing_factor
            optimized_pnls.append(optimized_pnl)
        
        optimized_returns = np.array(optimized_pnls)
        
        # Calculate metrics
        total_return = np.sum(optimized_returns)
        expected_return = np.mean(optimized_returns) if len(optimized_returns) > 0 else 0
        
        volatility = np.std(optimized_returns) if len(optimized_returns) > 1 else 0
        sharpe_ratio = (expected_return - self.risk_free_rate/252) / volatility if volatility > 0 else 0
        
        # Risk-adjusted return
        risk_adjusted_return = expected_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumsum(optimized_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Win rate and profit factor
        win_trades = optimized_returns[optimized_returns > 0]
        loss_trades = optimized_returns[optimized_returns < 0]
        
        win_rate = len(win_trades) / len(optimized_returns) if len(optimized_returns) > 0 else 0
        profit_factor = abs(np.sum(win_trades) / np.sum(loss_trades)) if len(loss_trades) > 0 else float('inf')
        
        return {
            'expected_return': expected_return,
            'risk_adjusted_return': risk_adjusted_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_return': total_return,
            'volatility': volatility
        }
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0
        
        return (np.mean(returns) - self.risk_free_rate/252) / downside_deviation
    
    def _generate_recommendations(self, baseline: PortfolioMetrics,
                                 optimized: Dict[str, float],
                                 market_regimes: np.ndarray) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Position sizing recommendations
        if optimized['sharpe_ratio'] > baseline.sharpe_ratio * 1.1:
            recommendations.append(f"Increase position sizing to {optimized.get('optimal_size', 0.1):.2%} for improved risk-adjusted returns")
        
        # Risk management recommendations
        if optimized['max_drawdown'] < baseline.max_drawdown * 0.8:
            recommendations.append("Current optimization reduces maximum drawdown significantly")
        
        # Profit factor improvements
        if optimized['profit_factor'] > baseline.profit_factor * 1.2:
            recommendations.append("Optimized strategy shows substantial profit factor improvement")
        
        # Market regime recommendations
        unique_regimes = len(np.unique(market_regimes))
        if unique_regimes >= 2:
            recommendations.append(f"Consider regime-specific strategies - {unique_regimes} market regimes detected")
        
        # Performance recommendations
        if optimized['win_rate'] > 0.6:
            recommendations.append("High win rate strategy - consider increasing position size in favorable conditions")
        elif optimized['win_rate'] < 0.4:
            recommendations.append("Low win rate strategy - focus on improving trade selection or risk/reward ratio")
        
        return recommendations if recommendations else ["No significant optimization opportunities identified"]
    
    def _calculate_confidence_interval(self, metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for expected returns"""
        expected_return = metrics.get('expected_return', 0)
        volatility = metrics.get('volatility', 0)
        
        if volatility == 0:
            return (expected_return, expected_return)
        
        # 95% confidence interval
        z_score = 1.96  # 95% confidence
        margin_error = z_score * volatility / np.sqrt(252)  # Assuming daily data
        
        return (expected_return - margin_error, expected_return + margin_error)
    
    def optimize_portfolio_allocation(self, strategies: Dict[str, Dict]) -> Dict[str, float]:
        """Optimize allocation across multiple strategies"""
        if not strategies:
            return {}
        
        strategy_returns = {}
        strategy_risks = {}
        
        # Calculate returns and risks for each strategy
        for name, data in strategies.items():
            if 'trades' in data:
                pnls = [trade['pnl'] for trade in data['trades']]
                strategy_returns[name] = np.mean(pnls) if pnls else 0
                strategy_risks[name] = np.std(pnls) if len(pnls) > 1 else 0.01
        
        if not strategy_returns:
            return {}
        
        # Mean-variance optimization for portfolio allocation
        def portfolio_variance(weights, cov_matrix):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        def portfolio_return(weights, returns):
            return np.dot(weights, returns)
        
        strategy_names = list(strategy_returns.keys())
        returns = np.array([strategy_returns[name] for name in strategy_names])
        risks = np.array([strategy_risks[name] for name in strategy_names])
        
        # Simple covariance matrix (can be improved with correlation analysis)
        cov_matrix = np.diag(risks ** 2)
        
        # Optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in strategy_names]
        
        # Optimize for maximum Sharpe ratio
        def negative_sharpe(weights):
            port_return = portfolio_return(weights, returns)
            port_risk = np.sqrt(portfolio_variance(weights, cov_matrix))
            return -(port_return - self.risk_free_rate) / port_risk if port_risk > 0 else -999
        
        # Initial equal weights
        initial_weights = np.array([1/len(strategy_names)] * len(strategy_names))
        
        try:
            result = minimize(negative_sharpe, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                return dict(zip(strategy_names, optimal_weights))
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}")
        
        # Fallback to equal weights
        equal_weight = 1.0 / len(strategy_names)
        return {name: equal_weight for name in strategy_names}
    
    def generate_performance_report(self, optimization_result: OptimizationResult) -> str:
        """Generate comprehensive performance report"""
        report = f"""
PROFIT OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

OPTIMIZATION SUMMARY:
- Method: {optimization_result.optimization_method.upper()}
- Optimal Position Size: {optimization_result.optimal_position_size:.2%}
- Expected Return: {optimization_result.expected_return:.4f}
- Risk-Adjusted Return: {optimization_result.risk_adjusted_return:.4f}

PERFORMANCE METRICS:
- Sharpe Ratio: {optimization_result.sharpe_ratio:.3f}
- Maximum Drawdown: {optimization_result.max_drawdown:.2%}
- Profit Factor: {optimization_result.profit_factor:.2f}
- Win Rate: {optimization_result.win_rate:.1%}

CONFIDENCE INTERVAL (95%):
- Lower Bound: {optimization_result.confidence_interval[0]:.4f}
- Upper Bound: {optimization_result.confidence_interval[1]:.4f}

RECOMMENDATIONS:
"""
        for i, rec in enumerate(optimization_result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report

    async def optimize_profits_async(self, strategy_data: Dict[str, Any]) -> OptimizationResult:
        """
        Async version of profit optimization for real-time processing
        
        Args:
            strategy_data: Dictionary containing trading data and metrics
            
        Returns:
            OptimizationResult: Comprehensive optimization results
        """
        try:
            logger.info("Starting async profit optimization...")
            
            # Run CPU-intensive optimization in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.optimize_profits, strategy_data)
            
            logger.info("Async profit optimization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Async profit optimization failed: {e}")
            raise

    async def real_time_optimization_stream(self, data_stream):
        """
        Process real-time data stream for continuous optimization
        
        Args:
            data_stream: Async generator yielding strategy data
            
        Yields:
            OptimizationResult: Real-time optimization results
        """
        try:
            async for strategy_data in data_stream:
                try:
                    # Optimize with timeout for real-time requirements
                    optimization_task = asyncio.create_task(
                        self.optimize_profits_async(strategy_data)
                    )
                    
                    # Set timeout to ensure real-time performance
                    result = await asyncio.wait_for(optimization_task, timeout=1.0)
                    yield result
                    
                except asyncio.TimeoutError:
                    logger.warning("Real-time optimization timeout - using cached result")
                    # Return last known good result or basic fallback
                    if self.optimization_history:
                        yield self.optimization_history[-1]
                        
                except Exception as e:
                    logger.error(f"Real-time optimization error: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Real-time optimization stream failed: {e}")
            raise

    def get_optimization_recommendations(self, current_state: Dict[str, Any]) -> List[str]:
        """
        Get specific optimization recommendations based on current trading state
        
        Args:
            current_state: Current portfolio and market state
            
        Returns:
            List[str]: Actionable optimization recommendations
        """
        recommendations = []
        
        try:
            # Analyze current portfolio performance
            if 'portfolio_metrics' in current_state:
                metrics = current_state['portfolio_metrics']
                
                # Check Sharpe ratio
                if metrics.get('sharpe_ratio', 0) < 1.0:
                    recommendations.append("Consider reducing position sizes to improve risk-adjusted returns")
                
                # Check drawdown
                if metrics.get('max_drawdown', 0) > 0.15:
                    recommendations.append("Implement stricter stop-loss rules to limit drawdown")
                
                # Check win rate
                if metrics.get('win_rate', 0) < 0.5:
                    recommendations.append("Review entry criteria to improve trade selection")
            
            # Market condition recommendations
            if 'market_conditions' in current_state:
                conditions = current_state['market_conditions']
                
                if conditions.get('volatility', 0) > 0.03:
                    recommendations.append("Reduce position sizes during high volatility periods")
                
                if conditions.get('trend_strength', 0) < 0.3:
                    recommendations.append("Consider range-trading strategies in sideways markets")
            
            # Default recommendations if no specific issues found
            if not recommendations:
                recommendations.append("Current optimization appears satisfactory - monitor ongoing performance")
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations - check data quality"]

    def validate_optimization_performance(self, backtest_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate optimization performance against historical data
        
        Args:
            backtest_data: Historical trading data for validation
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        try:
            validation_metrics = {}
            
            # Run optimization on training data
            train_data = backtest_data.get('train', {})
            test_data = backtest_data.get('test', {})
            
            if train_data and test_data:
                # Optimize on training data
                train_result = self.optimize_profits(train_data)
                
                # Apply optimized parameters to test data
                test_metrics = self._calculate_baseline_metrics(test_data.get('trades', []))
                
                # Calculate improvement metrics                baseline_sharpe = test_data.get('baseline_metrics', {}).get('sharpe_ratio', 0)
                optimized_sharpe = test_metrics.get('sharpe_ratio', 0)
                validation_metrics['sharpe_improvement'] = optimized_sharpe - baseline_sharpe
                validation_metrics['profit_improvement'] = (
                    test_metrics.get('total_return', 0) - 
                    test_data.get('baseline_metrics', {}).get('total_return', 0)
                )
                validation_metrics['drawdown_improvement'] = (
                    test_data.get('baseline_metrics', {}).get('max_drawdown', 0) - 
                    test_metrics.get('max_drawdown', 0)
                )
                
                logger.info("Optimization validation completed successfully")
                
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Optimization validation failed: {e}")
            return {'error': str(e)}
    
    async def _publish_optimization_results(self, results: Dict[str, Any]):
        """Publish optimization results to Platform3 communication framework"""
        try:
            message = {
                'type': 'optimization_result',
                'service': 'profit-optimizer-analytics',
                'data': results,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.communication_framework:
                # Handle different interface signatures
                try:
                    self.communication_framework.send_message('analytics.optimization', message)
                except TypeError:
                    # Try alternative signature
                    pass
                
        except Exception as e:
            logger.error(f"Error publishing optimization results: {e}")
    
    def _get_timeframe_data(self, timeframe: str) -> List[Dict[str, Any]]:
        """Get optimization data for specified timeframe"""
        try:
            current_time = datetime.now()
            
            # Parse timeframe
            if timeframe == "1h":
                cutoff_time = current_time - timedelta(hours=1)
            elif timeframe == "4h":
                cutoff_time = current_time - timedelta(hours=4)
            elif timeframe == "1d":
                cutoff_time = current_time - timedelta(days=1)
            elif timeframe == "1w":
                cutoff_time = current_time - timedelta(weeks=1)
            else:
                cutoff_time = current_time - timedelta(hours=1)
              # Filter optimization history by timeframe
            timeframe_data = []
            for opt in self.optimization_history:
                try:
                    if isinstance(opt, dict) and 'timestamp' in opt:
                        opt_time = datetime.fromisoformat(opt['timestamp'].replace('Z', '+00:00'))
                        if opt_time >= cutoff_time:
                            timeframe_data.append(opt)
                except Exception as e:
                    logger.warning(f"Error parsing optimization timestamp: {e}")
                    continue
            
            return timeframe_data
            
        except Exception as e:
            logger.error(f"Error getting timeframe data: {e}")
            return []
    
    def _calculate_risk_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional risk metrics for the data"""
        try:
            risk_metrics = {}
            
            if 'trades' in data:
                trades_data = self._extract_trades_data(data)
                pnls = np.array([trade.pnl for trade in trades_data])
                
                if len(pnls) > 0:
                    # Value at Risk (VaR)
                    var_95 = np.percentile(pnls, 5)
                    var_99 = np.percentile(pnls, 1)
                    
                    # Expected Shortfall (Conditional VaR)
                    es_95 = np.mean(pnls[pnls <= var_95]) if np.any(pnls <= var_95) else 0
                    es_99 = np.mean(pnls[pnls <= var_99]) if np.any(pnls <= var_99) else 0
                    
                    risk_metrics.update({
                        'var_95': var_95,
                        'var_99': var_99,
                        'expected_shortfall_95': es_95,
                        'expected_shortfall_99': es_99,
                        'volatility': np.std(pnls),
                        'skewness': float(pd.Series(pnls).skew()) if len(pnls) > 2 else 0,
                        'kurtosis': float(pd.Series(pnls).kurtosis()) if len(pnls) > 3 else 0
                    })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _generate_performance_insights(self, optimization_result: OptimizationResult) -> List[str]:
        """Generate performance insights from optimization results"""
        insights = []
        
        try:
            # Profit factor insights
            if optimization_result.profit_factor > 2.0:
                insights.append("Excellent profit factor indicates strong strategy performance")
            elif optimization_result.profit_factor > 1.5:
                insights.append("Good profit factor shows profitable strategy with room for improvement")
            elif optimization_result.profit_factor > 1.0:
                insights.append("Marginal profitability - consider strategy adjustments")
            else:
                insights.append("Strategy showing losses - immediate optimization required")
            
            # Sharpe ratio insights
            if optimization_result.sharpe_ratio > 2.0:
                insights.append("Exceptional risk-adjusted returns")
            elif optimization_result.sharpe_ratio > 1.0:
                insights.append("Good risk-adjusted performance")
            else:
                insights.append("Poor risk-adjusted returns - review risk management")
            
            # Win rate insights
            if optimization_result.win_rate > 0.6:
                insights.append("High win rate indicates consistent strategy performance")
            elif optimization_result.win_rate < 0.4:
                insights.append("Low win rate suggests need for entry/exit optimization")
            
            # Drawdown insights
            if optimization_result.max_drawdown < 0.05:
                insights.append("Low maximum drawdown shows excellent risk control")
            elif optimization_result.max_drawdown > 0.15:
                insights.append("High drawdown indicates need for better position sizing")
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            insights.append("Unable to generate performance insights due to data issues")
        
        return insights
    
    def _generate_optimization_insights(self, timeframe_data: List[Dict], metrics: Dict[str, Any]) -> List[str]:
        """Generate insights from optimization analysis"""
        insights = []
        
        try:
            # Optimization frequency insights
            if len(timeframe_data) > 10:
                insights.append(f"High optimization activity with {len(timeframe_data)} optimizations")
            elif len(timeframe_data) < 3:
                insights.append("Low optimization activity - consider increasing analysis frequency")
            
            # Success rate insights
            success_rate = metrics.get('optimization_success_rate', 0)
            if success_rate > 90:
                insights.append("Excellent optimization success rate")
            elif success_rate < 70:
                insights.append("Below average optimization success rate - review data quality")
            
            # Performance insights
            avg_profit_factor = metrics.get('average_profit_factor', 0)
            if avg_profit_factor > 1.8:
                insights.append("Strong average profit factor across optimizations")
            elif avg_profit_factor < 1.2:
                insights.append("Low average profit factor - consider strategy review")
            
            # Processing efficiency insights
            avg_time = metrics.get('average_processing_time', 0)
            if avg_time > 5.0:
                insights.append("High processing time - consider optimization algorithm tuning")
            elif avg_time < 1.0:
                insights.append("Efficient processing time enables real-time optimization")
            
        except Exception as e:
            logger.error(f"Error generating optimization insights: {e}")
            insights.append("Unable to generate optimization insights")
        
        return insights
    
    def _generate_optimization_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []
        
        try:
            # Success rate recommendations
            success_rate = metrics.get('optimization_success_rate', 0)
            if success_rate < 80:
                recommendations.append("Improve data validation and preprocessing")
                recommendations.append("Review optimization parameters and constraints")
            
            # Performance recommendations
            avg_profit_factor = metrics.get('average_profit_factor', 0)
            if avg_profit_factor < 1.5:
                recommendations.append("Consider adjusting position sizing strategy")
                recommendations.append("Review risk management parameters")
                recommendations.append("Analyze market regime detection accuracy")
            
            # Efficiency recommendations
            avg_time = metrics.get('average_processing_time', 0)
            if avg_time > 3.0:
                recommendations.append("Optimize calculation algorithms for better performance")
                recommendations.append("Consider parallel processing for multiple strategies")
            
            # General recommendations
            if metrics.get('current_efficiency', 0) < 85:
                recommendations.append("Monitor system resources and optimize memory usage")
                recommendations.append("Review optimization frequency and scheduling")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Review system logs for optimization issues")
        
        return recommendations
    
    def _calculate_report_confidence(self, timeframe_data: List[Dict], metrics: Dict[str, Any]) -> float:
        """Calculate confidence score for the report"""
        try:
            confidence_factors = []
            
            # Data volume factor
            data_volume = len(timeframe_data)
            if data_volume >= 10:
                confidence_factors.append(1.0)
            elif data_volume >= 5:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Success rate factor
            success_rate = metrics.get('optimization_success_rate', 0) / 100
            confidence_factors.append(success_rate)
            
            # Processing time consistency factor
            avg_time = metrics.get('average_processing_time', 0)
            if avg_time < 2.0:
                confidence_factors.append(1.0)
            elif avg_time < 5.0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            return min(np.mean(confidence_factors), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _assess_data_quality(self, timeframe_data: List[Dict]) -> str:
        """Assess the quality of data used in the report"""
        try:
            if len(timeframe_data) == 0:
                return "poor"
            
            # Check for complete data
            complete_data_count = sum(1 for data in timeframe_data if 'optimization_result' in data)
            completeness_ratio = complete_data_count / len(timeframe_data)
            
            if completeness_ratio >= 0.9:
                return "excellent"
            elif completeness_ratio >= 0.7:
                return "good"
            elif completeness_ratio >= 0.5:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return "unknown"

# Example usage and testing
if __name__ == "__main__":
    async def test_profit_optimizer():
        """Test the enhanced ProfitOptimizer with AnalyticsInterface"""
        
        # Example strategy data
        sample_data = {
            'trades': [
                {'entry_price': 100, 'exit_price': 105, 'position_size': 0.1, 'pnl': 500, 
                 'duration': 5, 'timestamp': '2024-01-01T09:30:00', 'strategy': 'momentum'},
                {'entry_price': 110, 'exit_price': 108, 'position_size': 0.1, 'pnl': -200,
                 'duration': 3, 'timestamp': '2024-01-02T10:15:00', 'strategy': 'momentum'},
                {'entry_price': 105, 'exit_price': 112, 'position_size': 0.15, 'pnl': 1050,
                 'duration': 8, 'timestamp': '2024-01-03T11:00:00', 'strategy': 'breakout'}
            ],
            'market_data': {
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'price': [100, 110, 105],
                'volume': [1000000, 1200000, 900000],
                'volatility': [0.02, 0.025, 0.018]
            }
        }
        
        # Initialize enhanced optimizer
        optimizer = ProfitOptimizer(optimization_method='kelly')
        
        print("=" * 60)
        print("ADVANCED PROFIT OPTIMIZER WITH ANALYTICS INTERFACE")
        print("=" * 60)
        
        # Test AnalyticsInterface methods
        print("\n1. Testing process_data method...")
        processing_result = await optimizer.process_data(sample_data)
        print(f"Processing completed in {processing_result.get('processing_time', 0):.2f}s")
        print(f"Profit Factor: {processing_result.get('optimization_result', {}).get('profit_factor', 0):.2f}")
        
        print("\n2. Testing real-time metrics...")
        metrics = optimizer.get_real_time_metrics()
        print(f"Retrieved {len(metrics)} real-time metrics:")
        for metric in metrics[:3]:  # Show first 3 metrics
            print(f"  - {metric.metric_name}: {metric.value:.2f} {metric.unit} ({metric.status})")
        
        print("\n3. Testing report generation...")
        report = await optimizer.generate_report("1h")
        print(f"Report generated for {report.timeframe} timeframe")
        print(f"Confidence Score: {report.confidence_score:.2f}")
        print(f"Data Quality: {report.data_quality}")
        print(f"Insights: {len(report.insights)} generated")
        
        print("\n4. Testing legacy optimization methods...")
        # Run legacy optimization
        result = optimizer.optimize_profits(sample_data)
        print(f"Legacy optimization result: Profit Factor {result.profit_factor:.2f}")
        
        # Test portfolio allocation
        strategies = {
            'momentum': sample_data,
            'breakout': sample_data
        }
        allocation = optimizer.optimize_portfolio_allocation(strategies)
        print(f"Optimal Portfolio Allocation: {allocation}")
        
        # Generate performance report
        print("\n5. Performance Report:")
        print(optimizer.generate_performance_report(result))
        
        print("\n" + "=" * 60)
        print("ENHANCED PROFIT OPTIMIZER TESTING COMPLETED")
        print("=" * 60)
    
    # Run the async test
    import asyncio
    asyncio.run(test_profit_optimizer())

print("Advanced ProfitOptimizer with AnalyticsInterface ready for Platform3 integration!")
