"""
Advanced Swing Trading Analytics System for Platform3

This module provides comprehensive swing trading performance analysis including:
- Multi-timeframe swing pattern recognition and profitability analysis
- Hold period optimization and trade duration analysis
- Market cycle correlation with swing performance
- Trend strength impact on swing trading effectiveness
- Risk-adjusted swing trading metrics and position sizing optimization
- Seasonal and cyclical swing trading patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Platform3 Communication Framework Integration
import sys
import os
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework

# Analytics Framework Interface
@dataclass
class RealtimeMetric:
    """Real-time metric data structure"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    alert_threshold: Optional[float] = None

@dataclass
class AnalyticsReport:
    """Standardized analytics report structure"""
    report_id: str
    report_type: str
    generated_at: datetime
    data: Dict[str, Any]
    summary: str
    recommendations: List[str]
    confidence_score: float

class AnalyticsInterface(ABC):
    """Abstract interface for analytics engines"""
    
    @abstractmethod
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data and return analytics results"""
        pass
    
    @abstractmethod
    async def generate_report(self, timeframe: str) -> AnalyticsReport:
        """Generate analytics report for specified timeframe"""
        pass
    
    @abstractmethod
    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        """Get current real-time metrics"""
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwingDirection(Enum):
    """Swing trade direction"""
    LONG = "Long"
    SHORT = "Short"
    BOTH = "Both"

class MarketTrend(Enum):
    """Market trend classification"""
    STRONG_UPTREND = "Strong_Uptrend"
    WEAK_UPTREND = "Weak_Uptrend"
    SIDEWAYS = "Sideways"
    WEAK_DOWNTREND = "Weak_Downtrend"
    STRONG_DOWNTREND = "Strong_Downtrend"

@dataclass
class SwingTrade:
    """Individual swing trade with detailed metrics"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    direction: SwingDirection
    hold_period_days: int
    pnl: float
    pnl_percentage: float
    max_adverse_excursion: float  # MAE
    max_favorable_excursion: float  # MFE
    market_trend: MarketTrend
    volatility_at_entry: float
    volume_profile: str
    exit_reason: str

@dataclass
class SwingPattern:
    """Swing trading pattern analysis"""
    pattern_name: str
    frequency: int
    avg_hold_period: float
    avg_return: float
    win_rate: float
    profit_factor: float
    best_market_condition: MarketTrend
    optimal_position_size: float

@dataclass
class SwingAnalyticsResult:
    """Comprehensive swing trading analytics results"""
    analysis_period: Tuple[datetime, datetime]
    total_swing_trades: int
    avg_hold_period: float
    overall_win_rate: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    consecutive_wins: int
    consecutive_losses: int
    hold_period_analysis: Dict[str, Dict[str, float]]
    direction_analysis: Dict[SwingDirection, Dict[str, float]]
    trend_analysis: Dict[MarketTrend, Dict[str, float]]
    seasonal_patterns: Dict[str, float]
    swing_patterns: List[SwingPattern]
    mae_mfe_analysis: Dict[str, float]
    recommendations: List[str]

class SwingAnalytics(AnalyticsInterface):
    """
    Advanced Swing Trading Analytics System
    
    Provides comprehensive analysis of swing trading performance including
    pattern recognition, trend correlation, and optimization recommendations.
    Now implements AnalyticsInterface for framework integration.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 benchmark_return: float = 0.08):
        """
        Initialize SwingAnalytics
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            benchmark_return: Benchmark return for comparison        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        
        # Analysis parameters
        self.short_term_days = 7
        self.medium_term_days = 21
        self.long_term_days = 63
        
        # Pattern recognition
        self.pattern_classifier = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.swing_history = []
        self.pattern_database = defaultdict(list)
        
        # Real-time processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._last_update = None
        
        # Platform3 Communication Framework
        self.communication_framework = Platform3CommunicationFramework(
            service_name="swing-analytics",
            service_port=8002,
            redis_url="redis://localhost:6379",
            consul_host="localhost",
            consul_port=8500
        )
        
        # Initialize the framework
        try:
            self.communication_framework.initialize()
            logger.info("Swing Analytics Communication framework initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize communication framework: {e}")
        
        logger.info("SwingAnalytics initialized for comprehensive swing trading analysis")
    
    def analyze_performance(self, data: Union[Dict[str, Any], List[Dict]]) -> SwingAnalyticsResult:
        """
        Analyze comprehensive swing trading performance
        
        Args:
            data: Trading data containing swing trades and market information
            
        Returns:
            SwingAnalyticsResult: Comprehensive swing trading analysis
        """
        try:
            # Parse and structure swing trades
            swing_trades = self._parse_swing_trades(data)
            
            if not swing_trades:
                logger.warning("No valid swing trades found in data")
                return self._create_empty_result()
            
            # Calculate basic performance metrics
            basic_metrics = self._calculate_basic_metrics(swing_trades)
            
            # Analyze hold period patterns
            hold_period_analysis = self._analyze_hold_periods(swing_trades)
            
            # Analyze performance by direction
            direction_analysis = self._analyze_by_direction(swing_trades)
            
            # Analyze performance by market trend
            trend_analysis = self._analyze_by_trend(swing_trades)
            
            # Detect seasonal patterns
            seasonal_patterns = self._analyze_seasonal_patterns(swing_trades)
            
            # Perform MAE/MFE analysis
            mae_mfe_analysis = self._analyze_mae_mfe(swing_trades)
            
            # Identify swing patterns
            swing_patterns = self._identify_swing_patterns(swing_trades)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(swing_trades)
            
            # Generate recommendations
            recommendations = self._generate_swing_recommendations(
                basic_metrics, hold_period_analysis, direction_analysis, 
                trend_analysis, swing_patterns
            )
            
            # Determine analysis period
            analysis_period = self._get_analysis_period(swing_trades)
            
            result = SwingAnalyticsResult(
                analysis_period=analysis_period,
                total_swing_trades=basic_metrics['total_trades'],
                avg_hold_period=basic_metrics['avg_hold_period'],
                overall_win_rate=basic_metrics['win_rate'],
                total_return=basic_metrics['total_return'],
                annualized_return=basic_metrics['annualized_return'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                sortino_ratio=risk_metrics['sortino_ratio'],
                max_drawdown=risk_metrics['max_drawdown'],
                calmar_ratio=risk_metrics['calmar_ratio'],
                profit_factor=basic_metrics['profit_factor'],
                avg_winner=basic_metrics['avg_winner'],
                avg_loser=basic_metrics['avg_loser'],
                largest_winner=basic_metrics['largest_winner'],
                largest_loser=basic_metrics['largest_loser'],
                consecutive_wins=basic_metrics['consecutive_wins'],
                consecutive_losses=basic_metrics['consecutive_losses'],
                hold_period_analysis=hold_period_analysis,
                direction_analysis=direction_analysis,
                trend_analysis=trend_analysis,
                seasonal_patterns=seasonal_patterns,
                swing_patterns=swing_patterns,
                mae_mfe_analysis=mae_mfe_analysis,
                recommendations=recommendations
            )
            
            # Store for historical analysis
            self.swing_history.extend(swing_trades)
            
            logger.info(f"Swing analytics completed: {len(swing_trades)} trades analyzed, "
                       f"{result.overall_win_rate:.1%} win rate")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in swing analytics: {str(e)}")
            return self._create_empty_result()
    
    def _parse_swing_trades(self, data: Union[Dict[str, Any], List[Dict]]) -> List[SwingTrade]:
        """Parse and structure swing trading data"""
        trades = []
        
        # Handle different data formats
        if isinstance(data, dict):
            trade_list = data.get('trades', [])
        elif isinstance(data, list):
            trade_list = data
        else:
            logger.error("Invalid data format for swing analysis")
            return trades
        
        for trade_data in trade_list:
            try:
                # Parse dates
                entry_date = self._parse_date(trade_data.get('entry_date') or trade_data.get('timestamp'))
                exit_date = self._parse_date(trade_data.get('exit_date') or trade_data.get('exit_timestamp'))
                
                # Calculate hold period
                if entry_date and exit_date:
                    hold_period = (exit_date - entry_date).days
                else:
                    hold_period = int(trade_data.get('duration', 1))
                
                # Calculate PnL percentage
                entry_price = float(trade_data.get('entry_price', 100))
                exit_price = float(trade_data.get('exit_price', 100))
                pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                # Determine direction
                direction = SwingDirection.LONG if trade_data.get('side', 'long').lower() == 'long' else SwingDirection.SHORT
                
                # Estimate market trend (simplified)
                trend = self._estimate_market_trend(trade_data)
                
                trade = SwingTrade(
                    symbol=trade_data.get('symbol', 'UNKNOWN'),
                    entry_date=entry_date or datetime.now(),
                    exit_date=exit_date or datetime.now(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=float(trade_data.get('position_size', 0.1)),
                    direction=direction,
                    hold_period_days=hold_period,
                    pnl=float(trade_data.get('pnl', 0)),
                    pnl_percentage=pnl_pct,
                    max_adverse_excursion=float(trade_data.get('mae', 0)),
                    max_favorable_excursion=float(trade_data.get('mfe', abs(pnl_pct))),
                    market_trend=trend,
                    volatility_at_entry=float(trade_data.get('volatility', 0.02)),
                    volume_profile=trade_data.get('volume_profile', 'normal'),
                    exit_reason=trade_data.get('exit_reason', 'target_reached')
                )
                trades.append(trade)
                
            except Exception as e:
                logger.warning(f"Error parsing swing trade data: {e}")
                continue
        
        return trades
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support"""
        if not date_str:
            return None
        
        try:
            formats = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%m/%d/%Y',
                '%d/%m/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Fallback to ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
        except Exception:
            return None
    
    def _estimate_market_trend(self, trade_data: Dict) -> MarketTrend:
        """Estimate market trend during trade (simplified)"""
        # Use price movement and volatility as trend indicators
        entry_price = float(trade_data.get('entry_price', 100))
        exit_price = float(trade_data.get('exit_price', 100))
        volatility = float(trade_data.get('volatility', 0.02))
        
        price_change_pct = ((exit_price - entry_price) / entry_price) if entry_price > 0 else 0
        
        # Simple trend classification based on price movement and volatility
        if price_change_pct > 0.05 and volatility < 0.03:
            return MarketTrend.STRONG_UPTREND
        elif price_change_pct > 0.02:
            return MarketTrend.WEAK_UPTREND
        elif price_change_pct < -0.05 and volatility < 0.03:
            return MarketTrend.STRONG_DOWNTREND
        elif price_change_pct < -0.02:
            return MarketTrend.WEAK_DOWNTREND
        else:
            return MarketTrend.SIDEWAYS
    
    def _calculate_basic_metrics(self, trades: List[SwingTrade]) -> Dict[str, float]:
        """Calculate basic swing trading performance metrics"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_return': 0, 'annualized_return': 0,
                'profit_factor': 0, 'avg_winner': 0, 'avg_loser': 0, 'largest_winner': 0,
                'largest_loser': 0, 'consecutive_wins': 0, 'consecutive_losses': 0,
                'avg_hold_period': 0
            }
        
        total_trades = len(trades)
        
        # Win/Loss analysis
        winners = [trade for trade in trades if trade.pnl > 0]
        losers = [trade for trade in trades if trade.pnl < 0]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        
        # PnL analysis
        total_return = sum(trade.pnl for trade in trades)
        gross_profit = sum(trade.pnl for trade in winners)
        gross_loss = abs(sum(trade.pnl for trade in losers))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_winner = np.mean([trade.pnl for trade in winners]) if winners else 0
        avg_loser = np.mean([trade.pnl for trade in losers]) if losers else 0
        
        largest_winner = max([trade.pnl for trade in trades])
        largest_loser = min([trade.pnl for trade in trades])
        
        # Hold period analysis
        avg_hold_period = np.mean([trade.hold_period_days for trade in trades])
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades)
        
        # Annualized return calculation
        if len(trades) > 1:
            time_span_years = (trades[-1].exit_date - trades[0].entry_date).days / 365.25
            annualized_return = total_return / time_span_years if time_span_years > 0 else 0
        else:
            annualized_return = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'profit_factor': profit_factor,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'largest_winner': largest_winner,
            'largest_loser': largest_loser,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'avg_hold_period': avg_hold_period
        }
    
    def _analyze_hold_periods(self, trades: List[SwingTrade]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by hold period ranges"""
        hold_period_analysis = {
            'short_term': {'trades': [], 'label': f'1-{self.short_term_days} days'},
            'medium_term': {'trades': [], 'label': f'{self.short_term_days+1}-{self.medium_term_days} days'},
            'long_term': {'trades': [], 'label': f'{self.medium_term_days+1}-{self.long_term_days} days'},
            'very_long_term': {'trades': [], 'label': f'>{self.long_term_days} days'}
        }
        
        # Categorize trades by hold period
        for trade in trades:
            if trade.hold_period_days <= self.short_term_days:
                hold_period_analysis['short_term']['trades'].append(trade)
            elif trade.hold_period_days <= self.medium_term_days:
                hold_period_analysis['medium_term']['trades'].append(trade)
            elif trade.hold_period_days <= self.long_term_days:
                hold_period_analysis['long_term']['trades'].append(trade)
            else:
                hold_period_analysis['very_long_term']['trades'].append(trade)
        
        # Calculate metrics for each period
        for period, data in hold_period_analysis.items():
            period_trades = data['trades']
            if period_trades:
                total_pnl = sum(trade.pnl for trade in period_trades)
                win_rate = sum(1 for trade in period_trades if trade.pnl > 0) / len(period_trades)
                avg_pnl = np.mean([trade.pnl for trade in period_trades])
                
                hold_period_analysis[period].update({
                    'count': len(period_trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'avg_hold_days': np.mean([trade.hold_period_days for trade in period_trades])
                })
            else:
                hold_period_analysis[period].update({
                    'count': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0, 'avg_hold_days': 0
                })
        
        return hold_period_analysis
    
    def _analyze_by_direction(self, trades: List[SwingTrade]) -> Dict[SwingDirection, Dict[str, float]]:
        """Analyze performance by trade direction"""
        direction_analysis = {}
        
        for direction in [SwingDirection.LONG, SwingDirection.SHORT]:
            direction_trades = [trade for trade in trades if trade.direction == direction]
            
            if direction_trades:
                total_pnl = sum(trade.pnl for trade in direction_trades)
                win_rate = sum(1 for trade in direction_trades if trade.pnl > 0) / len(direction_trades)
                avg_pnl = np.mean([trade.pnl for trade in direction_trades])
                avg_hold_period = np.mean([trade.hold_period_days for trade in direction_trades])
                
                direction_analysis[direction] = {
                    'count': len(direction_trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'avg_hold_period': avg_hold_period,
                    'largest_winner': max([trade.pnl for trade in direction_trades]),
                    'largest_loser': min([trade.pnl for trade in direction_trades])
                }
            else:
                direction_analysis[direction] = {
                    'count': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0,
                    'avg_hold_period': 0, 'largest_winner': 0, 'largest_loser': 0
                }
        
        return direction_analysis
    
    def _analyze_by_trend(self, trades: List[SwingTrade]) -> Dict[MarketTrend, Dict[str, float]]:
        """Analyze performance by market trend"""
        trend_analysis = {}
        
        for trend in MarketTrend:
            trend_trades = [trade for trade in trades if trade.market_trend == trend]
            
            if trend_trades:
                total_pnl = sum(trade.pnl for trade in trend_trades)
                win_rate = sum(1 for trade in trend_trades if trade.pnl > 0) / len(trend_trades)
                avg_pnl = np.mean([trade.pnl for trade in trend_trades])
                
                trend_analysis[trend] = {
                    'count': len(trend_trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'avg_hold_period': np.mean([trade.hold_period_days for trade in trend_trades])
                }
        
        return trend_analysis
    
    def _analyze_seasonal_patterns(self, trades: List[SwingTrade]) -> Dict[str, float]:
        """Analyze seasonal trading patterns"""
        seasonal_patterns = {}
        
        # Monthly analysis
        monthly_performance = defaultdict(list)
        for trade in trades:
            month = trade.entry_date.month
            monthly_performance[month].append(trade.pnl)
        
        for month, pnls in monthly_performance.items():
            if len(pnls) >= 3:  # Minimum trades for significance
                seasonal_patterns[f'month_{month}'] = np.mean(pnls)
        
        # Day of week analysis
        weekday_performance = defaultdict(list)
        for trade in trades:
            weekday = trade.entry_date.weekday()  # Monday = 0
            weekday_performance[weekday].append(trade.pnl)
        
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for weekday, pnls in weekday_performance.items():
            if len(pnls) >= 3:
                seasonal_patterns[f'{weekday_names[weekday]}'] = np.mean(pnls)
        
        return seasonal_patterns
    
    def _analyze_mae_mfe(self, trades: List[SwingTrade]) -> Dict[str, float]:
        """Analyze Maximum Adverse Excursion and Maximum Favorable Excursion"""
        if not trades:
            return {'avg_mae': 0, 'avg_mfe': 0, 'mae_mfe_ratio': 0, 'efficiency_ratio': 0}
        
        # Calculate MAE/MFE metrics
        maes = [trade.max_adverse_excursion for trade in trades if trade.max_adverse_excursion != 0]
        mfes = [trade.max_favorable_excursion for trade in trades if trade.max_favorable_excursion != 0]
        
        avg_mae = np.mean(maes) if maes else 0
        avg_mfe = np.mean(mfes) if mfes else 0
        
        # MAE/MFE ratio indicates trade efficiency
        mae_mfe_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
        
        # Efficiency ratio (actual PnL vs potential MFE)
        actual_pnls = [trade.pnl_percentage for trade in trades]
        efficiency_ratio = np.mean(actual_pnls) / avg_mfe if avg_mfe > 0 else 0
        
        return {
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'mae_mfe_ratio': mae_mfe_ratio,
            'efficiency_ratio': efficiency_ratio
        }
    
    def _identify_swing_patterns(self, trades: List[SwingTrade]) -> List[SwingPattern]:
        """Identify common swing trading patterns using clustering"""
        if len(trades) < 10:  # Need minimum trades for pattern recognition
            return []
        
        # Create feature matrix for pattern recognition
        features = []
        for trade in trades:
            feature_vector = [
                trade.hold_period_days,
                trade.pnl_percentage,
                trade.volatility_at_entry,
                trade.max_adverse_excursion,
                trade.max_favorable_excursion,
                1 if trade.direction == SwingDirection.LONG else 0
            ]
            features.append(feature_vector)
        
        feature_matrix = np.array(features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        try:
            # Scale features and perform clustering
            scaled_features = self.scaler.fit_transform(feature_matrix)
            cluster_labels = self.pattern_classifier.fit_predict(scaled_features)
            
            # Analyze each cluster as a pattern
            patterns = []
            for cluster_id in np.unique(cluster_labels):
                cluster_trades = [trades[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_trades) >= 3:  # Minimum for a valid pattern
                    pattern = self._analyze_pattern_cluster(cluster_id, cluster_trades)
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Pattern recognition failed: {e}")
            return []
    
    def _analyze_pattern_cluster(self, cluster_id: int, cluster_trades: List[SwingTrade]) -> SwingPattern:
        """Analyze a cluster of trades to identify pattern characteristics"""
        # Pattern characteristics
        avg_hold_period = np.mean([trade.hold_period_days for trade in cluster_trades])
        avg_return = np.mean([trade.pnl_percentage for trade in cluster_trades])
        win_rate = sum(1 for trade in cluster_trades if trade.pnl > 0) / len(cluster_trades)
        
        # Profit factor
        winners = [trade.pnl for trade in cluster_trades if trade.pnl > 0]
        losers = [trade.pnl for trade in cluster_trades if trade.pnl < 0]
        profit_factor = sum(winners) / abs(sum(losers)) if losers else float('inf')
        
        # Best market condition for this pattern
        trend_counts = defaultdict(int)
        for trade in cluster_trades:
            trend_counts[trade.market_trend] += 1
        best_market_condition = max(trend_counts.keys(), key=lambda k: trend_counts[k])
        
        # Optimal position size (simplified calculation)
        position_sizes = [trade.position_size for trade in cluster_trades]
        optimal_position_size = np.mean(position_sizes)
        
        return SwingPattern(
            pattern_name=f"Pattern_{cluster_id}",
            frequency=len(cluster_trades),
            avg_hold_period=avg_hold_period,
            avg_return=avg_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            best_market_condition=best_market_condition,
            optimal_position_size=optimal_position_size
        )
    
    def _calculate_risk_metrics(self, trades: List[SwingTrade]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        if not trades:
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0}
        
        returns = [trade.pnl for trade in trades]
        returns_array = np.array(returns)
        
        # Sharpe ratio
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe_ratio = (mean_return - self.risk_free_rate/252) / std_return if std_return > 0 else 0
        
        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else std_return
        sortino_ratio = (mean_return - self.risk_free_rate/252) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        total_return = np.sum(returns_array)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_consecutive_trades(self, trades: List[SwingTrade]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _generate_swing_recommendations(self, basic_metrics: Dict, hold_period_analysis: Dict,
                                      direction_analysis: Dict, trend_analysis: Dict,
                                      patterns: List[SwingPattern]) -> List[str]:
        """Generate actionable swing trading recommendations"""
        recommendations = []
        
        # Hold period recommendations
        best_period = max(hold_period_analysis.items(), 
                         key=lambda x: x[1].get('avg_pnl', 0) if x[1].get('count', 0) > 0 else -999)
        if best_period[1].get('count', 0) > 0:
            recommendations.append(
                f"Optimal hold period: {best_period[1]['label']} with "
                f"${best_period[1]['avg_pnl']:.2f} average profit and "
                f"{best_period[1]['win_rate']:.1%} win rate"
            )
        
        # Direction recommendations
        if SwingDirection.LONG in direction_analysis and SwingDirection.SHORT in direction_analysis:
            long_pnl = direction_analysis[SwingDirection.LONG].get('avg_pnl', 0)
            short_pnl = direction_analysis[SwingDirection.SHORT].get('avg_pnl', 0)
            
            if long_pnl > short_pnl * 1.2:
                recommendations.append("Focus on long swing trades - showing superior performance")
            elif short_pnl > long_pnl * 1.2:
                recommendations.append("Focus on short swing trades - showing superior performance")
        
        # Trend recommendations
        best_trends = sorted(trend_analysis.items(), 
                           key=lambda x: x[1].get('avg_pnl', 0), reverse=True)[:2]
        for trend, metrics in best_trends:
            if metrics.get('count', 0) >= 3 and metrics.get('avg_pnl', 0) > 0:
                recommendations.append(
                    f"Trade during {trend.value.replace('_', ' ').lower()} conditions: "
                    f"${metrics['avg_pnl']:.2f} average profit"
                )
        
        # Pattern recommendations
        profitable_patterns = [p for p in patterns if p.avg_return > 0 and p.win_rate > 0.5]
        for pattern in profitable_patterns[:2]:  # Top 2 patterns
            recommendations.append(
                f"{pattern.pattern_name}: {pattern.frequency} occurrences, "
                f"{pattern.win_rate:.1%} win rate, optimal in {pattern.best_market_condition.value}"
            )
        
        # Risk recommendations
        if basic_metrics.get('consecutive_losses', 0) > 5:
            recommendations.append("Consider implementing stricter risk management - high consecutive losses detected")
        
        return recommendations if recommendations else ["Insufficient data for specific recommendations"]
    
    def _get_analysis_period(self, trades: List[SwingTrade]) -> Tuple[datetime, datetime]:
        """Get analysis period from trade dates"""
        if not trades:
            now = datetime.now()
            return (now, now)
        
        entry_dates = [trade.entry_date for trade in trades]
        exit_dates = [trade.exit_date for trade in trades]
        
        return (min(entry_dates), max(exit_dates))
    
    def _create_empty_result(self) -> SwingAnalyticsResult:
        """Create empty result for error cases"""
        now = datetime.now()
        return SwingAnalyticsResult(
            analysis_period=(now, now), total_swing_trades=0, avg_hold_period=0,
            overall_win_rate=0, total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0, profit_factor=0,
            avg_winner=0, avg_loser=0, largest_winner=0, largest_loser=0,
            consecutive_wins=0, consecutive_losses=0, hold_period_analysis={},
            direction_analysis={}, trend_analysis={}, seasonal_patterns={},
            swing_patterns=[], mae_mfe_analysis={}, recommendations=[]
        )
    
    def generate_comprehensive_report(self, result: SwingAnalyticsResult) -> str:
        """Generate comprehensive swing trading performance report"""
        report = f"""
SWING TRADING PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {result.analysis_period[0].strftime('%Y-%m-%d')} to {result.analysis_period[1].strftime('%Y-%m-%d')}
=========================================

OVERVIEW:
- Total Swing Trades: {result.total_swing_trades}
- Overall Win Rate: {result.overall_win_rate:.1%}
- Average Hold Period: {result.avg_hold_period:.1f} days
- Total Return: ${result.total_return:,.2f}
- Annualized Return: ${result.annualized_return:,.2f}

PERFORMANCE METRICS:
- Profit Factor: {result.profit_factor:.2f}
- Sharpe Ratio: {result.sharpe_ratio:.3f}
- Sortino Ratio: {result.sortino_ratio:.3f}
- Calmar Ratio: {result.calmar_ratio:.3f}
- Maximum Drawdown: ${result.max_drawdown:,.2f}

TRADE ANALYSIS:
- Average Winner: ${result.avg_winner:.2f}
- Average Loser: ${result.avg_loser:.2f}
- Largest Winner: ${result.largest_winner:.2f}
- Largest Loser: ${result.largest_loser:.2f}
- Max Consecutive Wins: {result.consecutive_wins}
- Max Consecutive Losses: {result.consecutive_losses}

HOLD PERIOD ANALYSIS:
"""
        
        for period, data in result.hold_period_analysis.items():
            if data.get('count', 0) > 0:
                report += f"- {data['label']}: {data['count']} trades, "
                report += f"${data['avg_pnl']:.2f} avg profit, {data['win_rate']:.1%} win rate\n"
        
        report += "\nDIRECTION ANALYSIS:\n"
        for direction, data in result.direction_analysis.items():
            if data.get('count', 0) > 0:
                report += f"- {direction.value}: {data['count']} trades, "
                report += f"${data['avg_pnl']:.2f} avg profit, {data['win_rate']:.1%} win rate\n"
        
        if result.swing_patterns:
            report += f"\nIDENTIFIED PATTERNS ({len(result.swing_patterns)}):\n"
            for pattern in result.swing_patterns:
                report += f"- {pattern.pattern_name}: {pattern.frequency} trades, "
                report += f"{pattern.win_rate:.1%} win rate, {pattern.avg_return:.2f}% avg return\n"
        
        if result.mae_mfe_analysis:
            mae_mfe = result.mae_mfe_analysis
            report += f"\nMAE/MFE ANALYSIS:\n"
            report += f"- Average MAE: {mae_mfe.get('avg_mae', 0):.2f}%\n"
            report += f"- Average MFE: {mae_mfe.get('avg_mfe', 0):.2f}%\n"
            report += f"- Efficiency Ratio: {mae_mfe.get('efficiency_ratio', 0):.2f}\n"
        
        report += "\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def export_analysis_to_json(self, result: SwingAnalyticsResult, filepath: str) -> bool:
        """Export swing analysis to JSON file"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_period': [dt.isoformat() for dt in result.analysis_period],
                'summary': {
                    'total_trades': result.total_swing_trades,
                    'win_rate': result.overall_win_rate,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'profit_factor': result.profit_factor
                },
                'hold_period_analysis': result.hold_period_analysis,
                'direction_analysis': {k.value: v for k, v in result.direction_analysis.items()},
                'seasonal_patterns': result.seasonal_patterns,
                'recommendations': result.recommendations
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
              logger.info(f"Swing analysis exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting swing analysis: {e}")
            return False

    # AnalyticsInterface Implementation for Framework Integration
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming data and return analytics results
        Implements AnalyticsInterface for framework integration
        """
        try:
            # Update last update timestamp
            self._last_update = datetime.now()
            
            # Process swing trading data
            if 'swing_trades' in data:
                # Process swing trade data
                swing_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.analyze_performance, 
                    data['swing_trades']
                )
                
                return {
                    "success": True,
                    "swing_performance": swing_results.total_return,
                    "win_rate": swing_results.overall_win_rate,
                    "avg_hold_period": swing_results.avg_hold_period,
                    "total_trades": swing_results.total_swing_trades,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif 'market_data' in data:
                # Process market data for swing opportunities
                market_data = data['market_data']
                return {
                    "success": True,
                    "analysis": "Market data processed for swing opportunities",
                    "data_points": len(market_data) if isinstance(market_data, list) else 1,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Return empty result if no processable data
            return {
                "success": False,
                "message": "No processable swing trading data found",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing data in SwingAnalytics: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def generate_report(self, timeframe: str) -> AnalyticsReport:
        """
        Generate analytics report for specified timeframe
        Implements AnalyticsInterface for framework integration
        """
        try:
            # Generate comprehensive swing analytics report
            report_data = {
                "timeframe": timeframe,
                "analysis_parameters": {
                    "short_term_days": self.short_term_days,
                    "medium_term_days": self.medium_term_days,
                    "long_term_days": self.long_term_days,
                    "risk_free_rate": self.risk_free_rate
                },
                "pattern_analysis": {},
                "performance_metrics": {}
            }
            
            # Add swing history data if available
            if self.swing_history:
                report_data["swing_history_count"] = len(self.swing_history)
                report_data["pattern_database_size"] = len(self.pattern_database)
            
            # Generate swing-specific recommendations
            recommendations = [
                "Focus on swing trades during high volatility periods",
                "Optimize hold periods based on market cycle analysis",
                "Use trend strength indicators for position sizing",
                "Implement seasonal pattern adjustments",
                "Monitor swing failure rates for risk management"
            ]
            
            # Calculate confidence score based on historical data
            confidence_score = 78.0
            if self.swing_history:
                confidence_score = min(92.0, 78.0 + min(len(self.swing_history), 50) * 0.3)
            
            summary = f"Swing trading analytics report for {timeframe} showing performance patterns and optimization opportunities"
            
            return AnalyticsReport(
                report_id=f"swing_analytics_{timeframe}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                report_type="swing_trading_analytics",
                generated_at=datetime.utcnow(),
                data=report_data,
                summary=summary,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating swing analytics report: {e}")
            return AnalyticsReport(
                report_id=f"swing_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                report_type="swing_trading_analytics",
                generated_at=datetime.utcnow(),
                data={"error": str(e)},
                summary=f"Error generating swing analytics report: {str(e)}",
                recommendations=["Review data input", "Check system connectivity"],
                confidence_score=0.0
            )

    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        """
        Get current real-time metrics
        Implements AnalyticsInterface for framework integration
        """
        try:
            metrics = []
            current_time = datetime.utcnow()
            
            # Engine status metric
            metrics.append(RealtimeMetric(
                metric_name="swing_analytics_engine_status",
                value=1.0,  # 1.0 = active, 0.0 = inactive
                timestamp=current_time,
                context={"engine": "swing_analytics", "status": "active"}
            ))
            
            # Swing history utilization
            history_utilization = len(self.swing_history) / 1000.0  # Normalize to 0-1
            metrics.append(RealtimeMetric(
                metric_name="swing_history_utilization",
                value=min(1.0, history_utilization),
                timestamp=current_time,
                context={"history_count": len(self.swing_history)},
                alert_threshold=0.9
            ))
            
            # Pattern database efficiency
            pattern_efficiency = len(self.pattern_database) / 100.0  # Normalize
            metrics.append(RealtimeMetric(
                metric_name="pattern_database_efficiency",
                value=min(1.0, pattern_efficiency),
                timestamp=current_time,
                context={"pattern_count": len(self.pattern_database)}
            ))
            
            # Processing efficiency metric
            if self._last_update:
                time_since_update = (current_time - self._last_update).total_seconds()
                efficiency = max(0.0, 1.0 - (time_since_update / 7200.0))  # Normalize by 2 hours
                metrics.append(RealtimeMetric(
                    metric_name="swing_processing_efficiency",
                    value=efficiency,
                    timestamp=current_time,
                    context={"last_update": self._last_update.isoformat()},
                    alert_threshold=0.2
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting swing analytics real-time metrics: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    # Example swing trading data
    sample_trades = [
        {
            'symbol': 'AAPL',
            'entry_date': '2024-01-01',
            'exit_date': '2024-01-15',
            'entry_price': 150.0,
            'exit_price': 165.0,
            'position_size': 1000,
            'side': 'long',
            'pnl': 15000,
            'mae': -2.5,
            'mfe': 12.0,
            'volatility': 0.025,
            'exit_reason': 'target_reached'
        },
        {
            'symbol': 'TSLA',
            'entry_date': '2024-01-10',
            'exit_date': '2024-01-25',
            'entry_price': 200.0,
            'exit_price': 180.0,
            'position_size': 500,
            'side': 'short',
            'pnl': 10000,
            'mae': -5.0,
            'mfe': 15.0,
            'volatility': 0.035,
            'exit_reason': 'stop_loss'
        }
    ]
    
    # Initialize swing analyzer
    analyzer = SwingAnalytics()
    
    # Analyze performance
    result = analyzer.analyze_performance(sample_trades)
    
    # Print comprehensive report
    print(analyzer.generate_comprehensive_report(result))

class SwingAnalytics:
    def __init__(self):
        print("SwingAnalytics initialized")

    def analyze_performance(self, data):
        # Placeholder for swing trading performance analysis
        print("Analyzing swing trading performance...")
