"""
Session Analytics Module

This module provides comprehensive session-based performance analytics for forex trading,
analyzing performance across different trading sessions (Asian, London, NY, Overlaps).
Optimized for session-specific strategy development and optimization.

Features:
- Session-based performance breakdown
- Overlap period analysis
- Volatility and volume analysis by session
- Currency pair performance by session
- Time-of-day optimization
- Session transition analysis
- Holiday and news impact analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, time, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading session classification"""
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    WEEKEND = "weekend"

class SessionCharacteristics(Enum):
    """Session market characteristics"""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"
    TRENDING = "trending"
    RANGING = "ranging"

@dataclass
class SessionTrade:
    """Session-specific trade data"""
    timestamp: datetime
    session: TradingSession
    currency_pair: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    duration_minutes: int
    volatility_at_entry: float
    volume_at_entry: float

@dataclass
class SessionMetrics:
    """Session performance metrics"""
    session: TradingSession
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl_per_trade: float
    profit_factor: float
    average_trade_duration: timedelta
    best_currency_pairs: List[str]
    worst_currency_pairs: List[str]
    peak_performance_hours: List[int]
    average_volatility: float
    average_volume: float
    characteristics: List[SessionCharacteristics]

@dataclass
class SessionAnalyticsResults:
    """Complete session analytics results"""
    session_metrics: Dict[str, SessionMetrics]
    hourly_performance: Dict[int, float]
    currency_pair_by_session: Dict[str, Dict[str, float]]
    session_transitions: Dict[str, float]
    overlap_analysis: Dict[str, Dict[str, float]]
    volatility_analysis: Dict[str, Dict[str, float]]
    volume_analysis: Dict[str, Dict[str, float]]
    recommendations: Dict[str, List[str]]

class SessionAnalytics:
    """
    Session-Based Trading Analytics
    
    Provides comprehensive analytics for trading performance across
    different forex trading sessions and time periods.
    """
    
    def __init__(self, timezone: str = 'UTC'):
        """
        Initialize Session Analytics
        
        Args:
            timezone: Timezone for session calculations
        """
        self.timezone = pytz.timezone(timezone)
        
        # Trade history
        self.session_trades: List[SessionTrade] = []
        
        # Session time definitions (UTC)
        self.session_times = {
            TradingSession.ASIAN: (time(22, 0), time(8, 0)),  # 22:00-08:00 UTC
            TradingSession.LONDON: (time(8, 0), time(16, 0)),  # 08:00-16:00 UTC
            TradingSession.NY: (time(13, 0), time(22, 0)),     # 13:00-22:00 UTC
            TradingSession.OVERLAP_ASIAN_LONDON: (time(8, 0), time(9, 0)),    # 08:00-09:00 UTC
            TradingSession.OVERLAP_LONDON_NY: (time(13, 0), time(16, 0))      # 13:00-16:00 UTC
        }
        
        logger.info(f"SessionAnalytics initialized with timezone: {timezone}")
    
    def _identify_session(self, timestamp: datetime) -> TradingSession:
        """
        Identify trading session based on timestamp
        
        Args:
            timestamp: Trade timestamp
            
        Returns:
            TradingSession enum value
        """
        # Convert to UTC if needed
        if timestamp.tzinfo is None:
            utc_time = timestamp.time()
        else:
            utc_timestamp = timestamp.astimezone(pytz.UTC)
            utc_time = utc_timestamp.time()
        
        # Check for weekend
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return TradingSession.WEEKEND
        
        # Check for overlaps first (more specific)
        if time(13, 0) <= utc_time <= time(16, 0):
            return TradingSession.OVERLAP_LONDON_NY
        elif time(8, 0) <= utc_time <= time(9, 0):
            return TradingSession.OVERLAP_ASIAN_LONDON
        elif time(8, 0) <= utc_time < time(13, 0):
            return TradingSession.LONDON
        elif time(16, 0) < utc_time <= time(22, 0):
            return TradingSession.NY
        else:  # 22:00-08:00 (next day)
            return TradingSession.ASIAN
    
    def _classify_session_characteristics(self, session_trades: List[SessionTrade]) -> List[SessionCharacteristics]:
        """
        Classify session characteristics based on trade data
        
        Args:
            session_trades: List of trades for the session
            
        Returns:
            List of session characteristics
        """
        if not session_trades:
            return []
        
        characteristics = []
        
        # Volatility analysis
        avg_volatility = np.mean([t.volatility_at_entry for t in session_trades])
        if avg_volatility > 0.015:  # High volatility threshold
            characteristics.append(SessionCharacteristics.HIGH_VOLATILITY)
        elif avg_volatility < 0.005:  # Low volatility threshold
            characteristics.append(SessionCharacteristics.LOW_VOLATILITY)
        
        # Volume analysis
        avg_volume = np.mean([t.volume_at_entry for t in session_trades])
        if avg_volume > 1000000:  # High volume threshold
            characteristics.append(SessionCharacteristics.HIGH_VOLUME)
        elif avg_volume < 100000:  # Low volume threshold
            characteristics.append(SessionCharacteristics.LOW_VOLUME)
        
        # Trend analysis (based on trade success in one direction)
        long_trades = [t for t in session_trades if t.direction == 'long']
        short_trades = [t for t in session_trades if t.direction == 'short']
        
        long_success = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0
        short_success = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0
        
        if abs(long_success - short_success) > 0.3:  # Strong directional bias
            characteristics.append(SessionCharacteristics.TRENDING)
        else:
            characteristics.append(SessionCharacteristics.RANGING)
        
        return characteristics
    
    def add_session_trade(self,
                         timestamp: datetime,
                         currency_pair: str,
                         direction: str,
                         entry_price: float,
                         exit_price: float,
                         quantity: float,
                         commission: float = 0.0,
                         duration_minutes: int = 60,
                         volatility_at_entry: float = 0.01,
                         volume_at_entry: float = 500000) -> None:
        """
        Add a trade to session analytics
        
        Args:
            timestamp: Trade timestamp
            currency_pair: Currency pair (e.g., 'EURUSD')
            direction: 'long' or 'short'
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            commission: Trade commission
            duration_minutes: Trade duration in minutes
            volatility_at_entry: Market volatility at entry
            volume_at_entry: Market volume at entry
        """
        try:
            # Calculate P&L
            if direction.lower() == 'long':
                pnl = (exit_price - entry_price) * quantity
            else:  # short
                pnl = (entry_price - exit_price) * quantity
            
            # Identify session
            session = self._identify_session(timestamp)
            
            # Create session trade
            session_trade = SessionTrade(
                timestamp=timestamp,
                session=session,
                currency_pair=currency_pair.upper(),
                direction=direction.lower(),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl=pnl,
                commission=commission,
                duration_minutes=duration_minutes,
                volatility_at_entry=volatility_at_entry,
                volume_at_entry=volume_at_entry
            )
            
            self.session_trades.append(session_trade)
            
            logger.debug(f"Session trade added: {currency_pair} {direction} in {session.value}, "
                        f"P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding session trade: {str(e)}")
            raise
    
    def _calculate_session_metrics(self, session: TradingSession) -> SessionMetrics:
        """
        Calculate metrics for a specific session
        
        Args:
            session: Trading session to analyze
            
        Returns:
            SessionMetrics object
        """
        session_trades = [t for t in self.session_trades if t.session == session]
        
        if not session_trades:
            return SessionMetrics(
                session=session,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                average_pnl_per_trade=0.0,
                profit_factor=0.0,
                average_trade_duration=timedelta(),
                best_currency_pairs=[],
                worst_currency_pairs=[],
                peak_performance_hours=[],
                average_volatility=0.0,
                average_volume=0.0,
                characteristics=[]
            )
        
        # Basic metrics
        total_trades = len(session_trades)
        winning_trades = len([t for t in session_trades if t.pnl > 0])
        losing_trades = len([t for t in session_trades if t.pnl < 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L metrics
        total_pnl = sum(t.pnl - t.commission for t in session_trades)
        average_pnl_per_trade = total_pnl / total_trades
        
        wins = [t.pnl - t.commission for t in session_trades if t.pnl > 0]
        losses = [t.pnl - t.commission for t in session_trades if t.pnl < 0]
        
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Duration analysis
        durations = [timedelta(minutes=t.duration_minutes) for t in session_trades]
        average_trade_duration = sum(durations, timedelta()) / len(durations)
        
        # Currency pair analysis
        pair_performance = {}
        for trade in session_trades:
            pair = trade.currency_pair
            if pair not in pair_performance:
                pair_performance[pair] = 0.0
            pair_performance[pair] += trade.pnl - trade.commission
        
        sorted_pairs = sorted(pair_performance.items(), key=lambda x: x[1], reverse=True)
        best_currency_pairs = [pair[0] for pair in sorted_pairs[:3] if pair[1] > 0]
        worst_currency_pairs = [pair[0] for pair in sorted_pairs[-3:] if pair[1] < 0]
        
        # Hourly performance analysis
        hourly_pnl = {}
        for trade in session_trades:
            hour = trade.timestamp.hour
            if hour not in hourly_pnl:
                hourly_pnl[hour] = 0.0
            hourly_pnl[hour] += trade.pnl - trade.commission
        
        peak_performance_hours = sorted(hourly_pnl.items(), key=lambda x: x[1], reverse=True)[:2]
        peak_performance_hours = [hour[0] for hour in peak_performance_hours]
        
        # Market condition analysis
        average_volatility = np.mean([t.volatility_at_entry for t in session_trades])
        average_volume = np.mean([t.volume_at_entry for t in session_trades])
        
        # Session characteristics
        characteristics = self._classify_session_characteristics(session_trades)
        
        return SessionMetrics(
            session=session,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_pnl_per_trade=average_pnl_per_trade,
            profit_factor=profit_factor,
            average_trade_duration=average_trade_duration,
            best_currency_pairs=best_currency_pairs,
            worst_currency_pairs=worst_currency_pairs,
            peak_performance_hours=peak_performance_hours,
            average_volatility=average_volatility,
            average_volume=average_volume,
            characteristics=characteristics
        )
    
    def _analyze_hourly_performance(self) -> Dict[int, float]:
        """
        Analyze performance by hour of day
        
        Returns:
            Dictionary mapping hour to total P&L
        """
        hourly_performance = {}
        
        for trade in self.session_trades:
            hour = trade.timestamp.hour
            net_pnl = trade.pnl - trade.commission
            
            if hour not in hourly_performance:
                hourly_performance[hour] = 0.0
            hourly_performance[hour] += net_pnl
        
        return hourly_performance
    
    def _analyze_currency_pair_by_session(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze currency pair performance by session
        
        Returns:
            Nested dictionary: {currency_pair: {session: pnl}}
        """
        pair_session_performance = {}
        
        for trade in self.session_trades:
            pair = trade.currency_pair
            session = trade.session.value
            net_pnl = trade.pnl - trade.commission
            
            if pair not in pair_session_performance:
                pair_session_performance[pair] = {}
            
            if session not in pair_session_performance[pair]:
                pair_session_performance[pair][session] = 0.0
            
            pair_session_performance[pair][session] += net_pnl
        
        return pair_session_performance
    
    def _analyze_session_transitions(self) -> Dict[str, float]:
        """
        Analyze performance during session transitions
        
        Returns:
            Dictionary with transition performance
        """
        transition_performance = {}
        
        # Define transition periods (1 hour before and after session changes)
        transition_hours = {
            'asian_to_london': [7, 8, 9],
            'london_to_ny': [12, 13, 14],
            'ny_to_asian': [21, 22, 23]
        }
        
        for transition, hours in transition_hours.items():
            transition_pnl = 0.0
            for trade in self.session_trades:
                if trade.timestamp.hour in hours:
                    transition_pnl += trade.pnl - trade.commission
            transition_performance[transition] = transition_pnl
        
        return transition_performance
    
    def _analyze_overlap_periods(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance during session overlaps
        
        Returns:
            Dictionary with overlap analysis
        """
        overlap_analysis = {}
        
        overlap_sessions = [TradingSession.OVERLAP_ASIAN_LONDON, TradingSession.OVERLAP_LONDON_NY]
        
        for overlap_session in overlap_sessions:
            overlap_trades = [t for t in self.session_trades if t.session == overlap_session]
            
            if overlap_trades:
                total_pnl = sum(t.pnl - t.commission for t in overlap_trades)
                avg_volatility = np.mean([t.volatility_at_entry for t in overlap_trades])
                avg_volume = np.mean([t.volume_at_entry for t in overlap_trades])
                win_rate = len([t for t in overlap_trades if t.pnl > 0]) / len(overlap_trades) * 100
                
                overlap_analysis[overlap_session.value] = {
                    'total_pnl': total_pnl,
                    'average_volatility': avg_volatility,
                    'average_volume': avg_volume,
                    'win_rate': win_rate,
                    'trade_count': len(overlap_trades)
                }
        
        return overlap_analysis
    
    def calculate_session_analytics(self) -> SessionAnalyticsResults:
        """
        Calculate comprehensive session analytics
        
        Returns:
            SessionAnalyticsResults with complete analysis
        """
        try:
            if not self.session_trades:
                logger.warning("No session trades available for analysis")
                return self._create_empty_results()
            
            # Calculate metrics for each session
            session_metrics = {}
            for session in TradingSession:
                if session != TradingSession.WEEKEND:  # Skip weekend analysis
                    metrics = self._calculate_session_metrics(session)
                    session_metrics[session.value] = metrics
            
            # Additional analyses
            hourly_performance = self._analyze_hourly_performance()
            currency_pair_by_session = self._analyze_currency_pair_by_session()
            session_transitions = self._analyze_session_transitions()
            overlap_analysis = self._analyze_overlap_periods()
            
            # Volatility and volume analysis by session
            volatility_analysis = {}
            volume_analysis = {}
            
            for session_name, metrics in session_metrics.items():
                volatility_analysis[session_name] = {
                    'average_volatility': metrics.average_volatility,
                    'volatility_impact_on_pnl': self._calculate_volatility_correlation(session_name)
                }
                volume_analysis[session_name] = {
                    'average_volume': metrics.average_volume,
                    'volume_impact_on_pnl': self._calculate_volume_correlation(session_name)
                }
            
            # Generate recommendations
            recommendations = self._generate_session_recommendations(session_metrics)
            
            results = SessionAnalyticsResults(
                session_metrics=session_metrics,
                hourly_performance=hourly_performance,
                currency_pair_by_session=currency_pair_by_session,
                session_transitions=session_transitions,
                overlap_analysis=overlap_analysis,
                volatility_analysis=volatility_analysis,
                volume_analysis=volume_analysis,
                recommendations=recommendations
            )
            
            logger.info(f"Session analytics calculated for {len(self.session_trades)} trades")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating session analytics: {str(e)}")
            raise
    
    def _calculate_volatility_correlation(self, session_name: str) -> float:
        """Calculate correlation between volatility and P&L for a session"""
        session_trades = [t for t in self.session_trades if t.session.value == session_name]
        
        if len(session_trades) < 3:
            return 0.0
        
        volatilities = [t.volatility_at_entry for t in session_trades]
        pnls = [t.pnl - t.commission for t in session_trades]
        
        try:
            correlation = np.corrcoef(volatilities, pnls)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_volume_correlation(self, session_name: str) -> float:
        """Calculate correlation between volume and P&L for a session"""
        session_trades = [t for t in self.session_trades if t.session.value == session_name]
        
        if len(session_trades) < 3:
            return 0.0
        
        volumes = [t.volume_at_entry for t in session_trades]
        pnls = [t.pnl - t.commission for t in session_trades]
        
        try:
            correlation = np.corrcoef(volumes, pnls)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _generate_session_recommendations(self, session_metrics: Dict[str, SessionMetrics]) -> Dict[str, List[str]]:
        """Generate trading recommendations based on session analysis"""
        recommendations = {
            "best_sessions": [],
            "best_currency_pairs": [],
            "optimal_hours": [],
            "strategy_adjustments": []
        }
        
        # Best performing sessions
        profitable_sessions = [(name, metrics.total_pnl) for name, metrics in session_metrics.items() 
                              if metrics.total_pnl > 0]
        profitable_sessions.sort(key=lambda x: x[1], reverse=True)
        recommendations["best_sessions"] = [session[0] for session in profitable_sessions[:2]]
        
        # Best currency pairs overall
        all_pairs = {}
        for metrics in session_metrics.values():
            for pair in metrics.best_currency_pairs:
                all_pairs[pair] = all_pairs.get(pair, 0) + 1
        
        best_pairs = sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
        recommendations["best_currency_pairs"] = [pair[0] for pair in best_pairs]
        
        # Strategy adjustments
        for name, metrics in session_metrics.items():
            if metrics.win_rate < 45:
                recommendations["strategy_adjustments"].append(f"Improve entry timing for {name} session")
            if metrics.profit_factor < 1.2:
                recommendations["strategy_adjustments"].append(f"Enhance risk management for {name} session")
        
        return recommendations
    
    def _create_empty_results(self) -> SessionAnalyticsResults:
        """Create empty results for no-trade scenarios"""
        return SessionAnalyticsResults(
            session_metrics={},
            hourly_performance={},
            currency_pair_by_session={},
            session_transitions={},
            overlap_analysis={},
            volatility_analysis={},
            volume_analysis={},
            recommendations={}
        )
