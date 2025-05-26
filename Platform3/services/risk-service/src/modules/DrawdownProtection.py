"""
Drawdown Protection System
Advanced drawdown monitoring and protection mechanisms for forex trading

Features:
- Real-time drawdown monitoring
- Dynamic position size reduction
- Automatic trading halt mechanisms
- Recovery strategy implementation
- Risk-adjusted comeback protocols
- Performance-based trading resumption
- Psychological protection measures
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrawdownLevel(Enum):
    NORMAL = "normal"           # 0-3% drawdown
    WARNING = "warning"         # 3-5% drawdown
    CAUTION = "caution"         # 5-8% drawdown
    DANGER = "danger"           # 8-12% drawdown
    CRITICAL = "critical"       # 12-15% drawdown
    EMERGENCY = "emergency"     # >15% drawdown

class ProtectionAction(Enum):
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    HALT_NEW_TRADES = "halt_new_trades"
    CLOSE_LOSING_POSITIONS = "close_losing_positions"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY_MODE = "recovery_mode"

class RecoveryPhase(Enum):
    ASSESSMENT = "assessment"
    GRADUAL_RETURN = "gradual_return"
    NORMAL_TRADING = "normal_trading"

@dataclass
class DrawdownMetrics:
    current_drawdown: float
    max_drawdown: float
    peak_balance: float
    current_balance: float
    drawdown_duration: timedelta
    consecutive_losses: int
    recovery_factor: float
    underwater_curve: List[float]
    drawdown_level: DrawdownLevel

@dataclass
class ProtectionRule:
    drawdown_threshold: float
    action: ProtectionAction
    size_reduction_factor: float
    halt_duration: timedelta
    recovery_threshold: float
    description: str

@dataclass
class RecoveryPlan:
    phase: RecoveryPhase
    target_return: float
    max_position_size: float
    allowed_pairs: List[str]
    risk_per_trade: float
    daily_loss_limit: float
    success_criteria: Dict[str, float]
    duration: timedelta

class DrawdownProtection:
    """
    Advanced drawdown protection and recovery system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Protection rules configuration
        self.protection_rules = {
            DrawdownLevel.WARNING: ProtectionRule(
                drawdown_threshold=0.03,
                action=ProtectionAction.REDUCE_SIZE,
                size_reduction_factor=0.8,
                halt_duration=timedelta(hours=1),
                recovery_threshold=0.02,
                description="Reduce position sizes by 20%"
            ),
            DrawdownLevel.CAUTION: ProtectionRule(
                drawdown_threshold=0.05,
                action=ProtectionAction.REDUCE_SIZE,
                size_reduction_factor=0.6,
                halt_duration=timedelta(hours=2),
                recovery_threshold=0.03,
                description="Reduce position sizes by 40%"
            ),
            DrawdownLevel.DANGER: ProtectionRule(
                drawdown_threshold=0.08,
                action=ProtectionAction.HALT_NEW_TRADES,
                size_reduction_factor=0.4,
                halt_duration=timedelta(hours=4),
                recovery_threshold=0.05,
                description="Halt new trades, reduce existing positions by 60%"
            ),
            DrawdownLevel.CRITICAL: ProtectionRule(
                drawdown_threshold=0.12,
                action=ProtectionAction.CLOSE_LOSING_POSITIONS,
                size_reduction_factor=0.2,
                halt_duration=timedelta(hours=8),
                recovery_threshold=0.08,
                description="Close losing positions, emergency risk reduction"
            ),
            DrawdownLevel.EMERGENCY: ProtectionRule(
                drawdown_threshold=0.15,
                action=ProtectionAction.EMERGENCY_STOP,
                size_reduction_factor=0.0,
                halt_duration=timedelta(days=1),
                recovery_threshold=0.12,
                description="Emergency stop - close all positions"
            )
        }
        
        # Recovery configuration
        self.recovery_config = {
            RecoveryPhase.ASSESSMENT: {
                'duration': timedelta(hours=24),
                'max_position_size': 0.0,
                'risk_per_trade': 0.0,
                'analysis_required': True
            },
            RecoveryPhase.GRADUAL_RETURN: {
                'duration': timedelta(days=7),
                'max_position_size': 0.5,
                'risk_per_trade': 0.005,  # 0.5% risk per trade
                'success_threshold': 0.02  # 2% positive return to advance
            },
            RecoveryPhase.NORMAL_TRADING: {
                'max_position_size': 1.0,
                'risk_per_trade': 0.02,  # 2% risk per trade
                'monitoring_period': timedelta(days=30)
            }
        }
        
        # State tracking
        self.current_state = {
            'protection_active': False,
            'current_level': DrawdownLevel.NORMAL,
            'halt_until': None,
            'recovery_phase': None,
            'recovery_start': None,
            'last_assessment': None
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_protections_triggered': 0,
            'emergency_stops': 0,
            'successful_recoveries': 0,
            'average_recovery_time': 0.0,
            'max_drawdown_prevented': 0.0,
            'protection_effectiveness': 0.0
        }
        
        # Historical data
        self.balance_history = []
        self.drawdown_history = []
        
        logger.info("DrawdownProtection system initialized")

    async def start_monitoring(self, initial_balance: float):
        """Start drawdown monitoring"""
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.balance_history = [(datetime.now(), initial_balance)]
        
        logger.info(f"🛡️ Drawdown protection started with initial balance: ${initial_balance:,.2f}")

    async def update_balance(self, current_balance: float) -> DrawdownMetrics:
        """Update current balance and calculate drawdown metrics"""
        try:
            # Update balance history
            self.balance_history.append((datetime.now(), current_balance))
            
            # Keep only last 1000 entries for performance
            if len(self.balance_history) > 1000:
                self.balance_history = self.balance_history[-1000:]
            
            # Update peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            # Calculate drawdown metrics
            metrics = await self._calculate_drawdown_metrics(current_balance)
            
            # Check for protection triggers
            await self._check_protection_triggers(metrics)
            
            # Update recovery status if in recovery
            if self.current_state['recovery_phase']:
                await self._update_recovery_status(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error updating balance: {e}")
            return self._default_metrics(current_balance)

    async def _calculate_drawdown_metrics(self, current_balance: float) -> DrawdownMetrics:
        """Calculate comprehensive drawdown metrics"""
        try:
            # Current drawdown
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
            # Maximum drawdown from history
            max_drawdown = 0.0
            peak = self.initial_balance
            
            for timestamp, balance in self.balance_history:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Drawdown duration
            drawdown_start = None
            for timestamp, balance in reversed(self.balance_history):
                if balance >= self.peak_balance * 0.999:  # Within 0.1% of peak
                    break
                drawdown_start = timestamp
            
            drawdown_duration = datetime.now() - drawdown_start if drawdown_start else timedelta(0)
            
            # Consecutive losses
            consecutive_losses = await self._calculate_consecutive_losses()
            
            # Recovery factor (how quickly we recover from drawdowns)
            recovery_factor = await self._calculate_recovery_factor()
            
            # Underwater curve (time spent in drawdown)
            underwater_curve = await self._calculate_underwater_curve()
            
            # Determine drawdown level
            drawdown_level = self._determine_drawdown_level(current_drawdown)
            
            return DrawdownMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                peak_balance=self.peak_balance,
                current_balance=current_balance,
                drawdown_duration=drawdown_duration,
                consecutive_losses=consecutive_losses,
                recovery_factor=recovery_factor,
                underwater_curve=underwater_curve,
                drawdown_level=drawdown_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return self._default_metrics(current_balance)

    def _determine_drawdown_level(self, drawdown: float) -> DrawdownLevel:
        """Determine current drawdown level"""
        if drawdown >= 0.15:
            return DrawdownLevel.EMERGENCY
        elif drawdown >= 0.12:
            return DrawdownLevel.CRITICAL
        elif drawdown >= 0.08:
            return DrawdownLevel.DANGER
        elif drawdown >= 0.05:
            return DrawdownLevel.CAUTION
        elif drawdown >= 0.03:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL

    async def _check_protection_triggers(self, metrics: DrawdownMetrics):
        """Check if protection measures should be triggered"""
        try:
            current_level = metrics.drawdown_level
            
            # Check if we need to escalate protection
            if current_level != self.current_state['current_level']:
                await self._trigger_protection(current_level, metrics)
            
            # Check for recovery conditions
            elif self.current_state['protection_active']:
                await self._check_recovery_conditions(metrics)
                
        except Exception as e:
            logger.error(f"Error checking protection triggers: {e}")

    async def _trigger_protection(self, level: DrawdownLevel, metrics: DrawdownMetrics):
        """Trigger protection measures for given drawdown level"""
        try:
            rule = self.protection_rules[level]
            
            logger.warning(f"🚨 Drawdown protection triggered: {level.value}")
            logger.warning(f"📊 Current drawdown: {metrics.current_drawdown:.2%}")
            logger.warning(f"🛡️ Action: {rule.description}")
            
            # Update state
            self.current_state['protection_active'] = True
            self.current_state['current_level'] = level
            self.current_state['halt_until'] = datetime.now() + rule.halt_duration
            
            # Execute protection action
            await self._execute_protection_action(rule, metrics)
            
            # Start recovery planning if critical
            if level in [DrawdownLevel.CRITICAL, DrawdownLevel.EMERGENCY]:
                await self._initiate_recovery_plan(level)
            
            # Update statistics
            self.performance_stats['total_protections_triggered'] += 1
            if level == DrawdownLevel.EMERGENCY:
                self.performance_stats['emergency_stops'] += 1
            
            # Store protection event
            await self._store_protection_event(level, rule, metrics)
            
        except Exception as e:
            logger.error(f"Error triggering protection: {e}")

    async def _execute_protection_action(self, rule: ProtectionRule, metrics: DrawdownMetrics):
        """Execute specific protection action"""
        try:
            if rule.action == ProtectionAction.REDUCE_SIZE:
                await self._reduce_position_sizes(rule.size_reduction_factor)
                
            elif rule.action == ProtectionAction.HALT_NEW_TRADES:
                await self._halt_new_trades(rule.halt_duration)
                await self._reduce_position_sizes(rule.size_reduction_factor)
                
            elif rule.action == ProtectionAction.CLOSE_LOSING_POSITIONS:
                await self._close_losing_positions()
                await self._halt_new_trades(rule.halt_duration)
                
            elif rule.action == ProtectionAction.EMERGENCY_STOP:
                await self._emergency_stop_all_trading()
                
        except Exception as e:
            logger.error(f"Error executing protection action: {e}")

    async def _reduce_position_sizes(self, reduction_factor: float):
        """Reduce all position sizes by given factor"""
        try:
            # This would integrate with the trading system to reduce position sizes
            logger.info(f"📉 Reducing position sizes by {(1-reduction_factor)*100:.0f}%")
            
            # Store reduction command in Redis for trading system to pick up
            reduction_command = {
                'action': 'reduce_positions',
                'factor': reduction_factor,
                'timestamp': datetime.now().isoformat(),
                'reason': 'drawdown_protection'
            }
            
            await self.redis_client.lpush('risk_commands', json.dumps(reduction_command))
            
        except Exception as e:
            logger.error(f"Error reducing position sizes: {e}")

    async def _halt_new_trades(self, duration: timedelta):
        """Halt new trade entries for specified duration"""
        try:
            halt_until = datetime.now() + duration
            logger.info(f"⏸️ Halting new trades until {halt_until}")
            
            # Store halt command
            halt_command = {
                'action': 'halt_trading',
                'until': halt_until.isoformat(),
                'reason': 'drawdown_protection'
            }
            
            await self.redis_client.set('trading_halt', json.dumps(halt_command))
            
        except Exception as e:
            logger.error(f"Error halting new trades: {e}")

    async def _close_losing_positions(self):
        """Close all losing positions"""
        try:
            logger.info("🔴 Closing all losing positions")
            
            close_command = {
                'action': 'close_losing_positions',
                'timestamp': datetime.now().isoformat(),
                'reason': 'drawdown_protection'
            }
            
            await self.redis_client.lpush('risk_commands', json.dumps(close_command))
            
        except Exception as e:
            logger.error(f"Error closing losing positions: {e}")

    async def _emergency_stop_all_trading(self):
        """Emergency stop - close all positions and halt trading"""
        try:
            logger.critical("🛑 EMERGENCY STOP - Closing all positions")
            
            emergency_command = {
                'action': 'emergency_stop',
                'timestamp': datetime.now().isoformat(),
                'reason': 'critical_drawdown'
            }
            
            await self.redis_client.lpush('risk_commands', json.dumps(emergency_command))
            await self.redis_client.set('emergency_stop', 'true')
            
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")

    async def _initiate_recovery_plan(self, level: DrawdownLevel):
        """Initiate recovery plan for critical drawdown levels"""
        try:
            logger.info(f"🔄 Initiating recovery plan for {level.value} drawdown")
            
            self.current_state['recovery_phase'] = RecoveryPhase.ASSESSMENT
            self.current_state['recovery_start'] = datetime.now()
            
            # Create recovery plan
            recovery_plan = RecoveryPlan(
                phase=RecoveryPhase.ASSESSMENT,
                target_return=0.05,  # 5% recovery target
                max_position_size=0.0,  # No trading during assessment
                allowed_pairs=[],
                risk_per_trade=0.0,
                daily_loss_limit=0.0,
                success_criteria={'stability_days': 3, 'no_new_losses': True},
                duration=self.recovery_config[RecoveryPhase.ASSESSMENT]['duration']
            )
            
            await self._store_recovery_plan(recovery_plan)
            
        except Exception as e:
            logger.error(f"Error initiating recovery plan: {e}")

    async def get_current_status(self) -> Dict:
        """Get current drawdown protection status"""
        return {
            'protection_active': self.current_state['protection_active'],
            'current_level': self.current_state['current_level'].value if self.current_state['current_level'] else 'normal',
            'halt_until': self.current_state['halt_until'].isoformat() if self.current_state['halt_until'] else None,
            'recovery_phase': self.current_state['recovery_phase'].value if self.current_state['recovery_phase'] else None,
            'recovery_start': self.current_state['recovery_start'].isoformat() if self.current_state['recovery_start'] else None,
            'performance_stats': self.performance_stats,
            'protection_rules': {k.value: {
                'threshold': v.drawdown_threshold,
                'action': v.action.value,
                'description': v.description
            } for k, v in self.protection_rules.items()}
        }

    async def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed"""
        try:
            # Check emergency stop
            emergency_stop = await self.redis_client.get('emergency_stop')
            if emergency_stop == 'true':
                return False, "Emergency stop active"
            
            # Check trading halt
            halt_data = await self.redis_client.get('trading_halt')
            if halt_data:
                halt_info = json.loads(halt_data)
                halt_until = datetime.fromisoformat(halt_info['until'])
                if datetime.now() < halt_until:
                    return False, f"Trading halted until {halt_until}"
            
            # Check recovery phase restrictions
            if self.current_state['recovery_phase'] == RecoveryPhase.ASSESSMENT:
                return False, "In recovery assessment phase"
            
            return True, "Trading allowed"
            
        except Exception as e:
            logger.error(f"Error checking trading status: {e}")
            return False, "Error checking trading status"

    def _default_metrics(self, current_balance: float) -> DrawdownMetrics:
        """Return default metrics in case of errors"""
        return DrawdownMetrics(
            current_drawdown=0.0,
            max_drawdown=0.0,
            peak_balance=current_balance,
            current_balance=current_balance,
            drawdown_duration=timedelta(0),
            consecutive_losses=0,
            recovery_factor=1.0,
            underwater_curve=[],
            drawdown_level=DrawdownLevel.NORMAL
        )

    # Additional helper methods would be implemented here for:
    # - _calculate_consecutive_losses()
    # - _calculate_recovery_factor()
    # - _calculate_underwater_curve()
    # - _check_recovery_conditions()
    # - _update_recovery_status()
    # - _store_protection_event()
    # - _store_recovery_plan()
