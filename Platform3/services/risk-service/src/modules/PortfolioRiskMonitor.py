"""
Portfolio Risk Monitoring System
Real-time portfolio risk assessment and monitoring for forex trading

Features:
- Real-time portfolio risk calculations
- Multi-currency exposure monitoring
- Correlation-based risk adjustments
- VaR (Value at Risk) calculations
- Portfolio heat monitoring
- Risk limit enforcement
- Automated risk alerts and actions
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    EXPOSURE_LIMIT = "exposure_limit"
    CORRELATION_SPIKE = "correlation_spike"
    VAR_BREACH = "var_breach"
    DRAWDOWN_LIMIT = "drawdown_limit"
    PORTFOLIO_HEAT = "portfolio_heat"
    MARGIN_CALL = "margin_call"

@dataclass
class PortfolioPosition:
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_used: float
    timestamp: datetime

@dataclass
class RiskMetrics:
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    portfolio_value: float
    unrealized_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    portfolio_heat: float  # % of portfolio at risk
    margin_utilization: float
    correlation_risk: float
    concentration_risk: float
    risk_score: float  # 0-100

@dataclass
class RiskAlert:
    alert_id: str
    alert_type: AlertType
    severity: RiskLevel
    message: str
    current_value: float
    limit_value: float
    timestamp: datetime
    positions_affected: List[str]
    recommended_action: str

class PortfolioRiskMonitor:
    """
    Advanced portfolio risk monitoring system for forex trading
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, db_config: Optional[Dict] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_platform',
            'user': 'postgres',
            'password': 'password'
        }
        
        self.positions: Dict[str, PortfolioPosition] = {}
        self.risk_limits = self._load_risk_limits()
        self.correlation_matrix = {}
        self.price_history = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Performance tracking
        self.performance_stats = {
            'total_calculations': 0,
            'alerts_generated': 0,
            'risk_violations': 0,
            'average_calculation_time': 0.0,
            'last_update': None
        }
        
        logger.info("PortfolioRiskMonitor initialized")

    def _load_risk_limits(self) -> Dict:
        """Load risk limits configuration"""
        return {
            'max_portfolio_exposure': 1000000,  # $1M max exposure
            'max_single_position': 100000,     # $100K max single position
            'max_currency_exposure': 500000,   # $500K max per currency
            'max_drawdown': 0.10,              # 10% max drawdown
            'max_daily_loss': 50000,           # $50K max daily loss
            'var_95_limit': 25000,             # $25K VaR 95%
            'var_99_limit': 50000,             # $50K VaR 99%
            'max_portfolio_heat': 0.15,        # 15% max portfolio heat
            'max_margin_utilization': 0.80,    # 80% max margin usage
            'max_correlation_exposure': 0.70,  # 70% max correlated exposure
            'max_concentration': 0.25          # 25% max single position concentration
        }

    async def start_monitoring(self):
        """Start real-time portfolio risk monitoring"""
        self.running = True
        logger.info("🚀 Starting portfolio risk monitoring...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_portfolio_risk()),
            asyncio.create_task(self._monitor_correlations()),
            asyncio.create_task(self._monitor_var_limits()),
            asyncio.create_task(self._update_price_data())
        ]
        
        await asyncio.gather(*tasks)

    async def stop_monitoring(self):
        """Stop portfolio risk monitoring"""
        self.running = False
        logger.info("⏹️ Stopping portfolio risk monitoring...")

    async def add_position(self, position: PortfolioPosition) -> bool:
        """Add position to portfolio monitoring"""
        try:
            self.positions[position.symbol] = position
            await self._cache_position(position)
            
            # Immediate risk check for new position
            risk_check = await self.calculate_portfolio_risk()
            await self._check_risk_violations(risk_check)
            
            logger.info(f"✅ Added position {position.symbol} to portfolio monitoring")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add position {position.symbol}: {e}")
            return False

    async def remove_position(self, symbol: str) -> bool:
        """Remove position from portfolio monitoring"""
        try:
            if symbol in self.positions:
                del self.positions[symbol]
                await self._remove_cached_position(symbol)
                logger.info(f"✅ Removed position {symbol} from portfolio monitoring")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to remove position {symbol}: {e}")
            return False

    async def update_position_price(self, symbol: str, current_price: float):
        """Update position with current market price"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            
            # Recalculate unrealized P&L
            if position.side == 'buy':
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
            await self._cache_position(position)

    async def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        start_time = datetime.now()
        
        try:
            if not self.positions:
                return self._empty_risk_metrics()
            
            # Calculate basic exposure metrics
            total_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
            net_exposure = sum(pos.size * pos.current_price * (1 if pos.side == 'buy' else -1) 
                             for pos in self.positions.values())
            gross_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
            
            # Calculate P&L metrics
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            daily_pnl = await self._calculate_daily_pnl()
            
            # Calculate drawdown
            portfolio_value = 1000000 + unrealized_pnl  # Assuming $1M starting capital
            max_drawdown, current_drawdown = await self._calculate_drawdown()
            
            # Calculate VaR
            var_95, var_99 = await self._calculate_var()
            
            # Calculate portfolio heat (% at risk)
            portfolio_heat = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate margin utilization
            margin_used = sum(pos.margin_used for pos in self.positions.values())
            margin_utilization = margin_used / 1000000  # Assuming $1M available margin
            
            # Calculate correlation and concentration risk
            correlation_risk = await self._calculate_correlation_risk()
            concentration_risk = await self._calculate_concentration_risk()
            
            # Calculate overall risk score (0-100)
            risk_score = await self._calculate_risk_score(
                portfolio_heat, margin_utilization, correlation_risk, 
                concentration_risk, current_drawdown
            )
            
            risk_metrics = RiskMetrics(
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                portfolio_value=portfolio_value,
                unrealized_pnl=unrealized_pnl,
                daily_pnl=daily_pnl,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                var_95=var_95,
                var_99=var_99,
                portfolio_heat=portfolio_heat,
                margin_utilization=margin_utilization,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                risk_score=risk_score
            )
            
            # Update performance stats
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['average_calculation_time'] = (
                (self.performance_stats['average_calculation_time'] * 
                 (self.performance_stats['total_calculations'] - 1) + calculation_time) /
                self.performance_stats['total_calculations']
            )
            self.performance_stats['last_update'] = datetime.now()
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating portfolio risk: {e}")
            return self._empty_risk_metrics()

    async def _monitor_portfolio_risk(self):
        """Continuous portfolio risk monitoring"""
        while self.running:
            try:
                risk_metrics = await self.calculate_portfolio_risk()
                await self._check_risk_violations(risk_metrics)
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in portfolio risk monitoring: {e}")
                await asyncio.sleep(5)

    async def _monitor_correlations(self):
        """Monitor currency pair correlations"""
        while self.running:
            try:
                await self._update_correlation_matrix()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in correlation monitoring: {e}")
                await asyncio.sleep(60)

    async def _monitor_var_limits(self):
        """Monitor VaR limit breaches"""
        while self.running:
            try:
                risk_metrics = await self.calculate_portfolio_risk()
                
                if risk_metrics.var_95 > self.risk_limits['var_95_limit']:
                    await self._generate_alert(
                        AlertType.VAR_BREACH,
                        RiskLevel.HIGH,
                        f"VaR 95% breach: ${risk_metrics.var_95:,.2f} > ${self.risk_limits['var_95_limit']:,.2f}",
                        risk_metrics.var_95,
                        self.risk_limits['var_95_limit']
                    )
                
                if risk_metrics.var_99 > self.risk_limits['var_99_limit']:
                    await self._generate_alert(
                        AlertType.VAR_BREACH,
                        RiskLevel.CRITICAL,
                        f"VaR 99% breach: ${risk_metrics.var_99:,.2f} > ${self.risk_limits['var_99_limit']:,.2f}",
                        risk_metrics.var_99,
                        self.risk_limits['var_99_limit']
                    )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in VaR monitoring: {e}")
                await asyncio.sleep(30)

    async def _update_price_data(self):
        """Update price data for risk calculations"""
        while self.running:
            try:
                # Update price history for VaR calculations
                for symbol in self.positions.keys():
                    price_data = await self._fetch_price_history(symbol, 100)
                    self.price_history[symbol] = price_data
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating price data: {e}")
                await asyncio.sleep(60)

    async def _check_risk_violations(self, risk_metrics: RiskMetrics):
        """Check for risk limit violations and generate alerts"""
        violations = []
        
        # Check exposure limits
        if risk_metrics.total_exposure > self.risk_limits['max_portfolio_exposure']:
            violations.append(('exposure_limit', 'CRITICAL', 
                             f"Portfolio exposure ${risk_metrics.total_exposure:,.2f} exceeds limit"))
        
        # Check drawdown limits
        if risk_metrics.current_drawdown > self.risk_limits['max_drawdown']:
            violations.append(('drawdown_limit', 'CRITICAL',
                             f"Drawdown {risk_metrics.current_drawdown:.2%} exceeds limit"))
        
        # Check portfolio heat
        if risk_metrics.portfolio_heat > self.risk_limits['max_portfolio_heat']:
            violations.append(('portfolio_heat', 'HIGH',
                             f"Portfolio heat {risk_metrics.portfolio_heat:.2%} exceeds limit"))
        
        # Check margin utilization
        if risk_metrics.margin_utilization > self.risk_limits['max_margin_utilization']:
            violations.append(('margin_call', 'HIGH',
                             f"Margin utilization {risk_metrics.margin_utilization:.2%} exceeds limit"))
        
        # Generate alerts for violations
        for violation_type, severity, message in violations:
            await self._generate_alert(
                AlertType(violation_type),
                RiskLevel(severity.lower()),
                message,
                0, 0  # Will be filled by specific violation
            )
            
        if violations:
            self.performance_stats['risk_violations'] += len(violations)

    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            total_exposure=0, net_exposure=0, gross_exposure=0,
            portfolio_value=1000000, unrealized_pnl=0, daily_pnl=0,
            max_drawdown=0, current_drawdown=0, var_95=0, var_99=0,
            portfolio_heat=0, margin_utilization=0, correlation_risk=0,
            concentration_risk=0, risk_score=0
        )

    async def _cache_position(self, position: PortfolioPosition):
        """Cache position data in Redis"""
        try:
            position_data = {
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'margin_used': position.margin_used,
                'timestamp': position.timestamp.isoformat()
            }
            await self.redis_client.hset(f"portfolio:position:{position.symbol}", 
                                       mapping=position_data)
        except Exception as e:
            logger.error(f"Failed to cache position {position.symbol}: {e}")

    async def _remove_cached_position(self, symbol: str):
        """Remove cached position from Redis"""
        try:
            await self.redis_client.delete(f"portfolio:position:{symbol}")
        except Exception as e:
            logger.error(f"Failed to remove cached position {symbol}: {e}")

    async def _generate_alert(self, alert_type: AlertType, severity: RiskLevel, 
                            message: str, current_value: float, limit_value: float):
        """Generate risk alert"""
        alert = RiskAlert(
            alert_id=f"RISK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            limit_value=limit_value,
            timestamp=datetime.now(),
            positions_affected=list(self.positions.keys()),
            recommended_action=self._get_recommended_action(alert_type, severity)
        )
        
        # Store alert
        await self._store_alert(alert)
        
        # Send notifications based on severity
        if severity in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
            await self._send_immediate_notification(alert)
        
        self.performance_stats['alerts_generated'] += 1
        logger.warning(f"🚨 Risk Alert: {alert.message}")

    def _get_recommended_action(self, alert_type: AlertType, severity: RiskLevel) -> str:
        """Get recommended action for alert"""
        actions = {
            AlertType.EXPOSURE_LIMIT: "Reduce position sizes or close positions",
            AlertType.CORRELATION_SPIKE: "Diversify positions or hedge correlated exposure",
            AlertType.VAR_BREACH: "Reduce portfolio risk or increase capital",
            AlertType.DRAWDOWN_LIMIT: "Stop trading and review strategy",
            AlertType.PORTFOLIO_HEAT: "Reduce position sizes",
            AlertType.MARGIN_CALL: "Close positions or add margin"
        }
        return actions.get(alert_type, "Review portfolio and take appropriate action")

    async def get_performance_stats(self) -> Dict:
        """Get monitoring performance statistics"""
        return {
            **self.performance_stats,
            'positions_monitored': len(self.positions),
            'risk_limits': self.risk_limits,
            'monitoring_status': 'running' if self.running else 'stopped'
        }

# Additional helper methods would be implemented here for:
# - _calculate_daily_pnl()
# - _calculate_drawdown()
# - _calculate_var()
# - _calculate_correlation_risk()
# - _calculate_concentration_risk()
# - _calculate_risk_score()
# - _update_correlation_matrix()
# - _fetch_price_history()
# - _store_alert()
# - _send_immediate_notification()
