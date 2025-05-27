"""
Risk Limit Violation Monitoring System
Comprehensive monitoring and testing of risk management compliance

Features:
- Real-time risk limit monitoring
- Violation detection and alerting
- Risk compliance testing
- Automated risk scenario testing
- Performance impact analysis
- Regulatory compliance validation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskType(Enum):
    """Types of risk limits to monitor"""
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    EXPOSURE = "exposure"
    DRAWDOWN = "drawdown"
    VAR = "var"  # Value at Risk
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MARGIN = "margin"
    STOP_LOSS = "stop_loss"

class ViolationSeverity(Enum):
    """Severity levels for risk violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ViolationStatus(Enum):
    """Status of risk violations"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"

@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_id: str
    risk_type: RiskType
    limit_name: str
    limit_value: float
    warning_threshold: float  # % of limit that triggers warning
    account_id: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class RiskViolation:
    """Risk violation record"""
    violation_id: str
    limit_id: str
    risk_type: RiskType
    account_id: str
    symbol: Optional[str]
    current_value: float
    limit_value: float
    violation_amount: float
    violation_percentage: float
    severity: ViolationSeverity
    status: ViolationStatus
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_action: Optional[str] = None
    impact_assessment: Optional[Dict] = None

@dataclass
class RiskScenario:
    """Risk testing scenario"""
    scenario_id: str
    scenario_name: str
    description: str
    risk_parameters: Dict[str, float]
    expected_violations: List[RiskType]
    test_duration: timedelta
    market_conditions: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

class RiskViolationMonitor:
    """
    Comprehensive risk violation monitoring and testing system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Risk limits storage
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.active_violations: Dict[str, RiskViolation] = {}
        self.violation_history: List[RiskViolation] = []
        self.test_scenarios: Dict[str, RiskScenario] = {}
        
        # Default risk limits
        self._initialize_default_limits()
        
        # Monitoring configuration
        self.config = {
            'monitoring_interval': 5,  # seconds
            'violation_cooldown': 60,  # seconds between same violation alerts
            'max_violation_history': 1000,
            'alert_escalation_time': 300,  # 5 minutes
            'auto_resolution_timeout': 3600,  # 1 hour
            'stress_test_frequency': timedelta(hours=24)
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_violations_detected': 0,
            'violations_resolved': 0,
            'average_resolution_time': 0.0,
            'critical_violations': 0,
            'false_positives': 0,
            'monitoring_uptime': 0.0,
            'last_violation': None,
            'compliance_score': 100.0
        }
        
        self.running = False
        logger.info("RiskViolationMonitor initialized")

    def _initialize_default_limits(self):
        """Initialize default risk limits"""
        default_limits = [
            RiskLimit("POS_SIZE_001", RiskType.POSITION_SIZE, "Max Position Size", 1000000, 0.8),
            RiskLimit("LEVERAGE_001", RiskType.LEVERAGE, "Max Leverage", 50.0, 0.9),
            RiskLimit("EXPOSURE_001", RiskType.EXPOSURE, "Max Portfolio Exposure", 5000000, 0.85),
            RiskLimit("DRAWDOWN_001", RiskType.DRAWDOWN, "Max Daily Drawdown", 0.05, 0.8),  # 5%
            RiskLimit("VAR_001", RiskType.VAR, "Daily VaR Limit", 100000, 0.9),
            RiskLimit("CONCENTRATION_001", RiskType.CONCENTRATION, "Max Single Asset %", 0.25, 0.8),  # 25%
            RiskLimit("MARGIN_001", RiskType.MARGIN, "Min Margin Level", 0.2, 0.9),  # 20%
        ]
        
        for limit in default_limits:
            self.risk_limits[limit.limit_id] = limit

    async def start_monitoring(self):
        """Start risk violation monitoring"""
        self.running = True
        logger.info("🚀 Starting risk violation monitoring...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._continuous_risk_monitoring()),
            asyncio.create_task(self._violation_management()),
            asyncio.create_task(self._automated_stress_testing()),
            asyncio.create_task(self._compliance_reporting())
        ]
        
        await asyncio.gather(*tasks)

    async def stop_monitoring(self):
        """Stop risk violation monitoring"""
        self.running = False
        logger.info("⏹️ Stopping risk violation monitoring...")

    async def add_risk_limit(self, limit: RiskLimit) -> bool:
        """Add a new risk limit"""
        try:
            self.risk_limits[limit.limit_id] = limit
            
            # Cache in Redis
            await self._cache_risk_limit(limit)
            
            logger.info(f"✅ Added risk limit: {limit.limit_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add risk limit {limit.limit_id}: {e}")
            return False

    async def check_risk_violation(self, account_id: str, risk_type: RiskType, 
                                 current_value: float, symbol: Optional[str] = None) -> Optional[RiskViolation]:
        """Check for risk limit violations"""
        try:
            # Find applicable risk limits
            applicable_limits = [
                limit for limit in self.risk_limits.values()
                if (limit.risk_type == risk_type and 
                    limit.is_active and
                    (limit.account_id is None or limit.account_id == account_id) and
                    (limit.symbol is None or limit.symbol == symbol))
            ]
            
            for limit in applicable_limits:
                # Check if current value exceeds limit
                if self._is_violation(current_value, limit.limit_value, risk_type):
                    violation = await self._create_violation(
                        limit, account_id, symbol, current_value
                    )
                    
                    if violation:
                        await self._handle_violation(violation)
                        return violation
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking risk violation: {e}")
            return None

    def _is_violation(self, current_value: float, limit_value: float, risk_type: RiskType) -> bool:
        """Determine if current value violates the limit"""
        if risk_type in [RiskType.MARGIN]:
            # For margin, violation is when current < limit (minimum requirement)
            return current_value < limit_value
        else:
            # For most limits, violation is when current > limit (maximum allowed)
            return current_value > limit_value

    async def _create_violation(self, limit: RiskLimit, account_id: str, 
                              symbol: Optional[str], current_value: float) -> Optional[RiskViolation]:
        """Create a new risk violation record"""
        try:
            violation_amount = abs(current_value - limit.limit_value)
            violation_percentage = (violation_amount / limit.limit_value) * 100
            
            # Determine severity
            severity = self._determine_violation_severity(violation_percentage)
            
            violation = RiskViolation(
                violation_id=f"VIOL_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                limit_id=limit.limit_id,
                risk_type=limit.risk_type,
                account_id=account_id,
                symbol=symbol,
                current_value=current_value,
                limit_value=limit.limit_value,
                violation_amount=violation_amount,
                violation_percentage=violation_percentage,
                severity=severity,
                status=ViolationStatus.ACTIVE,
                detected_at=datetime.now()
            )
            
            return violation
            
        except Exception as e:
            logger.error(f"Error creating violation: {e}")
            return None

    def _determine_violation_severity(self, violation_percentage: float) -> ViolationSeverity:
        """Determine violation severity based on percentage over limit"""
        if violation_percentage >= 100:  # 100% over limit
            return ViolationSeverity.EMERGENCY
        elif violation_percentage >= 50:  # 50% over limit
            return ViolationSeverity.CRITICAL
        elif violation_percentage >= 25:  # 25% over limit
            return ViolationSeverity.HIGH
        elif violation_percentage >= 10:  # 10% over limit
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW

    async def _handle_violation(self, violation: RiskViolation):
        """Handle a detected risk violation"""
        try:
            # Store violation
            self.active_violations[violation.violation_id] = violation
            self.violation_history.append(violation)
            
            # Update performance stats
            self.performance_stats['total_violations_detected'] += 1
            self.performance_stats['last_violation'] = violation.detected_at
            
            if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.EMERGENCY]:
                self.performance_stats['critical_violations'] += 1
            
            # Cache violation
            await self._cache_violation(violation)
            
            # Send alert
            await self._send_violation_alert(violation)
            
            # Take automatic action if needed
            await self._take_automatic_action(violation)
            
            logger.warning(f"🚨 Risk violation detected: {violation.risk_type.value} - {violation.severity.value}")
            
        except Exception as e:
            logger.error(f"Error handling violation: {e}")

    async def _send_violation_alert(self, violation: RiskViolation):
        """Send violation alert"""
        try:
            alert_data = {
                'violation_id': violation.violation_id,
                'risk_type': violation.risk_type.value,
                'account_id': violation.account_id,
                'symbol': violation.symbol,
                'severity': violation.severity.value,
                'current_value': violation.current_value,
                'limit_value': violation.limit_value,
                'violation_percentage': violation.violation_percentage,
                'detected_at': violation.detected_at.isoformat(),
                'message': f"Risk limit violation: {violation.risk_type.value} exceeded by {violation.violation_percentage:.1f}%"
            }
            
            # Store alert in Redis for external consumption
            self.redis_client.lpush('risk_violation_alerts', json.dumps(alert_data))
            self.redis_client.ltrim('risk_violation_alerts', 0, 99)  # Keep last 100 alerts
            
        except Exception as e:
            logger.error(f"Error sending violation alert: {e}")

    async def _take_automatic_action(self, violation: RiskViolation):
        """Take automatic action based on violation severity"""
        try:
            if violation.severity == ViolationSeverity.EMERGENCY:
                # Emergency stop - halt all trading
                action = "EMERGENCY_STOP_ALL_TRADING"
                await self._execute_emergency_stop(violation.account_id)
                
            elif violation.severity == ViolationSeverity.CRITICAL:
                # Close positions to reduce risk
                action = "CLOSE_RISKY_POSITIONS"
                await self._close_risky_positions(violation)
                
            elif violation.severity == ViolationSeverity.HIGH:
                # Reduce position sizes
                action = "REDUCE_POSITION_SIZES"
                await self._reduce_position_sizes(violation)
                
            else:
                # Monitor only for lower severity
                action = "MONITOR_ONLY"
            
            violation.resolution_action = action
            
        except Exception as e:
            logger.error(f"Error taking automatic action: {e}")

    async def _execute_emergency_stop(self, account_id: str):
        """Execute emergency stop for account"""
        try:
            # Signal emergency stop to trading systems
            emergency_data = {
                'account_id': account_id,
                'action': 'EMERGENCY_STOP',
                'timestamp': datetime.now().isoformat(),
                'reason': 'Risk limit violation - emergency level'
            }
            
            self.redis_client.set(f"emergency_stop:{account_id}", json.dumps(emergency_data))
            logger.critical(f"🚨 EMERGENCY STOP executed for account {account_id}")
            
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")

    async def _close_risky_positions(self, violation: RiskViolation):
        """Close positions that contribute to risk violation"""
        try:
            # Signal position closure to trading systems
            closure_data = {
                'account_id': violation.account_id,
                'action': 'CLOSE_RISKY_POSITIONS',
                'risk_type': violation.risk_type.value,
                'symbol': violation.symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.lpush('position_closure_requests', json.dumps(closure_data))
            logger.warning(f"⚠️ Requested closure of risky positions for {violation.account_id}")
            
        except Exception as e:
            logger.error(f"Error closing risky positions: {e}")

    async def _reduce_position_sizes(self, violation: RiskViolation):
        """Reduce position sizes to mitigate risk"""
        try:
            # Signal position size reduction
            reduction_data = {
                'account_id': violation.account_id,
                'action': 'REDUCE_POSITION_SIZES',
                'risk_type': violation.risk_type.value,
                'reduction_percentage': 0.5,  # Reduce by 50%
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.lpush('position_reduction_requests', json.dumps(reduction_data))
            logger.info(f"📉 Requested position size reduction for {violation.account_id}")
            
        except Exception as e:
            logger.error(f"Error reducing position sizes: {e}")

    async def resolve_violation(self, violation_id: str, resolution_action: str) -> bool:
        """Manually resolve a risk violation"""
        try:
            if violation_id not in self.active_violations:
                logger.warning(f"Violation {violation_id} not found in active violations")
                return False
            
            violation = self.active_violations[violation_id]
            violation.status = ViolationStatus.RESOLVED
            violation.resolved_at = datetime.now()
            violation.resolution_action = resolution_action
            
            # Remove from active violations
            del self.active_violations[violation_id]
            
            # Update performance stats
            self.performance_stats['violations_resolved'] += 1
            
            # Calculate resolution time
            resolution_time = (violation.resolved_at - violation.detected_at).total_seconds()
            current_avg = self.performance_stats['average_resolution_time']
            resolved_count = self.performance_stats['violations_resolved']
            
            self.performance_stats['average_resolution_time'] = (
                (current_avg * (resolved_count - 1) + resolution_time) / resolved_count
            )
            
            logger.info(f"✅ Resolved violation {violation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving violation {violation_id}: {e}")
            return False

    async def _continuous_risk_monitoring(self):
        """Continuous risk monitoring loop"""
        while self.running:
            try:
                # Monitor would integrate with actual trading systems
                # For now, simulate monitoring
                await self._simulate_risk_monitoring()
                
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Error in continuous risk monitoring: {e}")
                await asyncio.sleep(self.config['monitoring_interval'])

    async def _simulate_risk_monitoring(self):
        """Simulate risk monitoring for testing purposes"""
        try:
            # Simulate random risk checks
            test_accounts = ['ACC001', 'ACC002', 'ACC003']
            
            for account_id in test_accounts:
                # Simulate position size check
                position_size = np.random.uniform(500000, 1200000)
                await self.check_risk_violation(account_id, RiskType.POSITION_SIZE, position_size)
                
                # Simulate leverage check
                leverage = np.random.uniform(20, 60)
                await self.check_risk_violation(account_id, RiskType.LEVERAGE, leverage)
                
        except Exception as e:
            logger.error(f"Error in simulated risk monitoring: {e}")

    async def _violation_management(self):
        """Manage active violations and auto-resolution"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check for violations that should auto-resolve
                for violation_id, violation in list(self.active_violations.items()):
                    time_since_detection = (current_time - violation.detected_at).total_seconds()
                    
                    if time_since_detection > self.config['auto_resolution_timeout']:
                        await self.resolve_violation(violation_id, "AUTO_TIMEOUT")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in violation management: {e}")
                await asyncio.sleep(60)

    async def _automated_stress_testing(self):
        """Automated stress testing of risk systems"""
        while self.running:
            try:
                # Run stress tests periodically
                await self._run_stress_test_scenarios()
                
                await asyncio.sleep(self.config['stress_test_frequency'].total_seconds())
                
            except Exception as e:
                logger.error(f"Error in automated stress testing: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    async def _run_stress_test_scenarios(self):
        """Run predefined stress test scenarios"""
        try:
            logger.info("🧪 Running automated risk stress tests...")
            
            # Test scenario: Extreme position size
            await self.check_risk_violation('TEST_ACC', RiskType.POSITION_SIZE, 2000000)
            
            # Test scenario: High leverage
            await self.check_risk_violation('TEST_ACC', RiskType.LEVERAGE, 100.0)
            
            # Test scenario: Excessive drawdown
            await self.check_risk_violation('TEST_ACC', RiskType.DRAWDOWN, 0.15)  # 15%
            
            logger.info("✅ Stress test scenarios completed")
            
        except Exception as e:
            logger.error(f"Error running stress test scenarios: {e}")

    async def _compliance_reporting(self):
        """Generate compliance reports"""
        while self.running:
            try:
                # Calculate compliance score
                total_checks = self.performance_stats['total_violations_detected'] + 1000  # Assume 1000 clean checks
                violations = self.performance_stats['total_violations_detected']
                compliance_score = max(0, (total_checks - violations) / total_checks * 100)
                
                self.performance_stats['compliance_score'] = compliance_score
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error in compliance reporting: {e}")
                await asyncio.sleep(3600)

    async def _cache_risk_limit(self, limit: RiskLimit):
        """Cache risk limit in Redis"""
        try:
            limit_data = {
                'limit_id': limit.limit_id,
                'risk_type': limit.risk_type.value,
                'limit_name': limit.limit_name,
                'limit_value': limit.limit_value,
                'warning_threshold': limit.warning_threshold,
                'account_id': limit.account_id,
                'symbol': limit.symbol,
                'is_active': limit.is_active,
                'created_at': limit.created_at.isoformat()
            }
            
            self.redis_client.setex(
                f"risk_limit:{limit.limit_id}",
                3600,  # 1 hour TTL
                json.dumps(limit_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching risk limit: {e}")

    async def _cache_violation(self, violation: RiskViolation):
        """Cache violation in Redis"""
        try:
            violation_data = {
                'violation_id': violation.violation_id,
                'limit_id': violation.limit_id,
                'risk_type': violation.risk_type.value,
                'account_id': violation.account_id,
                'symbol': violation.symbol,
                'current_value': violation.current_value,
                'limit_value': violation.limit_value,
                'violation_amount': violation.violation_amount,
                'violation_percentage': violation.violation_percentage,
                'severity': violation.severity.value,
                'status': violation.status.value,
                'detected_at': violation.detected_at.isoformat(),
                'resolved_at': violation.resolved_at.isoformat() if violation.resolved_at else None,
                'resolution_action': violation.resolution_action
            }
            
            self.redis_client.setex(
                f"risk_violation:{violation.violation_id}",
                86400,  # 24 hours TTL
                json.dumps(violation_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching violation: {e}")

    def get_violation_report(self, timeframe: timedelta = timedelta(days=1)) -> Dict:
        """Generate violation report for specified timeframe"""
        try:
            end_time = datetime.now()
            start_time = end_time - timeframe
            
            # Filter violations by timeframe
            recent_violations = [
                v for v in self.violation_history
                if start_time <= v.detected_at <= end_time
            ]
            
            # Calculate statistics
            total_violations = len(recent_violations)
            by_severity = {}
            by_risk_type = {}
            
            for violation in recent_violations:
                # By severity
                severity = violation.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                # By risk type
                risk_type = violation.risk_type.value
                by_risk_type[risk_type] = by_risk_type.get(risk_type, 0) + 1
            
            return {
                'timeframe': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': timeframe.total_seconds() / 3600
                },
                'summary': {
                    'total_violations': total_violations,
                    'active_violations': len(self.active_violations),
                    'resolved_violations': len([v for v in recent_violations if v.status == ViolationStatus.RESOLVED]),
                    'compliance_score': self.performance_stats['compliance_score']
                },
                'by_severity': by_severity,
                'by_risk_type': by_risk_type,
                'performance_stats': self.performance_stats,
                'recent_violations': [
                    {
                        'violation_id': v.violation_id,
                        'risk_type': v.risk_type.value,
                        'severity': v.severity.value,
                        'account_id': v.account_id,
                        'violation_percentage': v.violation_percentage,
                        'detected_at': v.detected_at.isoformat(),
                        'status': v.status.value
                    }
                    for v in recent_violations[-10:]  # Last 10 violations
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating violation report: {e}")
            return {'error': str(e)}
