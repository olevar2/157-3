"""
Risk Limit Violation Monitoring & Alerting System
Comprehensive monitoring for risk limit violations and compliance

Features:
- Real-time risk limit monitoring
- Violation detection and alerting
- Compliance tracking and reporting
- Risk breach escalation
- Automated remediation triggers
- Regulatory compliance monitoring
- Risk audit trail maintenance
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViolationType(Enum):
    POSITION_SIZE_LIMIT = "position_size_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    MARGIN_LIMIT = "margin_limit"
    CORRELATION_LIMIT = "correlation_limit"
    VAR_LIMIT = "var_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    FREQUENCY_LIMIT = "frequency_limit"

class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    BREACH = "breach"
    EMERGENCY = "emergency"

@dataclass
class RiskLimit:
    limit_id: str
    limit_type: ViolationType
    limit_value: float
    warning_threshold: float  # % of limit that triggers warning
    description: str
    applicable_accounts: List[str]
    enforcement_level: str  # 'soft', 'hard', 'emergency'
    created_date: datetime
    last_updated: datetime

@dataclass
class RiskViolation:
    violation_id: str
    account_id: str
    violation_type: ViolationType
    limit_id: str
    current_value: float
    limit_value: float
    severity: ViolationSeverity
    violation_percentage: float
    timestamp: datetime
    duration: timedelta
    description: str
    auto_remediated: bool
    remediation_action: Optional[str] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class ComplianceMetrics:
    account_id: str
    compliance_status: ComplianceStatus
    total_violations: int
    active_violations: int
    violation_rate: float  # violations per day
    avg_violation_duration: timedelta
    most_frequent_violation: ViolationType
    compliance_score: float  # 0-100
    last_violation: Optional[datetime]
    days_since_violation: int
    risk_score: float

class RiskViolationMonitor:
    """
    Comprehensive risk violation monitoring and alerting system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Risk limits configuration
        self.risk_limits = {
            ViolationType.POSITION_SIZE_LIMIT: RiskLimit(
                limit_id="PSL001",
                limit_type=ViolationType.POSITION_SIZE_LIMIT,
                limit_value=10.0,  # 10 lots max
                warning_threshold=0.8,  # 80% warning
                description="Maximum position size per trade",
                applicable_accounts=["all"],
                enforcement_level="hard",
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            ViolationType.DAILY_LOSS_LIMIT: RiskLimit(
                limit_id="DLL001",
                limit_type=ViolationType.DAILY_LOSS_LIMIT,
                limit_value=5000.0,  # $5000 max daily loss
                warning_threshold=0.8,
                description="Maximum daily loss limit",
                applicable_accounts=["all"],
                enforcement_level="emergency",
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            ViolationType.DRAWDOWN_LIMIT: RiskLimit(
                limit_id="DDL001",
                limit_type=ViolationType.DRAWDOWN_LIMIT,
                limit_value=0.10,  # 10% max drawdown
                warning_threshold=0.7,
                description="Maximum portfolio drawdown",
                applicable_accounts=["all"],
                enforcement_level="hard",
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            ViolationType.EXPOSURE_LIMIT: RiskLimit(
                limit_id="EXL001",
                limit_type=ViolationType.EXPOSURE_LIMIT,
                limit_value=100000.0,  # $100K max exposure
                warning_threshold=0.8,
                description="Maximum portfolio exposure",
                applicable_accounts=["all"],
                enforcement_level="hard",
                created_date=datetime.now(),
                last_updated=datetime.now()
            ),
            ViolationType.VAR_LIMIT: RiskLimit(
                limit_id="VAR001",
                limit_type=ViolationType.VAR_LIMIT,
                limit_value=2500.0,  # $2500 VaR limit
                warning_threshold=0.8,
                description="Value at Risk 95% limit",
                applicable_accounts=["all"],
                enforcement_level="hard",
                created_date=datetime.now(),
                last_updated=datetime.now()
            )
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            'check_interval': 10,  # seconds
            'violation_retention_days': 30,
            'alert_cooldown': timedelta(minutes=5),
            'escalation_threshold': 3,  # violations before escalation
            'auto_remediation_enabled': True,
            'compliance_reporting_interval': timedelta(hours=24)
        }
        
        # State tracking
        self.active_violations = {}
        self.violation_history = []
        self.compliance_metrics = {}
        self.alert_history = []
        self.running = False
        
        # Performance statistics
        self.performance_stats = {
            'total_violations_detected': 0,
            'violations_auto_remediated': 0,
            'average_violation_duration': 0.0,
            'compliance_rate': 100.0,
            'most_violated_limit': None,
            'accounts_monitored': 0,
            'last_compliance_check': None
        }
        
        logger.info("RiskViolationMonitor initialized")

    async def start_monitoring(self):
        """Start risk violation monitoring"""
        self.running = True
        logger.info("🚀 Starting risk violation monitoring...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_risk_violations()),
            asyncio.create_task(self._compliance_reporting()),
            asyncio.create_task(self._violation_escalation()),
            asyncio.create_task(self._cleanup_old_violations())
        ]
        
        await asyncio.gather(*tasks)

    async def stop_monitoring(self):
        """Stop risk violation monitoring"""
        self.running = False
        logger.info("⏹️ Stopping risk violation monitoring...")

    async def check_risk_limits(self, account_id: str, risk_metrics: Dict) -> List[RiskViolation]:
        """Check all risk limits for an account and detect violations"""
        violations = []
        
        try:
            for violation_type, limit in self.risk_limits.items():
                # Skip if limit doesn't apply to this account
                if "all" not in limit.applicable_accounts and account_id not in limit.applicable_accounts:
                    continue
                
                # Get current value for this risk metric
                current_value = await self._get_current_value(violation_type, risk_metrics)
                
                if current_value is None:
                    continue
                
                # Check for violation
                violation = await self._check_limit_violation(
                    account_id, violation_type, limit, current_value
                )
                
                if violation:
                    violations.append(violation)
                    await self._handle_violation(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"❌ Error checking risk limits for {account_id}: {e}")
            return []

    async def _get_current_value(self, violation_type: ViolationType, risk_metrics: Dict) -> Optional[float]:
        """Extract current value for violation type from risk metrics"""
        value_mapping = {
            ViolationType.POSITION_SIZE_LIMIT: risk_metrics.get('max_position_size', 0),
            ViolationType.DAILY_LOSS_LIMIT: abs(risk_metrics.get('daily_pnl', 0)) if risk_metrics.get('daily_pnl', 0) < 0 else 0,
            ViolationType.DRAWDOWN_LIMIT: risk_metrics.get('current_drawdown', 0),
            ViolationType.EXPOSURE_LIMIT: risk_metrics.get('total_exposure', 0),
            ViolationType.MARGIN_LIMIT: risk_metrics.get('margin_utilization', 0),
            ViolationType.CORRELATION_LIMIT: risk_metrics.get('correlation_risk', 0),
            ViolationType.VAR_LIMIT: risk_metrics.get('var_95', 0),
            ViolationType.CONCENTRATION_LIMIT: risk_metrics.get('concentration_risk', 0),
            ViolationType.LEVERAGE_LIMIT: risk_metrics.get('leverage', 0)
        }
        
        return value_mapping.get(violation_type)

    async def _check_limit_violation(self, account_id: str, violation_type: ViolationType,
                                   limit: RiskLimit, current_value: float) -> Optional[RiskViolation]:
        """Check if current value violates the limit"""
        try:
            # Check if limit is exceeded
            if current_value <= limit.limit_value:
                # Check if existing violation should be resolved
                await self._resolve_violation_if_exists(account_id, violation_type)
                return None
            
            # Calculate violation percentage
            violation_percentage = (current_value - limit.limit_value) / limit.limit_value
            
            # Determine severity
            severity = self._determine_violation_severity(violation_percentage, limit)
            
            # Check if this is a new violation or update existing
            existing_violation = await self._get_existing_violation(account_id, violation_type)
            
            if existing_violation:
                # Update existing violation
                existing_violation.current_value = current_value
                existing_violation.violation_percentage = violation_percentage
                existing_violation.severity = severity
                existing_violation.duration = datetime.now() - existing_violation.timestamp
                return existing_violation
            else:
                # Create new violation
                violation = RiskViolation(
                    violation_id=f"VIO_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    account_id=account_id,
                    violation_type=violation_type,
                    limit_id=limit.limit_id,
                    current_value=current_value,
                    limit_value=limit.limit_value,
                    severity=severity,
                    violation_percentage=violation_percentage,
                    timestamp=datetime.now(),
                    duration=timedelta(0),
                    description=f"{limit.description} exceeded: {current_value:.2f} > {limit.limit_value:.2f}",
                    auto_remediated=False
                )
                
                return violation
                
        except Exception as e:
            logger.error(f"Error checking limit violation: {e}")
            return None

    def _determine_violation_severity(self, violation_percentage: float, limit: RiskLimit) -> ViolationSeverity:
        """Determine violation severity based on percentage and limit type"""
        if limit.enforcement_level == "emergency":
            return ViolationSeverity.EMERGENCY
        elif violation_percentage > 0.5:  # 50% over limit
            return ViolationSeverity.CRITICAL
        elif violation_percentage > 0.25:  # 25% over limit
            return ViolationSeverity.HIGH
        elif violation_percentage > 0.1:   # 10% over limit
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW

    async def _handle_violation(self, violation: RiskViolation):
        """Handle detected violation"""
        try:
            # Store violation
            await self._store_violation(violation)
            
            # Add to active violations
            key = f"{violation.account_id}_{violation.violation_type.value}"
            self.active_violations[key] = violation
            
            # Generate alert
            await self._generate_violation_alert(violation)
            
            # Trigger auto-remediation if enabled and appropriate
            if (self.monitoring_config['auto_remediation_enabled'] and 
                violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.EMERGENCY]):
                await self._trigger_auto_remediation(violation)
            
            # Update statistics
            self.performance_stats['total_violations_detected'] += 1
            
            logger.warning(f"🚨 Risk violation detected: {violation.description}")
            
        except Exception as e:
            logger.error(f"Error handling violation: {e}")

    async def _trigger_auto_remediation(self, violation: RiskViolation):
        """Trigger automatic remediation for violation"""
        try:
            remediation_actions = {
                ViolationType.POSITION_SIZE_LIMIT: "reduce_position_size",
                ViolationType.DAILY_LOSS_LIMIT: "halt_trading",
                ViolationType.DRAWDOWN_LIMIT: "emergency_stop",
                ViolationType.EXPOSURE_LIMIT: "reduce_exposure",
                ViolationType.MARGIN_LIMIT: "close_positions",
                ViolationType.VAR_LIMIT: "reduce_risk"
            }
            
            action = remediation_actions.get(violation.violation_type)
            
            if action:
                # Send remediation command
                remediation_command = {
                    'action': action,
                    'account_id': violation.account_id,
                    'violation_id': violation.violation_id,
                    'severity': violation.severity.value,
                    'timestamp': datetime.now().isoformat(),
                    'reason': f"Auto-remediation for {violation.violation_type.value}"
                }
                
                await self.redis_client.lpush('remediation_commands', json.dumps(remediation_command))
                
                # Update violation
                violation.auto_remediated = True
                violation.remediation_action = action
                
                self.performance_stats['violations_auto_remediated'] += 1
                
                logger.info(f"🔧 Auto-remediation triggered: {action} for {violation.violation_id}")
                
        except Exception as e:
            logger.error(f"Error triggering auto-remediation: {e}")

    async def _monitor_risk_violations(self):
        """Continuous risk violation monitoring"""
        while self.running:
            try:
                # Get all active accounts
                active_accounts = await self._get_active_accounts()
                
                for account_id in active_accounts:
                    # Get current risk metrics
                    risk_metrics = await self._get_account_risk_metrics(account_id)
                    
                    if risk_metrics:
                        # Check for violations
                        violations = await self.check_risk_limits(account_id, risk_metrics)
                        
                        # Update compliance metrics
                        await self._update_compliance_metrics(account_id, violations)
                
                self.performance_stats['last_compliance_check'] = datetime.now()
                
                await asyncio.sleep(self.monitoring_config['check_interval'])
                
            except Exception as e:
                logger.error(f"Error in violation monitoring: {e}")
                await asyncio.sleep(self.monitoring_config['check_interval'])

    async def _compliance_reporting(self):
        """Generate compliance reports"""
        while self.running:
            try:
                # Generate compliance report for all accounts
                compliance_report = await self.generate_compliance_report()
                
                # Store report
                await self._store_compliance_report(compliance_report)
                
                # Check overall compliance status
                await self._check_overall_compliance(compliance_report)
                
                await asyncio.sleep(self.monitoring_config['compliance_reporting_interval'].total_seconds())
                
            except Exception as e:
                logger.error(f"Error in compliance reporting: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    async def generate_compliance_report(self, timeframe: timedelta = timedelta(days=1)) -> Dict:
        """Generate comprehensive compliance report"""
        try:
            end_time = datetime.now()
            start_time = end_time - timeframe
            
            # Get violations in timeframe
            violations_in_period = [
                v for v in self.violation_history 
                if start_time <= v.timestamp <= end_time
            ]
            
            # Calculate overall metrics
            total_violations = len(violations_in_period)
            active_violations = len(self.active_violations)
            
            # Violation breakdown by type
            violation_breakdown = {}
            for violation_type in ViolationType:
                count = len([v for v in violations_in_period if v.violation_type == violation_type])
                violation_breakdown[violation_type.value] = count
            
            # Account compliance status
            account_compliance = {}
            for account_id in await self._get_active_accounts():
                metrics = self.compliance_metrics.get(account_id)
                if metrics:
                    account_compliance[account_id] = {
                        'compliance_status': metrics.compliance_status.value,
                        'compliance_score': metrics.compliance_score,
                        'active_violations': metrics.active_violations,
                        'days_since_violation': metrics.days_since_violation
                    }
            
            # Calculate compliance rate
            compliant_accounts = len([
                m for m in self.compliance_metrics.values() 
                if m.compliance_status == ComplianceStatus.COMPLIANT
            ])
            total_accounts = len(self.compliance_metrics)
            compliance_rate = (compliant_accounts / total_accounts * 100) if total_accounts > 0 else 100
            
            self.performance_stats['compliance_rate'] = compliance_rate
            
            return {
                'report_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': timeframe.total_seconds() / 3600
                },
                'overall_compliance': {
                    'compliance_rate': compliance_rate,
                    'total_violations': total_violations,
                    'active_violations': active_violations,
                    'accounts_monitored': total_accounts,
                    'auto_remediation_rate': (
                        self.performance_stats['violations_auto_remediated'] / 
                        max(self.performance_stats['total_violations_detected'], 1) * 100
                    )
                },
                'violation_breakdown': violation_breakdown,
                'account_compliance': account_compliance,
                'risk_limits': {
                    limit.limit_id: {
                        'type': limit.limit_type.value,
                        'limit_value': limit.limit_value,
                        'description': limit.description,
                        'enforcement_level': limit.enforcement_level
                    }
                    for limit in self.risk_limits.values()
                },
                'performance_stats': self.performance_stats
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {'error': str(e)}

    async def get_violation_status(self, account_id: Optional[str] = None) -> Dict:
        """Get current violation status"""
        try:
            if account_id:
                # Get violations for specific account
                account_violations = {
                    k: v for k, v in self.active_violations.items() 
                    if v.account_id == account_id
                }
                compliance_metrics = self.compliance_metrics.get(account_id)
                
                return {
                    'account_id': account_id,
                    'active_violations': len(account_violations),
                    'violations': [
                        {
                            'violation_id': v.violation_id,
                            'type': v.violation_type.value,
                            'severity': v.severity.value,
                            'current_value': v.current_value,
                            'limit_value': v.limit_value,
                            'duration': str(v.duration),
                            'auto_remediated': v.auto_remediated
                        }
                        for v in account_violations.values()
                    ],
                    'compliance_metrics': {
                        'compliance_status': compliance_metrics.compliance_status.value if compliance_metrics else 'unknown',
                        'compliance_score': compliance_metrics.compliance_score if compliance_metrics else 0,
                        'days_since_violation': compliance_metrics.days_since_violation if compliance_metrics else 0
                    } if compliance_metrics else None
                }
            else:
                # Get overall violation status
                return {
                    'total_active_violations': len(self.active_violations),
                    'violations_by_severity': {
                        severity.value: len([
                            v for v in self.active_violations.values() 
                            if v.severity == severity
                        ])
                        for severity in ViolationSeverity
                    },
                    'violations_by_type': {
                        violation_type.value: len([
                            v for v in self.active_violations.values() 
                            if v.violation_type == violation_type
                        ])
                        for violation_type in ViolationType
                    },
                    'performance_stats': self.performance_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting violation status: {e}")
            return {'error': str(e)}

    # Additional helper methods would be implemented here for:
    # - _get_existing_violation()
    # - _resolve_violation_if_exists()
    # - _store_violation()
    # - _generate_violation_alert()
    # - _get_active_accounts()
    # - _get_account_risk_metrics()
    # - _update_compliance_metrics()
    # - _store_compliance_report()
    # - _check_overall_compliance()
    # - _violation_escalation()
    # - _cleanup_old_violations()
