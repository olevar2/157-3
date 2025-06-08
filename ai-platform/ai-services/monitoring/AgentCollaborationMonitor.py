"""
🏥 HUMANITARIAN AI PLATFORM - AGENT COLLABORATION MONITOR
💝 Real-time monitoring for agent communication and coordination

This system provides comprehensive monitoring of agent collaboration performance,
communication metrics, and coordination effectiveness for the Platform3 humanitarian
trading mission.

Key Features:
- Real-time agent communication latency tracking
- Dependency resolution performance monitoring  
- Agent-to-agent message flow analysis
- Coordination effectiveness metrics
- Humanitarian mission impact tracking
- Performance trend analysis and forecasting
- Intelligent alerting for coordination failures
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import threading
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of collaboration metrics"""
    COMMUNICATION_LATENCY = "communication_latency"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    MESSAGE_THROUGHPUT = "message_throughput"
    ERROR_RATE = "error_rate"
    COORDINATION_SUCCESS = "coordination_success"
    AGENT_AVAILABILITY = "agent_availability"
    RESOURCE_UTILIZATION = "resource_utilization"

class AlertLevel(Enum):
    """Alert severity levels for collaboration monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class CollaborationMetric:
    """Individual collaboration metric data point"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    agent_id: str
    target_agent_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentCollaborationSummary:
    """Summary of agent collaboration performance"""
    agent_id: str
    total_messages_sent: int
    total_messages_received: int
    average_response_time_ms: float
    error_rate_percent: float
    dependency_resolution_success_rate: float
    coordination_score: float
    last_activity: datetime
    status: str  # "active", "degraded", "offline"

@dataclass
class SystemCollaborationMetrics:
    """System-wide collaboration metrics"""
    total_active_agents: int
    total_messages_per_second: float
    average_coordination_latency_ms: float
    system_error_rate_percent: float
    coordination_success_rate_percent: float
    humanitarian_impact_score: float
    performance_trend: str  # "improving", "stable", "degrading"

class AgentCollaborationMonitor:
    """
    🏥 Agent Collaboration Monitor for Humanitarian AI Platform
    
    Provides comprehensive real-time monitoring of agent collaboration:
    - Communication latency and throughput metrics
    - Dependency resolution performance tracking
    - Agent coordination effectiveness monitoring
    - Humanitarian mission impact assessment
    - Performance trend analysis with forecasting
    - Intelligent alerting for coordination issues
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.agent_summaries = {}
        self.system_metrics = SystemCollaborationMetrics(
            total_active_agents=0,
            total_messages_per_second=0.0,
            average_coordination_latency_ms=0.0,
            system_error_rate_percent=0.0,
            coordination_success_rate_percent=0.0,
            humanitarian_impact_score=0.0,
            performance_trend="stable"
        )
        
        # Communication tracking
        self.agent_communications = defaultdict(lambda: defaultdict(list))  # from_agent -> to_agent -> [latencies]
        self.dependency_resolution_times = defaultdict(list)  # agent_id -> [resolution_times]
        self.message_throughput = defaultdict(lambda: deque(maxlen=1000))  # agent_id -> [timestamps]
        self.error_counts = defaultdict(int)  # agent_id -> error_count
        self.coordination_successes = defaultdict(int)  # agent_id -> success_count
        
        # Performance thresholds
        self.thresholds = {
            "communication_latency_warning_ms": 1000.0,
            "communication_latency_critical_ms": 5000.0,
            "dependency_resolution_warning_ms": 2000.0,
            "dependency_resolution_critical_ms": 10000.0,
            "error_rate_warning_percent": 2.0,
            "error_rate_critical_percent": 5.0,
            "coordination_success_warning_percent": 95.0,
            "coordination_success_critical_percent": 90.0
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.last_cleanup = time.time()
        
        logger.info("🏥 Agent Collaboration Monitor initialized")
        logger.info("💝 Monitoring agent coordination for humanitarian mission")
    
    def record_communication(self, from_agent: str, to_agent: str, latency_ms: float, success: bool = True, metadata: Dict[str, Any] = None):
        """Record agent-to-agent communication metrics"""
        timestamp = datetime.now()
        
        # Record communication latency
        metric = CollaborationMetric(
            metric_type=MetricType.COMMUNICATION_LATENCY,
            value=latency_ms,
            timestamp=timestamp,
            agent_id=from_agent,
            target_agent_id=to_agent,
            metadata=metadata
        )
        self.metrics_history[MetricType.COMMUNICATION_LATENCY].append(metric)
        
        # Update agent communication tracking
        self.agent_communications[from_agent][to_agent].append(latency_ms)
        if len(self.agent_communications[from_agent][to_agent]) > 100:
            self.agent_communications[from_agent][to_agent] = self.agent_communications[from_agent][to_agent][-100:]
        
        # Record message throughput
        self.message_throughput[from_agent].append(time.time())
        self.message_throughput[to_agent].append(time.time())
        
        # Update success/error tracking
        if success:
            self.coordination_successes[from_agent] += 1
        else:
            self.error_counts[from_agent] += 1
            
        # Check for alerts
        self._check_communication_alerts(from_agent, to_agent, latency_ms, success)
    
    def record_dependency_resolution(self, agent_id: str, dependency_agent: str, resolution_time_ms: float, success: bool = True, metadata: Dict[str, Any] = None):
        """Record dependency resolution performance"""
        timestamp = datetime.now()
        
        # Record resolution time
        metric = CollaborationMetric(
            metric_type=MetricType.DEPENDENCY_RESOLUTION,
            value=resolution_time_ms,
            timestamp=timestamp,
            agent_id=agent_id,
            target_agent_id=dependency_agent,
            metadata=metadata
        )
        self.metrics_history[MetricType.DEPENDENCY_RESOLUTION].append(metric)
        
        # Update dependency tracking
        self.dependency_resolution_times[agent_id].append(resolution_time_ms)
        if len(self.dependency_resolution_times[agent_id]) > 100:
            self.dependency_resolution_times[agent_id] = self.dependency_resolution_times[agent_id][-100:]
        
        # Update success tracking
        if success:
            self.coordination_successes[agent_id] += 1
        else:
            self.error_counts[agent_id] += 1
            
        # Check for alerts
        self._check_dependency_alerts(agent_id, dependency_agent, resolution_time_ms, success)
    
    def record_coordination_event(self, agent_id: str, event_type: str, success: bool, duration_ms: float = None, metadata: Dict[str, Any] = None):
        """Record general coordination events"""
        timestamp = datetime.now()
        
        # Record coordination success metric
        coordination_score = 1.0 if success else 0.0
        metric = CollaborationMetric(
            metric_type=MetricType.COORDINATION_SUCCESS,
            value=coordination_score,
            timestamp=timestamp,
            agent_id=agent_id,
            metadata={
                "event_type": event_type,
                "duration_ms": duration_ms,
                **(metadata or {})
            }
        )
        self.metrics_history[MetricType.COORDINATION_SUCCESS].append(metric)
        
        # Update tracking
        if success:
            self.coordination_successes[agent_id] += 1
        else:
            self.error_counts[agent_id] += 1
    
    def get_agent_collaboration_summary(self, agent_id: str) -> AgentCollaborationSummary:
        """Get collaboration summary for a specific agent"""
        now = time.time()
        hour_ago = now - 3600  # 1 hour ago
        
        # Calculate message counts
        messages_sent = len([t for t in self.message_throughput[agent_id] if t > hour_ago])
        
        # Calculate average response time
        latencies = []
        for target_agent, latency_list in self.agent_communications[agent_id].items():
            latencies.extend(latency_list)
        avg_response_time = np.mean(latencies) if latencies else 0.0
        
        # Calculate error rate
        total_operations = self.coordination_successes[agent_id] + self.error_counts[agent_id]
        error_rate = (self.error_counts[agent_id] / total_operations * 100) if total_operations > 0 else 0.0
        
        # Calculate dependency resolution success rate
        resolution_times = self.dependency_resolution_times[agent_id]
        dependency_success_rate = 100.0 if resolution_times else 0.0  # Assume success if resolved
        
        # Calculate coordination score (composite metric)
        coordination_score = self._calculate_coordination_score(agent_id)
        
        # Determine status
        if messages_sent == 0:
            status = "offline"
        elif error_rate > self.thresholds["error_rate_critical_percent"]:
            status = "degraded"
        else:
            status = "active"
        
        return AgentCollaborationSummary(
            agent_id=agent_id,
            total_messages_sent=messages_sent,
            total_messages_received=len([t for t in self.message_throughput.get(f"{agent_id}_received", []) if t > hour_ago]),
            average_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            dependency_resolution_success_rate=dependency_success_rate,
            coordination_score=coordination_score,
            last_activity=datetime.fromtimestamp(max(self.message_throughput[agent_id]) if self.message_throughput[agent_id] else 0),
            status=status
        )
    
    def get_system_collaboration_metrics(self) -> SystemCollaborationMetrics:
        """Get system-wide collaboration metrics"""
        now = time.time()
        hour_ago = now - 3600
        
        # Calculate active agents
        active_agents = len([agent for agent, timestamps in self.message_throughput.items() 
                           if timestamps and max(timestamps) > hour_ago])
        
        # Calculate messages per second
        total_messages = sum(len([t for t in timestamps if t > hour_ago]) 
                           for timestamps in self.message_throughput.values())
        messages_per_second = total_messages / 3600  # Messages in last hour / 3600 seconds
        
        # Calculate average coordination latency
        all_latencies = []
        for agent_comms in self.agent_communications.values():
            for latency_list in agent_comms.values():
                all_latencies.extend(latency_list)
        avg_latency = np.mean(all_latencies) if all_latencies else 0.0
        
        # Calculate system error rate
        total_successes = sum(self.coordination_successes.values())
        total_errors = sum(self.error_counts.values())
        total_operations = total_successes + total_errors
        system_error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0.0
        
        # Calculate coordination success rate
        coordination_success_rate = (total_successes / total_operations * 100) if total_operations > 0 else 100.0
        
        # Calculate humanitarian impact score
        humanitarian_impact_score = self._calculate_humanitarian_impact_score()
        
        # Determine performance trend
        performance_trend = self._calculate_performance_trend()
        
        self.system_metrics = SystemCollaborationMetrics(
            total_active_agents=active_agents,
            total_messages_per_second=messages_per_second,
            average_coordination_latency_ms=avg_latency,
            system_error_rate_percent=system_error_rate,
            coordination_success_rate_percent=coordination_success_rate,
            humanitarian_impact_score=humanitarian_impact_score,
            performance_trend=performance_trend
        )
        
        return self.system_metrics
    
    def _calculate_coordination_score(self, agent_id: str) -> float:
        """Calculate composite coordination score for an agent"""
        # Factors: response time, error rate, dependency resolution, availability
        
        # Response time score (0-100, lower latency = higher score)
        latencies = []
        for target_agent, latency_list in self.agent_communications[agent_id].items():
            latencies.extend(latency_list)
        avg_latency = np.mean(latencies) if latencies else 1000.0
        response_score = max(0, 100 - (avg_latency / 50))  # 50ms = perfect score
        
        # Error rate score (0-100, lower error rate = higher score)
        total_operations = self.coordination_successes[agent_id] + self.error_counts[agent_id]
        error_rate = (self.error_counts[agent_id] / total_operations) if total_operations > 0 else 0.0
        error_score = max(0, 100 - (error_rate * 100))
        
        # Dependency resolution score
        resolution_times = self.dependency_resolution_times[agent_id]
        avg_resolution = np.mean(resolution_times) if resolution_times else 2000.0
        resolution_score = max(0, 100 - (avg_resolution / 100))  # 100ms = perfect score
        
        # Availability score
        now = time.time()
        recent_activity = len([t for t in self.message_throughput[agent_id] if t > (now - 300)])  # Last 5 minutes
        availability_score = min(100, recent_activity * 10)  # 10+ messages in 5 min = perfect
        
        # Weighted composite score
        coordination_score = (
            response_score * 0.3 +
            error_score * 0.3 +
            resolution_score * 0.2 +
            availability_score * 0.2
        )
        
        return coordination_score
    
    def _calculate_humanitarian_impact_score(self) -> float:
        """Calculate humanitarian mission impact score based on coordination efficiency"""
        # Higher coordination efficiency -> better trading performance -> more humanitarian impact
        
        # Calculate metrics directly to avoid recursion
        total_successes = sum(self.coordination_successes.values())
        total_errors = sum(self.error_counts.values())
        total_operations = total_successes + total_errors
        
        # Base score from coordination success rate
        coordination_success_rate = (total_successes / total_operations * 100) if total_operations > 0 else 100.0
        base_score = coordination_success_rate
        
        # Calculate average latency directly
        all_latencies = []
        for agent_comms in self.agent_communications.values():
            for latency_list in agent_comms.values():
                all_latencies.extend(latency_list)
        avg_latency = np.mean(all_latencies) if all_latencies else 1000.0
        
        # Calculate messages per second directly
        now = time.time()
        hour_ago = now - 3600
        total_messages = sum(len([t for t in timestamps if t > hour_ago]) 
                           for timestamps in self.message_throughput.values())
        messages_per_second = total_messages / 3600
        
        # Calculate error rate directly
        system_error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0.0
        
        # Bonus for low latency (faster decisions = better trading opportunities)
        latency_bonus = max(0, 20 - (avg_latency / 100))  # Up to 20 points
        
        # Bonus for high message throughput (more active coordination)
        throughput_bonus = min(15, messages_per_second * 3)  # Up to 15 points
        
        # Penalty for high error rate
        error_penalty = system_error_rate * 2  # 2x penalty for errors
        
        humanitarian_score = base_score + latency_bonus + throughput_bonus - error_penalty
        return max(0, min(100, humanitarian_score))
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend based on recent metrics"""
        # Get recent coordination success metrics (last 30 data points)
        recent_success_metrics = list(self.metrics_history[MetricType.COORDINATION_SUCCESS])[-30:]
        
        if len(recent_success_metrics) < 10:
            return "stable"
        
        # Calculate trend in success rate
        success_values = [metric.value for metric in recent_success_metrics]
        time_points = np.arange(len(success_values))
        
        # Simple linear regression to find trend
        if len(success_values) > 1:
            trend_slope = np.polyfit(time_points, success_values, 1)[0]
            
            if trend_slope > 0.01:  # Improving
                return "improving"
            elif trend_slope < -0.01:  # Degrading
                return "degrading"
            else:
                return "stable"
        
        return "stable"
    
    def _check_communication_alerts(self, from_agent: str, to_agent: str, latency_ms: float, success: bool):
        """Check for communication-related alerts"""
        if not success:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Communication Failed: {from_agent} -> {to_agent}",
                f"Communication from {from_agent} to {to_agent} failed",
                {"from_agent": from_agent, "to_agent": to_agent, "latency_ms": latency_ms}
            )
        elif latency_ms > self.thresholds["communication_latency_critical_ms"]:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"High Communication Latency: {from_agent} -> {to_agent}",
                f"Communication latency {latency_ms:.1f}ms exceeds critical threshold",
                {"from_agent": from_agent, "to_agent": to_agent, "latency_ms": latency_ms}
            )
        elif latency_ms > self.thresholds["communication_latency_warning_ms"]:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Elevated Communication Latency: {from_agent} -> {to_agent}",
                f"Communication latency {latency_ms:.1f}ms exceeds warning threshold",
                {"from_agent": from_agent, "to_agent": to_agent, "latency_ms": latency_ms}
            )
    
    def _check_dependency_alerts(self, agent_id: str, dependency_agent: str, resolution_time_ms: float, success: bool):
        """Check for dependency resolution alerts"""
        if not success:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Dependency Resolution Failed: {agent_id}",
                f"Failed to resolve dependency on {dependency_agent}",
                {"agent_id": agent_id, "dependency_agent": dependency_agent, "resolution_time_ms": resolution_time_ms}
            )
        elif resolution_time_ms > self.thresholds["dependency_resolution_critical_ms"]:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Slow Dependency Resolution: {agent_id}",
                f"Dependency resolution took {resolution_time_ms:.1f}ms (critical threshold exceeded)",
                {"agent_id": agent_id, "dependency_agent": dependency_agent, "resolution_time_ms": resolution_time_ms}
            )
        elif resolution_time_ms > self.thresholds["dependency_resolution_warning_ms"]:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Slow Dependency Resolution: {agent_id}",
                f"Dependency resolution took {resolution_time_ms:.1f}ms (warning threshold exceeded)",
                {"agent_id": agent_id, "dependency_agent": dependency_agent, "resolution_time_ms": resolution_time_ms}
            )
    
    def _trigger_alert(self, level: AlertLevel, title: str, message: str, metadata: Dict[str, Any]):
        """Trigger an alert and notify callbacks"""
        alert_data = {
            "level": level.value,
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        logger.warning(f"🚨 Collaboration Alert ({level.value}): {title}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                asyncio.create_task(callback(alert_data))
            except Exception as e:
                logger.error(f"❌ Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for collaboration alerts"""
        self.alert_callbacks.append(callback)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for monitoring UI"""
        system_metrics = self.get_system_collaboration_metrics()
        
        # Get agent summaries
        agent_summaries = {}
        for agent_id in set(list(self.message_throughput.keys()) + list(self.agent_communications.keys())):
            if not agent_id.endswith("_received"):  # Skip internal tracking keys
                agent_summaries[agent_id] = asdict(self.get_agent_collaboration_summary(agent_id))
        
        # Get recent metrics for charts
        recent_metrics = {}
        for metric_type in MetricType:
            recent_data = list(self.metrics_history[metric_type])[-50:]  # Last 50 data points
            recent_metrics[metric_type.value] = [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value,
                    "agent_id": metric.agent_id,
                    "target_agent_id": metric.target_agent_id
                }
                for metric in recent_data
            ]
        
        # Calculate communication matrix
        communication_matrix = {}
        for from_agent, targets in self.agent_communications.items():
            communication_matrix[from_agent] = {}
            for to_agent, latencies in targets.items():
                communication_matrix[from_agent][to_agent] = {
                    "average_latency_ms": np.mean(latencies) if latencies else 0,
                    "message_count": len(latencies),
                    "max_latency_ms": np.max(latencies) if latencies else 0,
                    "min_latency_ms": np.min(latencies) if latencies else 0
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(system_metrics),
            "agent_summaries": agent_summaries,
            "recent_metrics": recent_metrics,
            "communication_matrix": communication_matrix,
            "thresholds": self.thresholds,
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "humanitarian_mission": {
                "status": "active",
                "coordination_impact": system_metrics.humanitarian_impact_score,
                "performance_trend": system_metrics.performance_trend,
                "mission_health": "excellent" if system_metrics.humanitarian_impact_score > 80 else "good" if system_metrics.humanitarian_impact_score > 60 else "needs_attention"
            }
        }
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.monitoring_active:
            logger.warning("⚠️ Collaboration monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("🚀 Agent collaboration monitoring started")
        logger.info("💝 Monitoring coordination for humanitarian mission success")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Agent collaboration monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for cleanup and analysis"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Periodic cleanup (every 30 minutes)
                if current_time - self.last_cleanup > 1800:
                    self._cleanup_old_metrics()
                    self.last_cleanup = current_time
                
                # Update system metrics
                self.get_system_collaboration_metrics()
                
                # Check for system-wide alerts
                await self._check_system_alerts()
                
                # Wait for next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Collaboration monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of data
        
        for metric_type, metrics in self.metrics_history.items():
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
        
        logger.info("🧹 Cleaned up old collaboration metrics")
    
    async def _check_system_alerts(self):
        """Check for system-wide collaboration alerts"""
        system_metrics = self.get_system_collaboration_metrics()
        
        # Check coordination success rate
        if system_metrics.coordination_success_rate_percent < self.thresholds["coordination_success_critical_percent"]:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                "Low System Coordination Success Rate",
                f"System coordination success rate {system_metrics.coordination_success_rate_percent:.1f}% below critical threshold",
                {"success_rate": system_metrics.coordination_success_rate_percent}
            )
        elif system_metrics.coordination_success_rate_percent < self.thresholds["coordination_success_warning_percent"]:
            self._trigger_alert(
                AlertLevel.WARNING,
                "Degraded System Coordination Success Rate",
                f"System coordination success rate {system_metrics.coordination_success_rate_percent:.1f}% below warning threshold",
                {"success_rate": system_metrics.coordination_success_rate_percent}
            )
        
        # Check system error rate
        if system_metrics.system_error_rate_percent > self.thresholds["error_rate_critical_percent"]:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                "High System Error Rate",
                f"System error rate {system_metrics.system_error_rate_percent:.1f}% exceeds critical threshold",
                {"error_rate": system_metrics.system_error_rate_percent}
            )
        elif system_metrics.system_error_rate_percent > self.thresholds["error_rate_warning_percent"]:
            self._trigger_alert(
                AlertLevel.WARNING,
                "Elevated System Error Rate",
                f"System error rate {system_metrics.system_error_rate_percent:.1f}% exceeds warning threshold",
                {"error_rate": system_metrics.system_error_rate_percent}
            )
        
        # Check humanitarian impact score
        if system_metrics.humanitarian_impact_score < 50:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                "Low Humanitarian Impact Score",
                f"Humanitarian impact score {system_metrics.humanitarian_impact_score:.1f} indicates coordination issues affecting mission",
                {"impact_score": system_metrics.humanitarian_impact_score}
            )
        elif system_metrics.humanitarian_impact_score < 70:
            self._trigger_alert(
                AlertLevel.WARNING,
                "Suboptimal Humanitarian Impact",
                f"Humanitarian impact score {system_metrics.humanitarian_impact_score:.1f} below optimal levels",
                {"impact_score": system_metrics.humanitarian_impact_score}
            )

# Global collaboration monitor instance
collaboration_monitor = None

def get_collaboration_monitor() -> AgentCollaborationMonitor:
    """Get or create global collaboration monitor"""
    global collaboration_monitor
    
    if collaboration_monitor is None:
        collaboration_monitor = AgentCollaborationMonitor()
    
    return collaboration_monitor

# Example usage and testing
if __name__ == "__main__":
    async def test_collaboration_monitor():
        print("🏥 Testing Agent Collaboration Monitor")
        print("💝 Monitoring agent coordination for humanitarian mission")
        
        # Initialize monitor
        monitor = AgentCollaborationMonitor()
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate agent communications
        print("\n📊 Simulating agent collaborations...")
        
        agents = ["genius_data_coordinator", "genius_market_analyzer", "genius_decision_master", "genius_risk_manager"]
        
        for i in range(50):
            from_agent = np.random.choice(agents)
            to_agent = np.random.choice([a for a in agents if a != from_agent])
            
            # Simulate communication with varying latency
            latency = np.random.uniform(50, 2000)  # 50ms to 2s
            success = np.random.random() > 0.05  # 95% success rate
            
            monitor.record_communication(from_agent, to_agent, latency, success)
            
            # Simulate dependency resolution
            if np.random.random() > 0.7:  # 30% chance of dependency resolution
                resolution_time = np.random.uniform(100, 5000)  # 100ms to 5s
                resolution_success = np.random.random() > 0.03  # 97% success rate
                monitor.record_dependency_resolution(from_agent, to_agent, resolution_time, resolution_success)
            
            # Small delay to simulate real-time
            await asyncio.sleep(0.01)
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        
        print(f"\n📋 Collaboration Monitoring Dashboard:")
        print(f"   Active Agents: {dashboard['system_metrics']['total_active_agents']}")
        print(f"   Messages/Second: {dashboard['system_metrics']['total_messages_per_second']:.2f}")
        print(f"   Avg Latency: {dashboard['system_metrics']['average_coordination_latency_ms']:.1f}ms")
        print(f"   Error Rate: {dashboard['system_metrics']['system_error_rate_percent']:.1f}%")
        print(f"   Coordination Success: {dashboard['system_metrics']['coordination_success_rate_percent']:.1f}%")
        print(f"   Humanitarian Impact: {dashboard['system_metrics']['humanitarian_impact_score']:.1f}")
        print(f"   Performance Trend: {dashboard['system_metrics']['performance_trend']}")
        
        print(f"\n👥 Agent Summaries:")
        for agent_id, summary in dashboard['agent_summaries'].items():
            print(f"   {agent_id}:")
            print(f"     Status: {summary['status']}")
            print(f"     Messages Sent: {summary['total_messages_sent']}")
            print(f"     Avg Response: {summary['average_response_time_ms']:.1f}ms")
            print(f"     Error Rate: {summary['error_rate_percent']:.1f}%")
            print(f"     Coordination Score: {summary['coordination_score']:.1f}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("\n🎯 Collaboration monitoring test completed!")
        print("💝 Agent coordination monitored for humanitarian mission success")
    
    # Run test
    asyncio.run(test_collaboration_monitor())