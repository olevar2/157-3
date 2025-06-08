"""
Advanced Analytics and Reporting Framework
Real-time analytics integration with standardized interfaces and automated reporting
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import aioredis
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Event system for real-time streaming
from events import EventEmitter

# Platform3 communication framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared', 'communication'))
from platform3_communication_framework import Platform3CommunicationFramework

# Import existing analytics services
from DayTradingAnalytics import DayTradingAnalytics
from SwingAnalytics import SwingAnalytics
from SessionAnalytics import SessionAnalytics
from ScalpingMetrics import ScalpingMetrics
from ProfitOptimizer import ProfitOptimizer

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AnalyticsEvent:
    """Standardized analytics event structure"""
    event_type: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

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

class AdvancedAnalyticsFramework:
    """
    Advanced Analytics and Reporting Framework
    Orchestrates multiple analytics engines with real-time streaming and automated reporting
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the advanced analytics framework"""
        self.redis_url = redis_url
        self.redis_client = None
        
        # Event system for real-time streaming
        self.event_emitter = EventEmitter()
        
        # Analytics engines registry
        self.engines: Dict[str, AnalyticsInterface] = {}
        
        # Performance cache with TTL
        self.cache = {}
        self.cache_ttl = {}
        self.cache_timeout = 300  # 5 minutes default TTL
        
        # Executor for background tasks
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Communication framework
        self.communication_framework = Platform3CommunicationFramework(
            service_name="advanced-analytics",
            service_port=8001,
            redis_url=redis_url
        )
        
        # Metrics collection
        self.realtime_metrics: Dict[str, RealtimeMetric] = {}
        self.metric_history: Dict[str, List[RealtimeMetric]] = {}
        
        # Report scheduling
        self.scheduled_reports = {}
        self.report_queue = asyncio.Queue()
        
        logger.info("Advanced Analytics Framework initialized")

    async def initialize(self):
        """Initialize the analytics framework"""
        try:
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Initialize communication framework
            self.communication_framework.initialize()
            
            # Register analytics engines
            await self._register_analytics_engines()
            
            # Start background tasks            asyncio.create_task(self._cache_cleanup_task())
            asyncio.create_task(self._metrics_collection_task())
            asyncio.create_task(self._report_generation_task())
            
            logger.info("Advanced Analytics Framework fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics framework: {e}")
            raise
    
    async def _register_analytics_engines(self):
        """Register all analytics engines with standardized interfaces"""
        try:
            # Day Trading Analytics Engine (Enhanced)
            day_trading_engine = DayTradingAnalytics()
            await self.register_engine("day_trading", day_trading_engine)
              
            # Swing Analytics Engine (Enhanced)
            swing_engine = SwingAnalytics()
            await self.register_engine("swing_trading", swing_engine)
              
            # Session Analytics Engine (Enhanced)
            session_engine = SessionAnalytics()
            await self.register_engine("session_analysis", session_engine)
            
            # Scalping Metrics Engine (Enhanced)
            scalping_engine = ScalpingMetrics()
            await self.register_engine("scalping", scalping_engine)
            
            # Profit Optimization Engine (Enhanced)
            profit_engine = ProfitOptimizer(optimization_method='kelly')
            await self.register_engine("profit_optimization", profit_engine)
            
            logger.info(f"Registered {len(self.engines)} enhanced analytics engines")
            
        except Exception as e:
            logger.error(f"Failed to register analytics engines: {e}")
            raise

    async def register_engine(self, name: str, engine: AnalyticsInterface):
        """Register an analytics engine"""
        self.engines[name] = engine
        logger.info(f"Registered analytics engine: {name}")

    async def stream_analytics_data(self, data: Dict[str, Any], source: str = "platform3"):
        """Stream analytics data to all registered engines"""
        try:
            # Create analytics event
            event = AnalyticsEvent(
                event_type="data_stream",
                timestamp=datetime.utcnow(),
                source=source,
                data=data
            )
            
            # Emit event for real-time listeners
            self.event_emitter.emit("analytics_data", event)
            
            # Process data through all engines
            results = {}
            for name, engine in self.engines.items():
                try:
                    result = await engine.process_data(data)
                    results[name] = result
                    
                    # Update real-time metrics
                    await self._update_realtime_metrics(name, result)
                    
                except Exception as e:
                    logger.error(f"Engine {name} failed to process data: {e}")
                    results[name] = {"error": str(e)}
            
            # Cache results
            cache_key = f"analytics_results_{source}_{datetime.utcnow().timestamp()}"
            await self._cache_set(cache_key, results, ttl=self.cache_timeout)
            
            # Publish to Redis for other services
            await self.redis_client.publish("analytics_stream", json.dumps(results, default=str))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to stream analytics data: {e}")
            raise

    async def _update_realtime_metrics(self, engine_name: str, result: Dict[str, Any]):
        """Update real-time metrics from engine results"""
        try:
            timestamp = datetime.utcnow()
            
            # Extract key metrics from result
            if 'performance_score' in result:
                metric = RealtimeMetric(
                    metric_name=f"{engine_name}_performance",
                    value=float(result['performance_score']),
                    timestamp=timestamp,
                    context={"engine": engine_name}
                )
                self.realtime_metrics[f"{engine_name}_performance"] = metric
                
                # Add to history
                if f"{engine_name}_performance" not in self.metric_history:
                    self.metric_history[f"{engine_name}_performance"] = []
                self.metric_history[f"{engine_name}_performance"].append(metric)
                
                # Keep only last 1000 metrics in history
                if len(self.metric_history[f"{engine_name}_performance"]) > 1000:
                    self.metric_history[f"{engine_name}_performance"] = \
                        self.metric_history[f"{engine_name}_performance"][-1000:]
            
            # Add more specific metrics based on result content
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['timestamp', 'error']:
                    metric_name = f"{engine_name}_{key}"
                    metric = RealtimeMetric(
                        metric_name=metric_name,
                        value=float(value),
                        timestamp=timestamp,
                        context={"engine": engine_name, "metric_type": key}
                    )
                    self.realtime_metrics[metric_name] = metric
                    
        except Exception as e:
            logger.error(f"Failed to update realtime metrics: {e}")

    async def generate_comprehensive_report(self, timeframe: str = "24h") -> AnalyticsReport:
        """Generate comprehensive analytics report across all engines"""
        try:
            report_id = f"comprehensive_{timeframe}_{datetime.utcnow().isoformat()}"
            
            # Collect reports from all engines
            engine_reports = {}
            for name, engine in self.engines.items():
                try:
                    report = await engine.generate_report(timeframe)
                    engine_reports[name] = asdict(report)
                except Exception as e:
                    logger.error(f"Engine {name} failed to generate report: {e}")
                    engine_reports[name] = {"error": str(e)}
            
            # Aggregate metrics
            aggregated_data = self._aggregate_engine_reports(engine_reports)
            
            # Generate summary and recommendations
            summary = self._generate_report_summary(aggregated_data)
            recommendations = self._generate_recommendations(aggregated_data)
            confidence_score = self._calculate_confidence_score(aggregated_data)
            
            # Create comprehensive report
            report = AnalyticsReport(
                report_id=report_id,
                report_type="comprehensive",
                generated_at=datetime.utcnow(),
                data={
                    "timeframe": timeframe,
                    "engine_reports": engine_reports,
                    "aggregated_metrics": aggregated_data,
                    "realtime_metrics": {k: asdict(v) for k, v in self.realtime_metrics.items()}
                },
                summary=summary,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            # Cache report
            await self._cache_set(f"report_{report_id}", asdict(report), ttl=3600)
            
            # Emit report generation event
            self.event_emitter.emit("report_generated", report)
            
            logger.info(f"Generated comprehensive report: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise

    def _aggregate_engine_reports(self, engine_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate metrics from multiple engine reports"""
        aggregated = {
            "total_engines": len(engine_reports),
            "successful_engines": 0,
            "failed_engines": 0,
            "average_performance": 0.0,
            "combined_metrics": {}
        }
        
        performance_scores = []
        
        for name, report in engine_reports.items():
            if "error" in report:
                aggregated["failed_engines"] += 1
            else:
                aggregated["successful_engines"] += 1
                
                # Extract performance metrics
                if "data" in report and "performance_score" in report["data"]:
                    performance_scores.append(report["data"]["performance_score"])
                
                # Combine other metrics
                if "data" in report:
                    for key, value in report["data"].items():
                        if isinstance(value, (int, float)):
                            if key not in aggregated["combined_metrics"]:
                                aggregated["combined_metrics"][key] = []
                            aggregated["combined_metrics"][key].append(value)
        
        # Calculate averages
        if performance_scores:
            aggregated["average_performance"] = np.mean(performance_scores)
        
        for key, values in aggregated["combined_metrics"].items():
            aggregated["combined_metrics"][f"{key}_avg"] = np.mean(values)
            aggregated["combined_metrics"][f"{key}_std"] = np.std(values)
            aggregated["combined_metrics"][f"{key}_min"] = np.min(values)
            aggregated["combined_metrics"][f"{key}_max"] = np.max(values)
        
        return aggregated

    def _generate_report_summary(self, aggregated_data: Dict[str, Any]) -> str:
        """Generate human-readable summary from aggregated data"""
        successful = aggregated_data["successful_engines"]
        total = aggregated_data["total_engines"]
        avg_performance = aggregated_data["average_performance"]
        
        return f"Analytics Report: {successful}/{total} engines successful. " \
               f"Average performance: {avg_performance:.2f}. " \
               f"Processed {len(aggregated_data.get('combined_metrics', {}))} metric types."

    def _generate_recommendations(self, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on aggregated data"""
        recommendations = []
        
        avg_performance = aggregated_data["average_performance"]
        
        if avg_performance < 0.6:
            recommendations.append("Performance below optimal - consider strategy adjustments")
        elif avg_performance > 0.8:
            recommendations.append("Excellent performance - maintain current strategies")
        
        failed_engines = aggregated_data["failed_engines"]
        if failed_engines > 0:
            recommendations.append(f"Address {failed_engines} failed analytics engines")
        
        return recommendations

    def _calculate_confidence_score(self, aggregated_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the report"""
        successful_ratio = aggregated_data["successful_engines"] / aggregated_data["total_engines"]
        performance_factor = min(aggregated_data["average_performance"], 1.0)
        
        return (successful_ratio * 0.6 + performance_factor * 0.4) * 100

    async def _cache_set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache with TTL"""
        self.cache[key] = value
        self.cache_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl or self.cache_timeout)

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            if datetime.utcnow() < self.cache_ttl[key]:
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.cache_ttl[key]
        return None

    async def _cache_cleanup_task(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                current_time = datetime.utcnow()
                expired_keys = [k for k, exp_time in self.cache_ttl.items() if current_time >= exp_time]
                
                for key in expired_keys:
                    del self.cache[key]
                    del self.cache_ttl[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                logger.error(f"Cache cleanup task error: {e}")

    async def _metrics_collection_task(self):
        """Background task for metrics collection and monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Update real-time metrics
                for metric_name, value in system_metrics.items():
                    metric = RealtimeMetric(
                        metric_name=metric_name,
                        value=value,
                        timestamp=datetime.utcnow(),
                        context={"type": "system"}
                    )
                    self.realtime_metrics[metric_name] = metric
                
                # Publish metrics to Redis
                await self.redis_client.publish(
                    "metrics_stream", 
                    json.dumps({k: asdict(v) for k, v in self.realtime_metrics.items()}, default=str)
                )
                
            except Exception as e:
                logger.error(f"Metrics collection task error: {e}")

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        return {
            "cache_size": len(self.cache),
            "active_engines": len(self.engines),
            "realtime_metrics_count": len(self.realtime_metrics),
            "uptime_hours": (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
        }

    async def _report_generation_task(self):
        """Background task for scheduled report generation"""
        while True:
            try:
                # Check for scheduled reports
                await self._process_scheduled_reports()
                
                # Wait for next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Report generation task error: {e}")

    async def _process_scheduled_reports(self):
        """Process any scheduled reports that are due"""
        current_time = datetime.utcnow()
        
        for report_id, schedule in list(self.scheduled_reports.items()):
            if current_time >= schedule.get("next_run", datetime.min):
                try:
                    # Generate report
                    report = await self.generate_comprehensive_report(schedule.get("timeframe", "24h"))
                    
                    # Save report
                    await self._save_report(report)
                    
                    # Update next run time
                    interval = schedule.get("interval_hours", 24)
                    self.scheduled_reports[report_id]["next_run"] = current_time + timedelta(hours=interval)
                    
                    logger.info(f"Generated scheduled report: {report_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate scheduled report {report_id}: {e}")

    async def _save_report(self, report: AnalyticsReport):
        """Save report to file system"""
        try:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            filename = f"{report.report_id}.json"
            filepath = reports_dir / filename
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(asdict(report), indent=2, default=str))
            
            logger.info(f"Saved report to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    async def schedule_report(self, report_id: str, timeframe: str = "24h", interval_hours: int = 24):
        """Schedule automatic report generation"""
        self.scheduled_reports[report_id] = {
            "timeframe": timeframe,
            "interval_hours": interval_hours,
            "next_run": datetime.utcnow() + timedelta(hours=interval_hours)
        }
        logger.info(f"Scheduled report: {report_id} every {interval_hours} hours")

    def get_realtime_metrics(self) -> Dict[str, RealtimeMetric]:
        """Get current real-time metrics"""
        return self.realtime_metrics.copy()

    def subscribe_to_events(self, event_type: str, callback):
        """Subscribe to analytics events"""
        self.event_emitter.on(event_type, callback)

    async def shutdown(self):
        """Gracefully shutdown the analytics framework"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            self.executor.shutdown(wait=True)
            logger.info("Advanced Analytics Framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Analytics Engine Wrappers for standardized interface
# DayTradingAnalyticsEngine, SwingAnalyticsEngine, and SessionAnalyticsEngine are now directly integrated

class ScalpingMetricsEngine(AnalyticsInterface):
    """Wrapper for ScalpingMetrics with standardized interface"""
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"performance_score": 0.85, "scalp_trades": 0}
    
    async def generate_report(self, timeframe: str) -> AnalyticsReport:
        return AnalyticsReport(
            report_id=f"scalping_{timeframe}_{datetime.utcnow().isoformat()}",
            report_type="scalping",
            generated_at=datetime.utcnow(),
            data={"timeframe": timeframe, "performance_score": 0.85},
            summary="Scalping metrics report",
            recommendations=["Reduce holding time", "Increase trade frequency"],
            confidence_score=88.0
        )
    
    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        return []

class ProfitOptimizationEngine(AnalyticsInterface):
    """Wrapper for ProfitOptimizer with standardized interface"""
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"performance_score": 0.9, "optimization_score": 95}
    
    async def generate_report(self, timeframe: str) -> AnalyticsReport:
        return AnalyticsReport(
            report_id=f"profit_opt_{timeframe}_{datetime.utcnow().isoformat()}",
            report_type="profit_optimization",
            generated_at=datetime.utcnow(),
            data={"timeframe": timeframe, "performance_score": 0.9},
            summary="Profit optimization report",
            recommendations=["Implement dynamic position sizing", "Optimize exit strategies"],
            confidence_score=92.0
        )
    
    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        return []

# Main execution
if __name__ == "__main__":
    async def main():
        framework = AdvancedAnalyticsFramework()
        await framework.initialize()
        
        # Example usage
        sample_data = {
            "trades": [
                {"symbol": "EURUSD", "entry_price": 1.1000, "exit_price": 1.1050, "quantity": 1000},
                {"symbol": "GBPUSD", "entry_price": 1.3000, "exit_price": 1.2950, "quantity": 800}
            ]
        }
        
        # Stream data
        results = await framework.stream_analytics_data(sample_data)
        print(f"Analytics results: {results}")
        
        # Generate report
        report = await framework.generate_comprehensive_report("1h")
        print(f"Generated report: {report.report_id}")
        
        # Schedule automatic reports
        await framework.schedule_report("daily_report", "24h", 24)
        
        await framework.shutdown()
    
    asyncio.run(main())
