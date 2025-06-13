"""
Platform3 AI Platform Integration Layer
Central orchestration and management for the entire AI platform
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading

# Import all AI platform services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "ai-services"))

from ai_services.model_registry.model_registry import ModelRegistry, ModelType, ModelStatus
from ai_services.coordination.ai_coordinator import AICoordinator, TaskPriority
from ai_services.performance_monitoring.performance_monitor import PerformanceMonitor, MetricType
from ai_services.mlops.mlops_service import MLOpsService, DeploymentStage

@dataclass
class PlatformStatus:
    """Overall platform status"""
    total_models: int
    active_models: int
    running_tasks: int
    pending_tasks: int
    system_cpu_usage: float
    system_memory_usage: float
    active_alerts: int
    timestamp: datetime

class AIPlatformManager:
    """
    Central manager for the entire AI platform
    Coordinates all services and provides unified interface
    """
    
    def __init__(self):        # Initialize all services
        self.registry = ModelRegistry()
        self.coordinator = AICoordinator()
        self.performance_monitor = PerformanceMonitor()
        self.mlops = MLOpsService()
        
        self.logger = logging.getLogger(__name__)
        self.startup_time = datetime.now()
        
        # Platform-wide settings
        self.settings = {
            'auto_scaling_enabled': True,
            'performance_monitoring_enabled': True,
            'alert_notifications_enabled': True,
            'model_auto_deployment': False,
            'load_balancing_enabled': True
        }
        
        self.logger.info("AI Platform Manager initialized successfully")
    
    def get_platform_status(self) -> PlatformStatus:
        """Get comprehensive platform status"""
        
        # Get registry status
        registry_summary = self.registry.get_registry_summary()
        
        # Get coordination status
        queue_status = self.coordinator.get_queue_status()
        
        # Get system performance
        system_perf = self.performance_monitor.get_system_performance()
        
        # Get alerts
        alerts = self.performance_monitor.get_active_alerts()
        
        return PlatformStatus(
            total_models=registry_summary['total_models'],
            active_models=registry_summary['active_models'],
            running_tasks=queue_status['running_tasks'],
            pending_tasks=queue_status['pending_tasks'],
            system_cpu_usage=system_perf.get('cpu_usage', {}).get('current', 0),
            system_memory_usage=system_perf.get('memory_usage', {}).get('current', 0),
            active_alerts=len(alerts),
            timestamp=datetime.now()
        )
    
    def execute_prediction(self, 
                          model_id: str,
                          function_name: str = "predict",
                          parameters: Dict[str, Any] = None,
                          priority: TaskPriority = TaskPriority.MEDIUM,
                          timeout: Optional[int] = None,
                          track_performance: bool = True) -> str:
        """Execute a prediction with full platform integration"""
        
        if parameters is None:
            parameters = {}
        
        # Validate model
        model_info = self.registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        if model_info.status != ModelStatus.ACTIVE:
            raise ValueError(f"Model {model_id} is not active")
        
        # Generate request ID for tracking
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Start performance tracking
        if track_performance:
            self.performance_monitor.start_request(model_id, request_id)
        
        try:
            # Submit task to coordinator
            task_id = self.coordinator.submit_task(
                model_id=model_id,
                function_name=function_name,
                parameters=parameters,
                priority=priority,
                timeout=timeout
            )
            
            self.logger.info(f"Prediction task {task_id} submitted for model {model_id}")
            return task_id
            
        except Exception as e:
            # End performance tracking with error
            if track_performance:
                self.performance_monitor.end_request(model_id, request_id, success=False)
            raise
    
    def execute_ensemble_prediction(self,
                                  model_ids: List[str],
                                  function_name: str = "predict",
                                  parameters: Dict[str, Any] = None,
                                  aggregation_method: str = "average",
                                  min_success_rate: float = 0.7) -> Dict[str, Any]:
        """Execute ensemble prediction across multiple models"""
        
        if parameters is None:
            parameters = {}
        
        # Validate all models
        valid_models = []
        for model_id in model_ids:
            model_info = self.registry.get_model(model_id)
            if model_info and model_info.status == ModelStatus.ACTIVE:
                valid_models.append(model_id)
            else:
                self.logger.warning(f"Skipping inactive model {model_id}")
        
        if not valid_models:
            raise ValueError("No valid active models found for ensemble")
        
        # Execute ensemble prediction
        result = self.coordinator.execute_ensemble_prediction(
            model_ids=valid_models,
            function_name=function_name,
            parameters=parameters,
            aggregation_method=aggregation_method
        )
        
        # Check success rate
        if result['success_rate'] < min_success_rate:
            self.logger.warning(f"Ensemble success rate {result['success_rate']:.2f} below threshold {min_success_rate}")
        
        return result
    
    def get_model_performance(self, model_id: str, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive model performance data"""
        
        # Get current performance snapshot
        current_perf = self.performance_monitor.get_current_performance(model_id)
        
        # Get performance summary
        perf_summary = self.performance_monitor.get_performance_summary(model_id, hours)
        
        # Get model usage stats from coordinator
        model_stats = self.coordinator.get_model_stats()
        
        # Get MLOps information
        latest_version = self.mlops.get_latest_version(model_id, DeploymentStage.PRODUCTION)
        
        return {
            'model_id': model_id,
            'current_performance': current_perf.__dict__ if current_perf else None,
            'performance_summary': perf_summary,
            'usage_stats': model_stats.get(model_id, {}),
            'latest_version': latest_version.__dict__ if latest_version else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_platform_health(self) -> Dict[str, Any]:
        """Get comprehensive platform health report"""
        
        status = self.get_platform_status()
        alerts = self.performance_monitor.get_active_alerts()
        system_perf = self.performance_monitor.get_system_performance()
        
        # Calculate health score
        health_factors = {
            'model_availability': min(status.active_models / max(status.total_models, 1), 1.0),
            'system_resources': (100 - status.system_cpu_usage) / 100 * (100 - status.system_memory_usage) / 100,
            'task_processing': 1.0 if status.running_tasks < 10 else max(0.5, 1.0 - (status.running_tasks - 10) / 50),
            'alert_status': 1.0 if status.active_alerts == 0 else max(0.3, 1.0 - status.active_alerts / 20)
        }
        
        overall_health = sum(health_factors.values()) / len(health_factors)
        
        return {
            'overall_health_score': overall_health,
            'health_factors': health_factors,
            'status': status.__dict__,
            'system_performance': system_perf,
            'active_alerts': [alert.__dict__ for alert in alerts],
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
    
    def auto_scale_models(self) -> Dict[str, Any]:
        """Automatically scale models based on performance metrics"""
        
        if not self.settings['auto_scaling_enabled']:
            return {'status': 'disabled', 'message': 'Auto-scaling is disabled'}
        
        scaling_actions = []
        
        # Get all active models
        active_models = self.registry.get_models_by_status(ModelStatus.ACTIVE)
        
        for model_info in active_models:
            model_id = model_info.model_id
            
            # Get current performance
            current_perf = self.performance_monitor.get_current_performance(model_id)
            
            if current_perf:
                # Check if scaling is needed
                action = None
                
                # Scale up conditions
                if (current_perf.latency_ms and current_perf.latency_ms > 2000) or \
                   (current_perf.cpu_percent and current_perf.cpu_percent > 85) or \
                   (current_perf.active_requests > 20):
                    action = 'scale_up'
                
                # Scale down conditions
                elif (current_perf.latency_ms and current_perf.latency_ms < 200) and \
                     (current_perf.cpu_percent and current_perf.cpu_percent < 30) and \
                     (current_perf.active_requests < 5):
                    action = 'scale_down'
                
                if action:
                    # In a real implementation, this would trigger actual scaling
                    scaling_actions.append({
                        'model_id': model_id,
                        'action': action,
                        'reason': f"Latency: {current_perf.latency_ms}ms, CPU: {current_perf.cpu_percent}%, Requests: {current_perf.active_requests}",
                        'timestamp': datetime.now().isoformat()
                    })
        
        return {
            'status': 'completed',
            'actions_taken': len(scaling_actions),
            'scaling_actions': scaling_actions,
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_model_placement(self) -> Dict[str, Any]:
        """Optimize model placement across available resources"""
        
        optimization_actions = []
        
        # Get system performance
        system_perf = self.performance_monitor.get_system_performance()
        
        # Get model performance data
        active_models = self.registry.get_models_by_status(ModelStatus.ACTIVE)
        
        model_metrics = []
        for model_info in active_models:
            current_perf = self.performance_monitor.get_current_performance(model_info.model_id)
            if current_perf:
                model_metrics.append({
                    'model_id': model_info.model_id,
                    'model_type': model_info.model_type,
                    'latency': current_perf.latency_ms or 0,
                    'memory': current_perf.memory_mb or 0,
                    'cpu': current_perf.cpu_percent or 0,
                    'requests': current_perf.active_requests
                })
        
        # Sort by resource usage (highest first)
        model_metrics.sort(key=lambda x: x['memory'] + x['cpu'], reverse=True)
        
        # Simple optimization: suggest moving high-resource models
        if system_perf.get('memory_usage', {}).get('current', 0) > 80:
            for model_metric in model_metrics[:3]:  # Top 3 resource consumers
                optimization_actions.append({
                    'model_id': model_metric['model_id'],
                    'action': 'move_to_dedicated_node',
                    'reason': f"High resource usage: {model_metric['memory']}MB memory, {model_metric['cpu']}% CPU",
                    'priority': 'high' if model_metric['memory'] > 1000 else 'medium'
                })
        
        return {
            'status': 'completed',
            'system_memory_usage': system_perf.get('memory_usage', {}).get('current', 0),
            'optimization_actions': optimization_actions,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_model_backup(self, model_id: str) -> Dict[str, Any]:
        """Create a backup of a model and its metadata"""
        
        try:
            # Get model info
            model_info = self.registry.get_model(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            # Get latest version from MLOps
            latest_version = self.mlops.get_latest_version(model_id)
            
            if latest_version:
                # Export model
                project_root = os.path.dirname(os.path.dirname(__file__))
                backup_path = f"{project_root}/ai-platform/backups/{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                success = self.mlops.export_model(model_id, latest_version.version, backup_path)
                
                if success:
                    return {
                        'status': 'success',
                        'model_id': model_id,
                        'version': latest_version.version,
                        'backup_path': backup_path,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'status': 'error',
                'message': 'Could not create backup',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_platform_summary(self) -> Dict[str, Any]:
        """Get comprehensive platform summary"""
        
        # Get basic status
        status = self.get_platform_status()
        
        # Get model breakdown by type
        models_by_type = {}
        for model_info in self.registry.list_models():
            model_type = model_info.model_type.value
            if model_type not in models_by_type:
                models_by_type[model_type] = {'total': 0, 'active': 0}
            models_by_type[model_type]['total'] += 1
            if model_info.status == ModelStatus.ACTIVE:
                models_by_type[model_type]['active'] += 1
        
        # Get performance overview
        total_requests = sum(
            stats.get('total_calls', 0) 
            for stats in self.coordinator.get_model_stats().values()
        )
        
        return {
            'platform_status': status.__dict__,
            'models_by_type': models_by_type,
            'total_requests_processed': total_requests,
            'settings': self.settings,
            'uptime_hours': (datetime.now() - self.startup_time).total_seconds() / 3600,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_platform_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update platform-wide settings"""
        
        updated = {}
        for key, value in settings.items():
            if key in self.settings:
                old_value = self.settings[key]
                self.settings[key] = value
                updated[key] = {'old': old_value, 'new': value}
                self.logger.info(f"Updated setting {key}: {old_value} -> {value}")
        
        return {
            'status': 'success',
            'updated_settings': updated,
            'current_settings': self.settings,
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown the AI platform gracefully"""
        
        self.logger.info("Shutting down AI Platform...")
        
        # Shutdown services
        self.coordinator.shutdown()
        self.performance_monitor.shutdown()
        
        self.logger.info("AI Platform shutdown completed")

# Global platform manager instance
_platform_manager = None

def get_platform_manager() -> AIPlatformManager:
    """Get global platform manager instance"""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = AIPlatformManager()
    return _platform_manager

if __name__ == "__main__":
    # Test the platform manager
    platform = AIPlatformManager()
    
    # Get platform summary
    summary = platform.get_platform_summary()
    print(f"Platform Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Get platform health
    health = platform.get_platform_health()
    print(f"\nPlatform Health Score: {health['overall_health_score']:.2f}")
