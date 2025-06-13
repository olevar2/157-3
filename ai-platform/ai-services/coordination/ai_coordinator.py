"""
Platform3 AI Coordination Service
Enterprise-grade AI model coordination and orchestration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

import sys
from pathlib import Path
from ai_platform.ai_services.model_registry import AIModelRegistry, ModelType, ModelStatus, get_registry

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AITask:
    """AI task structure"""
    task_id: str
    model_id: str
    function_name: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    timeout: Optional[int] = None

class AICoordinationService:
    """
    Centralized coordination service for all AI models
    Handles task scheduling, load balancing, and model orchestration
    """
    
    def __init__(self, max_workers: int = 8):
        self.registry = get_registry()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue: List[AITask] = []
        self.running_tasks: Dict[str, AITask] = {}
        self.completed_tasks: Dict[str, AITask] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Task scheduling thread
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._task_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def submit_task(self, 
                   model_id: str, 
                   function_name: str, 
                   parameters: Dict[str, Any],
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   timeout: Optional[int] = None) -> str:
        """Submit a task to the AI coordination service"""
        
        # Validate model exists
        model_info = self.registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if model_info.status != ModelStatus.ACTIVE:
            raise ValueError(f"Model {model_id} is not active (status: {model_info.status.value})")
        
        # Create task
        task_id = f"{model_id}_{function_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        task = AITask(
            task_id=task_id,
            model_id=model_id,
            function_name=function_name,
            parameters=parameters,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            timeout=timeout
        )
        
        with self.lock:
            self.task_queue.append(task)
            # Sort by priority (highest first)
            self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        self.logger.info(f"Task {task_id} submitted for model {model_id}")
        return task_id
    
    def _task_scheduler(self):
        """Background task scheduler"""
        while self.scheduler_running:
            try:
                # Check for pending tasks
                with self.lock:
                    if self.task_queue and len(self.running_tasks) < self.max_workers:
                        task = self.task_queue.pop(0)
                        self.running_tasks[task.task_id] = task
                
                if 'task' in locals():
                    # Execute task
                    future = self.executor.submit(self._execute_task, task)
                    # Don't wait for completion here, let it run asynchronously
                    del task
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Sleep briefly to avoid busy waiting
                asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {e}")
                asyncio.sleep(1)
    
    def _execute_task(self, task: AITask):
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Load model if not already loaded
            model = self._get_or_load_model(task.model_id)
            
            # Get function from model
            if not hasattr(model, task.function_name):
                raise AttributeError(f"Model {task.model_id} does not have function {task.function_name}")
            
            func = getattr(model, task.function_name)
            
            # Execute function
            if callable(func):
                result = func(**task.parameters)
            else:
                raise ValueError(f"Attribute {task.function_name} is not callable")
            
            # Update task
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update model usage stats
            self._update_model_stats(task.model_id, True)
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            # Update model usage stats
            self._update_model_stats(task.model_id, False)
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Move from running to completed
            with self.lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
    
    def _get_or_load_model(self, model_id: str):
        """Get model from cache or load it"""
        if model_id not in self.loaded_models:
            self.loaded_models[model_id] = self.registry.load_model(model_id)
        return self.loaded_models[model_id]
    
    def _update_model_stats(self, model_id: str, success: bool):
        """Update model usage statistics"""
        if model_id not in self.model_usage_stats:
            self.model_usage_stats[model_id] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'last_used': None,
                'average_response_time': 0.0
            }
        
        stats = self.model_usage_stats[model_id]
        stats['total_calls'] += 1
        stats['last_used'] = datetime.now()
        
        if success:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks to prevent memory buildup"""
        current_time = datetime.now()
        
        # Keep completed tasks for 1 hour
        with self.lock:
            to_remove = []
            for task_id, task in self.completed_tasks.items():
                if task.completed_at and (current_time - task.completed_at).seconds > 3600:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.completed_tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[AITask]:
        """Get status of a task"""
        with self.lock:
            if task_id in self.running_tasks:
                return self.running_tasks[task_id]
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            else:
                # Check if it's still in queue
                for task in self.task_queue:
                    if task.task_id == task_id:
                        return task
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        with self.lock:
            # Remove from queue if pending
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.task_queue.pop(i)
                    self.completed_tasks[task_id] = task
                    return True
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self.lock:
            return {
                'pending_tasks': len(self.task_queue),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'queue_by_priority': {
                    priority.name: len([t for t in self.task_queue if t.priority == priority])
                    for priority in TaskPriority
                }
            }
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get model usage statistics"""
        return self.model_usage_stats.copy()
    
    def execute_ensemble_prediction(self, 
                                  model_ids: List[str], 
                                  function_name: str, 
                                  parameters: Dict[str, Any],
                                  aggregation_method: str = "average") -> Dict[str, Any]:
        """Execute ensemble prediction across multiple models"""
        
        # Submit tasks for all models
        task_ids = []
        for model_id in model_ids:
            try:
                task_id = self.submit_task(
                    model_id, 
                    function_name, 
                    parameters, 
                    TaskPriority.HIGH
                )
                task_ids.append(task_id)
            except Exception as e:
                self.logger.warning(f"Could not submit task for model {model_id}: {e}")
        
        # Wait for all tasks to complete
        results = {}
        errors = {}
        
        # Poll for completion (in production, use async/await)
        max_wait_time = 300  # 5 minutes
        start_time = datetime.now()
        
        while len(results) + len(errors) < len(task_ids):
            if (datetime.now() - start_time).seconds > max_wait_time:
                break
            
            for task_id in task_ids:
                if task_id not in results and task_id not in errors:
                    task = self.get_task_status(task_id)
                    if task and task.status == TaskStatus.COMPLETED:
                        results[task_id] = task.result
                    elif task and task.status == TaskStatus.FAILED:
                        errors[task_id] = task.error
            
            asyncio.sleep(0.5)
        
        # Aggregate results
        if results:
            aggregated_result = self._aggregate_ensemble_results(
                list(results.values()), 
                aggregation_method
            )
        else:
            aggregated_result = None
        
        return {
            'aggregated_result': aggregated_result,
            'individual_results': results,
            'errors': errors,
            'success_rate': len(results) / len(task_ids) if task_ids else 0
        }
    
    def _aggregate_ensemble_results(self, results: List[Any], method: str) -> Any:
        """Aggregate ensemble results"""
        if not results:
            return None
        
        if method == "average":
            # Try to average numeric results
            try:
                if all(isinstance(r, (int, float)) for r in results):
                    return sum(results) / len(results)
                elif all(isinstance(r, dict) for r in results):
                    # Average dict values
                    keys = set()
                    for r in results:
                        keys.update(r.keys())
                    
                    averaged = {}
                    for key in keys:
                        values = [r.get(key, 0) for r in results if key in r]
                        if values and all(isinstance(v, (int, float)) for v in values):
                            averaged[key] = sum(values) / len(values)
                    return averaged
            except:
                pass
        
        elif method == "majority_vote":
            # Return most common result
            from collections import Counter
            counter = Counter(str(r) for r in results)
            most_common = counter.most_common(1)[0][0]
            # Try to convert back to original type
            for r in results:
                if str(r) == most_common:
                    return r
        
        elif method == "best_confidence":
            # If results have confidence scores, return highest
            try:
                if all(isinstance(r, dict) and 'confidence' in r for r in results):
                    return max(results, key=lambda x: x['confidence'])
            except:
                pass
        
        # Default: return first result
        return results[0]
    
    def shutdown(self):
        """Shutdown the coordination service"""
        self.scheduler_running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        self.executor.shutdown(wait=True)

# Global coordination service instance
_coordination_service = None

def get_coordination_service() -> AICoordinationService:
    """Get global coordination service instance"""
    global _coordination_service
    if _coordination_service is None:
        _coordination_service = AICoordinationService()
    return _coordination_service

if __name__ == "__main__":
    # Test the coordination service
    service = AICoordinationService()
    
    # Get queue status
    status = service.get_queue_status()
    print(f"Queue Status: {status}")
    
    # Get model stats
    stats = service.get_model_stats()
    print(f"Model Stats: {stats}")
