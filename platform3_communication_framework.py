"""
Platform3 Communication Framework
Central communication framework for inter-module messaging and coordination
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio
import threading

class MessageBus:
    """Central message bus for inter-module communication"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.subscribers = {}
        self.message_history = []
        self.lock = threading.Lock()
        
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic with a callback function"""
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
            
    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic"""
        with self.lock:
            # Add to history
            self.message_history.append({
                'topic': topic,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Notify subscribers
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(f"Error in subscriber callback: {e}")

class CommunicationFramework:
    """Main communication framework class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.message_bus = MessageBus()
        self.services = {}
        self.status = 'active'
        
        self.logger.info("Platform3 Communication Framework initialized")
    
    def register_service(self, name: str, service: Any):
        """Register a service for communication"""
        self.services[name] = service
        self.logger.info(f"Service '{name}' registered")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service"""
        return self.services.get(name)
    
    def broadcast_message(self, topic: str, message: Dict[str, Any]):
        """Broadcast a message to all subscribers"""
        self.message_bus.publish(topic, message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get framework status"""
        return {
            'status': self.status,
            'services': list(self.services.keys()),
            'topics': list(self.message_bus.subscribers.keys()),
            'message_count': len(self.message_bus.message_history)
        }

class Platform3CommunicationFramework(CommunicationFramework):
    """Platform3-specific communication framework with extended features"""
    
    def __init__(self, service_name: Optional[str] = None, service_port: Optional[int] = None, 
                 redis_url: Optional[str] = None, consul_host: Optional[str] = None, 
                 consul_port: Optional[int] = None, **kwargs):
        super().__init__()
        self.service_name = service_name or "platform3_communication"
        self.service_port = service_port or 8000
        self.redis_url = redis_url or "redis://localhost:6379"
        self.consul_host = consul_host or "localhost" 
        self.consul_port = consul_port or 8500
        
        self.ai_services = {}
        self.indicator_pipeline = None
        self.market_data_streams = {}
        
        self.logger.info(f"Platform3 Communication Framework initialized for service '{self.service_name}'")
    
    def initialize_ai_services(self):
        """Initialize AI services integration"""
        try:
            from ai_services import get_model_registry
            self.ai_services['model_registry'] = get_model_registry()
            self.logger.info("AI services initialized")
        except ImportError:
            self.logger.warning("AI services not available")
    
    def get_ai_service(self, service_name: str):
        """Get an AI service by name"""
        return self.ai_services.get(service_name)
    
    def register_indicator_pipeline(self, pipeline):
        """Register the indicator pipeline"""
        self.indicator_pipeline = pipeline
        self.logger.info("Indicator pipeline registered")
    
    def initialize(self):
        """Initialize the communication framework"""
        self.logger.info("Platform3 Communication Framework initialization started")
        self.status = 'initialized'
        
        # Initialize AI services if available
        self.initialize_ai_services()
        
        # Initialize message bus
        if not hasattr(self.message_bus, 'subscribers'):
            self.message_bus.subscribers = {}
            
        self.logger.info("Platform3 Communication Framework initialization completed")
        return True

# Global framework instance
communication_framework = Platform3CommunicationFramework()

# Convenience functions
def get_framework() -> CommunicationFramework:
    """Get the global communication framework instance"""
    return communication_framework

def register_service(name: str, service: Any):
    """Register a service with the global framework"""
    return communication_framework.register_service(name, service)

def get_service(name: str) -> Optional[Any]:
    """Get a service from the global framework"""
    return communication_framework.get_service(name)

def broadcast(topic: str, message: Dict[str, Any]):
    """Broadcast a message using the global framework"""
    return communication_framework.broadcast_message(topic, message)
