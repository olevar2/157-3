"""
Winston Configuration Factory for Platform3
Provides standardized logger configurations for different environments and services
"""

import os
from typing import Dict, Any, Optional
from .platform3_logger import Platform3Logger


class LoggerConfigFactory:
    """Factory for creating standardized logger configurations"""
    
    # Default configurations for different environments
    ENVIRONMENTS = {
        'development': {
            'level': 'DEBUG',
            'log_to_file': True,
            'colorize': True,
            'max_file_size': '10MB',
            'max_files': 7
        },
        'staging': {
            'level': 'INFO', 
            'log_to_file': True,
            'colorize': False,
            'max_file_size': '20MB',
            'max_files': 14
        },
        'production': {
            'level': 'INFO',
            'log_to_file': True,
            'colorize': False,
            'max_file_size': '50MB',
            'max_files': 30
        },
        'testing': {
            'level': 'WARNING',
            'log_to_file': False,
            'colorize': False,
            'max_file_size': '5MB',
            'max_files': 3
        }
    }
    
    # Service-specific configurations
    SERVICE_CONFIGS = {
        'ai-models': {
            'service_name': 'Platform3-AI',
            'log_directory': './logs/ai-models'
        },
        'trading-service': {
            'service_name': 'Platform3-Trading',
            'log_directory': './logs/trading'
        },
        'event-system': {
            'service_name': 'Platform3-Events',
            'log_directory': './logs/events'
        },
        'payment-service': {
            'service_name': 'Platform3-Payments',
            'log_directory': './logs/payments'
        },
        'user-service': {
            'service_name': 'Platform3-Users',
            'log_directory': './logs/users'
        },
        'dashboard': {
            'service_name': 'Platform3-Dashboard',
            'log_directory': './logs/dashboard'
        }
    }
    
    @classmethod
    def create_config(
        self,
        environment: str = None,
        service: str = None,
        custom_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create logger configuration for environment and service
        
        Args:
            environment: Environment name (development, staging, production, testing)
            service: Service name (ai-models, trading-service, etc.)
            custom_config: Custom configuration overrides
            
        Returns:
            Configuration dictionary for Platform3Logger
        """
        # Detect environment from ENV vars if not specified
        if environment is None:
            environment = os.getenv('NODE_ENV', 'development')
        
        # Start with environment defaults
        config = self.ENVIRONMENTS.get(environment, self.ENVIRONMENTS['development']).copy()
        
        # Apply service-specific config
        if service and service in self.SERVICE_CONFIGS:
            config.update(self.SERVICE_CONFIGS[service])
        
        # Apply environment variable overrides
        env_overrides = {
            'level': os.getenv('LOG_LEVEL'),
            'log_to_file': os.getenv('LOG_TO_FILE'),
            'log_directory': os.getenv('LOG_DIRECTORY'),
            'max_file_size': os.getenv('LOG_MAX_FILE_SIZE'),
            'max_files': os.getenv('LOG_MAX_FILES')
        }
        
        for key, value in env_overrides.items():
            if value is not None:
                if key == 'log_to_file':
                    config[key] = value.lower() not in ('false', '0', 'no')
                elif key == 'max_files':
                    try:
                        config[key] = int(value)
                    except ValueError:
                        pass
                else:
                    config[key] = value
        
        # Apply custom overrides
        if custom_config:
            config.update(custom_config)
        
        return config
    
    @classmethod
    def create_logger(
        self,
        context: str,
        environment: str = None,
        service: str = None,
        custom_config: Dict[str, Any] = None
    ) -> Platform3Logger:
        """
        Create configured Platform3Logger instance
        
        Args:
            context: Logger context name
            environment: Environment name
            service: Service name
            custom_config: Custom configuration overrides
            
        Returns:
            Configured Platform3Logger instance
        """
        config = self.create_config(environment, service, custom_config)
        return Platform3Logger(context=context, **config)
    
    @classmethod
    def create_ai_model_logger(self, model_name: str) -> Platform3Logger:
        """Create logger specifically for AI models"""
        return self.create_logger(
            context=f"AI-Model-{model_name}",
            service='ai-models'
        )
    
    @classmethod
    def create_trading_logger(self, strategy_name: str) -> Platform3Logger:
        """Create logger specifically for trading strategies"""
        return self.create_logger(
            context=f"Trading-{strategy_name}",
            service='trading-service'
        )
    
    @classmethod
    def create_pattern_logger(self, pattern_name: str) -> Platform3Logger:
        """Create logger specifically for pattern recognition"""
        return self.create_logger(
            context=f"Pattern-{pattern_name}",
            service='ai-models',
            custom_config={'log_directory': './logs/patterns'}
        )


# Convenience functions for common logger types
def create_ai_model_logger(model_name: str) -> Platform3Logger:
    """Create AI model logger"""
    return LoggerConfigFactory.create_ai_model_logger(model_name)


def create_trading_logger(strategy_name: str) -> Platform3Logger:
    """Create trading strategy logger"""
    return LoggerConfigFactory.create_trading_logger(strategy_name)


def create_pattern_logger(pattern_name: str) -> Platform3Logger:
    """Create pattern recognition logger"""
    return LoggerConfigFactory.create_pattern_logger(pattern_name)
