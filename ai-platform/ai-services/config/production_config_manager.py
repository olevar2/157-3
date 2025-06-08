"""
🏥 HUMANITARIAN AI PLATFORM - PRODUCTION CONFIGURATION MANAGER
💝 Production-ready configuration management for charitable trading mission

This service manages all production configurations for the humanitarian AI trading platform.
Ensures optimal settings for generating profits for medical aid, children's surgeries, and poverty relief.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum
import asyncio
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class RiskLevel(Enum):
    """Risk tolerance levels for charitable fund protection"""
    CONSERVATIVE = "conservative"  # 5-15% risk for charity protection
    MODERATE = "moderate"         # 15-25% risk for balanced growth
    AGGRESSIVE = "aggressive"     # 25-35% risk for rapid growth

@dataclass
class HumanitarianConfig:
    """Configuration for humanitarian mission tracking"""
    charitable_allocation_percentage: float = 50.0  # 50% of profits to charity
    medical_aid_cost_per_unit: float = 25.0        # Cost per medical aid package
    surgery_cost_per_unit: float = 500.0           # Cost per children's surgery
    family_support_cost_per_unit: float = 100.0    # Cost to feed family for month
    monthly_target_usd: float = 50000.0            # Monthly charitable funding target
    emergency_stop_loss_percentage: float = 15.0   # Emergency stop for fund protection
    
@dataclass
class AIModelConfig:
    """Configuration for AI models"""
    inference_timeout_ms: int = 2                  # Max inference time
    ensemble_weights: Dict[str, float] = None      # Model ensemble weights
    confidence_threshold: float = 0.75             # Minimum prediction confidence
    model_update_frequency_hours: int = 24         # Model retraining frequency
    performance_tracking_enabled: bool = True      # Enable performance monitoring
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "ScalpingEnsemble": 0.35,
                "PatternRecognition": 0.30,
                "RiskGenius": 0.25,
                "TrendFollower": 0.10
            }

@dataclass
class TradingConfig:
    """Configuration for trading operations"""
    max_positions: int = 10                        # Maximum concurrent positions
    position_size_percentage: float = 2.0          # Position size as % of capital
    stop_loss_percentage: float = 1.5              # Stop loss percentage
    take_profit_percentage: float = 3.0            # Take profit percentage
    max_daily_trades: int = 100                    # Maximum trades per day
    min_profit_per_trade_usd: float = 10.0        # Minimum profit threshold
    
@dataclass
class BrokerConfig:
    """Configuration for broker integration"""
    primary_broker: str = "MT5_Demo"               # Primary broker
    backup_brokers: List[str] = None               # Backup brokers
    connection_timeout_seconds: int = 30           # Connection timeout
    heartbeat_interval_seconds: int = 60           # Heartbeat check interval
    max_reconnection_attempts: int = 5             # Max reconnection tries
    
    def __post_init__(self):
        if self.backup_brokers is None:
            self.backup_brokers = ["MT4_Demo", "cTrader_Demo"]

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    target_inference_time_ms: float = 2.0          # Target inference time
    target_sub_millisecond_rate: float = 70.0      # Target % sub-millisecond
    max_memory_usage_mb: int = 2048                # Maximum memory usage
    cpu_usage_threshold: float = 80.0              # CPU usage alert threshold
    latency_monitoring_enabled: bool = True        # Enable latency monitoring
    performance_optimization_enabled: bool = True   # Enable auto-optimization

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerting"""
    prometheus_enabled: bool = True                 # Enable Prometheus metrics
    grafana_enabled: bool = True                   # Enable Grafana dashboards
    alert_email_enabled: bool = True               # Enable email alerts
    alert_slack_enabled: bool = False              # Enable Slack alerts
    health_check_interval_seconds: int = 30        # Health check frequency
    log_level: str = "INFO"                        # Logging level

class ProductionConfigManager:
    """
    🏥 Production Configuration Manager for Humanitarian AI Platform
    
    Manages all production configurations for optimal charitable mission execution:
    - Environment-specific settings (dev/staging/production)
    - Humanitarian mission parameters (charitable allocation, targets)
    - AI model optimization settings
    - Trading risk management
    - Broker integration settings
    - Performance optimization
    - Monitoring and alerting
    """
    
    def __init__(self, environment: Environment = Environment.PRODUCTION):
        self.environment = environment
        self.config_cache = {}
        self.config_file_path = self._get_config_file_path()
        self.load_configuration()
        
        logger.info(f"🏥 Production Config Manager initialized for {environment.value}")
        logger.info("💝 Optimized for humanitarian charitable mission")
    
    def _get_config_file_path(self) -> Path:
        """Get configuration file path based on environment"""
        base_path = Path(__file__).parent
        config_files = {
            Environment.DEVELOPMENT: base_path / "configs" / "development.yaml",
            Environment.STAGING: base_path / "configs" / "staging.yaml", 
            Environment.PRODUCTION: base_path / "configs" / "production.yaml",
            Environment.TESTING: base_path / "configs" / "testing.yaml"
        }
        return config_files[self.environment]
    
    def load_configuration(self) -> None:
        """Load configuration from file or create default"""
        try:
            if self.config_file_path.exists():
                with open(self.config_file_path, 'r') as file:
                    self.config_cache = yaml.safe_load(file)
                logger.info(f"✅ Configuration loaded from {self.config_file_path}")
            else:
                self.config_cache = self._create_default_config()
                self.save_configuration()
                logger.info("✅ Default configuration created and saved")
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            self.config_cache = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration based on environment"""
        base_config = {
            "humanitarian": asdict(HumanitarianConfig()),
            "ai_models": asdict(AIModelConfig()),
            "trading": asdict(TradingConfig()),
            "brokers": asdict(BrokerConfig()),
            "performance": asdict(PerformanceConfig()),
            "monitoring": asdict(MonitoringConfig())
        }
        
        # Environment-specific overrides
        if self.environment == Environment.DEVELOPMENT:
            base_config["humanitarian"]["monthly_target_usd"] = 5000.0
            base_config["trading"]["max_positions"] = 3
            base_config["brokers"]["primary_broker"] = "MockBroker"
            base_config["monitoring"]["prometheus_enabled"] = False
            
        elif self.environment == Environment.STAGING:
            base_config["humanitarian"]["monthly_target_usd"] = 25000.0
            base_config["trading"]["max_positions"] = 5
            base_config["brokers"]["primary_broker"] = "MT5_Demo"
            
        elif self.environment == Environment.PRODUCTION:
            # Production settings optimized for maximum charitable impact
            base_config["humanitarian"]["monthly_target_usd"] = 50000.0
            base_config["trading"]["max_positions"] = 10
            base_config["brokers"]["primary_broker"] = "MT5_Live"
            base_config["performance"]["target_inference_time_ms"] = 1.0
            base_config["performance"]["target_sub_millisecond_rate"] = 80.0
            
        return base_config
    
    def save_configuration(self) -> None:
        """Save current configuration to file"""
        try:
            # Ensure directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file_path, 'w') as file:
                yaml.dump(self.config_cache, file, default_flow_style=False, sort_keys=False)
            logger.info(f"✅ Configuration saved to {self.config_file_path}")
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
    
    def get_humanitarian_config(self) -> HumanitarianConfig:
        """Get humanitarian mission configuration"""
        config_data = self.config_cache.get("humanitarian", {})
        return HumanitarianConfig(**config_data)
    
    def get_ai_model_config(self) -> AIModelConfig:
        """Get AI model configuration"""
        config_data = self.config_cache.get("ai_models", {})
        return AIModelConfig(**config_data)
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        config_data = self.config_cache.get("trading", {})
        return TradingConfig(**config_data)
    
    def get_broker_config(self) -> BrokerConfig:
        """Get broker configuration"""
        config_data = self.config_cache.get("brokers", {})
        return BrokerConfig(**config_data)
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        config_data = self.config_cache.get("performance", {})
        return PerformanceConfig(**config_data)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        config_data = self.config_cache.get("monitoring", {})
        return MonitoringConfig(**config_data)
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section"""
        if section not in self.config_cache:
            self.config_cache[section] = {}
        
        self.config_cache[section].update(updates)
        self.save_configuration()
        logger.info(f"✅ Configuration section '{section}' updated")
    
    def get_database_url(self) -> str:
        """Get database connection URL based on environment"""
        db_configs = {
            Environment.DEVELOPMENT: "sqlite:///humanitarian_ai_dev.db",
            Environment.STAGING: "postgresql://user:pass@staging-db:5432/humanitarian_ai",
            Environment.PRODUCTION: os.getenv("DATABASE_URL", "postgresql://user:pass@prod-db:5432/humanitarian_ai"),
            Environment.TESTING: "sqlite:///:memory:"
        }
        return db_configs[self.environment]
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL based on environment"""
        redis_configs = {
            Environment.DEVELOPMENT: "redis://localhost:6379/0",
            Environment.STAGING: "redis://staging-redis:6379/0", 
            Environment.PRODUCTION: os.getenv("REDIS_URL", "redis://prod-redis:6379/0"),
            Environment.TESTING: "redis://localhost:6379/1"
        }
        return redis_configs[self.environment]
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys from environment variables"""
        return {
            "mt5_login": os.getenv("MT5_LOGIN", ""),
            "mt5_password": os.getenv("MT5_PASSWORD", ""),
            "mt5_server": os.getenv("MT5_SERVER", ""),
            "oanda_api_key": os.getenv("OANDA_API_KEY", ""),
            "ib_client_id": os.getenv("IB_CLIENT_ID", ""),
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL", "")
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any issues"""
        issues = []
        
        try:
            humanitarian_config = self.get_humanitarian_config()
            if humanitarian_config.charitable_allocation_percentage < 30:
                issues.append("Charitable allocation below recommended 30%")
            if humanitarian_config.monthly_target_usd < 1000:
                issues.append("Monthly target too low for meaningful impact")
                
            ai_config = self.get_ai_model_config()
            if ai_config.inference_timeout_ms > 10:
                issues.append("Inference timeout too high for real-time trading")
                
            trading_config = self.get_trading_config()
            if trading_config.stop_loss_percentage > 5:
                issues.append("Stop loss too high - risk to charitable funds")
                
            performance_config = self.get_performance_config()
            if performance_config.target_sub_millisecond_rate < 50:
                issues.append("Sub-millisecond target too low for competitive edge")
                
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get security-related settings"""
        return {
            "api_rate_limit": 1000,  # Requests per minute
            "max_login_attempts": 5,
            "session_timeout_minutes": 30,
            "encryption_enabled": self.environment == Environment.PRODUCTION,
            "audit_logging_enabled": True,
            "ip_whitelist_enabled": self.environment == Environment.PRODUCTION
        }
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information"""
        return {
            "environment": self.environment.value,
            "version": "1.0.0",
            "deployment_date": datetime.now().isoformat(),
            "config_file": str(self.config_file_path),
            "charitable_mission": "Medical aid, children's surgeries, poverty relief",
            "target_monthly_impact": self.get_humanitarian_config().monthly_target_usd
        }

# Global config manager instance
config_manager = None

def get_config_manager(environment: Environment = None) -> ProductionConfigManager:
    """Get or create global configuration manager"""
    global config_manager
    
    if config_manager is None:
        if environment is None:
            environment = Environment(os.getenv("APP_ENV", "production"))
        config_manager = ProductionConfigManager(environment)
    
    return config_manager

def initialize_production_config(environment: str = "production") -> ProductionConfigManager:
    """Initialize production configuration manager"""
    env = Environment(environment)
    manager = ProductionConfigManager(env)
    
    logger.info("🏥 Production Configuration Manager initialized")
    logger.info("💝 Ready to serve humanitarian mission")
    logger.info(f"🎯 Target: ${manager.get_humanitarian_config().monthly_target_usd:,.0f}/month for charity")
    
    return manager

# Example usage and testing
if __name__ == "__main__":
    # Test configuration manager
    print("🏥 Testing Production Configuration Manager")
    print("💝 Optimizing for humanitarian charitable mission")
    
    # Initialize for different environments
    for env in Environment:
        print(f"\n📋 Testing {env.value} environment:")
        manager = ProductionConfigManager(env)
        
        humanitarian_config = manager.get_humanitarian_config()
        print(f"   💰 Monthly target: ${humanitarian_config.monthly_target_usd:,.0f}")
        print(f"   💝 Charity allocation: {humanitarian_config.charitable_allocation_percentage}%")
        
        ai_config = manager.get_ai_model_config()
        print(f"   🤖 Inference timeout: {ai_config.inference_timeout_ms}ms")
        
        performance_config = manager.get_performance_config()
        print(f"   ⚡ Sub-ms target: {performance_config.target_sub_millisecond_rate}%")
        
        # Validate configuration
        issues = manager.validate_configuration()
        if issues:
            print(f"   ⚠️  Issues: {len(issues)}")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("   ✅ Configuration valid")
    
    print("\n🎯 Production Configuration Manager ready for deployment!")
    print("💝 Platform optimized for maximum charitable impact")
