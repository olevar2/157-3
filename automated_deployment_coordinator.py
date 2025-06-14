"""
Phase 6.1: Automated Deployment Coordinator
Orchestrates automated deployment processes with comprehensive monitoring and rollback capabilities.
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import subprocess
import shutil
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import tempfile
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

class DeploymentStage(Enum):
    """Deployment stage enumeration."""
    PREPARATION = "preparation"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    VERIFICATION = "verification"
    CLEANUP = "cleanup"

@dataclass
class DeploymentConfig:
    """Deployment configuration parameters."""
    name: str
    version: str
    environment: str
    source_path: str
    target_path: str
    build_commands: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    deploy_commands: List[str] = field(default_factory=list)
    verification_commands: List[str] = field(default_factory=list)
    rollback_commands: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 3600
    auto_rollback: bool = True
    backup_enabled: bool = True
    notifications_enabled: bool = True

@dataclass
class DeploymentStep:
    """Individual deployment step."""
    stage: DeploymentStage
    command: str
    description: str
    timeout: int = 300
    required: bool = True
    retry_count: int = 3
    retry_delay: int = 5

@dataclass
class DeploymentResult:
    """Deployment execution result."""
    step: DeploymentStep
    status: str
    output: str = ""
    error: str = ""
    duration: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentRecord:
    """Complete deployment record."""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[DeploymentResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    backup_path: Optional[str] = None

class AutomatedDeploymentCoordinator:
    """
    Automated Deployment Coordinator
    Orchestrates comprehensive deployment processes with monitoring and rollback.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/deployment_config.yaml"
        self.deployments: Dict[str, DeploymentRecord] = {}
        self.active_deployments: Dict[str, asyncio.Task] = {}
        self.deployment_history: List[DeploymentRecord] = []
        self.notification_handlers: List[Callable] = []
        
    async def initialize(self) -> bool:
        """Initialize the deployment coordinator."""
        try:
            logger.info("Initializing Automated Deployment Coordinator...")
            
            # Create necessary directories
            await self._create_directories()
            
            # Load deployment configurations
            await self._load_configurations()
            
            # Initialize monitoring
            await self._setup_monitoring()
            
            # Load deployment history
            await self._load_deployment_history()
            
            logger.info("Automated Deployment Coordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Deployment Coordinator: {e}")
            return False
    
    async def _create_directories(self):
        """Create necessary deployment directories."""
        directories = [
            "deployments", "backups", "logs/deployments", 
            "config", "staging", "monitoring"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def _load_configurations(self):
        """Load deployment configurations."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    # Process configuration data
            else:
                # Create default configuration
                await self._create_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
    
    async def _create_default_config(self):
        """Create default deployment configuration."""
        default_config = {
            "environments": {
                "staging": {
                    "target_path": "/opt/platform3/staging",
                    "backup_retention_days": 7,
                    "auto_rollback": True
                },
                "production": {
                    "target_path": "/opt/platform3/production",
                    "backup_retention_days": 30,
                    "auto_rollback": True
                }
            },
            "global_settings": {
                "max_concurrent_deployments": 3,
                "default_timeout": 3600,
                "notification_channels": ["email", "slack"],
                "monitoring_enabled": True
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    async def _setup_monitoring(self):
        """Setup deployment monitoring."""
        logger.info("Setting up deployment monitoring...")
        # Implementation would include metrics collection,
        # health checks, alerting setup, etc.
    
    async def _load_deployment_history(self):
        """Load deployment history."""
        try:
            history_file = "deployments/deployment_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    # Process history data
                    logger.info(f"Loaded {len(history_data)} deployment records")
                    
        except Exception as e:
            logger.error(f"Failed to load deployment history: {e}")
    
    async def create_deployment(self, config: DeploymentConfig) -> str:
        """Create a new deployment."""
        try:
            deployment_id = self._generate_deployment_id(config)
            
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                config=config,
                status=DeploymentStatus.PENDING,
                start_time=datetime.now(timezone.utc)
            )
            
            self.deployments[deployment_id] = deployment_record
            
            logger.info(f"Created deployment: {deployment_id}")
            await self._notify_deployment_created(deployment_record)
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            raise
    
    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """Generate unique deployment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = f"{config.name}_{config.version}_{config.environment}_{timestamp}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"deploy_{timestamp}_{hash_suffix}"
    
    async def execute_deployment(self, deployment_id: str) -> bool:
        """Execute a deployment."""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        deployment_record = self.deployments[deployment_id]
        
        # Check if already running
        if deployment_id in self.active_deployments:
            logger.warning(f"Deployment already running: {deployment_id}")
            return False
        
        # Start deployment task
        task = asyncio.create_task(self._execute_deployment_task(deployment_record))
        self.active_deployments[deployment_id] = task
        
        try:
            result = await task
            return result
        finally:
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
    
    async def _execute_deployment_task(self, deployment_record: DeploymentRecord) -> bool:
        """Execute deployment task."""
        deployment_id = deployment_record.deployment_id
        config = deployment_record.config
        
        try:
            logger.info(f"Starting deployment: {deployment_id}")
            deployment_record.status = DeploymentStatus.IN_PROGRESS
            await self._notify_deployment_status_change(deployment_record)
            
            # Create deployment steps
            steps = await self._create_deployment_steps(config)
            
            # Execute deployment stages
            success = True
            for step in steps:
                result = await self._execute_deployment_step(step, config)
                deployment_record.results.append(result)
                
                if result.status != "SUCCESS" and step.required:
                    logger.error(f"Required step failed: {step.description}")
                    success = False
                    break
            
            if success:
                # Verify deployment
                verification_success = await self._verify_deployment(deployment_record)
                if verification_success:
                    deployment_record.status = DeploymentStatus.COMPLETED
                    logger.info(f"Deployment completed successfully: {deployment_id}")
                else:
                    success = False
                    deployment_record.status = DeploymentStatus.FAILED
                    logger.error(f"Deployment verification failed: {deployment_id}")
            
            if not success:
                deployment_record.status = DeploymentStatus.FAILED
                
                # Auto-rollback if enabled
                if config.auto_rollback:
                    logger.info(f"Initiating auto-rollback for: {deployment_id}")
                    rollback_success = await self._rollback_deployment(deployment_record)
                    if rollback_success:
                        deployment_record.status = DeploymentStatus.ROLLED_BACK
            
            deployment_record.end_time = datetime.now(timezone.utc)
            await self._save_deployment_record(deployment_record)
            await self._notify_deployment_completed(deployment_record)
            
            return success
            
        except Exception as e:
            logger.error(f"Deployment task failed: {e}")
            deployment_record.status = DeploymentStatus.FAILED
            deployment_record.end_time = datetime.now(timezone.utc)
            await self._save_deployment_record(deployment_record)
            return False
    
    async def _create_deployment_steps(self, config: DeploymentConfig) -> List[DeploymentStep]:
        """Create deployment steps based on configuration."""
        steps = []
        
        # Preparation steps
        steps.append(DeploymentStep(
            stage=DeploymentStage.PREPARATION,
            command="prepare_deployment",
            description="Prepare deployment environment"
        ))
        
        if config.backup_enabled:
            steps.append(DeploymentStep(
                stage=DeploymentStage.PREPARATION,
                command="create_backup",
                description="Create deployment backup"
            ))
        
        # Build steps
        for command in config.build_commands:
            steps.append(DeploymentStep(
                stage=DeploymentStage.BUILD,
                command=command,
                description=f"Build: {command}"
            ))
        
        # Test steps
        for command in config.test_commands:
            steps.append(DeploymentStep(
                stage=DeploymentStage.TEST,
                command=command,
                description=f"Test: {command}"
            ))
        
        # Deploy steps
        for command in config.deploy_commands:
            steps.append(DeploymentStep(
                stage=DeploymentStage.DEPLOY,
                command=command,
                description=f"Deploy: {command}"
            ))
        
        # Verification steps
        for command in config.verification_commands:
            steps.append(DeploymentStep(
                stage=DeploymentStage.VERIFICATION,
                command=command,
                description=f"Verify: {command}"
            ))
        
        return steps
    
    async def _execute_deployment_step(self, step: DeploymentStep, config: DeploymentConfig) -> DeploymentResult:
        """Execute a single deployment step."""
        logger.info(f"Executing step: {step.description}")
        
        start_time = datetime.now(timezone.utc)
        result = DeploymentResult(step=step, status="RUNNING")
        
        try:
            # Handle special commands
            if step.command == "prepare_deployment":
                await self._prepare_deployment_environment(config)
                output = "Deployment environment prepared"
                
            elif step.command == "create_backup":
                backup_path = await self._create_backup(config)
                output = f"Backup created: {backup_path}"
                
            else:
                # Execute shell command
                output, error = await self._execute_command(
                    step.command, 
                    config.environment_variables,
                    step.timeout
                )
                
                if error:
                    result.status = "FAILED"
                    result.error = error
                    logger.error(f"Step failed: {step.description} - {error}")
                else:
                    result.status = "SUCCESS"
                    result.output = output
            
            if result.status != "FAILED":
                result.status = "SUCCESS"
                result.output = output
                
        except Exception as e:
            result.status = "ERROR"
            result.error = str(e)
            logger.error(f"Step error: {step.description} - {e}")
        
        finally:
            end_time = datetime.now(timezone.utc)
            result.duration = (end_time - start_time).total_seconds()
            result.timestamp = end_time
        
        return result
    
    async def _prepare_deployment_environment(self, config: DeploymentConfig):
        """Prepare deployment environment."""
        # Create staging directory
        staging_path = f"staging/{config.name}_{config.version}"
        os.makedirs(staging_path, exist_ok=True)
        
        # Copy source files
        if os.path.exists(config.source_path):
            shutil.copytree(config.source_path, staging_path, dirs_exist_ok=True)
    
    async def _create_backup(self, config: DeploymentConfig) -> str:
        """Create deployment backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{config.name}_{config.environment}_{timestamp}"
        backup_path = f"backups/{backup_name}"
        
        # Create backup directory
        os.makedirs(backup_path, exist_ok=True)
        
        # Backup current deployment if exists
        if os.path.exists(config.target_path):
            shutil.copytree(config.target_path, backup_path, dirs_exist_ok=True)
        
        return backup_path
    
    async def _execute_command(self, command: str, env_vars: Dict[str, str], timeout: int) -> Tuple[str, str]:
        """Execute shell command with timeout."""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(env_vars)
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            output = stdout.decode('utf-8') if stdout else ""
            error = stderr.decode('utf-8') if stderr else ""
            
            if process.returncode != 0:
                error = error or f"Command failed with return code {process.returncode}"
            
            return output, error
            
        except asyncio.TimeoutError:
            return "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return "", f"Command execution error: {e}"
    
    async def _verify_deployment(self, deployment_record: DeploymentRecord) -> bool:
        """Verify deployment success."""
        try:
            config = deployment_record.config
            
            # Basic file system checks
            if not os.path.exists(config.target_path):
                logger.error(f"Target path does not exist: {config.target_path}")
                return False
            
            # Run verification commands
            for command in config.verification_commands:
                output, error = await self._execute_command(command, config.environment_variables, 60)
                if error:
                    logger.error(f"Verification failed: {command} - {error}")
                    return False
            
            logger.info("Deployment verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Deployment verification error: {e}")
            return False
    
    async def _rollback_deployment(self, deployment_record: DeploymentRecord) -> bool:
        """Rollback failed deployment."""
        try:
            config = deployment_record.config
            
            logger.info(f"Rolling back deployment: {deployment_record.deployment_id}")
            
            # Execute rollback commands
            for command in config.rollback_commands:
                output, error = await self._execute_command(command, config.environment_variables, 300)
                if error:
                    logger.error(f"Rollback command failed: {command} - {error}")
                    return False
            
            # Restore from backup if available
            if deployment_record.backup_path and os.path.exists(deployment_record.backup_path):
                if os.path.exists(config.target_path):
                    shutil.rmtree(config.target_path)
                shutil.copytree(deployment_record.backup_path, config.target_path)
                logger.info("Restored from backup")
            
            logger.info("Deployment rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def _save_deployment_record(self, deployment_record: DeploymentRecord):
        """Save deployment record."""
        try:
            record_path = f"deployments/{deployment_record.deployment_id}.json"
            
            record_data = {
                "deployment_id": deployment_record.deployment_id,
                "config": deployment_record.config.__dict__,
                "status": deployment_record.status.value,
                "start_time": deployment_record.start_time.isoformat(),
                "end_time": deployment_record.end_time.isoformat() if deployment_record.end_time else None,
                "backup_path": deployment_record.backup_path,
                "results": [
                    {
                        "step": {
                            "stage": result.step.stage.value,
                            "command": result.step.command,
                            "description": result.step.description
                        },
                        "status": result.status,
                        "output": result.output,
                        "error": result.error,
                        "duration": result.duration,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for result in deployment_record.results
                ]
            }
            
            with open(record_path, 'w') as f:
                json.dump(record_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save deployment record: {e}")
    
    async def _notify_deployment_created(self, deployment_record: DeploymentRecord):
        """Notify that deployment was created."""
        await self._send_notification(
            f"Deployment Created: {deployment_record.deployment_id}",
            f"Created deployment for {deployment_record.config.name} v{deployment_record.config.version}"
        )
    
    async def _notify_deployment_status_change(self, deployment_record: DeploymentRecord):
        """Notify deployment status change."""
        await self._send_notification(
            f"Deployment Status: {deployment_record.status.value}",
            f"Deployment {deployment_record.deployment_id} status changed to {deployment_record.status.value}"
        )
    
    async def _notify_deployment_completed(self, deployment_record: DeploymentRecord):
        """Notify deployment completion."""
        status_emoji = "✅" if deployment_record.status == DeploymentStatus.COMPLETED else "❌"
        await self._send_notification(
            f"Deployment {deployment_record.status.value}: {deployment_record.deployment_id}",
            f"{status_emoji} Deployment {deployment_record.deployment_id} {deployment_record.status.value}"
        )
    
    async def _send_notification(self, title: str, message: str):
        """Send notification through configured channels."""
        try:
            for handler in self.notification_handlers:
                await handler(title, message)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler."""
        self.notification_handlers.append(handler)
    
    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel active deployment."""
        if deployment_id not in self.active_deployments:
            logger.warning(f"No active deployment to cancel: {deployment_id}")
            return False
        
        try:
            task = self.active_deployments[deployment_id]
            task.cancel()
            
            deployment_record = self.deployments[deployment_id]
            deployment_record.status = DeploymentStatus.CANCELLED
            deployment_record.end_time = datetime.now(timezone.utc)
            
            await self._save_deployment_record(deployment_record)
            logger.info(f"Deployment cancelled: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel deployment: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status."""
        if deployment_id not in self.deployments:
            return None
        
        deployment_record = self.deployments[deployment_id]
        return {
            "deployment_id": deployment_id,
            "status": deployment_record.status.value,
            "config": deployment_record.config.__dict__,
            "start_time": deployment_record.start_time.isoformat(),
            "end_time": deployment_record.end_time.isoformat() if deployment_record.end_time else None,
            "steps_completed": len(deployment_record.results),
            "is_active": deployment_id in self.active_deployments
        }
    
    def list_deployments(self, status_filter: Optional[DeploymentStatus] = None) -> List[Dict[str, Any]]:
        """List deployments with optional status filter."""
        deployments = []
        for deployment_record in self.deployments.values():
            if status_filter is None or deployment_record.status == status_filter:
                deployments.append({
                    "deployment_id": deployment_record.deployment_id,
                    "name": deployment_record.config.name,
                    "version": deployment_record.config.version,
                    "environment": deployment_record.config.environment,
                    "status": deployment_record.status.value,
                    "start_time": deployment_record.start_time.isoformat(),
                    "duration": (
                        (deployment_record.end_time or datetime.now(timezone.utc)) - 
                        deployment_record.start_time
                    ).total_seconds()
                })
        
        return sorted(deployments, key=lambda x: x["start_time"], reverse=True)

async def main():
    """Main execution function for testing."""
    coordinator = AutomatedDeploymentCoordinator()
    
    # Initialize
    if not await coordinator.initialize():
        print("Failed to initialize deployment coordinator")
        return
    
    # Create test deployment config
    config = DeploymentConfig(
        name="Platform3",
        version="1.0.0",
        environment="staging",
        source_path="./src",
        target_path="./deployment/staging",
        build_commands=["echo 'Building application...'"],
        test_commands=["echo 'Running tests...'"],
        deploy_commands=["echo 'Deploying application...'"],
        verification_commands=["echo 'Verifying deployment...'"],
        rollback_commands=["echo 'Rolling back deployment...'"]
    )
    
    # Create and execute deployment
    deployment_id = await coordinator.create_deployment(config)
    print(f"Created deployment: {deployment_id}")
    
    success = await coordinator.execute_deployment(deployment_id)
    print(f"Deployment result: {'SUCCESS' if success else 'FAILED'}")
    
    # Show deployment status
    status = coordinator.get_deployment_status(deployment_id)
    print(f"Final status: {status}")

if __name__ == "__main__":
    asyncio.run(main())
