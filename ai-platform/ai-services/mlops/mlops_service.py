"""
Platform3 MLOps Service
Model lifecycle management, versioning, and deployment automation
"""

import os
import json
import logging
import shutil
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import subprocess

import sys
from ai_platform.ai_services.model_registry import get_registry, ModelStatus

class DeploymentStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelVersion(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"

@dataclass
class ModelVersionInfo:
    """Model version information"""
    model_id: str
    version: str
    stage: DeploymentStage
    created_at: datetime
    created_by: str
    description: str
    model_path: str
    config_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None
    parent_version: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_id: str
    version: str
    stage: DeploymentStage
    resource_requirements: Dict[str, Any]
    environment_variables: Dict[str, str]
    health_check_config: Dict[str, Any]
    scaling_config: Dict[str, Any]

class MLOpsService:
    """
    MLOps service for model lifecycle management
    Handles versioning, deployment, monitoring, and automation
    """
    
    def __init__(self, mlops_root: str = None):
        if mlops_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            mlops_root = os.path.join(project_root, "ai-platform", "mlops")
        self.mlops_root = Path(mlops_root)
        self.models_store = self.mlops_root / "model-store"
        self.configs_store = self.mlops_root / "configs"
        self.deployments_store = self.mlops_root / "deployments"
        self.metadata_store = self.mlops_root / "metadata"
        
        # Create directories
        for directory in [self.models_store, self.configs_store, 
                         self.deployments_store, self.metadata_store]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.registry = get_registry()
        self.versions: Dict[str, List[ModelVersionInfo]] = {}
        self.deployments: Dict[str, Dict[str, DeploymentConfig]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing MLOps metadata"""
        try:
            # Load versions
            versions_file = self.metadata_store / "versions.json"
            if versions_file.exists():
                with open(versions_file, 'r') as f:
                    data = json.load(f)
                    for model_id, versions_data in data.items():
                        self.versions[model_id] = [
                            ModelVersionInfo(
                                **{
                                    **version_data,
                                    'stage': DeploymentStage(version_data['stage']),
                                    'created_at': datetime.fromisoformat(version_data['created_at'])
                                }
                            )
                            for version_data in versions_data
                        ]
            
            # Load deployments
            deployments_file = self.metadata_store / "deployments.json"
            if deployments_file.exists():
                with open(deployments_file, 'r') as f:
                    data = json.load(f)
                    for model_id, stages_data in data.items():
                        self.deployments[model_id] = {}
                        for stage, deployment_data in stages_data.items():
                            self.deployments[model_id][stage] = DeploymentConfig(
                                **{
                                    **deployment_data,
                                    'stage': DeploymentStage(deployment_data['stage'])
                                }
                            )
            
        except Exception as e:
            self.logger.error(f"Error loading MLOps metadata: {e}")
    
    def _save_metadata(self):
        """Save MLOps metadata"""
        try:
            # Save versions
            versions_data = {}
            for model_id, versions in self.versions.items():
                versions_data[model_id] = [
                    {
                        **asdict(version),
                        'stage': version.stage.value,
                        'created_at': version.created_at.isoformat()
                    }
                    for version in versions
                ]
            
            with open(self.metadata_store / "versions.json", 'w') as f:
                json.dump(versions_data, f, indent=2)
            
            # Save deployments
            deployments_data = {}
            for model_id, stages in self.deployments.items():
                deployments_data[model_id] = {}
                for stage, deployment in stages.items():
                    deployments_data[model_id][stage] = {
                        **asdict(deployment),
                        'stage': deployment.stage.value
                    }
            
            with open(self.metadata_store / "deployments.json", 'w') as f:
                json.dump(deployments_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving MLOps metadata: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def create_model_version(self, 
                           model_id: str,
                           source_path: str,
                           version_type: ModelVersion = ModelVersion.MINOR,
                           description: str = "",
                           created_by: str = "system",
                           config: Optional[Dict[str, Any]] = None,
                           metrics: Optional[Dict[str, float]] = None,
                           tags: Optional[List[str]] = None) -> str:
        """Create a new model version"""
        
        with self.lock:
            # Get current version
            current_versions = self.versions.get(model_id, [])
            
            if current_versions:
                latest_version = max(current_versions, key=lambda v: v.created_at)
                current_version_parts = latest_version.version.split('.')
            else:
                current_version_parts = ['0', '0', '0']
            
            # Calculate new version
            major, minor, patch = map(int, current_version_parts)
            
            if version_type == ModelVersion.MAJOR:
                major += 1
                minor = 0
                patch = 0
            elif version_type == ModelVersion.MINOR:
                minor += 1
                patch = 0
            else:  # PATCH
                patch += 1
            
            new_version = f"{major}.{minor}.{patch}"
            
            # Create model store directory
            version_dir = self.models_store / model_id / new_version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            if os.path.isfile(source_path):
                target_path = version_dir / os.path.basename(source_path)
                shutil.copy2(source_path, target_path)
                model_path = str(target_path)
            else:
                target_path = version_dir / "model"
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                model_path = str(target_path)
            
            # Calculate checksum
            if os.path.isfile(model_path):
                checksum = self._calculate_checksum(model_path)
            else:
                # For directories, calculate combined checksum
                checksums = []
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        checksums.append(self._calculate_checksum(file_path))
                checksum = hashlib.sha256(''.join(sorted(checksums)).encode()).hexdigest()
            
            # Save config if provided
            config_path = None
            if config:
                config_path = str(version_dir / "config.json")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Create version info
            version_info = ModelVersionInfo(
                model_id=model_id,
                version=new_version,
                stage=DeploymentStage.DEVELOPMENT,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                model_path=model_path,
                config_path=config_path,
                metrics=metrics or {},
                tags=tags or [],
                parent_version=current_versions[-1].version if current_versions else None,
                checksum=checksum
            )
            
            # Store version
            if model_id not in self.versions:
                self.versions[model_id] = []
            self.versions[model_id].append(version_info)
            
            # Save metadata
            self._save_metadata()
            
            self.logger.info(f"Created version {new_version} for model {model_id}")
            return new_version
    
    def promote_model(self, 
                     model_id: str, 
                     version: str, 
                     target_stage: DeploymentStage) -> bool:
        """Promote a model version to a different stage"""
        
        with self.lock:
            # Find version
            version_info = self._get_version_info(model_id, version)
            if not version_info:
                raise ValueError(f"Version {version} not found for model {model_id}")
            
            # Validate promotion path
            current_stage = version_info.stage
            valid_promotions = {
                DeploymentStage.DEVELOPMENT: [DeploymentStage.STAGING],
                DeploymentStage.STAGING: [DeploymentStage.PRODUCTION, DeploymentStage.DEVELOPMENT],
                DeploymentStage.PRODUCTION: [DeploymentStage.ARCHIVED, DeploymentStage.STAGING],
                DeploymentStage.ARCHIVED: [DeploymentStage.DEVELOPMENT]
            }
            
            if target_stage not in valid_promotions.get(current_stage, []):
                raise ValueError(f"Cannot promote from {current_stage.value} to {target_stage.value}")
            
            # Update stage
            version_info.stage = target_stage
            
            # Save metadata
            self._save_metadata()
            
            self.logger.info(f"Promoted model {model_id} version {version} to {target_stage.value}")
            return True
    
    def _get_version_info(self, model_id: str, version: str) -> Optional[ModelVersionInfo]:
        """Get version information"""
        if model_id in self.versions:
            for version_info in self.versions[model_id]:
                if version_info.version == version:
                    return version_info
        return None
    
    def get_model_versions(self, model_id: str) -> List[ModelVersionInfo]:
        """Get all versions for a model"""
        return self.versions.get(model_id, [])
    
    def get_latest_version(self, model_id: str, stage: Optional[DeploymentStage] = None) -> Optional[ModelVersionInfo]:
        """Get latest version for a model, optionally filtered by stage"""
        versions = self.get_model_versions(model_id)
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        if versions:
            return max(versions, key=lambda v: v.created_at)
        
        return None
    
    def create_deployment(self, 
                         model_id: str,
                         version: str,
                         stage: DeploymentStage,
                         resource_requirements: Optional[Dict[str, Any]] = None,
                         environment_variables: Optional[Dict[str, str]] = None,
                         health_check_config: Optional[Dict[str, Any]] = None,
                         scaling_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a deployment configuration"""
        
        # Validate version exists
        version_info = self._get_version_info(model_id, version)
        if not version_info:
            raise ValueError(f"Version {version} not found for model {model_id}")
        
        # Default configurations
        default_resources = {
            'cpu': '500m',
            'memory': '1Gi',
            'gpu': None
        }
        
        default_env = {
            'MODEL_ID': model_id,
            'MODEL_VERSION': version,
            'DEPLOYMENT_STAGE': stage.value
        }
        
        default_health_check = {
            'endpoint': '/health',
            'interval_seconds': 30,
            'timeout_seconds': 10,
            'failure_threshold': 3
        }
        
        default_scaling = {
            'min_replicas': 1,
            'max_replicas': 5,
            'target_cpu_utilization': 70
        }
        
        # Create deployment config
        deployment_config = DeploymentConfig(
            model_id=model_id,
            version=version,
            stage=stage,
            resource_requirements=resource_requirements or default_resources,
            environment_variables={**default_env, **(environment_variables or {})},
            health_check_config=health_check_config or default_health_check,
            scaling_config=scaling_config or default_scaling
        )
        
        with self.lock:
            if model_id not in self.deployments:
                self.deployments[model_id] = {}
            
            self.deployments[model_id][stage.value] = deployment_config
            
            # Save deployment configuration to file
            deployment_file = self.deployments_store / f"{model_id}_{stage.value}_deployment.json"
            with open(deployment_file, 'w') as f:
                json.dump({
                    **asdict(deployment_config),
                    'stage': deployment_config.stage.value
                }, f, indent=2)
            
            # Save metadata
            self._save_metadata()
        
        self.logger.info(f"Created deployment for {model_id} v{version} in {stage.value}")
        return True
    
    def deploy_model(self, model_id: str, stage: DeploymentStage) -> bool:
        """Deploy a model to the specified stage"""
        
        # Get deployment config
        if model_id not in self.deployments or stage.value not in self.deployments[model_id]:
            raise ValueError(f"No deployment configuration found for {model_id} in {stage.value}")
        
        deployment_config = self.deployments[model_id][stage.value]
        
        try:
            # For now, this is a simplified deployment
            # In a real implementation, this would interact with Kubernetes, Docker, etc.
            
            # Update model status in registry
            model_info = self.registry.get_model(model_id)
            if model_info:
                if stage == DeploymentStage.PRODUCTION:
                    self.registry.update_model_status(model_id, ModelStatus.ACTIVE)
                else:
                    self.registry.update_model_status(model_id, ModelStatus.INACTIVE)
            
            # Create deployment manifest (simplified)
            manifest = self._create_deployment_manifest(deployment_config)
            manifest_file = self.deployments_store / f"{model_id}_{stage.value}_manifest.yaml"
            
            with open(manifest_file, 'w') as f:
                f.write(manifest)
            
            self.logger.info(f"Deployed {model_id} to {stage.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying {model_id} to {stage.value}: {e}")
            return False
    
    def _create_deployment_manifest(self, deployment_config: DeploymentConfig) -> str:
        """Create Kubernetes deployment manifest"""
        
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_config.model_id}-{deployment_config.stage.value}
  labels:
    app: {deployment_config.model_id}
    version: {deployment_config.version}
    stage: {deployment_config.stage.value}
spec:
  replicas: {deployment_config.scaling_config['min_replicas']}
  selector:
    matchLabels:
      app: {deployment_config.model_id}
      stage: {deployment_config.stage.value}
  template:
    metadata:
      labels:
        app: {deployment_config.model_id}
        version: {deployment_config.version}
        stage: {deployment_config.stage.value}
    spec:
      containers:
      - name: model-server
        image: platform3/model-server:latest
        ports:
        - containerPort: 8080
        env:
"""
        
        # Add environment variables
        for key, value in deployment_config.environment_variables.items():
            manifest += f"        - name: {key}\n          value: \"{value}\"\n"
        
        # Add resource requirements
        manifest += f"""
        resources:
          requests:
            cpu: {deployment_config.resource_requirements['cpu']}
            memory: {deployment_config.resource_requirements['memory']}
          limits:
            cpu: {deployment_config.resource_requirements['cpu']}
            memory: {deployment_config.resource_requirements['memory']}
        livenessProbe:
          httpGet:
            path: {deployment_config.health_check_config['endpoint']}
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: {deployment_config.health_check_config['interval_seconds']}
          timeoutSeconds: {deployment_config.health_check_config['timeout_seconds']}
          failureThreshold: {deployment_config.health_check_config['failure_threshold']}
        readinessProbe:
          httpGet:
            path: {deployment_config.health_check_config['endpoint']}
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: {deployment_config.model_id}-{deployment_config.stage.value}-service
spec:
  selector:
    app: {deployment_config.model_id}
    stage: {deployment_config.stage.value}
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
"""
        
        return manifest
    
    def rollback_model(self, model_id: str, stage: DeploymentStage, target_version: str) -> bool:
        """Rollback a model to a previous version"""
        
        # Validate target version exists
        version_info = self._get_version_info(model_id, target_version)
        if not version_info:
            raise ValueError(f"Target version {target_version} not found for model {model_id}")
        
        # Update deployment configuration
        if model_id in self.deployments and stage.value in self.deployments[model_id]:
            deployment_config = self.deployments[model_id][stage.value]
            deployment_config.version = target_version
            
            # Re-deploy
            success = self.deploy_model(model_id, stage)
            
            if success:
                self.logger.info(f"Rolled back {model_id} in {stage.value} to version {target_version}")
            
            return success
        
        return False
    
    def compare_versions(self, model_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        
        v1_info = self._get_version_info(model_id, version1)
        v2_info = self._get_version_info(model_id, version2)
        
        if not v1_info or not v2_info:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'model_id': model_id,
            'version1': {
                'version': v1_info.version,
                'stage': v1_info.stage.value,
                'created_at': v1_info.created_at.isoformat(),
                'metrics': v1_info.metrics,
                'checksum': v1_info.checksum
            },
            'version2': {
                'version': v2_info.version,
                'stage': v2_info.stage.value,
                'created_at': v2_info.created_at.isoformat(),
                'metrics': v2_info.metrics,
                'checksum': v2_info.checksum
            },
            'differences': {
                'checksum_changed': v1_info.checksum != v2_info.checksum,
                'stage_changed': v1_info.stage != v2_info.stage,
                'metrics_comparison': {}
            }
        }
        
        # Compare metrics
        if v1_info.metrics and v2_info.metrics:
            all_metrics = set(v1_info.metrics.keys()) | set(v2_info.metrics.keys())
            for metric in all_metrics:
                v1_value = v1_info.metrics.get(metric, 0)
                v2_value = v2_info.metrics.get(metric, 0)
                comparison['differences']['metrics_comparison'][metric] = {
                    'version1': v1_value,
                    'version2': v2_value,
                    'difference': v2_value - v1_value,
                    'percentage_change': ((v2_value - v1_value) / v1_value * 100) if v1_value != 0 else None
                }
        
        return comparison
    
    def get_deployment_status(self, model_id: str, stage: DeploymentStage) -> Dict[str, Any]:
        """Get deployment status"""
        
        if model_id not in self.deployments or stage.value not in self.deployments[model_id]:
            return {'status': 'not_deployed', 'message': 'No deployment found'}
        
        deployment_config = self.deployments[model_id][stage.value]
        
        # In a real implementation, this would query the actual deployment status
        # For now, return configuration info
        return {
            'status': 'deployed',
            'model_id': model_id,
            'version': deployment_config.version,
            'stage': deployment_config.stage.value,
            'resource_requirements': deployment_config.resource_requirements,
            'scaling_config': deployment_config.scaling_config,
            'last_updated': datetime.now().isoformat()
        }
    
    def cleanup_old_versions(self, model_id: str, keep_versions: int = 5) -> int:
        """Clean up old model versions"""
        
        versions = self.get_model_versions(model_id)
        if len(versions) <= keep_versions:
            return 0
        
        # Sort by creation time, keep latest versions
        sorted_versions = sorted(versions, key=lambda v: v.created_at, reverse=True)
        versions_to_remove = sorted_versions[keep_versions:]
        
        removed_count = 0
        
        with self.lock:
            for version_info in versions_to_remove:
                # Don't remove if it's in production or staging
                if version_info.stage in [DeploymentStage.PRODUCTION, DeploymentStage.STAGING]:
                    continue
                
                try:
                    # Remove model files
                    version_dir = Path(version_info.model_path).parent
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                    
                    # Remove from tracking
                    self.versions[model_id].remove(version_info)
                    removed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error removing version {version_info.version}: {e}")
            
            # Save metadata
            self._save_metadata()
        
        self.logger.info(f"Cleaned up {removed_count} old versions for model {model_id}")
        return removed_count
    
    def export_model(self, model_id: str, version: str, export_path: str) -> bool:
        """Export a model version for external use"""
        
        version_info = self._get_version_info(model_id, version)
        if not version_info:
            raise ValueError(f"Version {version} not found for model {model_id}")
        
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            model_source = Path(version_info.model_path)
            if model_source.is_file():
                shutil.copy2(model_source, export_dir / model_source.name)
            else:
                shutil.copytree(model_source, export_dir / "model", dirs_exist_ok=True)
            
            # Copy config if exists
            if version_info.config_path:
                shutil.copy2(version_info.config_path, export_dir / "config.json")
            
            # Create metadata file
            metadata = {
                'model_id': model_id,
                'version': version,
                'export_timestamp': datetime.now().isoformat(),
                'checksum': version_info.checksum,
                'metrics': version_info.metrics,
                'description': version_info.description
            }
            
            with open(export_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Exported {model_id} v{version} to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return False

# Global MLOps service instance
_mlops_service = None

def get_mlops_service() -> MLOpsService:
    """Get global MLOps service instance"""
    global _mlops_service
    if _mlops_service is None:
        _mlops_service = MLOpsService()
    return _mlops_service

if __name__ == "__main__":
    # Test the MLOps service
    mlops = MLOpsService()
    
    # Get model versions (example)
    print("MLOps Service initialized successfully")
