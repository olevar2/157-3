"""
Production Deployment Configuration for Advanced Analytics Framework
Sets up production-ready deployment with Redis clustering, load balancing, and monitoring
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class AnalyticsProductionConfig:
    """Production configuration for analytics framework"""
    
    def __init__(self, environment: str = "production"):
        """Initialize production configuration"""
        self.environment = environment
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load production configuration"""
        return {
            "environment": self.environment,
            "redis": {
                "cluster_mode": True,
                "nodes": [
                    {"host": "redis-analytics-1", "port": 6379},
                    {"host": "redis-analytics-2", "port": 6379},
                    {"host": "redis-analytics-3", "port": 6379}
                ],
                "password": os.getenv("REDIS_PASSWORD", ""),
                "ssl": True,
                "health_check_interval": 30,
                "failover_timeout": 5000
            },
            "websocket": {
                "load_balancer": {
                    "enabled": True,
                    "strategy": "round_robin",
                    "health_check": "/health",
                    "instances": [
                        {"host": "analytics-ws-1", "port": 8080},
                        {"host": "analytics-ws-2", "port": 8080},
                        {"host": "analytics-ws-3", "port": 8080}
                    ]
                },
                "connection_limits": {
                    "max_connections": 10000,
                    "max_connections_per_ip": 100,
                    "connection_timeout": 30,
                    "heartbeat_interval": 30
                }
            },
            "api": {
                "load_balancer": {
                    "enabled": True,
                    "strategy": "least_connections",
                    "instances": [
                        {"host": "analytics-api-1", "port": 8000},
                        {"host": "analytics-api-2", "port": 8000},
                        {"host": "analytics-api-3", "port": 8000}
                    ]
                },
                "rate_limiting": {
                    "requests_per_minute": 1000,
                    "burst_limit": 100,
                    "whitelist_ips": []
                },
                "security": {
                    "cors_origins": ["https://platform3.com", "https://api.platform3.com"],
                    "api_key_required": True,
                    "jwt_verification": True
                }
            },
            "database": {
                "connection_pool": {
                    "min_connections": 10,
                    "max_connections": 100,
                    "connection_timeout": 30,
                    "idle_timeout": 300
                },
                "read_replicas": [
                    {"host": "db-analytics-read-1", "port": 5432},
                    {"host": "db-analytics-read-2", "port": 5432}
                ],
                "write_primary": {"host": "db-analytics-write", "port": 5432},
                "backup": {
                    "enabled": True,
                    "interval": "6h",
                    "retention_days": 30
                }
            },
            "monitoring": {
                "metrics": {
                    "prometheus_endpoint": "/metrics",
                    "collection_interval": 15,
                    "retention_days": 7
                },
                "alerting": {
                    "webhook_url": os.getenv("ALERT_WEBHOOK_URL", ""),
                    "thresholds": {
                        "cpu_usage": 80,
                        "memory_usage": 85,
                        "error_rate": 5,
                        "response_time": 1000
                    }
                },
                "logging": {
                    "level": "INFO",
                    "format": "json",
                    "aggregation": "elasticsearch",
                    "retention_days": 30
                }
            },
            "scaling": {
                "auto_scaling": {
                    "enabled": True,
                    "min_instances": 3,
                    "max_instances": 20,
                    "scale_up_threshold": 70,
                    "scale_down_threshold": 30,
                    "cooldown_minutes": 5
                },
                "resource_limits": {
                    "cpu": "2000m",
                    "memory": "4Gi",
                    "storage": "50Gi"
                }
            }
        }

def generate_docker_compose_production():
    """Generate production Docker Compose configuration"""
    config = {
        "version": "3.8",
        "services": {
            # Redis Cluster
            "redis-analytics-1": {
                "image": "redis:7-alpine",
                "command": [
                    "redis-server",
                    "--cluster-enabled", "yes",
                    "--cluster-config-file", "nodes.conf",
                    "--cluster-node-timeout", "5000",
                    "--appendonly", "yes",
                    "--requirepass", "${REDIS_PASSWORD}"
                ],
                "ports": ["6379:6379"],
                "volumes": [
                    "redis-analytics-1-data:/data",
                    "./redis/redis.conf:/usr/local/etc/redis/redis.conf"
                ],
                "networks": ["analytics-network"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "1G", "cpus": "0.5"},
                        "reservations": {"memory": "512M", "cpus": "0.25"}
                    }
                }
            },
            "redis-analytics-2": {
                "image": "redis:7-alpine",
                "command": [
                    "redis-server",
                    "--cluster-enabled", "yes",
                    "--cluster-config-file", "nodes.conf",
                    "--cluster-node-timeout", "5000",
                    "--appendonly", "yes",
                    "--requirepass", "${REDIS_PASSWORD}"
                ],
                "ports": ["6380:6379"],
                "volumes": [
                    "redis-analytics-2-data:/data",
                    "./redis/redis.conf:/usr/local/etc/redis/redis.conf"
                ],
                "networks": ["analytics-network"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "1G", "cpus": "0.5"},
                        "reservations": {"memory": "512M", "cpus": "0.25"}
                    }
                }
            },
            "redis-analytics-3": {
                "image": "redis:7-alpine",
                "command": [
                    "redis-server",
                    "--cluster-enabled", "yes",
                    "--cluster-config-file", "nodes.conf",
                    "--cluster-node-timeout", "5000",
                    "--appendonly", "yes",
                    "--requirepass", "${REDIS_PASSWORD}"
                ],
                "ports": ["6381:6379"],
                "volumes": [
                    "redis-analytics-3-data:/data",
                    "./redis/redis.conf:/usr/local/etc/redis/redis.conf"
                ],
                "networks": ["analytics-network"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "1G", "cpus": "0.5"},
                        "reservations": {"memory": "512M", "cpus": "0.25"}
                    }
                }
            },
            
            # Analytics API Instances
            "analytics-api-1": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.analytics-api"
                },
                "environment": [
                    "REDIS_CLUSTER_NODES=redis-analytics-1:6379,redis-analytics-2:6379,redis-analytics-3:6379",
                    "REDIS_PASSWORD=${REDIS_PASSWORD}",
                    "DATABASE_URL=${DATABASE_URL}",
                    "JWT_SECRET=${JWT_SECRET}",
                    "LOG_LEVEL=INFO"
                ],
                "ports": ["8001:8000"],
                "networks": ["analytics-network"],
                "depends_on": ["redis-analytics-1", "redis-analytics-2", "redis-analytics-3"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "2G", "cpus": "1.0"},
                        "reservations": {"memory": "1G", "cpus": "0.5"}
                    }
                },
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            },
            "analytics-api-2": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.analytics-api"
                },
                "environment": [
                    "REDIS_CLUSTER_NODES=redis-analytics-1:6379,redis-analytics-2:6379,redis-analytics-3:6379",
                    "REDIS_PASSWORD=${REDIS_PASSWORD}",
                    "DATABASE_URL=${DATABASE_URL}",
                    "JWT_SECRET=${JWT_SECRET}",
                    "LOG_LEVEL=INFO"
                ],
                "ports": ["8002:8000"],
                "networks": ["analytics-network"],
                "depends_on": ["redis-analytics-1", "redis-analytics-2", "redis-analytics-3"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "2G", "cpus": "1.0"},
                        "reservations": {"memory": "1G", "cpus": "0.5"}
                    }
                },
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            },
            
            # WebSocket Instances
            "analytics-websocket-1": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.analytics-websocket"
                },
                "environment": [
                    "REDIS_CLUSTER_NODES=redis-analytics-1:6379,redis-analytics-2:6379,redis-analytics-3:6379",
                    "REDIS_PASSWORD=${REDIS_PASSWORD}",
                    "LOG_LEVEL=INFO"
                ],
                "ports": ["8081:8080"],
                "networks": ["analytics-network"],
                "depends_on": ["redis-analytics-1", "redis-analytics-2", "redis-analytics-3"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "1G", "cpus": "0.5"},
                        "reservations": {"memory": "512M", "cpus": "0.25"}
                    }
                }
            },
            "analytics-websocket-2": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.analytics-websocket"
                },
                "environment": [
                    "REDIS_CLUSTER_NODES=redis-analytics-1:6379,redis-analytics-2:6379,redis-analytics-3:6379",
                    "REDIS_PASSWORD=${REDIS_PASSWORD}",
                    "LOG_LEVEL=INFO"
                ],
                "ports": ["8082:8080"],
                "networks": ["analytics-network"],
                "depends_on": ["redis-analytics-1", "redis-analytics-2", "redis-analytics-3"],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "1G", "cpus": "0.5"},
                        "reservations": {"memory": "512M", "cpus": "0.25"}
                    }
                }
            },
            
            # Load Balancer
            "analytics-lb": {
                "image": "nginx:alpine",
                "ports": ["80:80", "443:443"],
                "volumes": [
                    "./nginx/nginx.conf:/etc/nginx/nginx.conf",
                    "./nginx/ssl:/etc/nginx/ssl"
                ],
                "networks": ["analytics-network"],
                "depends_on": [
                    "analytics-api-1", "analytics-api-2",
                    "analytics-websocket-1", "analytics-websocket-2"
                ],
                "deploy": {
                    "resources": {
                        "limits": {"memory": "512M", "cpus": "0.5"},
                        "reservations": {"memory": "256M", "cpus": "0.25"}
                    }
                }
            },
            
            # Monitoring
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": [
                    "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                    "prometheus-data:/prometheus"
                ],
                "networks": ["analytics-network"],
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.path=/prometheus",
                    "--web.console.libraries=/etc/prometheus/console_libraries",
                    "--web.console.templates=/etc/prometheus/consoles",
                    "--storage.tsdb.retention.time=168h",
                    "--web.enable-lifecycle"
                ]
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}"
                ],
                "volumes": [
                    "grafana-data:/var/lib/grafana",
                    "./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards",
                    "./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources"
                ],
                "networks": ["analytics-network"],
                "depends_on": ["prometheus"]
            }
        },
        
        "networks": {
            "analytics-network": {
                "driver": "bridge",
                "ipam": {
                    "config": [{"subnet": "172.20.0.0/16"}]
                }
            }
        },
        
        "volumes": {
            "redis-analytics-1-data": {},
            "redis-analytics-2-data": {},
            "redis-analytics-3-data": {},
            "prometheus-data": {},
            "grafana-data": {}
        }
    }
    
    return config

def generate_nginx_config():
    """Generate Nginx load balancer configuration"""
    config = """
upstream analytics_api {
    least_conn;
    server analytics-api-1:8000 max_fails=3 fail_timeout=30s;
    server analytics-api-2:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream analytics_websocket {
    ip_hash;
    server analytics-websocket-1:8080 max_fails=3 fail_timeout=30s;
    server analytics-websocket-2:8080 max_fails=3 fail_timeout=30s;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=ws_limit:10m rate=50r/m;

server {
    listen 80;
    server_name analytics.platform3.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name analytics.platform3.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/analytics.crt;
    ssl_certificate_key /etc/nginx/ssl/analytics.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://analytics_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
    }
    
    # WebSocket endpoints
    location /ws/ {
        limit_req zone=ws_limit burst=10 nodelay;
        
        proxy_pass http://analytics_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific settings
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering off;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
    
    # Metrics endpoint (restricted access)
    location /metrics {
        allow 172.20.0.0/16;  # Only allow internal network
        deny all;
        
        proxy_pass http://analytics_api;
        proxy_set_header Host $host;
    }
}
"""
    return config

def generate_prometheus_config():
    """Generate Prometheus monitoring configuration"""
    config = {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s"
        },
        "alerting": {
            "alertmanagers": [
                {
                    "static_configs": [
                        {"targets": ["alertmanager:9093"]}
                    ]
                }
            ]
        },
        "rule_files": [
            "alert_rules.yml"
        ],
        "scrape_configs": [
            {
                "job_name": "analytics-api",
                "static_configs": [
                    {"targets": ["analytics-api-1:8000", "analytics-api-2:8000"]}
                ],
                "metrics_path": "/metrics",
                "scrape_interval": "15s"
            },
            {
                "job_name": "analytics-websocket",
                "static_configs": [
                    {"targets": ["analytics-websocket-1:8080", "analytics-websocket-2:8080"]}
                ],
                "metrics_path": "/metrics",
                "scrape_interval": "15s"
            },
            {
                "job_name": "redis-cluster",
                "static_configs": [
                    {"targets": ["redis-analytics-1:6379", "redis-analytics-2:6379", "redis-analytics-3:6379"]}
                ],
                "metrics_path": "/metrics",
                "scrape_interval": "30s"
            },
            {
                "job_name": "nginx",
                "static_configs": [
                    {"targets": ["analytics-lb:80"]}
                ],
                "metrics_path": "/nginx_status",
                "scrape_interval": "30s"
            }
        ]
    }
    return config

def generate_kubernetes_manifests():
    """Generate Kubernetes deployment manifests"""
    manifests = {
        "namespace.yaml": {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "platform3-analytics"
            }
        },
        
        "redis-cluster.yaml": {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "redis-cluster",
                "namespace": "platform3-analytics"
            },
            "spec": {
                "serviceName": "redis-cluster",
                "replicas": 3,
                "selector": {
                    "matchLabels": {"app": "redis-cluster"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "redis-cluster"}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "redis",
                                "image": "redis:7-alpine",
                                "ports": [{"containerPort": 6379}],
                                "resources": {
                                    "limits": {"memory": "1Gi", "cpu": "500m"},
                                    "requests": {"memory": "512Mi", "cpu": "250m"}
                                },
                                "volumeMounts": [
                                    {
                                        "name": "redis-data",
                                        "mountPath": "/data"
                                    }
                                ]
                            }
                        ]
                    }
                },
                "volumeClaimTemplates": [
                    {
                        "metadata": {"name": "redis-data"},
                        "spec": {
                            "accessModes": ["ReadWriteOnce"],
                            "resources": {"requests": {"storage": "10Gi"}}
                        }
                    }
                ]
            }
        },
        
        "analytics-api.yaml": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "analytics-api",
                "namespace": "platform3-analytics"
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {"app": "analytics-api"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "analytics-api"}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "analytics-api",
                                "image": "platform3/analytics-api:latest",
                                "ports": [{"containerPort": 8000}],
                                "resources": {
                                    "limits": {"memory": "2Gi", "cpu": "1000m"},
                                    "requests": {"memory": "1Gi", "cpu": "500m"}
                                },
                                "env": [
                                    {"name": "REDIS_CLUSTER_NODES", "value": "redis-cluster-0:6379,redis-cluster-1:6379,redis-cluster-2:6379"},
                                    {"name": "LOG_LEVEL", "value": "INFO"}
                                ],
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ]
                    }
                }
            }
        }
    }
    
    return manifests

def create_deployment_script():
    """Create deployment script"""
    script = """#!/bin/bash

# Platform3 Analytics Production Deployment Script

set -e

echo "Starting Platform3 Analytics Production Deployment..."

# Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Create necessary directories
echo "Creating deployment directories..."
mkdir -p ./redis ./nginx ./monitoring/grafana/dashboards ./monitoring/grafana/datasources ./ssl

# Generate configurations
echo "Generating configuration files..."
python3 generate_configs.py

# Set up SSL certificates (self-signed for development)
if [ ! -f "./ssl/analytics.crt" ]; then
    echo "Generating SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\
        -keyout ./ssl/analytics.key \\
        -out ./ssl/analytics.crt \\
        -subj "/C=US/ST=State/L=City/O=Platform3/OU=Analytics/CN=analytics.platform3.com"
fi

# Initialize Redis cluster
echo "Initializing Redis cluster..."
docker-compose up -d redis-analytics-1 redis-analytics-2 redis-analytics-3
sleep 10

# Create Redis cluster
docker exec -it redis-analytics-1 redis-cli --cluster create \\
    redis-analytics-1:6379 redis-analytics-2:6379 redis-analytics-3:6379 \\
    --cluster-replicas 0 --cluster-yes

# Deploy remaining services
echo "Deploying analytics services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Run health checks
echo "Running health checks..."
curl -f http://localhost/health || { echo "Health check failed"; exit 1; }

echo "Deployment completed successfully!"
echo "Analytics API available at: https://localhost/api/"
echo "WebSocket endpoint available at: wss://localhost/ws/"
echo "Grafana dashboard available at: http://localhost:3000"
echo "Prometheus metrics available at: http://localhost:9090"

# Show service status
docker-compose ps
"""
    return script

async def main():
    """Generate all production deployment files"""
    logger.info("Generating production deployment configuration")
    
    # Create deployment directory
    deployment_dir = Path("deployment/production")
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Docker Compose
    docker_compose = generate_docker_compose_production()
    with open(deployment_dir / "docker-compose.yml", 'w') as f:
        yaml.dump(docker_compose, f, default_flow_style=False)
    
    # Generate Nginx config
    nginx_config = generate_nginx_config()
    nginx_dir = deployment_dir / "nginx"
    nginx_dir.mkdir(exist_ok=True)
    with open(nginx_dir / "nginx.conf", 'w') as f:
        f.write(nginx_config)
    
    # Generate Prometheus config
    prometheus_config = generate_prometheus_config()
    monitoring_dir = deployment_dir / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)
    with open(monitoring_dir / "prometheus.yml", 'w') as f:
        yaml.dump(prometheus_config, f, default_flow_style=False)
    
    # Generate Kubernetes manifests
    k8s_manifests = generate_kubernetes_manifests()
    k8s_dir = deployment_dir / "kubernetes"
    k8s_dir.mkdir(exist_ok=True)
    
    for filename, manifest in k8s_manifests.items():
        with open(k8s_dir / filename, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    
    # Generate deployment script
    deploy_script = create_deployment_script()
    with open(deployment_dir / "deploy.sh", 'w') as f:
        f.write(deploy_script)
    
    # Make script executable
    os.chmod(deployment_dir / "deploy.sh", 0o755)
    
    # Generate production configuration
    prod_config = AnalyticsProductionConfig()
    with open(deployment_dir / "production_config.json", 'w') as f:
        json.dump(prod_config.config, f, indent=2, default=str)
    
    logger.info(f"Production deployment configuration generated in {deployment_dir}")
    
    return {
        "deployment_directory": str(deployment_dir),
        "files_generated": [
            "docker-compose.yml",
            "nginx/nginx.conf", 
            "monitoring/prometheus.yml",
            "kubernetes/namespace.yaml",
            "kubernetes/redis-cluster.yaml",
            "kubernetes/analytics-api.yaml",
            "deploy.sh",
            "production_config.json"
        ],
        "status": "completed"
    }

if __name__ == "__main__":
    asyncio.run(main())
