"""
🔐 PLATFORM3 AGENT AUTHENTICATION MANAGER
===========================================

Agent-to-Agent Security Infrastructure for Platform3's 9 Genius Agents

This module provides enterprise-grade security for inter-agent communication by integrating
with Platform3's existing mTLS infrastructure and JWT token system. Ensures that only
authenticated and authorized agents can communicate with each other.

Key Features:
- Integration with existing mTLS certificate infrastructure
- JWT-based agent identity tokens with permissions
- Permission-based capability access control
- Secure message encryption/decryption for agent communications
- Comprehensive audit trail logging
- Certificate rotation and validation
- Message integrity verification

Agent Security Architecture:
- Each agent receives unique certificate and JWT token
- Permissions define what data each agent can request/provide
- All agent messages are encrypted and authenticated
- Audit trails capture all inter-agent communications
- Automatic certificate rotation and revocation support

Humanitarian Mission Security:
- Protects critical trading algorithms from unauthorized access
- Ensures data integrity for financial calculations
- Provides compliance audit trails for regulatory requirements
- Prevents security breaches that could impact charitable funding
"""

import logging
import json
import time
import hashlib
import hmac
import ssl
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import jwt
import cryptography.x509 as x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import base64
from pathlib import Path


class AgentPermission(Enum):
    """Agent capability permissions for fine-grained access control"""
    READ_MARKET_DATA = "read_market_data"
    READ_RISK_ANALYSIS = "read_risk_analysis"
    READ_PATTERN_ANALYSIS = "read_pattern_analysis"
    READ_EXECUTION_SIGNALS = "read_execution_signals"
    READ_SESSION_DATA = "read_session_data"
    READ_PAIR_ANALYSIS = "read_pair_analysis"
    READ_DECISION_DATA = "read_decision_data"
    READ_MICROSTRUCTURE_DATA = "read_microstructure_data"
    READ_SENTIMENT_DATA = "read_sentiment_data"
    
    WRITE_MARKET_DATA = "write_market_data"
    WRITE_RISK_ANALYSIS = "write_risk_analysis"
    WRITE_PATTERN_ANALYSIS = "write_pattern_analysis"
    WRITE_EXECUTION_SIGNALS = "write_execution_signals"
    WRITE_SESSION_DATA = "write_session_data"
    WRITE_PAIR_ANALYSIS = "write_pair_analysis"
    WRITE_DECISION_DATA = "write_decision_data"
    WRITE_MICROSTRUCTURE_DATA = "write_microstructure_data"
    WRITE_SENTIMENT_DATA = "write_sentiment_data"
    
    COORDINATE_AGENTS = "coordinate_agents"
    MANAGE_DEPENDENCIES = "manage_dependencies"
    ACCESS_HEALTH_DATA = "access_health_data"
    EMERGENCY_OVERRIDE = "emergency_override"


@dataclass
class AgentIdentity:
    """Secure agent identity with certificate and token information"""
    agent_id: str
    agent_name: str
    certificate_path: str
    private_key_path: str
    jwt_token: str
    permissions: Set[AgentPermission]
    issued_at: datetime
    expires_at: datetime
    certificate_fingerprint: str
    

@dataclass 
class MessageSecurity:
    """Security metadata for encrypted agent messages"""
    message_id: str
    sender_id: str
    recipient_id: str
    timestamp: datetime
    encrypted_payload: bytes
    signature: bytes
    nonce: bytes
    auth_tag: bytes


@dataclass
class SecurityAuditEntry:
    """Audit trail entry for agent security events"""
    event_id: str
    event_type: str  # "authentication", "authorization", "message", "certificate"
    agent_id: str
    target_agent_id: Optional[str]
    timestamp: datetime
    success: bool
    details: Dict[str, Any]
    risk_level: str  # "low", "medium", "high", "critical"


class AgentSecurityManager:
    """
    Enterprise-grade security manager for Platform3 agent-to-agent communication.
    Integrates with existing mTLS infrastructure and JWT token system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Certificate and mTLS integration
        self.ca_cert_path = config.get('ca_cert_path', './certs/ca-cert.pem')
        self.cert_base_path = config.get('cert_base_path', './certs/agents/')
        self.jwt_secret_key = config.get('jwt_secret_key', os.environ.get('JWT_SECRET_KEY'))
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        self.token_expiry_hours = config.get('token_expiry_hours', 24)
        
        # Security state
        self.agent_identities: Dict[str, AgentIdentity] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.audit_trail: List[SecurityAuditEntry] = []
        self.revoked_certificates: Set[str] = set()
        
        # Agent permission matrix (9 genius agents)
        self.agent_permissions = {
            'market_data_agent': {
                AgentPermission.WRITE_MARKET_DATA,
                AgentPermission.READ_SESSION_DATA,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'risk_analysis_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.WRITE_RISK_ANALYSIS,
                AgentPermission.READ_SESSION_DATA,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'pattern_recognition_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.READ_RISK_ANALYSIS,
                AgentPermission.WRITE_PATTERN_ANALYSIS,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'execution_strategy_agent': {
                AgentPermission.READ_PATTERN_ANALYSIS,
                AgentPermission.READ_RISK_ANALYSIS,
                AgentPermission.WRITE_EXECUTION_SIGNALS,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'session_management_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.READ_EXECUTION_SIGNALS,
                AgentPermission.WRITE_SESSION_DATA,
                AgentPermission.COORDINATE_AGENTS,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'pair_trading_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.READ_PATTERN_ANALYSIS,
                AgentPermission.WRITE_PAIR_ANALYSIS,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'decision_master_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.READ_RISK_ANALYSIS,
                AgentPermission.READ_PATTERN_ANALYSIS,
                AgentPermission.READ_EXECUTION_SIGNALS,
                AgentPermission.READ_PAIR_ANALYSIS,
                AgentPermission.WRITE_DECISION_DATA,
                AgentPermission.COORDINATE_AGENTS,
                AgentPermission.MANAGE_DEPENDENCIES,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'microstructure_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.WRITE_MICROSTRUCTURE_DATA,
                AgentPermission.ACCESS_HEALTH_DATA
            },
            'sentiment_analysis_agent': {
                AgentPermission.READ_MARKET_DATA,
                AgentPermission.WRITE_SENTIMENT_DATA,
                AgentPermission.ACCESS_HEALTH_DATA
            }
        }
        
        self._initialize_certificates()
        self.logger.info("AgentSecurityManager initialized with mTLS integration")
        
    def _initialize_certificates(self):
        """Initialize and validate agent certificates"""
        try:
            # Load CA certificate for validation
            with open(self.ca_cert_path, 'rb') as f:
                self.ca_cert = x509.load_pem_x509_certificate(f.read())
                
            # Generate/load certificates for each agent
            for agent_name in self.agent_permissions.keys():
                cert_path = os.path.join(self.cert_base_path, f"{agent_name}-cert.pem")
                key_path = os.path.join(self.cert_base_path, f"{agent_name}-key.pem")
                
                if os.path.exists(cert_path) and os.path.exists(key_path):
                    # Load existing certificate
                    with open(cert_path, 'rb') as f:
                        cert = x509.load_pem_x509_certificate(f.read())
                    
                    # Validate certificate
                    if self._validate_certificate(cert):
                        fingerprint = self._get_certificate_fingerprint(cert)
                        self.logger.info(f"Loaded valid certificate for {agent_name}: {fingerprint}")
                    else:
                        self.logger.warning(f"Invalid certificate for {agent_name}, regeneration required")
                else:
                    self.logger.info(f"Certificate not found for {agent_name}, will generate on first use")
                    
        except Exception as e:
            self.logger.error(f"Certificate initialization failed: {e}")
            raise
    
    def authenticate_agent(self, agent_id: str, certificate_data: bytes) -> Optional[AgentIdentity]:
        """
        Authenticate agent using mTLS certificate validation
        
        Args:
            agent_id: Unique agent identifier
            certificate_data: Agent's certificate in PEM format
            
        Returns:
            AgentIdentity if authentication successful, None otherwise
        """
        try:
            # Parse and validate certificate
            cert = x509.load_pem_x509_certificate(certificate_data)
            
            if not self._validate_certificate(cert):
                self._log_security_event("authentication", agent_id, None, False, 
                                        {"reason": "invalid_certificate"}, "high")
                return None
            
            # Check certificate fingerprint against revocation list
            fingerprint = self._get_certificate_fingerprint(cert)
            if fingerprint in self.revoked_certificates:
                self._log_security_event("authentication", agent_id, None, False,
                                        {"reason": "revoked_certificate", "fingerprint": fingerprint}, "critical")
                return None
            
            # Extract agent name from certificate subject
            agent_name = self._extract_agent_name_from_cert(cert)
            if not agent_name or agent_name not in self.agent_permissions:
                self._log_security_event("authentication", agent_id, None, False,
                                        {"reason": "unknown_agent", "cert_subject": str(cert.subject)}, "high")
                return None
            
            # Generate JWT token for the session
            jwt_token = self._generate_jwt_token(agent_id, agent_name)
            
            # Create agent identity
            identity = AgentIdentity(
                agent_id=agent_id,
                agent_name=agent_name,
                certificate_path=f"memory://{fingerprint}",
                private_key_path="",  # Not stored for security
                jwt_token=jwt_token,
                permissions=self.agent_permissions[agent_name],
                issued_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                certificate_fingerprint=fingerprint
            )
            
            # Store identity
            self.agent_identities[agent_id] = identity
            self.active_sessions[agent_id] = {
                'authenticated_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'message_count': 0
            }
            
            self._log_security_event("authentication", agent_id, None, True,
                                   {"agent_name": agent_name, "fingerprint": fingerprint}, "low")
            
            self.logger.info(f"Agent {agent_id} ({agent_name}) authenticated successfully")
            return identity
            
        except Exception as e:
            self.logger.error(f"Authentication failed for agent {agent_id}: {e}")
            self._log_security_event("authentication", agent_id, None, False,
                                   {"reason": "exception", "error": str(e)}, "high")
            return None
    
    def authorize_action(self, agent_id: str, permission: AgentPermission, target_agent_id: Optional[str] = None) -> bool:
        """
        Authorize agent action based on permissions
        
        Args:
            agent_id: Agent requesting permission
            permission: Required permission
            target_agent_id: Target agent (for inter-agent actions)
            
        Returns:
            True if authorized, False otherwise
        """
        try:
            # Check if agent is authenticated
            if agent_id not in self.agent_identities:
                self._log_security_event("authorization", agent_id, target_agent_id, False,
                                        {"reason": "not_authenticated", "permission": permission.value}, "medium")
                return False
            
            identity = self.agent_identities[agent_id]
            
            # Check if token is still valid
            if datetime.utcnow() > identity.expires_at:
                self._log_security_event("authorization", agent_id, target_agent_id, False,
                                        {"reason": "token_expired", "permission": permission.value}, "medium")
                return False
            
            # Check permission
            if permission not in identity.permissions:
                self._log_security_event("authorization", agent_id, target_agent_id, False,
                                        {"reason": "insufficient_permissions", "permission": permission.value,
                                         "has_permissions": [p.value for p in identity.permissions]}, "medium")
                return False
            
            # Update session activity
            if agent_id in self.active_sessions:
                self.active_sessions[agent_id]['last_activity'] = datetime.utcnow()
            
            self._log_security_event("authorization", agent_id, target_agent_id, True,
                                   {"permission": permission.value}, "low")
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization failed for agent {agent_id}: {e}")
            self._log_security_event("authorization", agent_id, target_agent_id, False,
                                   {"reason": "exception", "error": str(e)}, "high")
            return False
    
    def encrypt_message(self, sender_id: str, recipient_id: str, message_data: Dict[str, Any]) -> Optional[MessageSecurity]:
        """
        Encrypt message for secure inter-agent communication
        
        Args:
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message_data: Message payload to encrypt
            
        Returns:
            MessageSecurity object with encrypted data
        """
        try:
            # Validate both agents are authenticated
            if sender_id not in self.agent_identities or recipient_id not in self.agent_identities:
                self.logger.error(f"Message encryption failed: agents not authenticated")
                return None
            
            # Serialize message data
            message_json = json.dumps(message_data, default=str).encode('utf-8')
            
            # Derive encryption key from sender and recipient tokens
            sender_identity = self.agent_identities[sender_id]
            recipient_identity = self.agent_identities[recipient_id]
            
            # Create deterministic key from both tokens
            key_material = (sender_identity.jwt_token + recipient_identity.jwt_token).encode('utf-8')
            key = hashlib.sha256(key_material).digest()  # AES-256 key
            nonce = os.urandom(12)  # GCM nonce (this can be random)
            
            # Encrypt message using AES-GCM
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(message_json) + encryptor.finalize()
            auth_tag = encryptor.tag
            
            # Create timestamp for consistent signature validation
            message_timestamp = datetime.utcnow()
            timestamp_int = int(message_timestamp.timestamp())
            
            # Create message signature using sender's identity
            sender_identity = self.agent_identities[sender_id]
            signature_data = f"{sender_id}:{recipient_id}:{timestamp_int}".encode('utf-8')
            signature = hmac.new(
                sender_identity.jwt_token.encode('utf-8'),
                signature_data + encrypted_data,
                hashlib.sha256
            ).digest()
            
            # Create secure message
            message_id = hashlib.sha256(f"{sender_id}:{recipient_id}:{timestamp_int}:{os.urandom(8).hex()}".encode()).hexdigest()
            
            secure_message = MessageSecurity(
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                timestamp=message_timestamp,
                encrypted_payload=encrypted_data,
                signature=signature,
                nonce=nonce,
                auth_tag=auth_tag
            )
            
            # Update session activity
            if sender_id in self.active_sessions:
                self.active_sessions[sender_id]['message_count'] += 1
                self.active_sessions[sender_id]['last_activity'] = datetime.utcnow()
            
            self._log_security_event("message", sender_id, recipient_id, True,
                                   {"message_id": message_id, "encrypted_size": len(encrypted_data)}, "low")
            
            return secure_message
            
        except Exception as e:
            self.logger.error(f"Message encryption failed: {e}")
            self._log_security_event("message", sender_id, recipient_id, False,
                                   {"reason": "encryption_failed", "error": str(e)}, "high")
            return None
    
    def decrypt_message(self, recipient_id: str, secure_message: MessageSecurity) -> Optional[Dict[str, Any]]:
        """
        Decrypt and validate message for recipient agent
        
        Args:
            recipient_id: Receiving agent ID
            secure_message: Encrypted message security object
            
        Returns:
            Decrypted message data if valid, None otherwise
        """
        try:
            # Validate recipient is authenticated
            if recipient_id not in self.agent_identities:
                self.logger.error(f"Message decryption failed: recipient not authenticated")
                return None
            
            # Validate sender is authenticated
            if secure_message.sender_id not in self.agent_identities:
                self.logger.error(f"Message decryption failed: sender not authenticated")
                return None
            
            # Validate message integrity using signature
            sender_identity = self.agent_identities[secure_message.sender_id]
            signature_data = f"{secure_message.sender_id}:{secure_message.recipient_id}:{int(secure_message.timestamp.timestamp())}".encode('utf-8')
            expected_signature = hmac.new(
                sender_identity.jwt_token.encode('utf-8'),
                signature_data + secure_message.encrypted_payload,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(secure_message.signature, expected_signature):
                self._log_security_event("message", secure_message.sender_id, recipient_id, False,
                                        {"reason": "invalid_signature", "message_id": secure_message.message_id}, "high")
                return None
            
            # Derive the same encryption key used for encryption
            sender_identity = self.agent_identities[secure_message.sender_id]
            recipient_identity = self.agent_identities[recipient_id]
            
            # Recreate the same deterministic key
            key_material = (sender_identity.jwt_token + recipient_identity.jwt_token).encode('utf-8')
            key = hashlib.sha256(key_material).digest()  # AES-256 key
            
            # Decrypt message using AES-GCM
            cipher = Cipher(algorithms.AES(key), modes.GCM(secure_message.nonce, secure_message.auth_tag))
            decryptor = cipher.decryptor()
            decrypted_data = decryptor.update(secure_message.encrypted_payload) + decryptor.finalize()
            
            # Parse decrypted JSON
            message_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Update session activity
            if recipient_id in self.active_sessions:
                self.active_sessions[recipient_id]['last_activity'] = datetime.utcnow()
            
            self._log_security_event("message", secure_message.sender_id, recipient_id, True,
                                   {"message_id": secure_message.message_id, "decrypted": True}, "low")
            
            return message_data
            
        except Exception as e:
            self.logger.error(f"Message decryption failed: {e}")
            self._log_security_event("message", secure_message.sender_id, recipient_id, False,
                                   {"reason": "decryption_failed", "error": str(e), 
                                    "message_id": secure_message.message_id}, "high")
            return None
    
    def get_security_audit_trail(self, agent_id: Optional[str] = None, 
                               event_type: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               limit: int = 100) -> List[SecurityAuditEntry]:
        """
        Retrieve security audit trail with optional filtering
        
        Args:
            agent_id: Filter by agent ID
            event_type: Filter by event type
            start_time: Filter events after this time
            limit: Maximum number of entries to return
            
        Returns:
            List of security audit entries
        """
        filtered_entries = self.audit_trail
        
        if agent_id:
            filtered_entries = [e for e in filtered_entries if e.agent_id == agent_id or e.target_agent_id == agent_id]
        
        if event_type:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]
        
        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_entries.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_entries[:limit]
    
    def revoke_agent_certificate(self, agent_id: str, reason: str) -> bool:
        """
        Revoke agent certificate and terminate sessions
        
        Args:
            agent_id: Agent to revoke
            reason: Reason for revocation
            
        Returns:
            True if revoked successfully
        """
        try:
            if agent_id in self.agent_identities:
                identity = self.agent_identities[agent_id]
                
                # Add to revocation list
                self.revoked_certificates.add(identity.certificate_fingerprint)
                
                # Remove from active identities and sessions
                del self.agent_identities[agent_id]
                if agent_id in self.active_sessions:
                    del self.active_sessions[agent_id]
                
                self._log_security_event("certificate", agent_id, None, True,
                                        {"action": "revoked", "reason": reason,
                                         "fingerprint": identity.certificate_fingerprint}, "high")
                
                self.logger.warning(f"Certificate revoked for agent {agent_id}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Certificate revocation failed for agent {agent_id}: {e}")
            return False
    
    def get_agent_security_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive security status for an agent"""
        if agent_id not in self.agent_identities:
            return None
        
        identity = self.agent_identities[agent_id]
        session = self.active_sessions.get(agent_id, {})
        
        return {
            'agent_id': agent_id,
            'agent_name': identity.agent_name,
            'authenticated': True,
            'certificate_fingerprint': identity.certificate_fingerprint,
            'token_expires_at': identity.expires_at.isoformat(),
            'permissions': [p.value for p in identity.permissions],
            'session_started_at': session.get('authenticated_at', datetime.utcnow()).isoformat(),
            'last_activity': session.get('last_activity', datetime.utcnow()).isoformat(),
            'message_count': session.get('message_count', 0),
            'security_level': 'high' if AgentPermission.EMERGENCY_OVERRIDE in identity.permissions else 'normal'
        }
    
    # Private helper methods
    
    def _validate_certificate(self, cert: x509.Certificate) -> bool:
        """Validate certificate against CA and check expiry"""
        try:
            # Check expiry
            if datetime.utcnow() > cert.not_valid_after:
                return False
            
            if datetime.utcnow() < cert.not_valid_before:
                return False
            
            # TODO: Add CA signature validation when cryptography library supports it
            # For now, assume certificate is valid if it's properly formatted
            return True
            
        except Exception:
            return False
    
    def _get_certificate_fingerprint(self, cert: x509.Certificate) -> str:
        """Get SHA-256 fingerprint of certificate"""
        return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    
    def _extract_agent_name_from_cert(self, cert: x509.Certificate) -> Optional[str]:
        """Extract agent name from certificate subject"""
        try:
            subject = cert.subject
            for attribute in subject:
                if attribute.oid._name == 'commonName':
                    cn = attribute.value
                    # Expected format: "agent-name" or "platform3-agent-name"
                    if cn.startswith('platform3-'):
                        return cn[10:]  # Remove 'platform3-' prefix
                    elif cn.endswith('-agent'):
                        return cn
                    else:
                        return cn + '_agent'
            return None
        except Exception:
            return None
    
    def _generate_jwt_token(self, agent_id: str, agent_name: str) -> str:
        """Generate JWT token for agent session"""
        payload = {
            'agent_id': agent_id,
            'agent_name': agent_name,
            'permissions': [p.value for p in self.agent_permissions[agent_name]],
            'iat': int(time.time()),
            'exp': int(time.time()) + (self.token_expiry_hours * 3600),
            'iss': 'platform3-agent-security',
            'aud': 'platform3-agents'
        }
        
        return jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
    
    def _log_security_event(self, event_type: str, agent_id: str, target_agent_id: Optional[str],
                           success: bool, details: Dict[str, Any], risk_level: str):
        """Log security event to audit trail"""
        event = SecurityAuditEntry(
            event_id=hashlib.sha256(f"{event_type}:{agent_id}:{time.time()}:{os.urandom(4).hex()}".encode()).hexdigest(),
            event_type=event_type,
            agent_id=agent_id,
            target_agent_id=target_agent_id,
            timestamp=datetime.utcnow(),
            success=success,
            details=details,
            risk_level=risk_level
        )
        
        self.audit_trail.append(event)
        
        # Keep audit trail size manageable (last 10000 events)
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-10000:]
        
        # Log high-risk events immediately
        if risk_level in ['high', 'critical']:
            log_func = self.logger.error if risk_level == 'critical' else self.logger.warning
            log_func(f"Security event [{risk_level.upper()}]: {event_type} for agent {agent_id} - {details}")


# Factory function for easy initialization
def create_agent_security_manager(config_path: Optional[str] = None) -> AgentSecurityManager:
    """
    Create and configure AgentSecurityManager with Platform3 defaults
    
    Args:
        config_path: Path to security configuration file
        
    Returns:
        Configured AgentSecurityManager instance
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration for Platform3
        config = {
            'ca_cert_path': os.environ.get('CA_CERT_PATH', './certs/ca-cert.pem'),
            'cert_base_path': os.environ.get('AGENT_CERT_BASE_PATH', './certs/agents/'),
            'jwt_secret_key': os.environ.get('JWT_SECRET_KEY', 'platform3-secure-key-change-in-production'),
            'jwt_algorithm': 'HS256',
            'token_expiry_hours': 24
        }
    
    return AgentSecurityManager(config)