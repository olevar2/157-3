#!/usr/bin/env python3
"""
PLATFORM3 TEST CERTIFICATE GENERATOR
=====================================

Generates test certificates for agent authentication development and testing.
This creates a complete CA hierarchy with agent certificates for all 9 genius agents.

SECURITY NOTE: These are for TESTING ONLY - never use in production!
"""

import os
import sys
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

def generate_private_key():
    """Generate a private key for certificates"""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

def generate_ca_certificate():
    """Generate a Certificate Authority (CA) certificate"""
    # Generate CA private key
    ca_key = generate_private_key()
    
    # Create CA certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Digital"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Cloud"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Platform3 Humanitarian Trading"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Platform3 Test CA"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        ca_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("platform3-ca"),
        ]),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=True,
            crl_sign=True,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    ).sign(ca_key, hashes.SHA256())
    
    return ca_key, cert

def generate_agent_certificate(agent_name, ca_key, ca_cert):
    """Generate a certificate for a specific agent"""
    # Generate agent private key
    agent_key = generate_private_key()
    
    # Create agent certificate
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Digital"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Cloud"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Platform3 Humanitarian Trading"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Intelligent Agents"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"platform3-{agent_name}"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        ca_cert.issuer
    ).public_key(
        agent_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=30)  # Short expiry for testing
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName(f"{agent_name}.platform3.local"),
            x509.DNSName(f"platform3-{agent_name}"),
        ]),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=False, path_length=None),
        critical=True,
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    ).add_extension(
        x509.ExtendedKeyUsage([
            x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
        ]),
        critical=True,
    ).sign(ca_key, hashes.SHA256())
    
    return agent_key, cert

def save_certificate_and_key(cert, key, cert_path, key_path):
    """Save certificate and private key to files"""
    # Save certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Save private key
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

def main():
    """Generate all test certificates for Platform3 agents"""
    print("PLATFORM3 TEST CERTIFICATE GENERATOR")
    print("="*50)
    
    # Certificate directory
    certs_dir = "./certs"
    agents_dir = os.path.join(certs_dir, "agents")
    
    # Ensure directories exist
    os.makedirs(certs_dir, exist_ok=True)
    os.makedirs(agents_dir, exist_ok=True)
    
    print("Generating Certificate Authority (CA)...")
    
    # Generate CA certificate
    ca_key, ca_cert = generate_ca_certificate()
    
    # Save CA certificate and key
    ca_cert_path = os.path.join(certs_dir, "ca-cert.pem")
    ca_key_path = os.path.join(certs_dir, "ca-key.pem")
    
    save_certificate_and_key(ca_cert, ca_key, ca_cert_path, ca_key_path)
    print(f"CA Certificate: {ca_cert_path}")
    print(f"CA Private Key: {ca_key_path}")
    
    # Define all 9 genius agents
    agents = [
        'market_data_agent',
        'risk_analysis_agent', 
        'pattern_recognition_agent',
        'execution_strategy_agent',
        'session_management_agent',
        'pair_trading_agent',
        'decision_master_agent',
        'microstructure_agent',
        'sentiment_analysis_agent'
    ]
    
    print(f"\nGenerating certificates for {len(agents)} agents...")
    
    # Generate agent certificates
    for agent_name in agents:
        print(f"Generating certificate for {agent_name}...")
        
        agent_key, agent_cert = generate_agent_certificate(agent_name, ca_key, ca_cert)
        
        # Save agent certificate and key
        agent_cert_path = os.path.join(agents_dir, f"{agent_name}-cert.pem")
        agent_key_path = os.path.join(agents_dir, f"{agent_name}-key.pem")
        
        save_certificate_and_key(agent_cert, agent_key, agent_cert_path, agent_key_path)
        print(f"   Certificate: {agent_cert_path}")
        print(f"   Private Key: {agent_key_path}")
    
    print(f"\nCertificate generation complete!")
    print(f"All certificates saved to: {os.path.abspath(certs_dir)}")
    print("\nWARNING: These certificates are for TESTING ONLY!")
    print("Never use these certificates in production environments.")
    
    # Create a summary file
    summary_path = os.path.join(certs_dir, "certificate_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PLATFORM3 TEST CERTIFICATE SUMMARY\n")
        f.write("="*40 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"CA Certificate: ca-cert.pem\n")
        f.write(f"CA Private Key: ca-key.pem\n")
        f.write(f"Total Agent Certificates: {len(agents)}\n\n")
        f.write("Agent Certificates:\n")
        for agent in agents:
            f.write(f"  - {agent}-cert.pem / {agent}-key.pem\n")
        f.write("\nWARNING: FOR TESTING ONLY!\n")
    
    print(f"Certificate summary: {summary_path}")

if __name__ == "__main__":
    main()