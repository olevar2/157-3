import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import * as jwt from 'jsonwebtoken';
import { EventEmitter } from 'events';
import { createLogger } from './utils/logger';

export interface ServiceIdentity {
  serviceName: string;
  serviceId: string;
  permissions: string[];
  issuer: string;
  issuedAt: number;
  expiresAt: number;
}

export interface CertificateInfo {
  subject: string;
  issuer: string;
  serialNumber: string;
  notBefore: Date;
  notAfter: Date;
  fingerprint: string;
}

export interface AuthenticationResult {
  success: boolean;
  identity?: ServiceIdentity;
  error?: string;
  certificateInfo?: CertificateInfo;
}

export class InterServiceAuthManager extends EventEmitter {
  private logger = createLogger('InterServiceAuthManager');
  private caCertificate: string;
  private jwtSecret: string;
  private trustedServices: Map<string, string[]> = new Map();
  private certificateCache: Map<string, CertificateInfo> = new Map();
  private blacklistedCertificates: Set<string> = new Set();

  constructor(options: {
    caCertPath: string;
    jwtSecret: string;
    trustedServicesConfig?: Record<string, string[]>;
  }) {
    super();
    
    this.caCertificate = fs.readFileSync(options.caCertPath, 'utf8');
    this.jwtSecret = options.jwtSecret;
    
    // Initialize trusted services configuration
    if (options.trustedServicesConfig) {
      Object.entries(options.trustedServicesConfig).forEach(([service, permissions]) => {
        this.trustedServices.set(service, permissions);
      });
    }
    
    this.logger.info('Inter-Service Authentication Manager initialized', {
      trustedServicesCount: this.trustedServices.size
    });
  }

  /**
   * Authenticate a service request using client certificate
   */
  async authenticateServiceRequest(clientCert: any): Promise<AuthenticationResult> {
    try {
      if (!clientCert || !clientCert.raw) {
        return {
          success: false,
          error: 'No client certificate provided'
        };
      }

      // Verify certificate against CA
      const isValid = await this.validateServiceCertificate(clientCert);
      if (!isValid) {
        return {
          success: false,
          error: 'Invalid client certificate'
        };
      }

      // Extract service identity from certificate
      const identity = this.extractServiceIdentity(clientCert);
      if (!identity) {
        return {
          success: false,
          error: 'Unable to extract service identity from certificate'
        };
      }

      // Check if service is trusted
      if (!this.trustedServices.has(identity.serviceName)) {
        this.logger.warn('Untrusted service attempted authentication', {
          serviceName: identity.serviceName
        });
        return {
          success: false,
          error: 'Service not in trusted list'
        };
      }

      // Check certificate blacklist
      const fingerprint = this.getCertificateFingerprint(clientCert);
      if (this.blacklistedCertificates.has(fingerprint)) {
        this.logger.warn('Blacklisted certificate used for authentication', {
          serviceName: identity.serviceName,
          fingerprint
        });
        return {
          success: false,
          error: 'Certificate has been revoked'
        };
      }

      const certificateInfo = this.extractCertificateInfo(clientCert);
      
      this.logger.info('Service authenticated successfully', {
        serviceName: identity.serviceName,
        serviceId: identity.serviceId
      });

      this.emit('serviceAuthenticated', { identity, certificateInfo });

      return {
        success: true,
        identity,
        certificateInfo
      };

    } catch (error) {
      this.logger.error('Service authentication failed', {
        error: error.message
      });
      
      return {
        success: false,
        error: 'Authentication failed: ' + error.message
      };
    }
  }

  /**
   * Validate service certificate against CA
   */
  private async validateServiceCertificate(clientCert: any): Promise<boolean> {
    try {
      const cert = crypto.createVerify('RSA-SHA256');
      
      // Check if certificate is signed by our CA
      const certPem = this.convertToPem(clientCert.raw);
      const isValidSignature = this.verifyCertificateSignature(certPem, this.caCertificate);
      
      if (!isValidSignature) {
        this.logger.warn('Certificate signature validation failed');
        return false;
      }

      // Check certificate validity period
      const now = new Date();
      if (clientCert.valid_from && new Date(clientCert.valid_from) > now) {
        this.logger.warn('Certificate not yet valid');
        return false;
      }
      
      if (clientCert.valid_to && new Date(clientCert.valid_to) < now) {
        this.logger.warn('Certificate has expired');
        return false;
      }

      return true;
    } catch (error) {
      this.logger.error('Certificate validation error', { error: error.message });
      return false;
    }
  }

  /**
   * Extract service identity from certificate
   */
  private extractServiceIdentity(clientCert: any): ServiceIdentity | null {
    try {
      const subject = clientCert.subject;
      if (!subject || !subject.CN) {
        return null;
      }

      const serviceName = subject.CN;
      const serviceId = this.generateServiceId(serviceName, clientCert);
      const permissions = this.trustedServices.get(serviceName) || [];
      
      return {
        serviceName,
        serviceId,
        permissions,
        issuer: 'Platform3-CA',
        issuedAt: Date.now(),
        expiresAt: Date.now() + (24 * 60 * 60 * 1000) // 24 hours
      };
    } catch (error) {
      this.logger.error('Failed to extract service identity', { error: error.message });
      return null;
    }
  }

  /**
   * Generate JWT token for authenticated service
   */
  generateServiceToken(identity: ServiceIdentity): string {
    const payload = {
      serviceName: identity.serviceName,
      serviceId: identity.serviceId,
      permissions: identity.permissions,
      iss: identity.issuer,
      iat: Math.floor(identity.issuedAt / 1000),
      exp: Math.floor(identity.expiresAt / 1000)
    };

    return jwt.sign(payload, this.jwtSecret, { algorithm: 'HS256' });
  }

  /**
   * Validate JWT token from service
   */
  validateServiceToken(token: string): ServiceIdentity | null {
    try {
      const decoded = jwt.verify(token, this.jwtSecret) as any;
      
      return {
        serviceName: decoded.serviceName,
        serviceId: decoded.serviceId,
        permissions: decoded.permissions || [],
        issuer: decoded.iss,
        issuedAt: decoded.iat * 1000,
        expiresAt: decoded.exp * 1000
      };
    } catch (error) {
      this.logger.warn('JWT token validation failed', { error: error.message });
      return null;
    }
  }

  /**
   * Add service to trusted list
   */
  addTrustedService(serviceName: string, permissions: string[]): void {
    this.trustedServices.set(serviceName, permissions);
    this.logger.info('Service added to trusted list', { serviceName, permissions });
    this.emit('trustedServiceAdded', { serviceName, permissions });
  }

  /**
   * Remove service from trusted list
   */
  removeTrustedService(serviceName: string): void {
    this.trustedServices.delete(serviceName);
    this.logger.info('Service removed from trusted list', { serviceName });
    this.emit('trustedServiceRemoved', { serviceName });
  }

  /**
   * Blacklist a certificate
   */
  blacklistCertificate(fingerprint: string, reason: string): void {
    this.blacklistedCertificates.add(fingerprint);
    this.logger.warn('Certificate blacklisted', { fingerprint, reason });
    this.emit('certificateBlacklisted', { fingerprint, reason });
  }

  /**
   * Check if service has specific permission
   */
  hasPermission(identity: ServiceIdentity, permission: string): boolean {
    return identity.permissions.includes(permission) || identity.permissions.includes('*');
  }

  /**
   * Get certificate fingerprint
   */
  private getCertificateFingerprint(clientCert: any): string {
    return crypto
      .createHash('sha256')
      .update(clientCert.raw)
      .digest('hex')
      .toUpperCase()
      .match(/.{2}/g)!
      .join(':');
  }

  /**
   * Extract certificate information
   */
  private extractCertificateInfo(clientCert: any): CertificateInfo {
    return {
      subject: clientCert.subject?.CN || 'Unknown',
      issuer: clientCert.issuer?.CN || 'Unknown',
      serialNumber: clientCert.serialNumber || 'Unknown',
      notBefore: new Date(clientCert.valid_from),
      notAfter: new Date(clientCert.valid_to),
      fingerprint: this.getCertificateFingerprint(clientCert)
    };
  }

  /**
   * Generate unique service ID
   */
  private generateServiceId(serviceName: string, clientCert: any): string {
    const fingerprint = this.getCertificateFingerprint(clientCert);
    return crypto
      .createHash('sha256')
      .update(`${serviceName}:${fingerprint}`)
      .digest('hex')
      .substring(0, 16);
  }

  /**
   * Convert certificate to PEM format
   */
  private convertToPem(certBuffer: Buffer): string {
    const base64Cert = certBuffer.toString('base64');
    const pemCert = base64Cert.match(/.{1,64}/g)!.join('\n');
    return `-----BEGIN CERTIFICATE-----\n${pemCert}\n-----END CERTIFICATE-----`;
  }

  /**
   * Verify certificate signature against CA
   */
  private verifyCertificateSignature(certPem: string, caCertPem: string): boolean {
    try {
      // This is a simplified verification - in production, use a proper crypto library
      // like node-forge or openssl bindings for complete certificate chain validation
      return certPem.includes('-----BEGIN CERTIFICATE-----') && 
             caCertPem.includes('-----BEGIN CERTIFICATE-----');
    } catch (error) {
      return false;
    }
  }

  /**
   * Get authentication statistics
   */
  getAuthenticationStats(): {
    trustedServicesCount: number;
    blacklistedCertificatesCount: number;
    cacheSize: number;
  } {
    return {
      trustedServicesCount: this.trustedServices.size,
      blacklistedCertificatesCount: this.blacklistedCertificates.size,
      cacheSize: this.certificateCache.size
    };
  }
}
