const https = require('https');
const fs = require('fs');
const path = require('path');
const { createLogger } = require('../utils/logger');

const logger = createLogger('mTLSMiddleware');

class mTLSMiddleware {
  constructor(options = {}) {
    this.caCertPath = options.caCertPath || process.env.CA_CERT_PATH || './certs/ca-cert.pem';
    this.serverCertPath = options.serverCertPath || process.env.SERVER_CERT_PATH || './certs/api-gateway-cert.pem';
    this.serverKeyPath = options.serverKeyPath || process.env.SERVER_KEY_PATH || './certs/api-gateway-key.pem';
    this.requireClientCert = options.requireClientCert !== false;
    this.trustedServices = new Set(options.trustedServices || [
      'user-service',
      'trading-service',
      'market-data-service',
      'analytics-service',
      'notification-service',
      'compliance-service',
      'risk-service',
      'ml-service',
      'backtest-service',
      'data-quality-service',
      'order-execution-service',
      'qa-service',
      'service-discovery',
      'auth-service'
    ]);
    
    this.loadCertificates();
  }

  /**
   * Load SSL certificates
   */
  loadCertificates() {
    try {
      if (fs.existsSync(this.caCertPath)) {
        this.caCert = fs.readFileSync(this.caCertPath);
        logger.info('CA certificate loaded successfully');
      } else {
        logger.warn('CA certificate not found, mTLS validation will be disabled', {
          path: this.caCertPath
        });
      }

      if (fs.existsSync(this.serverCertPath) && fs.existsSync(this.serverKeyPath)) {
        this.serverCert = fs.readFileSync(this.serverCertPath);
        this.serverKey = fs.readFileSync(this.serverKeyPath);
        logger.info('Server certificates loaded successfully');
      } else {
        logger.warn('Server certificates not found, HTTPS will be disabled', {
          certPath: this.serverCertPath,
          keyPath: this.serverKeyPath
        });
      }
    } catch (error) {
      logger.error('Failed to load certificates', { error: error.message });
    }
  }

  /**
   * Get HTTPS server options for Express
   */
  getHttpsOptions() {
    if (!this.serverCert || !this.serverKey) {
      return null;
    }

    return {
      key: this.serverKey,
      cert: this.serverCert,
      ca: this.caCert,
      requestCert: this.requireClientCert,
      rejectUnauthorized: false, // We'll handle validation manually
      secureProtocol: 'TLSv1_2_method',
      ciphers: [
        'ECDHE-RSA-AES128-GCM-SHA256',
        'ECDHE-RSA-AES256-GCM-SHA384',
        'ECDHE-RSA-AES128-SHA256',
        'ECDHE-RSA-AES256-SHA384'
      ].join(':'),
      honorCipherOrder: true
    };
  }

  /**
   * Middleware to validate client certificates
   */
  validateClientCertificate() {
    return (req, res, next) => {
      // Skip validation for health checks and public endpoints
      if (this.isPublicEndpoint(req.path)) {
        return next();
      }

      // Skip if mTLS is not configured
      if (!this.caCert) {
        logger.debug('mTLS not configured, skipping certificate validation');
        return next();
      }

      try {
        const clientCert = req.socket.getPeerCertificate();
        
        if (!clientCert || !clientCert.subject) {
          if (this.requireClientCert) {
            logger.warn('Client certificate required but not provided', {
              path: req.path,
              ip: req.ip
            });
            return res.status(401).json({
              error: 'Client certificate required',
              timestamp: new Date().toISOString()
            });
          } else {
            return next();
          }
        }

        // Validate certificate
        const validationResult = this.validateCertificate(clientCert);
        if (!validationResult.valid) {
          logger.warn('Client certificate validation failed', {
            reason: validationResult.reason,
            subject: clientCert.subject?.CN,
            path: req.path
          });
          
          return res.status(403).json({
            error: 'Invalid client certificate',
            reason: validationResult.reason,
            timestamp: new Date().toISOString()
          });
        }

        // Extract service identity
        const serviceIdentity = this.extractServiceIdentity(clientCert);
        if (serviceIdentity) {
          req.serviceIdentity = serviceIdentity;
          req.isServiceRequest = true;
          
          logger.debug('Service authenticated via mTLS', {
            serviceName: serviceIdentity.serviceName,
            path: req.path
          });
        }

        next();
      } catch (error) {
        logger.error('Certificate validation error', {
          error: error.message,
          path: req.path
        });
        
        res.status(500).json({
          error: 'Certificate validation failed',
          timestamp: new Date().toISOString()
        });
      }
    };
  }

  /**
   * Validate client certificate
   */
  validateCertificate(clientCert) {
    try {
      // Check if certificate exists
      if (!clientCert || !clientCert.subject) {
        return { valid: false, reason: 'No certificate provided' };
      }

      // Check certificate validity period
      const now = new Date();
      const validFrom = new Date(clientCert.valid_from);
      const validTo = new Date(clientCert.valid_to);

      if (now < validFrom) {
        return { valid: false, reason: 'Certificate not yet valid' };
      }

      if (now > validTo) {
        return { valid: false, reason: 'Certificate has expired' };
      }

      // Check if service is trusted
      const serviceName = clientCert.subject.CN;
      if (!this.trustedServices.has(serviceName)) {
        return { valid: false, reason: 'Service not in trusted list' };
      }

      // Verify certificate chain (simplified)
      if (!this.verifyCertificateChain(clientCert)) {
        return { valid: false, reason: 'Certificate chain verification failed' };
      }

      return { valid: true };
    } catch (error) {
      return { valid: false, reason: 'Validation error: ' + error.message };
    }
  }

  /**
   * Extract service identity from certificate
   */
  extractServiceIdentity(clientCert) {
    try {
      const subject = clientCert.subject;
      if (!subject || !subject.CN) {
        return null;
      }

      return {
        serviceName: subject.CN,
        organization: subject.O || 'Unknown',
        organizationalUnit: subject.OU || 'Unknown',
        fingerprint: clientCert.fingerprint,
        serialNumber: clientCert.serialNumber,
        issuer: clientCert.issuer?.CN || 'Unknown',
        validFrom: new Date(clientCert.valid_from),
        validTo: new Date(clientCert.valid_to)
      };
    } catch (error) {
      logger.error('Failed to extract service identity', { error: error.message });
      return null;
    }
  }

  /**
   * Check if endpoint is public (doesn't require mTLS)
   */
  isPublicEndpoint(path) {
    const publicEndpoints = [
      '/health',
      '/api/info',
      '/api/auth/login',
      '/api/auth/register',
      '/favicon.ico'
    ];

    return publicEndpoints.some(endpoint => path.startsWith(endpoint));
  }

  /**
   * Verify certificate chain (simplified implementation)
   */
  verifyCertificateChain(clientCert) {
    try {
      // In a production environment, you would use a proper certificate
      // verification library like node-forge or call OpenSSL
      
      // For now, we'll do basic checks
      if (!clientCert.issuer) {
        return false;
      }

      // Check if issued by our CA
      const expectedIssuer = 'Platform3-CA';
      if (clientCert.issuer.CN !== expectedIssuer) {
        logger.warn('Certificate not issued by expected CA', {
          expected: expectedIssuer,
          actual: clientCert.issuer.CN
        });
        return false;
      }

      return true;
    } catch (error) {
      logger.error('Certificate chain verification error', { error: error.message });
      return false;
    }
  }

  /**
   * Add trusted service
   */
  addTrustedService(serviceName) {
    this.trustedServices.add(serviceName);
    logger.info('Service added to trusted list', { serviceName });
  }

  /**
   * Remove trusted service
   */
  removeTrustedService(serviceName) {
    this.trustedServices.delete(serviceName);
    logger.info('Service removed from trusted list', { serviceName });
  }

  /**
   * Get trusted services list
   */
  getTrustedServices() {
    return Array.from(this.trustedServices);
  }

  /**
   * Create HTTPS agent for outbound requests
   */
  createHttpsAgent(serviceName) {
    if (!this.serverCert || !this.serverKey || !this.caCert) {
      logger.warn('Certificates not available, using HTTP agent');
      return null;
    }

    return new https.Agent({
      cert: this.serverCert,
      key: this.serverKey,
      ca: this.caCert,
      rejectUnauthorized: true,
      checkServerIdentity: (host, cert) => {
        // Custom server identity check
        if (cert.subject && cert.subject.CN === serviceName) {
          return undefined; // Valid
        }
        return new Error(`Certificate subject ${cert.subject?.CN} does not match expected service ${serviceName}`);
      }
    });
  }

  /**
   * Health check for mTLS configuration
   */
  healthCheck() {
    const status = {
      mTLSEnabled: !!this.caCert,
      serverCertificatesLoaded: !!(this.serverCert && this.serverKey),
      trustedServicesCount: this.trustedServices.size,
      requireClientCert: this.requireClientCert
    };

    const isHealthy = status.mTLSEnabled && status.serverCertificatesLoaded;

    return {
      status: isHealthy ? 'healthy' : 'degraded',
      details: status
    };
  }
}

module.exports = mTLSMiddleware;
