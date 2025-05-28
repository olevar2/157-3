import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { EventEmitter } from 'events';
import { createLogger } from './utils/logger';

export interface TLSConfiguration {
  serviceName: string;
  certPath: string;
  keyPath: string;
  caCertPath: string;
  enabled: boolean;
  requireClientCert: boolean;
  allowedCiphers: string[];
  protocols: string[];
}

export interface CertificateRotationConfig {
  enabled: boolean;
  checkInterval: number; // milliseconds
  renewBeforeExpiry: number; // days
  backupOldCerts: boolean;
}

export class mTLSConfigService extends EventEmitter {
  private logger = createLogger('mTLSConfigService');
  private configurations: Map<string, TLSConfiguration> = new Map();
  private rotationConfig: CertificateRotationConfig;
  private rotationTimer?: NodeJS.Timeout;
  private certificateWatchers: Map<string, fs.FSWatcher> = new Map();

  constructor(options: {
    rotationConfig?: Partial<CertificateRotationConfig>;
    autoWatch?: boolean;
  } = {}) {
    super();
    
    this.rotationConfig = {
      enabled: true,
      checkInterval: 24 * 60 * 60 * 1000, // 24 hours
      renewBeforeExpiry: 30, // 30 days
      backupOldCerts: true,
      ...options.rotationConfig
    };

    if (options.autoWatch !== false) {
      this.startCertificateWatching();
    }

    if (this.rotationConfig.enabled) {
      this.startRotationMonitoring();
    }

    this.logger.info('mTLS Configuration Service initialized', {
      rotationEnabled: this.rotationConfig.enabled,
      checkInterval: this.rotationConfig.checkInterval
    });
  }

  /**
   * Register TLS configuration for a service
   */
  registerServiceConfig(config: TLSConfiguration): void {
    try {
      // Validate configuration
      this.validateConfiguration(config);
      
      // Store configuration
      this.configurations.set(config.serviceName, config);
      
      // Start watching certificate files
      if (config.enabled) {
        this.watchCertificateFiles(config);
      }
      
      this.logger.info('Service TLS configuration registered', {
        serviceName: config.serviceName,
        enabled: config.enabled
      });
      
      this.emit('configRegistered', config);
    } catch (error) {
      this.logger.error('Failed to register service configuration', {
        serviceName: config.serviceName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Get TLS configuration for a service
   */
  getServiceConfig(serviceName: string): TLSConfiguration | null {
    return this.configurations.get(serviceName) || null;
  }

  /**
   * Update TLS configuration for a service
   */
  updateServiceConfig(serviceName: string, updates: Partial<TLSConfiguration>): void {
    const existing = this.configurations.get(serviceName);
    if (!existing) {
      throw new Error(`No configuration found for service: ${serviceName}`);
    }

    const updated = { ...existing, ...updates };
    this.validateConfiguration(updated);
    
    this.configurations.set(serviceName, updated);
    
    this.logger.info('Service TLS configuration updated', {
      serviceName,
      updates: Object.keys(updates)
    });
    
    this.emit('configUpdated', { serviceName, config: updated });
  }

  /**
   * Generate TLS options for Node.js HTTPS server
   */
  generateTLSOptions(serviceName: string): any {
    const config = this.configurations.get(serviceName);
    if (!config || !config.enabled) {
      return null;
    }

    try {
      const options: any = {
        key: fs.readFileSync(config.keyPath),
        cert: fs.readFileSync(config.certPath),
        ca: fs.readFileSync(config.caCertPath),
        requestCert: config.requireClientCert,
        rejectUnauthorized: false, // We handle validation manually
        secureProtocol: 'TLSv1_2_method'
      };

      if (config.allowedCiphers && config.allowedCiphers.length > 0) {
        options.ciphers = config.allowedCiphers.join(':');
        options.honorCipherOrder = true;
      }

      return options;
    } catch (error) {
      this.logger.error('Failed to generate TLS options', {
        serviceName,
        error: error.message
      });
      return null;
    }
  }

  /**
   * Check certificate expiry for all services
   */
  async checkCertificateExpiry(): Promise<Map<string, { daysUntilExpiry: number; needsRenewal: boolean }>> {
    const results = new Map();

    for (const [serviceName, config] of this.configurations) {
      if (!config.enabled) continue;

      try {
        const certInfo = await this.getCertificateInfo(config.certPath);
        const now = new Date();
        const expiry = new Date(certInfo.notAfter);
        const daysUntilExpiry = Math.ceil((expiry.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
        const needsRenewal = daysUntilExpiry <= this.rotationConfig.renewBeforeExpiry;

        results.set(serviceName, { daysUntilExpiry, needsRenewal });

        if (needsRenewal) {
          this.logger.warn('Certificate needs renewal', {
            serviceName,
            daysUntilExpiry,
            expiryDate: expiry.toISOString()
          });
          
          this.emit('certificateNeedsRenewal', {
            serviceName,
            daysUntilExpiry,
            expiryDate: expiry
          });
        }
      } catch (error) {
        this.logger.error('Failed to check certificate expiry', {
          serviceName,
          error: error.message
        });
      }
    }

    return results;
  }

  /**
   * Rotate certificate for a service
   */
  async rotateCertificate(serviceName: string, newCertPath: string, newKeyPath: string): Promise<void> {
    const config = this.configurations.get(serviceName);
    if (!config) {
      throw new Error(`No configuration found for service: ${serviceName}`);
    }

    try {
      // Validate new certificate
      await this.validateCertificateFiles(newCertPath, newKeyPath, config.caCertPath);

      // Backup old certificates if enabled
      if (this.rotationConfig.backupOldCerts) {
        await this.backupCertificates(config);
      }

      // Update configuration with new paths
      const oldCertPath = config.certPath;
      const oldKeyPath = config.keyPath;
      
      config.certPath = newCertPath;
      config.keyPath = newKeyPath;

      this.logger.info('Certificate rotated successfully', {
        serviceName,
        oldCertPath,
        newCertPath
      });

      this.emit('certificateRotated', {
        serviceName,
        oldCertPath,
        newCertPath,
        timestamp: new Date()
      });
    } catch (error) {
      this.logger.error('Certificate rotation failed', {
        serviceName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Get certificate information
   */
  private async getCertificateInfo(certPath: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const cert = fs.readFileSync(certPath);
      const certString = cert.toString();
      
      // Parse certificate using crypto module
      try {
        const x509 = crypto.X509Certificate ? new crypto.X509Certificate(cert) : null;
        if (x509) {
          resolve({
            subject: x509.subject,
            issuer: x509.issuer,
            notBefore: x509.validFrom,
            notAfter: x509.validTo,
            serialNumber: x509.serialNumber,
            fingerprint: x509.fingerprint
          });
        } else {
          // Fallback for older Node.js versions
          reject(new Error('X509Certificate not available in this Node.js version'));
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Validate TLS configuration
   */
  private validateConfiguration(config: TLSConfiguration): void {
    if (!config.serviceName) {
      throw new Error('Service name is required');
    }

    if (config.enabled) {
      if (!config.certPath || !config.keyPath || !config.caCertPath) {
        throw new Error('Certificate paths are required when TLS is enabled');
      }

      // Check if files exist
      if (!fs.existsSync(config.certPath)) {
        throw new Error(`Certificate file not found: ${config.certPath}`);
      }
      
      if (!fs.existsSync(config.keyPath)) {
        throw new Error(`Private key file not found: ${config.keyPath}`);
      }
      
      if (!fs.existsSync(config.caCertPath)) {
        throw new Error(`CA certificate file not found: ${config.caCertPath}`);
      }
    }
  }

  /**
   * Validate certificate files
   */
  private async validateCertificateFiles(certPath: string, keyPath: string, caCertPath: string): Promise<void> {
    // Check if files exist
    if (!fs.existsSync(certPath)) {
      throw new Error(`Certificate file not found: ${certPath}`);
    }
    
    if (!fs.existsSync(keyPath)) {
      throw new Error(`Private key file not found: ${keyPath}`);
    }
    
    if (!fs.existsSync(caCertPath)) {
      throw new Error(`CA certificate file not found: ${caCertPath}`);
    }

    // Additional validation could include:
    // - Verify certificate and key match
    // - Verify certificate is signed by CA
    // - Check certificate validity period
  }

  /**
   * Watch certificate files for changes
   */
  private watchCertificateFiles(config: TLSConfiguration): void {
    const watchPaths = [config.certPath, config.keyPath, config.caCertPath];
    
    watchPaths.forEach(filePath => {
      if (this.certificateWatchers.has(filePath)) {
        return; // Already watching
      }

      try {
        const watcher = fs.watch(filePath, (eventType) => {
          if (eventType === 'change') {
            this.logger.info('Certificate file changed', {
              serviceName: config.serviceName,
              filePath
            });
            
            this.emit('certificateFileChanged', {
              serviceName: config.serviceName,
              filePath,
              timestamp: new Date()
            });
          }
        });

        this.certificateWatchers.set(filePath, watcher);
      } catch (error) {
        this.logger.warn('Failed to watch certificate file', {
          filePath,
          error: error.message
        });
      }
    });
  }

  /**
   * Start certificate watching
   */
  private startCertificateWatching(): void {
    this.logger.info('Starting certificate file watching');
  }

  /**
   * Start rotation monitoring
   */
  private startRotationMonitoring(): void {
    this.rotationTimer = setInterval(async () => {
      try {
        await this.checkCertificateExpiry();
      } catch (error) {
        this.logger.error('Certificate expiry check failed', {
          error: error.message
        });
      }
    }, this.rotationConfig.checkInterval);

    this.logger.info('Certificate rotation monitoring started', {
      interval: this.rotationConfig.checkInterval
    });
  }

  /**
   * Backup certificates
   */
  private async backupCertificates(config: TLSConfiguration): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupDir = path.join(path.dirname(config.certPath), 'backup', timestamp);
    
    fs.mkdirSync(backupDir, { recursive: true });
    
    fs.copyFileSync(config.certPath, path.join(backupDir, path.basename(config.certPath)));
    fs.copyFileSync(config.keyPath, path.join(backupDir, path.basename(config.keyPath)));
    
    this.logger.info('Certificates backed up', {
      serviceName: config.serviceName,
      backupDir
    });
  }

  /**
   * Stop all watchers and timers
   */
  shutdown(): void {
    if (this.rotationTimer) {
      clearInterval(this.rotationTimer);
    }

    for (const watcher of this.certificateWatchers.values()) {
      watcher.close();
    }
    
    this.certificateWatchers.clear();
    
    this.logger.info('mTLS Configuration Service shutdown completed');
  }
}
