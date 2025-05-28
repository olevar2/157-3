const { InterServiceAuthManager } = require('../dist/InterServiceAuthManager');
const { mTLSConfigService } = require('../dist/mTLSConfigService');
const fs = require('fs');
const path = require('path');

describe('mTLS Authentication Tests', () => {
  let authManager;
  let configService;

  const mockCertificate = {
    subject: { CN: 'test-service', O: 'Platform3', OU: 'Services' },
    issuer: { CN: 'Platform3-CA' },
    valid_from: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), // Yesterday
    valid_to: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString(), // Next year
    serialNumber: '123456789',
    fingerprint: 'AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD',
    raw: Buffer.from('mock-certificate-data')
  };

  beforeAll(() => {
    // Mock CA certificate for testing
    const mockCaCert = '-----BEGIN CERTIFICATE-----\nMOCK_CA_CERTIFICATE\n-----END CERTIFICATE-----';

    // Mock fs.readFileSync for CA certificate BEFORE creating instances
    jest.spyOn(fs, 'readFileSync').mockReturnValue(mockCaCert);

    authManager = new InterServiceAuthManager({
      caCertPath: './test-ca-cert.pem',
      jwtSecret: 'test-secret-key',
      trustedServicesConfig: {
        'test-service': ['read', 'write'],
        'api-gateway': ['*'],
        'user-service': ['user:read', 'user:write']
      }
    });

    configService = new mTLSConfigService({
      rotationConfig: {
        enabled: false // Disable for testing
      },
      autoWatch: false
    });
  });

  afterAll(() => {
    jest.restoreAllMocks();
    if (configService) {
      configService.shutdown();
    }
  });

  describe('InterServiceAuthManager', () => {
    test('should authenticate valid service certificate', async () => {
      const result = await authManager.authenticateServiceRequest(mockCertificate);

      expect(result.success).toBe(true);
      expect(result.identity).toBeDefined();
      expect(result.identity.serviceName).toBe('test-service');
      expect(result.identity.permissions).toEqual(['read', 'write']);
    });

    test('should reject certificate from untrusted service', async () => {
      const untrustedCert = {
        ...mockCertificate,
        subject: { CN: 'untrusted-service', O: 'Platform3', OU: 'Services' }
      };

      const result = await authManager.authenticateServiceRequest(untrustedCert);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Service not in trusted list');
    });

    test('should reject expired certificate', async () => {
      const expiredCert = {
        ...mockCertificate,
        valid_to: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString() // Yesterday
      };

      const result = await authManager.authenticateServiceRequest(expiredCert);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid client certificate');
    });

    test('should reject certificate not yet valid', async () => {
      const futureCert = {
        ...mockCertificate,
        valid_from: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString() // Tomorrow
      };

      const result = await authManager.authenticateServiceRequest(futureCert);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid client certificate');
    });

    test('should generate and validate JWT tokens', () => {
      const identity = {
        serviceName: 'test-service',
        serviceId: 'test-123',
        permissions: ['read', 'write'],
        issuer: 'Platform3-CA',
        issuedAt: Date.now(),
        expiresAt: Date.now() + 24 * 60 * 60 * 1000
      };

      const token = authManager.generateServiceToken(identity);
      expect(token).toBeDefined();
      expect(typeof token).toBe('string');

      const validatedIdentity = authManager.validateServiceToken(token);
      expect(validatedIdentity).toBeDefined();
      expect(validatedIdentity.serviceName).toBe('test-service');
      expect(validatedIdentity.permissions).toEqual(['read', 'write']);
    });

    test('should check permissions correctly', () => {
      const identity = {
        serviceName: 'test-service',
        serviceId: 'test-123',
        permissions: ['read', 'write'],
        issuer: 'Platform3-CA',
        issuedAt: Date.now(),
        expiresAt: Date.now() + 24 * 60 * 60 * 1000
      };

      expect(authManager.hasPermission(identity, 'read')).toBe(true);
      expect(authManager.hasPermission(identity, 'write')).toBe(true);
      expect(authManager.hasPermission(identity, 'admin')).toBe(false);
    });

    test('should handle wildcard permissions', () => {
      const identity = {
        serviceName: 'api-gateway',
        serviceId: 'gateway-123',
        permissions: ['*'],
        issuer: 'Platform3-CA',
        issuedAt: Date.now(),
        expiresAt: Date.now() + 24 * 60 * 60 * 1000
      };

      expect(authManager.hasPermission(identity, 'read')).toBe(true);
      expect(authManager.hasPermission(identity, 'write')).toBe(true);
      expect(authManager.hasPermission(identity, 'admin')).toBe(true);
      expect(authManager.hasPermission(identity, 'any-permission')).toBe(true);
    });

    test('should manage trusted services', () => {
      const initialStats = authManager.getAuthenticationStats();
      const initialCount = initialStats.trustedServicesCount;

      authManager.addTrustedService('new-service', ['read']);

      const updatedStats = authManager.getAuthenticationStats();
      expect(updatedStats.trustedServicesCount).toBe(initialCount + 1);

      authManager.removeTrustedService('new-service');

      const finalStats = authManager.getAuthenticationStats();
      expect(finalStats.trustedServicesCount).toBe(initialCount);
    });

    test('should blacklist certificates', () => {
      const fingerprint = 'AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD';

      authManager.blacklistCertificate(fingerprint, 'Security breach');

      const stats = authManager.getAuthenticationStats();
      expect(stats.blacklistedCertificatesCount).toBeGreaterThan(0);
    });
  });

  describe('mTLSConfigService', () => {
    test('should register service configuration', () => {
      const config = {
        serviceName: 'test-service',
        certPath: './test-cert.pem',
        keyPath: './test-key.pem',
        caCertPath: './test-ca.pem',
        enabled: true,
        requireClientCert: true,
        allowedCiphers: ['ECDHE-RSA-AES256-GCM-SHA384'],
        protocols: ['TLSv1.2']
      };

      // Mock file existence
      jest.spyOn(fs, 'existsSync').mockReturnValue(true);

      expect(() => {
        configService.registerServiceConfig(config);
      }).not.toThrow();

      const retrievedConfig = configService.getServiceConfig('test-service');
      expect(retrievedConfig).toEqual(config);
    });

    test('should validate configuration requirements', () => {
      const invalidConfig = {
        serviceName: 'invalid-service',
        enabled: true,
        // Missing required paths
      };

      expect(() => {
        configService.registerServiceConfig(invalidConfig);
      }).toThrow('Certificate paths are required when TLS is enabled');
    });

    test('should update service configuration', () => {
      const config = {
        serviceName: 'update-test-service',
        certPath: './test-cert.pem',
        keyPath: './test-key.pem',
        caCertPath: './test-ca.pem',
        enabled: true,
        requireClientCert: true,
        allowedCiphers: [],
        protocols: []
      };

      jest.spyOn(fs, 'existsSync').mockReturnValue(true);

      configService.registerServiceConfig(config);

      configService.updateServiceConfig('update-test-service', {
        requireClientCert: false,
        allowedCiphers: ['ECDHE-RSA-AES256-GCM-SHA384']
      });

      const updatedConfig = configService.getServiceConfig('update-test-service');
      expect(updatedConfig.requireClientCert).toBe(false);
      expect(updatedConfig.allowedCiphers).toEqual(['ECDHE-RSA-AES256-GCM-SHA384']);
    });

    test('should generate TLS options for Node.js', () => {
      const config = {
        serviceName: 'tls-options-test',
        certPath: './test-cert.pem',
        keyPath: './test-key.pem',
        caCertPath: './test-ca.pem',
        enabled: true,
        requireClientCert: true,
        allowedCiphers: ['ECDHE-RSA-AES256-GCM-SHA384'],
        protocols: ['TLSv1.2']
      };

      jest.spyOn(fs, 'existsSync').mockReturnValue(true);
      jest.spyOn(fs, 'readFileSync').mockReturnValue('mock-cert-content');

      configService.registerServiceConfig(config);

      const tlsOptions = configService.generateTLSOptions('tls-options-test');

      expect(tlsOptions).toBeDefined();
      expect(tlsOptions.requestCert).toBe(true);
      expect(tlsOptions.rejectUnauthorized).toBe(false);
      expect(tlsOptions.ciphers).toBe('ECDHE-RSA-AES256-GCM-SHA384');
    });

    test('should return null for disabled service', () => {
      const config = {
        serviceName: 'disabled-service',
        certPath: './test-cert.pem',
        keyPath: './test-key.pem',
        caCertPath: './test-ca.pem',
        enabled: false,
        requireClientCert: false,
        allowedCiphers: [],
        protocols: []
      };

      configService.registerServiceConfig(config);

      const tlsOptions = configService.generateTLSOptions('disabled-service');
      expect(tlsOptions).toBeNull();
    });
  });

  describe('Integration Tests', () => {
    test('should handle complete authentication flow', async () => {
      // Register service configuration
      const config = {
        serviceName: 'integration-test-service',
        certPath: './test-cert.pem',
        keyPath: './test-key.pem',
        caCertPath: './test-ca.pem',
        enabled: true,
        requireClientCert: true,
        allowedCiphers: [],
        protocols: []
      };

      jest.spyOn(fs, 'existsSync').mockReturnValue(true);
      configService.registerServiceConfig(config);

      // Add service to trusted list
      authManager.addTrustedService('integration-test-service', ['read', 'write', 'admin']);

      // Authenticate with certificate
      const testCert = {
        ...mockCertificate,
        subject: { CN: 'integration-test-service', O: 'Platform3', OU: 'Services' }
      };

      const authResult = await authManager.authenticateServiceRequest(testCert);
      expect(authResult.success).toBe(true);

      // Generate JWT token
      const token = authManager.generateServiceToken(authResult.identity);
      expect(token).toBeDefined();

      // Validate token
      const validatedIdentity = authManager.validateServiceToken(token);
      expect(validatedIdentity.serviceName).toBe('integration-test-service');

      // Check permissions
      expect(authManager.hasPermission(validatedIdentity, 'read')).toBe(true);
      expect(authManager.hasPermission(validatedIdentity, 'admin')).toBe(true);
    });
  });
});
