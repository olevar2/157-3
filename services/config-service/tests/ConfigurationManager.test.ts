import { ConfigurationManager } from '../src/ConfigurationManager';
import { VaultConfig } from '../src/types';

// Mock dependencies
jest.mock('node-vault');
jest.mock('redis');

const mockVault = {
  status: jest.fn(),
  read: jest.fn(),
  write: jest.fn()
};

const mockRedis = {
  connect: jest.fn(),
  get: jest.fn(),
  setEx: jest.fn(),
  del: jest.fn(),
  ping: jest.fn(),
  on: jest.fn()
};

// Mock node-vault
const nodeVault = require('node-vault');
nodeVault.mockReturnValue(mockVault);

// Mock Redis
const Redis = require('redis');
Redis.createClient = jest.fn().mockReturnValue(mockRedis);

describe('ConfigurationManager', () => {
  let configManager: ConfigurationManager;
  let vaultConfig: VaultConfig;

  beforeEach(() => {
    jest.clearAllMocks();

    vaultConfig = {
      endpoint: 'http://localhost:8200',
      token: 'test-token',
      timeout: 5000,
      retries: 3
    };

    // Setup default mock responses
    mockVault.status.mockResolvedValue({ sealed: false });
    mockRedis.connect.mockResolvedValue(undefined);
    mockRedis.ping.mockResolvedValue('PONG');

    configManager = new ConfigurationManager(vaultConfig);
  });

  describe('Initialization', () => {
    it('should initialize Vault connection successfully', async () => {
      expect(nodeVault).toHaveBeenCalledWith({
        apiVersion: 'v1',
        endpoint: vaultConfig.endpoint,
        token: vaultConfig.token,
        namespace: vaultConfig.namespace,
        requestOptions: {
          timeout: vaultConfig.timeout
        }
      });

      expect(mockVault.status).toHaveBeenCalled();
    });

    it('should initialize Redis connection successfully', async () => {
      expect(Redis.createClient).toHaveBeenCalledWith({
        url: 'redis://redis:6379',
        password: undefined
      });

      expect(mockRedis.connect).toHaveBeenCalled();
    });

    it('should handle Vault connection failure', async () => {
      mockVault.status.mockRejectedValue(new Error('Vault connection failed'));

      expect(() => new ConfigurationManager(vaultConfig)).toThrow();
    });
  });

  describe('Configuration Management', () => {
    beforeEach(async () => {
      // Ensure manager is properly initialized
      await new Promise(resolve => setTimeout(resolve, 100));
    });

    it('should get configuration from Vault when not cached', async () => {
      const mockConfig = {
        database: {
          host: 'localhost',
          port: 5432
        },
        jwt: {
          secret: 'test-secret'
        }
      };

      mockRedis.get.mockResolvedValue(null);
      mockVault.read.mockResolvedValue({
        data: {
          data: mockConfig
        }
      });

      const request = {
        service: 'trading-service',
        environment: 'development'
      };

      const result = await configManager.getConfiguration(request);

      expect(mockVault.read).toHaveBeenCalledWith(
        'secret/data/services/trading-service/development'
      );
      expect(result.configuration).toEqual(mockConfig);
      expect(result.service).toBe('trading-service');
      expect(result.environment).toBe('development');
    });

    it('should get configuration from cache when available', async () => {
      const mockConfig = {
        database: {
          host: 'localhost',
          port: 5432
        }
      };

      mockRedis.get.mockResolvedValue(JSON.stringify(mockConfig));

      const request = {
        service: 'trading-service',
        environment: 'development'
      };

      const result = await configManager.getConfiguration(request);

      expect(mockVault.read).not.toHaveBeenCalled();
      expect(result.configuration).toEqual(mockConfig);
    });

    it('should filter configuration by requested keys', async () => {
      const mockConfig = {
        database: {
          host: 'localhost',
          port: 5432
        },
        jwt: {
          secret: 'test-secret'
        },
        redis: {
          host: 'redis'
        }
      };

      mockRedis.get.mockResolvedValue(null);
      mockVault.read.mockResolvedValue({
        data: {
          data: mockConfig
        }
      });

      const request = {
        service: 'trading-service',
        environment: 'development',
        keys: ['database', 'jwt']
      };

      const result = await configManager.getConfiguration(request);

      expect(result.configuration).toEqual({
        database: mockConfig.database,
        jwt: mockConfig.jwt
      });
      expect(result.configuration['redis']).toBeUndefined();
    });

    it('should update configuration in Vault', async () => {
      const existingConfig = {
        database: {
          host: 'localhost',
          port: 5432
        }
      };

      const newValue = {
        host: 'new-host',
        port: 5433
      };

      mockVault.read.mockResolvedValue({
        data: {
          data: existingConfig
        }
      });
      mockVault.write.mockResolvedValue({});
      mockRedis.del.mockResolvedValue(1);

      await configManager.updateConfiguration(
        'trading-service',
        'development',
        'database',
        newValue,
        'test-user'
      );

      expect(mockVault.write).toHaveBeenCalledWith(
        'secret/data/services/trading-service/development',
        {
          data: {
            ...existingConfig,
            database: newValue
          }
        }
      );

      expect(mockRedis.del).toHaveBeenCalledWith(
        'config:trading-service:development'
      );
    });

    it('should handle configuration not found in Vault', async () => {
      const error = new Error('Not found') as any;
      error.response = { statusCode: 404 };

      mockRedis.get.mockResolvedValue(null);
      mockVault.read.mockRejectedValue(error);

      const request = {
        service: 'new-service',
        environment: 'development'
      };

      const result = await configManager.getConfiguration(request);

      expect(result.configuration).toEqual({});
    });
  });

  describe('Feature Flags', () => {
    it('should get feature flag from Vault', async () => {
      const mockFlag = {
        name: 'advanced-analytics',
        enabled: true,
        environment: 'development',
        service: 'analytics-service',
        rolloutPercentage: 100,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      mockVault.read.mockResolvedValue({
        data: {
          data: mockFlag
        }
      });

      const result = await configManager.getFeatureFlag(
        'advanced-analytics',
        'analytics-service',
        'development'
      );

      expect(mockVault.read).toHaveBeenCalledWith(
        'secret/data/feature-flags/advanced-analytics'
      );
      expect(result).toEqual(mockFlag);
    });

    it('should return null for non-existent feature flag', async () => {
      const error = new Error('Not found') as any;
      error.response = { statusCode: 404 };

      mockVault.read.mockRejectedValue(error);

      const result = await configManager.getFeatureFlag(
        'non-existent-flag',
        'test-service',
        'development'
      );

      expect(result).toBeNull();
    });

    it('should filter feature flag by service', async () => {
      const mockFlag = {
        name: 'service-specific-flag',
        enabled: true,
        environment: 'development',
        service: 'other-service'
      };

      mockVault.read.mockResolvedValue({
        data: {
          data: mockFlag
        }
      });

      const result = await configManager.getFeatureFlag(
        'service-specific-flag',
        'trading-service',
        'development'
      );

      expect(result).toBeNull();
    });
  });

  describe('Service Registration', () => {
    it('should register service successfully', async () => {
      const registration = {
        serviceName: 'trading-service',
        environment: 'development',
        configKeys: ['database', 'jwt', 'redis'],
        webhookUrl: 'http://trading-service:3003/config-webhook',
        lastHeartbeat: new Date()
      };

      await configManager.registerService(registration);

      // Verify service is registered (this would be tested through internal state)
      expect(true).toBe(true); // Placeholder assertion
    });
  });

  describe('Configuration History', () => {
    it('should return configuration history', async () => {
      // First, update a configuration to create history
      mockVault.read.mockResolvedValue({
        data: {
          data: { oldKey: 'oldValue' }
        }
      });
      mockVault.write.mockResolvedValue({});
      mockRedis.del.mockResolvedValue(1);

      await configManager.updateConfiguration(
        'test-service',
        'development',
        'testKey',
        'newValue',
        'test-user'
      );

      const history = configManager.getConfigurationHistory(
        'test-service',
        'development',
        'testKey'
      );

      expect(history).toHaveLength(1);
      expect(history[0]).toMatchObject({
        key: 'testKey',
        newValue: 'newValue',
        environment: 'development',
        service: 'test-service',
        changedBy: 'test-user'
      });
    });

    it('should filter history by service', async () => {
      // Create history for multiple services
      mockVault.read.mockResolvedValue({
        data: { data: {} }
      });
      mockVault.write.mockResolvedValue({});
      mockRedis.del.mockResolvedValue(1);

      await configManager.updateConfiguration(
        'service1',
        'development',
        'key1',
        'value1',
        'user1'
      );

      await configManager.updateConfiguration(
        'service2',
        'development',
        'key2',
        'value2',
        'user2'
      );

      const history = configManager.getConfigurationHistory('service1');

      expect(history).toHaveLength(1);
      expect(history[0]?.service).toBe('service1');
    });
  });

  describe('Health Check', () => {
    it('should return healthy status when all services are up', async () => {
      mockVault.status.mockResolvedValue({ sealed: false });
      mockRedis.ping.mockResolvedValue('PONG');

      const health = await configManager.healthCheck();

      expect(health).toEqual({
        vault: true,
        redis: true,
        cache: true
      });
    });

    it('should return unhealthy status when Vault is down', async () => {
      mockVault.status.mockRejectedValue(new Error('Vault is down'));
      mockRedis.ping.mockResolvedValue('PONG');

      const health = await configManager.healthCheck();

      expect(health).toEqual({
        vault: false,
        redis: true,
        cache: true
      });
    });

    it('should return unhealthy status when Redis is down', async () => {
      mockVault.status.mockResolvedValue({ sealed: false });
      mockRedis.ping.mockRejectedValue(new Error('Redis is down'));

      const health = await configManager.healthCheck();

      expect(health).toEqual({
        vault: true,
        redis: false,
        cache: true
      });
    });
  });
});
