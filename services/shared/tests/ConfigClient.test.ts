import { ConfigClient } from '../src/ConfigClient';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ConfigClient', () => {
  let configClient: ConfigClient;
  let mockAxiosInstance: any;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock axios instance
    mockAxiosInstance = {
      post: jest.fn(),
      get: jest.fn(),
      interceptors: {
        request: {
          use: jest.fn()
        },
        response: {
          use: jest.fn()
        }
      }
    };

    mockedAxios.create.mockReturnValue(mockAxiosInstance);

    configClient = new ConfigClient({
      serviceUrl: 'http://config-service:3007',
      apiKey: 'test-api-key',
      serviceName: 'test-service',
      environment: 'development',
      refreshInterval: 60000,
      retryAttempts: 3,
      retryDelay: 1000
    });
  });

  afterEach(() => {
    configClient.stop();
  });

  describe('Initialization', () => {
    it('should create axios instance with correct configuration', () => {
      expect(mockedAxios.create).toHaveBeenCalledWith({
        baseURL: 'http://config-service:3007',
        headers: {
          'X-API-Key': 'test-api-key',
          'Content-Type': 'application/json'
        },
        timeout: 10000
      });
    });

    it('should initialize successfully with valid configuration', async () => {
      const mockConfig = {
        database: {
          host: 'localhost',
          port: 5432
        },
        jwt: {
          secret: 'test-secret'
        }
      };

      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: {
              configuration: mockConfig
            }
          }
        })
        .mockResolvedValueOnce({
          data: {
            success: true,
            message: 'Service registered successfully'
          }
        });

      await configClient.initialize();

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/config', {
        service: 'test-service',
        environment: 'development'
      });

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/register', {
        serviceName: 'test-service',
        environment: 'development',
        configKeys: ['database', 'jwt'],
        lastHeartbeat: expect.any(Date)
      });
    });

    it('should emit initialized event after successful initialization', async () => {
      const mockConfig = { key1: 'value1' };
      
      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: mockConfig }
          }
        })
        .mockResolvedValueOnce({
          data: { success: true }
        });

      const initPromise = new Promise(resolve => {
        configClient.on('initialized', resolve);
      });

      await configClient.initialize();
      await initPromise;

      expect(true).toBe(true); // Event was emitted
    });
  });

  describe('Configuration Access', () => {
    beforeEach(async () => {
      const mockConfig = {
        stringValue: 'test-string',
        numberValue: '42',
        booleanValue: 'true',
        arrayValue: '["item1", "item2", "item3"]',
        objectValue: '{"nested": {"key": "value"}}',
        csvArray: 'item1,item2,item3'
      };

      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: mockConfig }
          }
        })
        .mockResolvedValueOnce({
          data: { success: true }
        });

      await configClient.initialize();
    });

    it('should get string configuration value', () => {
      const value = configClient.getString('stringValue');
      expect(value).toBe('test-string');
    });

    it('should get string with default value', () => {
      const value = configClient.getString('nonExistentKey', 'default-value');
      expect(value).toBe('default-value');
    });

    it('should get number configuration value', () => {
      const value = configClient.getNumber('numberValue');
      expect(value).toBe(42);
    });

    it('should get number with default value', () => {
      const value = configClient.getNumber('nonExistentKey', 100);
      expect(value).toBe(100);
    });

    it('should throw error for invalid number', () => {
      expect(() => {
        configClient.getNumber('stringValue');
      }).toThrow("Configuration key 'stringValue' is not a valid number");
    });

    it('should get boolean configuration value', () => {
      const value = configClient.getBoolean('booleanValue');
      expect(value).toBe(true);
    });

    it('should get boolean with default value', () => {
      const value = configClient.getBoolean('nonExistentKey', false);
      expect(value).toBe(false);
    });

    it('should get array configuration value from JSON', () => {
      const value = configClient.getArray('arrayValue');
      expect(value).toEqual(['item1', 'item2', 'item3']);
    });

    it('should get array configuration value from CSV', () => {
      const value = configClient.getArray('csvArray');
      expect(value).toEqual(['item1', 'item2', 'item3']);
    });

    it('should get object configuration value', () => {
      const value = configClient.getObject('objectValue');
      expect(value).toEqual({ nested: { key: 'value' } });
    });

    it('should check if configuration key exists', () => {
      expect(configClient.has('stringValue')).toBe(true);
      expect(configClient.has('nonExistentKey')).toBe(false);
    });

    it('should get all configuration keys', () => {
      const keys = configClient.keys();
      expect(keys).toContain('stringValue');
      expect(keys).toContain('numberValue');
      expect(keys).toContain('booleanValue');
    });

    it('should get all configuration as object', () => {
      const allConfig = configClient.getAll();
      expect(allConfig).toHaveProperty('stringValue', 'test-string');
      expect(allConfig).toHaveProperty('numberValue', '42');
    });
  });

  describe('Feature Flags', () => {
    beforeEach(async () => {
      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: {} }
          }
        })
        .mockResolvedValueOnce({
          data: { success: true }
        });

      await configClient.initialize();
    });

    it('should check if feature flag is enabled', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: {
          success: true,
          data: {
            name: 'test-feature',
            enabled: true
          }
        }
      });

      const isEnabled = await configClient.isFeatureEnabled('test-feature');
      
      expect(isEnabled).toBe(true);
      expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        '/api/v1/feature-flags/test-feature?service=test-service&environment=development'
      );
    });

    it('should return false for disabled feature flag', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: {
          success: true,
          data: {
            name: 'test-feature',
            enabled: false
          }
        }
      });

      const isEnabled = await configClient.isFeatureEnabled('test-feature');
      expect(isEnabled).toBe(false);
    });

    it('should return false for non-existent feature flag', async () => {
      mockAxiosInstance.get.mockRejectedValue(new Error('Not found'));

      const isEnabled = await configClient.isFeatureEnabled('non-existent-feature');
      expect(isEnabled).toBe(false);
    });
  });

  describe('Configuration Refresh', () => {
    beforeEach(async () => {
      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: { key1: 'value1' } }
          }
        })
        .mockResolvedValueOnce({
          data: { success: true }
        });

      await configClient.initialize();
    });

    it('should refresh configuration from server', async () => {
      const newConfig = { key1: 'updated-value1', key2: 'value2' };
      
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          success: true,
          data: { configuration: newConfig }
        }
      });

      await configClient.refresh();

      expect(configClient.get('key1')).toBe('updated-value1');
      expect(configClient.get('key2')).toBe('value2');
    });

    it('should emit refreshed event after successful refresh', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          success: true,
          data: { configuration: { key1: 'refreshed-value' } }
        }
      });

      const refreshPromise = new Promise(resolve => {
        configClient.on('refreshed', resolve);
      });

      await configClient.refresh();
      await refreshPromise;

      expect(true).toBe(true); // Event was emitted
    });

    it('should retry on configuration load failure', async () => {
      mockAxiosInstance.post
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: { key1: 'retry-success' } }
          }
        });

      await configClient.refresh();

      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(3);
      expect(configClient.get('key1')).toBe('retry-success');
    });
  });

  describe('Client Status', () => {
    it('should return correct status before initialization', () => {
      const status = configClient.getStatus();
      
      expect(status).toEqual({
        initialized: false,
        configCount: 0,
        lastRefresh: null,
        serviceName: 'test-service',
        environment: 'development'
      });
    });

    it('should return correct status after initialization', async () => {
      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: { key1: 'value1', key2: 'value2' } }
          }
        })
        .mockResolvedValueOnce({
          data: { success: true }
        });

      await configClient.initialize();

      const status = configClient.getStatus();
      
      expect(status.initialized).toBe(true);
      expect(status.configCount).toBe(2);
      expect(status.lastRefresh).toBeInstanceOf(Date);
      expect(status.serviceName).toBe('test-service');
      expect(status.environment).toBe('development');
    });
  });

  describe('Error Handling', () => {
    it('should handle initialization failure', async () => {
      mockAxiosInstance.post.mockRejectedValue(new Error('Service unavailable'));

      await expect(configClient.initialize()).rejects.toThrow('Service unavailable');
    });

    it('should emit error event on configuration load failure', async () => {
      mockAxiosInstance.post
        .mockResolvedValueOnce({
          data: {
            success: true,
            data: { configuration: {} }
          }
        })
        .mockResolvedValueOnce({
          data: { success: true }
        });

      await configClient.initialize();

      mockAxiosInstance.post.mockRejectedValue(new Error('Refresh failed'));

      const errorPromise = new Promise(resolve => {
        configClient.on('error', resolve);
      });

      await expect(configClient.refresh()).rejects.toThrow('Refresh failed');
      await errorPromise;

      expect(true).toBe(true); // Error event was emitted
    });
  });
});
