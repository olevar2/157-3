import vault from 'node-vault';
import Redis from 'redis';
import logger from './utils/logger';
import {
  VaultConfig,
  ConfigurationItem,
  ConfigurationRequest,
  ConfigurationResponse,
  FeatureFlag,
  ConfigurationHistory,
  ServiceRegistration,
  ConfigurationSchema,
  ConfigurationEvent
} from './types';

export class ConfigurationManager {
  private vaultClient: any;
  private redisClient: any;
  private configCache: Map<string, any> = new Map();
  private serviceRegistrations: Map<string, ServiceRegistration> = new Map();
  private configSchemas: Map<string, ConfigurationSchema> = new Map();
  private configHistory: ConfigurationHistory[] = [];

  constructor(private config: VaultConfig) {
    this.initializeVault();
    this.initializeRedis();
  }

  private async initializeVault(): Promise<void> {
    try {
      this.vaultClient = vault({
        apiVersion: 'v1',
        endpoint: this.config.endpoint,
        token: this.config.token,
        namespace: this.config.namespace,
        requestOptions: {
          timeout: this.config.timeout
        }
      });

      // Test connection
      await this.vaultClient.status();
      logger.info('✅ Vault connection established successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize Vault connection', { error });
      throw new Error(`Vault initialization failed: ${error}`);
    }
  }

  private async initializeRedis(): Promise<void> {
    try {
      this.redisClient = Redis.createClient({
        url: process.env.REDIS_URL || 'redis://redis:6379',
        password: process.env.REDIS_PASSWORD || undefined
      });

      this.redisClient.on('error', (err: Error) => {
        logger.error('Redis connection error', { error: err });
      });

      await this.redisClient.connect();
      logger.info('✅ Redis connection established successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize Redis connection', { error });
      // Continue without Redis cache
    }
  }

  /**
   * Get configuration for a service and environment
   */
  async getConfiguration(request: ConfigurationRequest): Promise<ConfigurationResponse> {
    const cacheKey = `config:${request.service}:${request.environment}`;

    try {
      // Check cache first
      let config = await this.getCachedConfig(cacheKey);

      if (!config) {
        // Fetch from Vault
        config = await this.fetchConfigFromVault(request);

        // Cache the result
        await this.setCachedConfig(cacheKey, config);
      }

      // Log configuration access
      await this.logConfigurationEvent({
        type: 'FETCH',
        service: request.service,
        environment: request.environment,
        key: 'all',
        timestamp: new Date()
      });

      return {
        service: request.service,
        environment: request.environment,
        configuration: config,
        version: 1,
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Failed to get configuration', {
        service: request.service,
        environment: request.environment,
        error
      });
      throw error;
    }
  }

  /**
   * Update configuration value
   */
  async updateConfiguration(
    service: string,
    environment: string,
    key: string,
    value: any,
    userId?: string
  ): Promise<void> {
    try {
      const vaultPath = `secret/data/services/${service}/${environment}`;

      // Get current configuration
      let currentConfig = {};
      try {
        const response = await this.vaultClient.read(vaultPath);
        currentConfig = response.data?.data || {};
      } catch (error) {
        // Configuration doesn't exist yet, start with empty object
      }

      // Store old value for history
      const oldValue = currentConfig[key];

      // Update configuration
      const updatedConfig = {
        ...currentConfig,
        [key]: value
      };

      // Write to Vault
      await this.vaultClient.write(vaultPath, { data: updatedConfig });

      // Invalidate cache
      const cacheKey = `config:${service}:${environment}`;
      await this.invalidateCache(cacheKey);

      // Record history
      this.recordConfigurationHistory({
        id: `${Date.now()}-${Math.random()}`,
        key,
        oldValue,
        newValue: value,
        environment,
        service,
        changedBy: userId || 'system',
        changedAt: new Date()
      });

      // Log event
      await this.logConfigurationEvent({
        type: 'UPDATE',
        service,
        environment,
        key,
        timestamp: new Date(),
        userId
      });

      logger.info('Configuration updated successfully', {
        service,
        environment,
        key,
        userId
      });

      // Notify registered services
      await this.notifyServiceOfConfigChange(service, environment, key);

    } catch (error) {
      logger.error('Failed to update configuration', {
        service,
        environment,
        key,
        error
      });
      throw error;
    }
  }

  /**
   * Get feature flag status
   */
  async getFeatureFlag(name: string, service?: string, environment?: string): Promise<FeatureFlag | null> {
    try {
      const flagPath = `secret/data/feature-flags/${name}`;
      const response = await this.vaultClient.read(flagPath);

      if (!response.data?.data) {
        return null;
      }

      const flag = response.data.data as FeatureFlag;

      // Check if flag applies to specific service/environment
      if (service && flag.service && flag.service !== service) {
        return null;
      }

      if (environment && flag.environment !== environment) {
        return null;
      }

      return flag;
    } catch (error) {
      logger.error('Failed to get feature flag', { name, service, environment, error });
      return null;
    }
  }

  /**
   * Register service for configuration updates
   */
  async registerService(registration: ServiceRegistration): Promise<void> {
    const key = `${registration.serviceName}:${registration.environment}`;
    this.serviceRegistrations.set(key, registration);

    logger.info('Service registered for configuration updates', {
      service: registration.serviceName,
      environment: registration.environment,
      configKeys: registration.configKeys
    });
  }

  /**
   * Get configuration history
   */
  getConfigurationHistory(service?: string, environment?: string, key?: string): ConfigurationHistory[] {
    let history = this.configHistory;

    if (service) {
      history = history.filter(h => h.service === service);
    }

    if (environment) {
      history = history.filter(h => h.environment === environment);
    }

    if (key) {
      history = history.filter(h => h.key === key);
    }

    return history.sort((a, b) => b.changedAt.getTime() - a.changedAt.getTime());
  }

  private async fetchConfigFromVault(request: ConfigurationRequest): Promise<Record<string, any>> {
    const vaultPath = `secret/data/services/${request.service}/${request.environment}`;

    try {
      const response = await this.vaultClient.read(vaultPath);
      const config = response.data?.data || {};

      // If specific keys requested, filter the configuration
      if (request.keys && request.keys.length > 0) {
        const filteredConfig: Record<string, any> = {};
        for (const key of request.keys) {
          if (config[key] !== undefined) {
            filteredConfig[key] = config[key];
          }
        }
        return filteredConfig;
      }

      return config;
    } catch (error: any) {
      if (error.response?.statusCode === 404) {
        logger.warn('Configuration not found in Vault', {
          service: request.service,
          environment: request.environment
        });
        return {};
      }
      throw error;
    }
  }

  private async getCachedConfig(cacheKey: string): Promise<Record<string, any> | null> {
    if (!this.redisClient) {
      return this.configCache.get(cacheKey) || null;
    }

    try {
      const cached = await this.redisClient.get(cacheKey);
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      logger.warn('Failed to get cached configuration', { cacheKey, error });
      return null;
    }
  }

  private async setCachedConfig(cacheKey: string, config: Record<string, any>): Promise<void> {
    if (!this.redisClient) {
      this.configCache.set(cacheKey, config);
      return;
    }

    try {
      await this.redisClient.setEx(cacheKey, 300, JSON.stringify(config)); // 5 minute TTL
    } catch (error) {
      logger.warn('Failed to cache configuration', { cacheKey, error });
    }
  }

  private async invalidateCache(cacheKey: string): Promise<void> {
    if (!this.redisClient) {
      this.configCache.delete(cacheKey);
      return;
    }

    try {
      await this.redisClient.del(cacheKey);
    } catch (error) {
      logger.warn('Failed to invalidate cache', { cacheKey, error });
    }
  }

  private recordConfigurationHistory(history: ConfigurationHistory): void {
    this.configHistory.push(history);

    // Keep only last 1000 entries
    if (this.configHistory.length > 1000) {
      this.configHistory = this.configHistory.slice(-1000);
    }
  }

  private async logConfigurationEvent(event: ConfigurationEvent): Promise<void> {
    logger.info('Configuration event', event);
  }

  private async notifyServiceOfConfigChange(service: string, environment: string, key: string): Promise<void> {
    const registrationKey = `${service}:${environment}`;
    const registration = this.serviceRegistrations.get(registrationKey);

    if (registration?.webhookUrl) {
      try {
        // Implementation for webhook notification would go here
        logger.info('Configuration change notification sent', {
          service,
          environment,
          key,
          webhook: registration.webhookUrl
        });
      } catch (error) {
        logger.error('Failed to send configuration change notification', {
          service,
          environment,
          key,
          error
        });
      }
    }
  }

  /**
   * Health check method
   */
  async healthCheck(): Promise<{ vault: boolean; redis: boolean; cache: boolean }> {
    const health = {
      vault: false,
      redis: false,
      cache: true
    };

    try {
      await this.vaultClient.status();
      health.vault = true;
    } catch (error) {
      logger.error('Vault health check failed', { error });
    }

    try {
      if (this.redisClient) {
        await this.redisClient.ping();
        health.redis = true;
      }
    } catch (error) {
      logger.error('Redis health check failed', { error });
    }

    return health;
  }
}
