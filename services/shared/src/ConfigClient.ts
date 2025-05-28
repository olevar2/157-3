import axios, { AxiosInstance } from 'axios';
import EventEmitter from 'events';

export interface ConfigClientOptions {
  serviceUrl: string;
  apiKey: string;
  serviceName: string;
  environment: string;
  refreshInterval?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

export interface ConfigValue {
  key: string;
  value: any;
  lastUpdated: Date;
}

export class ConfigClient extends EventEmitter {
  private httpClient: AxiosInstance;
  private config: Map<string, ConfigValue> = new Map();
  private refreshTimer?: NodeJS.Timeout;
  private isInitialized = false;

  constructor(private options: ConfigClientOptions) {
    super();
    
    this.httpClient = axios.create({
      baseURL: options.serviceUrl,
      headers: {
        'X-API-Key': options.apiKey,
        'Content-Type': 'application/json'
      },
      timeout: 10000
    });

    // Set up request/response interceptors for logging
    this.httpClient.interceptors.request.use(
      (config) => {
        this.emit('debug', `Making request to ${config.url}`);
        return config;
      },
      (error) => {
        this.emit('error', `Request error: ${error.message}`);
        return Promise.reject(error);
      }
    );

    this.httpClient.interceptors.response.use(
      (response) => {
        this.emit('debug', `Response received from ${response.config.url}`);
        return response;
      },
      (error) => {
        this.emit('error', `Response error: ${error.message}`);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Initialize the configuration client
   */
  async initialize(): Promise<void> {
    try {
      await this.loadConfiguration();
      await this.registerService();
      this.startRefreshTimer();
      this.isInitialized = true;
      this.emit('initialized');
    } catch (error) {
      this.emit('error', `Failed to initialize ConfigClient: ${error}`);
      throw error;
    }
  }

  /**
   * Get configuration value by key
   */
  get<T = any>(key: string, defaultValue?: T): T {
    const configValue = this.config.get(key);
    
    if (configValue) {
      return configValue.value as T;
    }
    
    if (defaultValue !== undefined) {
      return defaultValue;
    }
    
    throw new Error(`Configuration key '${key}' not found and no default value provided`);
  }

  /**
   * Get configuration value with type safety
   */
  getString(key: string, defaultValue?: string): string {
    return this.get<string>(key, defaultValue);
  }

  getNumber(key: string, defaultValue?: number): number {
    const value = this.get<any>(key, defaultValue);
    const numValue = Number(value);
    
    if (isNaN(numValue)) {
      throw new Error(`Configuration key '${key}' is not a valid number: ${value}`);
    }
    
    return numValue;
  }

  getBoolean(key: string, defaultValue?: boolean): boolean {
    const value = this.get<any>(key, defaultValue);
    
    if (typeof value === 'boolean') {
      return value;
    }
    
    if (typeof value === 'string') {
      return value.toLowerCase() === 'true';
    }
    
    return Boolean(value);
  }

  getArray<T = any>(key: string, defaultValue?: T[]): T[] {
    const value = this.get<any>(key, defaultValue);
    
    if (Array.isArray(value)) {
      return value;
    }
    
    if (typeof value === 'string') {
      try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed)) {
          return parsed;
        }
      } catch {
        // If JSON parsing fails, try splitting by comma
        return value.split(',').map(item => item.trim()) as T[];
      }
    }
    
    throw new Error(`Configuration key '${key}' is not a valid array: ${value}`);
  }

  getObject<T = any>(key: string, defaultValue?: T): T {
    const value = this.get<any>(key, defaultValue);
    
    if (typeof value === 'object' && value !== null) {
      return value;
    }
    
    if (typeof value === 'string') {
      try {
        return JSON.parse(value);
      } catch {
        throw new Error(`Configuration key '${key}' is not valid JSON: ${value}`);
      }
    }
    
    throw new Error(`Configuration key '${key}' is not a valid object: ${value}`);
  }

  /**
   * Check if configuration key exists
   */
  has(key: string): boolean {
    return this.config.has(key);
  }

  /**
   * Get all configuration keys
   */
  keys(): string[] {
    return Array.from(this.config.keys());
  }

  /**
   * Get all configuration as object
   */
  getAll(): Record<string, any> {
    const result: Record<string, any> = {};
    
    for (const [key, configValue] of this.config.entries()) {
      result[key] = configValue.value;
    }
    
    return result;
  }

  /**
   * Refresh configuration from server
   */
  async refresh(): Promise<void> {
    try {
      await this.loadConfiguration();
      this.emit('refreshed');
    } catch (error) {
      this.emit('error', `Failed to refresh configuration: ${error}`);
      throw error;
    }
  }

  /**
   * Check if a feature flag is enabled
   */
  async isFeatureEnabled(flagName: string): Promise<boolean> {
    try {
      const response = await this.httpClient.get(
        `/api/v1/feature-flags/${flagName}?service=${this.options.serviceName}&environment=${this.options.environment}`
      );
      
      return response.data.success && response.data.data?.enabled === true;
    } catch (error) {
      this.emit('debug', `Feature flag '${flagName}' not found or error occurred: ${error}`);
      return false;
    }
  }

  /**
   * Stop the configuration client
   */
  stop(): void {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = undefined;
    }
    
    this.isInitialized = false;
    this.emit('stopped');
  }

  /**
   * Get client status
   */
  getStatus(): {
    initialized: boolean;
    configCount: number;
    lastRefresh: Date | null;
    serviceName: string;
    environment: string;
  } {
    const configValues = Array.from(this.config.values());
    const lastRefresh = configValues.length > 0 
      ? new Date(Math.max(...configValues.map(cv => cv.lastUpdated.getTime())))
      : null;

    return {
      initialized: this.isInitialized,
      configCount: this.config.size,
      lastRefresh,
      serviceName: this.options.serviceName,
      environment: this.options.environment
    };
  }

  private async loadConfiguration(): Promise<void> {
    const retryAttempts = this.options.retryAttempts || 3;
    const retryDelay = this.options.retryDelay || 1000;

    for (let attempt = 1; attempt <= retryAttempts; attempt++) {
      try {
        const response = await this.httpClient.post('/api/v1/config', {
          service: this.options.serviceName,
          environment: this.options.environment
        });

        if (response.data.success) {
          const configuration = response.data.data.configuration;
          const now = new Date();

          // Update configuration cache
          this.config.clear();
          for (const [key, value] of Object.entries(configuration)) {
            this.config.set(key, {
              key,
              value,
              lastUpdated: now
            });
          }

          this.emit('configurationLoaded', configuration);
          return;
        }

        throw new Error('Invalid response from configuration service');
      } catch (error) {
        if (attempt === retryAttempts) {
          throw error;
        }

        this.emit('debug', `Configuration load attempt ${attempt} failed, retrying in ${retryDelay}ms`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }

  private async registerService(): Promise<void> {
    try {
      await this.httpClient.post('/api/v1/register', {
        serviceName: this.options.serviceName,
        environment: this.options.environment,
        configKeys: Array.from(this.config.keys()),
        lastHeartbeat: new Date()
      });

      this.emit('debug', 'Service registered successfully');
    } catch (error) {
      this.emit('error', `Failed to register service: ${error}`);
      // Don't throw here as registration is not critical for basic functionality
    }
  }

  private startRefreshTimer(): void {
    const refreshInterval = this.options.refreshInterval || 300000; // 5 minutes default

    this.refreshTimer = setInterval(async () => {
      try {
        await this.refresh();
      } catch (error) {
        this.emit('error', `Scheduled refresh failed: ${error}`);
      }
    }, refreshInterval);
  }
}

// Factory function for easy instantiation
export function createConfigClient(options: ConfigClientOptions): ConfigClient {
  return new ConfigClient(options);
}

// Default export
export default ConfigClient;
