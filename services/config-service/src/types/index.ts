// Platform3 Configuration Service Types

export interface VaultConfig {
  endpoint: string;
  token: string;
  namespace?: string;
  timeout: number;
  retries: number;
}

export interface ConfigurationItem {
  key: string;
  value: any;
  environment: string;
  service: string;
  version: number;
  encrypted: boolean;
  lastModified: Date;
  modifiedBy: string;
}

export interface ConfigurationRequest {
  service: string;
  environment: string;
  keys?: string[];
}

export interface ConfigurationResponse {
  service: string;
  environment: string;
  configuration: Record<string, any>;
  version: number;
  timestamp: Date;
}

export interface FeatureFlag {
  name: string;
  enabled: boolean;
  environment: string;
  service?: string;
  conditions?: Record<string, any>;
  rolloutPercentage?: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface ConfigurationHistory {
  id: string;
  key: string;
  oldValue: any;
  newValue: any;
  environment: string;
  service: string;
  changedBy: string;
  changedAt: Date;
  reason?: string;
}

export interface ServiceRegistration {
  serviceName: string;
  environment: string;
  configKeys: string[];
  webhookUrl?: string;
  lastHeartbeat: Date;
}

export interface ConfigurationValidation {
  key: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required: boolean;
  defaultValue?: any;
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    enum?: any[];
  };
}

export interface ConfigurationSchema {
  service: string;
  environment: string;
  schema: Record<string, ConfigurationValidation>;
  version: number;
}

export interface ConfigurationEvent {
  type: 'CREATE' | 'UPDATE' | 'DELETE' | 'FETCH';
  service: string;
  environment: string;
  key: string;
  timestamp: Date;
  userId?: string;
  metadata?: Record<string, any>;
}

export interface CacheConfig {
  ttl: number;
  maxSize: number;
  enabled: boolean;
}

export interface NotificationConfig {
  webhooks: string[];
  email?: string[];
  slack?: {
    webhook: string;
    channel: string;
  };
}

export interface ConfigServiceConfig {
  port: number;
  environment: string;
  vault: VaultConfig;
  cache: CacheConfig;
  notifications: NotificationConfig;
  security: {
    apiKey: string;
    allowedOrigins: string[];
    rateLimiting: {
      windowMs: number;
      maxRequests: number;
    };
  };
}
