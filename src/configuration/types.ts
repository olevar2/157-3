export interface ConfigurationOptions {
  vault: {
    endpoint: string;
    token: string;
  };
  redis: {
    host: string;
    port: number;
    password?: string;
  };
}

export interface FeatureFlags {
  [key: string]: boolean;
}

export interface ConfigurationData {
  [key: string]: any;
}

export interface ConfigClientOptions {
  serviceUrl: string;
  serviceId: string;
  refreshInterval?: number;
}
