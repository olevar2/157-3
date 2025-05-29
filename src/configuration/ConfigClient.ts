import { ConfigClientOptions, ConfigurationData } from './types';

export class ConfigClient {
  private options: ConfigClientOptions;
  private listeners: Map<string, Function[]> = new Map();

  constructor(options: ConfigClientOptions) {
    this.options = options;
  }

  async getConfig(key: string): Promise<ConfigurationData> {
    // Mock implementation for testing
    if (key === 'database') {
      return {
        host: 'localhost',
        port: 5432,
        username: 'platform3',
        password: 'dev-secret'
      };
    }
    
    return {};
  }

  onConfigChange(key: string, callback: (config: ConfigurationData) => void): void {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, []);
    }
    this.listeners.get(key)!.push(callback);
  }

  destroy(): void {
    this.listeners.clear();
  }
}
