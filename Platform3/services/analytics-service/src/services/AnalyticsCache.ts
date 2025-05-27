import { Logger } from 'winston';

export class AnalyticsCache {
  private cache = new Map<string, { data: any; expires: number }>();
  private ready = false;

  constructor(private logger: Logger) {}

  async initialize(): Promise<void> {
    this.logger.info('Initializing Analytics Cache...');
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  async get(key: string): Promise<any> {
    const item = this.cache.get(key);
    if (!item) return null;
    
    if (Date.now() > item.expires) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data;
  }

  async set(key: string, data: any, ttlSeconds: number): Promise<void> {
    const expires = Date.now() + (ttlSeconds * 1000);
    this.cache.set(key, { data, expires });
  }

  async cleanup(): Promise<void> {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expires) {
        this.cache.delete(key);
      }
    }
    this.logger.info(`Cache cleanup completed. ${this.cache.size} items remaining.`);
  }
}
