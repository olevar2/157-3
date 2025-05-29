import { ConfigurationManager } from '../../src/configuration/ConfigurationManager';
import { ConfigClient } from '../../src/configuration/ConfigClient';
import axios from 'axios';

describe('Configuration Management Integration Tests', () => {
    let configManager: ConfigurationManager;
    let configClient: ConfigClient;
    
    beforeAll(async () => {
        // Wait for services to be ready
        await waitForService('http://localhost:3001/health', 30000);
        
        configManager = new ConfigurationManager({
            vault: {
                endpoint: 'http://localhost:8200',
                token: process.env.VAULT_TOKEN || 'dev-token'
            },
            redis: {
                host: 'localhost',
                port: 6379
            }
        });
        
        configClient = new ConfigClient({
            serviceUrl: 'http://localhost:3001',
            serviceId: 'test-service',
            refreshInterval: 1000
        });
        
        await configManager.initialize();
    });
    
    afterAll(async () => {
        await configManager.close();
        configClient.destroy();
    });
    
    describe('Configuration Retrieval', () => {
        it('should retrieve configuration from Vault', async () => {
            const config = await configManager.getConfiguration('database');
            
            expect(config).toBeDefined();
            expect(config.host).toBe('localhost');
            expect(config.port).toBe(5432);
        });
        
        it('should cache configuration in Redis', async () => {
            // First call - from Vault
            const config1 = await configManager.getConfiguration('database');
            
            // Second call - should be from cache
            const config2 = await configManager.getConfiguration('database');
            
            expect(config1).toEqual(config2);
        });
        
        it('should retrieve configuration via HTTP API', async () => {
            const response = await axios.get('http://localhost:3001/api/config/database', {
                headers: { 'x-service-id': 'test-service' }
            });
            
            expect(response.status).toBe(200);
            expect(response.data.host).toBe('localhost');
        });
    });
    
    describe('Feature Flags', () => {
        it('should retrieve feature flags', async () => {
            const flags = await configManager.getFeatureFlags();
            
            expect(flags['new-ui']).toBe(true);
            expect(flags['api-v2']).toBe(false);
        });
        
        it('should check individual feature flag', async () => {
            const isEnabled = await configManager.isFeatureEnabled('new-ui');
            expect(isEnabled).toBe(true);
        });
    });
    
    describe('Configuration Updates', () => {
        it('should update configuration', async () => {
            await configManager.updateConfiguration('test-config', {
                value: 'test-value',
                updated: new Date().toISOString()
            });
            
            const config = await configManager.getConfiguration('test-config');
            expect(config.value).toBe('test-value');
        });
        
        it('should invalidate cache on update', async () => {
            // Get initial config
            const initial = await configManager.getConfiguration('test-config');
            
            // Update config
            await configManager.updateConfiguration('test-config', {
                value: 'updated-value'
            });
            
            // Get updated config
            const updated = await configManager.getConfiguration('test-config');
            expect(updated.value).toBe('updated-value');
            expect(updated.value).not.toBe(initial.value);
        });
    });
    
    describe('ConfigClient Integration', () => {
        it('should retrieve configuration via client', async () => {
            const config = await configClient.getConfig('database');
            
            expect(config).toBeDefined();
            expect(config.host).toBe('localhost');
        });
        
        it('should auto-refresh configuration', (done) => {
            configClient.onConfigChange('test-config', (newConfig) => {
                expect(newConfig).toBeDefined();
                done();
            });
            
            // Trigger config change
            setTimeout(async () => {
                await configManager.updateConfiguration('test-config', {
                    value: 'auto-refresh-test'
                });
            }, 500);
        }, 10000);
    });
});

async function waitForService(url: string, timeout: number): Promise<void> {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        try {
            await axios.get(url);
            return;
        } catch (error) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    throw new Error(`Service at ${url} not ready within ${timeout}ms`);
}
