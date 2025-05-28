const request = require('supertest');
const { ConsulServiceRegistry } = require('../dist/ConsulServiceRegistry');
const { ServiceDiscoveryClient } = require('../dist/ServiceDiscoveryClient');
const { app } = require('../dist/index');

describe('Service Discovery Integration Tests', () => {
  let registry;
  let client;
  
  beforeAll(async () => {
    // Initialize registry and client for testing
    registry = new ConsulServiceRegistry({
      host: process.env.CONSUL_HOST || 'localhost',
      port: process.env.CONSUL_PORT || '8500'
    });
    
    client = new ServiceDiscoveryClient(registry);
    
    // Wait for Consul to be ready
    await new Promise(resolve => setTimeout(resolve, 2000));
  });

  afterAll(async () => {
    if (registry) {
      await registry.shutdown();
    }
  });

  describe('Service Registration', () => {
    test('should register a service successfully', async () => {
      const registration = {
        name: 'test-service',
        address: 'localhost',
        port: 3001,
        tags: ['test', 'http'],
        meta: { version: '1.0.0' },
        check: {
          http: 'http://localhost:3001/health',
          interval: '10s'
        }
      };

      const response = await request(app)
        .post('/services/register')
        .send(registration)
        .expect(201);

      expect(response.body.success).toBe(true);
      expect(response.body.serviceId).toBeDefined();
    });

    test('should discover registered service', async () => {
      const response = await request(app)
        .get('/services/test-service/instances')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.instances).toHaveLength(1);
      expect(response.body.instances[0].name).toBe('test-service');
    });

    test('should get service instance with load balancing', async () => {
      const response = await request(app)
        .get('/services/test-service/instance')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.instance.name).toBe('test-service');
      expect(response.body.instance.address).toBe('localhost');
      expect(response.body.instance.port).toBe(3001);
    });

    test('should get service URL', async () => {
      const response = await request(app)
        .get('/services/test-service/url')
        .query({ path: '/api/test' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.url).toBe('http://localhost:3001/api/test');
    });
  });

  describe('Service Discovery Client', () => {
    test('should discover service using client', async () => {
      const instance = await client.discoverService('test-service');
      
      expect(instance).toBeDefined();
      expect(instance.name).toBe('test-service');
      expect(instance.address).toBe('localhost');
      expect(instance.port).toBe(3001);
    });

    test('should get service URL using client', async () => {
      const url = await client.getServiceUrl('test-service', '/api/test');
      
      expect(url).toBe('http://localhost:3001/api/test');
    });

    test('should return null for non-existent service', async () => {
      const instance = await client.discoverService('non-existent-service');
      expect(instance).toBeNull();
    });
  });

  describe('Load Balancing', () => {
    beforeAll(async () => {
      // Register multiple instances of the same service
      const registrations = [
        {
          name: 'multi-service',
          address: 'localhost',
          port: 3002,
          tags: ['test', 'http'],
          check: { http: 'http://localhost:3002/health', interval: '10s' }
        },
        {
          name: 'multi-service',
          address: 'localhost',
          port: 3003,
          tags: ['test', 'http'],
          check: { http: 'http://localhost:3003/health', interval: '10s' }
        }
      ];

      for (const reg of registrations) {
        await request(app)
          .post('/services/register')
          .send(reg)
          .expect(201);
      }
    });

    test('should distribute requests across multiple instances', async () => {
      const instances = new Set();
      
      // Make multiple requests to see load balancing in action
      for (let i = 0; i < 10; i++) {
        const response = await request(app)
          .get('/services/multi-service/instance')
          .expect(200);
        
        instances.add(response.body.instance.port);
      }
      
      // Should have hit both instances
      expect(instances.size).toBeGreaterThan(1);
    });
  });

  describe('Health Checks', () => {
    test('should return healthy status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body.status).toBe('healthy');
      expect(response.body.service).toBe('service-discovery');
    });
  });

  describe('Cache Management', () => {
    test('should return cache statistics', async () => {
      // First, make a request to populate cache
      await request(app)
        .get('/services/test-service/instances')
        .expect(200);

      const response = await request(app)
        .get('/cache/stats')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.cacheStats).toBeDefined();
    });

    test('should clear cache successfully', async () => {
      const response = await request(app)
        .delete('/cache')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.message).toBe('Cache cleared successfully');
    });
  });

  describe('Load Balancer Strategy', () => {
    test('should set round-robin strategy', async () => {
      const response = await request(app)
        .post('/loadbalancer/strategy')
        .send({ strategy: 'round-robin' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.message).toContain('round-robin');
    });

    test('should set least-connections strategy', async () => {
      const response = await request(app)
        .post('/loadbalancer/strategy')
        .send({ strategy: 'least-connections' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.message).toContain('least-connections');
    });

    test('should reject invalid strategy', async () => {
      const response = await request(app)
        .post('/loadbalancer/strategy')
        .send({ strategy: 'invalid-strategy' })
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('Invalid strategy');
    });
  });

  describe('Error Handling', () => {
    test('should handle service not found', async () => {
      const response = await request(app)
        .get('/services/non-existent/instance')
        .expect(404);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toContain('No healthy instances found');
    });

    test('should handle invalid registration data', async () => {
      const response = await request(app)
        .post('/services/register')
        .send({ invalid: 'data' })
        .expect(500);

      expect(response.body.success).toBe(false);
    });
  });
});
