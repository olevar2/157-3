const { RoundRobinStrategy, LeastConnectionsStrategy, RandomStrategy } = require('../dist/ServiceDiscoveryClient');

describe('Service Discovery Unit Tests', () => {
  describe('Load Balancer Strategies', () => {
    const mockInstances = [
      { id: 'service-1', name: 'test-service', address: 'localhost', port: 3001, tags: [], meta: {}, health: 'passing' },
      { id: 'service-2', name: 'test-service', address: 'localhost', port: 3002, tags: [], meta: {}, health: 'passing' },
      { id: 'service-3', name: 'test-service', address: 'localhost', port: 3003, tags: [], meta: {}, health: 'passing' }
    ];

    describe('RoundRobinStrategy', () => {
      test('should distribute requests evenly across instances', () => {
        const strategy = new RoundRobinStrategy();
        const selections = [];

        // Make 6 selections (2 rounds)
        for (let i = 0; i < 6; i++) {
          const selected = strategy.selectInstance(mockInstances);
          selections.push(selected.port);
        }

        // Should cycle through all instances twice
        expect(selections).toEqual([3001, 3002, 3003, 3001, 3002, 3003]);
      });

      test('should handle empty instances array', () => {
        const strategy = new RoundRobinStrategy();
        const selected = strategy.selectInstance([]);
        expect(selected).toBeNull();
      });

      test('should handle single instance', () => {
        const strategy = new RoundRobinStrategy();
        const singleInstance = [mockInstances[0]];
        
        const selected1 = strategy.selectInstance(singleInstance);
        const selected2 = strategy.selectInstance(singleInstance);
        
        expect(selected1.port).toBe(3001);
        expect(selected2.port).toBe(3001);
      });
    });

    describe('LeastConnectionsStrategy', () => {
      test('should select instance with least connections', () => {
        const strategy = new LeastConnectionsStrategy();
        
        // Simulate connections to first instance
        strategy.incrementConnections('service-1');
        strategy.incrementConnections('service-1');
        
        // Should select instance with no connections
        const selected = strategy.selectInstance(mockInstances);
        expect(selected.id).not.toBe('service-1');
      });

      test('should handle connection increment and decrement', () => {
        const strategy = new LeastConnectionsStrategy();
        
        strategy.incrementConnections('service-1');
        strategy.incrementConnections('service-1');
        expect(strategy.connections.get('service-1')).toBe(2);
        
        strategy.decrementConnections('service-1');
        expect(strategy.connections.get('service-1')).toBe(1);
        
        // Should not go below 0
        strategy.decrementConnections('service-1');
        strategy.decrementConnections('service-1');
        expect(strategy.connections.get('service-1')).toBe(0);
      });

      test('should handle empty instances array', () => {
        const strategy = new LeastConnectionsStrategy();
        const selected = strategy.selectInstance([]);
        expect(selected).toBeNull();
      });
    });

    describe('RandomStrategy', () => {
      test('should select random instances', () => {
        const strategy = new RandomStrategy();
        const selections = new Set();
        
        // Make multiple selections to test randomness
        for (let i = 0; i < 20; i++) {
          const selected = strategy.selectInstance(mockInstances);
          selections.add(selected.port);
        }
        
        // Should have selected from multiple instances (high probability)
        expect(selections.size).toBeGreaterThan(1);
      });

      test('should handle empty instances array', () => {
        const strategy = new RandomStrategy();
        const selected = strategy.selectInstance([]);
        expect(selected).toBeNull();
      });

      test('should handle single instance', () => {
        const strategy = new RandomStrategy();
        const singleInstance = [mockInstances[0]];
        
        const selected = strategy.selectInstance(singleInstance);
        expect(selected.port).toBe(3001);
      });
    });
  });

  describe('Service Instance Validation', () => {
    test('should validate service instance structure', () => {
      const instance = {
        id: 'test-service-1',
        name: 'test-service',
        address: 'localhost',
        port: 3001,
        tags: ['http', 'api'],
        meta: { version: '1.0.0' },
        health: 'passing'
      };

      expect(instance).toHaveProperty('id');
      expect(instance).toHaveProperty('name');
      expect(instance).toHaveProperty('address');
      expect(instance).toHaveProperty('port');
      expect(instance).toHaveProperty('tags');
      expect(instance).toHaveProperty('meta');
      expect(instance).toHaveProperty('health');
      
      expect(typeof instance.id).toBe('string');
      expect(typeof instance.name).toBe('string');
      expect(typeof instance.address).toBe('string');
      expect(typeof instance.port).toBe('number');
      expect(Array.isArray(instance.tags)).toBe(true);
      expect(typeof instance.meta).toBe('object');
      expect(['passing', 'warning', 'critical']).toContain(instance.health);
    });
  });

  describe('Service Registration Validation', () => {
    test('should validate service registration structure', () => {
      const registration = {
        name: 'test-service',
        address: 'localhost',
        port: 3001,
        tags: ['http', 'api'],
        meta: { version: '1.0.0', environment: 'test' },
        check: {
          http: 'http://localhost:3001/health',
          interval: '10s',
          timeout: '5s',
          deregisterCriticalServiceAfter: '30s'
        }
      };

      expect(registration).toHaveProperty('name');
      expect(registration).toHaveProperty('address');
      expect(registration).toHaveProperty('port');
      expect(registration.check).toHaveProperty('http');
      expect(registration.check).toHaveProperty('interval');
      
      expect(typeof registration.name).toBe('string');
      expect(typeof registration.address).toBe('string');
      expect(typeof registration.port).toBe('number');
      expect(typeof registration.check.http).toBe('string');
      expect(typeof registration.check.interval).toBe('string');
    });
  });
});

// Mock console methods to reduce test noise
beforeAll(() => {
  jest.spyOn(console, 'log').mockImplementation(() => {});
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterAll(() => {
  jest.restoreAllMocks();
});
