import request from 'supertest';
import app from '../src/index';

// Integration tests for Configuration Service
describe('Configuration Service Integration Tests', () => {
  const validApiKey = 'config-service-key-1';

  describe('Health Check', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('service', 'config-service');
      expect(response.body).toHaveProperty('version', '1.0.0');
      expect(response.body).toHaveProperty('checks');
    });
  });

  describe('Configuration API', () => {
    describe('POST /api/v1/config', () => {
      it('should require API key', async () => {
        await request(app)
          .post('/api/v1/config')
          .send({
            service: 'test-service',
            environment: 'development'
          })
          .expect(401);
      });

      it('should validate required fields', async () => {
        const response = await request(app)
          .post('/api/v1/config')
          .set('X-API-Key', validApiKey)
          .send({})
          .expect(400);

        expect(response.body.error).toContain('Missing required fields');
      });

      it('should return configuration for valid request', async () => {
        const response = await request(app)
          .post('/api/v1/config')
          .set('X-API-Key', validApiKey)
          .send({
            service: 'test-service',
            environment: 'development'
          })
          .expect(200);

        expect(response.body).toHaveProperty('success', true);
        expect(response.body).toHaveProperty('data');
        expect(response.body.data).toHaveProperty('service', 'test-service');
        expect(response.body.data).toHaveProperty('environment', 'development');
        expect(response.body.data).toHaveProperty('configuration');
      });

      it('should handle specific configuration keys', async () => {
        const response = await request(app)
          .post('/api/v1/config')
          .set('X-API-Key', validApiKey)
          .send({
            service: 'test-service',
            environment: 'development',
            keys: ['database', 'jwt']
          })
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.configuration).toBeDefined();
      });
    });

    describe('PUT /api/v1/config/:service/:environment/:key', () => {
      it('should require API key', async () => {
        await request(app)
          .put('/api/v1/config/test-service/development/test-key')
          .send({ value: 'test-value' })
          .expect(401);
      });

      it('should validate required value field', async () => {
        const response = await request(app)
          .put('/api/v1/config/test-service/development/test-key')
          .set('X-API-Key', validApiKey)
          .send({})
          .expect(400);

        expect(response.body.error).toContain('Missing required field: value');
      });

      it('should update configuration successfully', async () => {
        const response = await request(app)
          .put('/api/v1/config/test-service/development/test-key')
          .set('X-API-Key', validApiKey)
          .set('X-User-Id', 'test-user')
          .send({ value: 'updated-value' })
          .expect(200);

        expect(response.body).toHaveProperty('success', true);
        expect(response.body).toHaveProperty('message', 'Configuration updated successfully');
      });
    });
  });

  describe('Feature Flags API', () => {
    describe('GET /api/v1/feature-flags/:name', () => {
      it('should require API key', async () => {
        await request(app)
          .get('/api/v1/feature-flags/test-flag')
          .expect(401);
      });

      it('should return 404 for non-existent feature flag', async () => {
        await request(app)
          .get('/api/v1/feature-flags/non-existent-flag')
          .set('X-API-Key', validApiKey)
          .expect(404);
      });

      it('should handle service and environment query parameters', async () => {
        await request(app)
          .get('/api/v1/feature-flags/test-flag')
          .query({
            service: 'test-service',
            environment: 'development'
          })
          .set('X-API-Key', validApiKey)
          .expect(404); // Assuming flag doesn't exist
      });
    });
  });

  describe('Service Registration API', () => {
    describe('POST /api/v1/register', () => {
      it('should require API key', async () => {
        await request(app)
          .post('/api/v1/register')
          .send({
            serviceName: 'test-service',
            environment: 'development',
            configKeys: ['database', 'jwt']
          })
          .expect(401);
      });

      it('should validate required fields', async () => {
        const response = await request(app)
          .post('/api/v1/register')
          .set('X-API-Key', validApiKey)
          .send({})
          .expect(400);

        expect(response.body.error).toContain('Missing required fields');
      });

      it('should register service successfully', async () => {
        const response = await request(app)
          .post('/api/v1/register')
          .set('X-API-Key', validApiKey)
          .send({
            serviceName: 'test-service',
            environment: 'development',
            configKeys: ['database', 'jwt', 'redis'],
            webhookUrl: 'http://test-service:3000/config-webhook'
          })
          .expect(200);

        expect(response.body).toHaveProperty('success', true);
        expect(response.body).toHaveProperty('message', 'Service registered successfully');
      });
    });
  });

  describe('Configuration History API', () => {
    describe('GET /api/v1/history', () => {
      it('should require API key', async () => {
        await request(app)
          .get('/api/v1/history')
          .expect(401);
      });

      it('should return configuration history', async () => {
        const response = await request(app)
          .get('/api/v1/history')
          .set('X-API-Key', validApiKey)
          .expect(200);

        expect(response.body).toHaveProperty('success', true);
        expect(response.body).toHaveProperty('data');
        expect(Array.isArray(response.body.data)).toBe(true);
      });

      it('should filter history by query parameters', async () => {
        const response = await request(app)
          .get('/api/v1/history')
          .query({
            service: 'test-service',
            environment: 'development',
            key: 'test-key'
          })
          .set('X-API-Key', validApiKey)
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(Array.isArray(response.body.data)).toBe(true);
      });
    });
  });

  describe('Metrics Endpoint', () => {
    it('should return Prometheus metrics', async () => {
      const response = await request(app)
        .get('/metrics')
        .expect(200);

      expect(response.text).toContain('config_service_requests_total');
      expect(response.text).toContain('config_service_vault_connection');
      expect(response.text).toContain('config_service_redis_connection');
      expect(response.headers['content-type']).toContain('text/plain');
    });
  });

  describe('Error Handling', () => {
    it('should handle 404 for unknown routes', async () => {
      const response = await request(app)
        .get('/api/v1/unknown-endpoint')
        .set('X-API-Key', validApiKey)
        .expect(404);

      expect(response.body).toHaveProperty('error', 'Not found');
      expect(response.body.message).toContain('Route GET /api/v1/unknown-endpoint not found');
    });

    it('should handle invalid JSON in request body', async () => {
      await request(app)
        .post('/api/v1/config')
        .set('X-API-Key', validApiKey)
        .set('Content-Type', 'application/json')
        .send('invalid json')
        .expect(400);
    });

    it('should handle large request bodies', async () => {
      const largeObject = {
        service: 'test-service',
        environment: 'development',
        data: 'x'.repeat(11 * 1024 * 1024) // 11MB - should exceed 10MB limit
      };

      await request(app)
        .post('/api/v1/config')
        .set('X-API-Key', validApiKey)
        .send(largeObject)
        .expect(413); // Payload too large
    });
  });

  describe('Security', () => {
    it('should include security headers', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      // Check for helmet security headers
      expect(response.headers).toHaveProperty('x-content-type-options');
      expect(response.headers).toHaveProperty('x-frame-options');
      expect(response.headers).toHaveProperty('x-xss-protection');
    });

    it('should reject requests with invalid API keys', async () => {
      await request(app)
        .post('/api/v1/config')
        .set('X-API-Key', 'invalid-key')
        .send({
          service: 'test-service',
          environment: 'development'
        })
        .expect(401);
    });

    it('should handle CORS properly', async () => {
      const response = await request(app)
        .options('/api/v1/config')
        .set('Origin', 'http://localhost:3000')
        .expect(204);

      expect(response.headers).toHaveProperty('access-control-allow-origin');
    });
  });

  describe('Performance', () => {
    it('should respond to health check quickly', async () => {
      const start = Date.now();

      await request(app)
        .get('/health')
        .expect(200);

      const duration = Date.now() - start;
      expect(duration).toBeLessThan(1000); // Should respond within 1 second
    });

    it('should handle concurrent requests', async () => {
      const requests = Array(10).fill(null).map(() =>
        request(app)
          .post('/api/v1/config')
          .set('X-API-Key', validApiKey)
          .send({
            service: 'test-service',
            environment: 'development'
          })
      );

      const responses = await Promise.all(requests);

      responses.forEach((response: any) => {
        expect(response.status).toBe(200);
        expect(response.body.success).toBe(true);
      });
    });
  });
});
