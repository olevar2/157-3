/**
 * API Test Setup for Platform3 Phase 2
 * Configures environment for testing REST and WebSocket APIs
 */

const supertest = require('supertest');
const WebSocket = require('ws');

// Global API testing utilities
global.apiUtils = {
  // REST API utilities using supertest
  rest: {
    createTestApp: (app) => {
      return supertest(app);
    },
    
    // Common API test patterns
    testEndpoint: async (request, endpoint, method = 'GET', data = null, expectedStatus = 200) => {
      let req = request[method.toLowerCase()](endpoint);
      
      // Add common headers
      req = req.set('Accept', 'application/json')
               .set('x-test-mode', 'true')
               .set('x-request-id', `test-${Date.now()}`);
      
      // Add data if provided
      if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
        req = req.send(data);
      }
      
      const response = await req.expect(expectedStatus);
      
      return {
        status: response.status,
        body: response.body,
        headers: response.headers,
        text: response.text
      };
    },
    
    // Authentication helpers
    withAuth: (request, token) => {
      return request.set('Authorization', `Bearer ${token}`);
    },
    
    // Common test scenarios
    testHealthEndpoint: async (request, endpoint = '/health') => {
      const response = await global.apiUtils.rest.testEndpoint(request, endpoint, 'GET', null, 200);
      
      expect(response.body).toHaveProperty('status');
      expect(response.body.status).toBe('healthy');
      expect(response.body).toHaveProperty('timestamp');
      
      return response;
    },
    
    testUnauthorizedAccess: async (request, endpoint, method = 'GET') => {
      const response = await global.apiUtils.rest.testEndpoint(request, endpoint, method, null, 401);
      
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toMatch(/unauthorized|authentication/i);
      
      return response;
    },
    
    testNotFoundEndpoint: async (request, endpoint) => {
      const response = await global.apiUtils.rest.testEndpoint(request, endpoint, 'GET', null, 404);
      
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toMatch(/not found/i);
      
      return response;
    },
    
    testRateLimit: async (request, endpoint, limit = 10) => {
      const requests = [];
      
      // Make multiple requests rapidly
      for (let i = 0; i < limit + 5; i++) {
        requests.push(
          request.get(endpoint)
                 .set('x-test-mode', 'true')
                 .set('x-client-ip', '192.168.1.100') // Simulate same IP
        );
      }
      
      const responses = await Promise.allSettled(requests);
      
      // Check that some requests were rate limited
      const rateLimitedResponses = responses.filter(
        result => result.status === 'fulfilled' && result.value.status === 429
      );
      
      expect(rateLimitedResponses.length).toBeGreaterThan(0);
      
      return responses;
    }
  },
  
  // WebSocket testing utilities
  websocket: {
    createConnection: (url, protocols = []) => {
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(url, protocols);
        
        ws.on('open', () => {
          ws.testUtils = {
            messages: [],
            errors: [],
            
            sendMessage: (data) => {
              const message = typeof data === 'string' ? data : JSON.stringify(data);
              ws.send(message);
            },
            
            waitForMessage: (timeout = 5000) => {
              return new Promise((msgResolve, msgReject) => {
                const timer = setTimeout(() => {
                  msgReject(new Error(`No message received within ${timeout}ms`));
                }, timeout);
                
                const onMessage = (data) => {
                  clearTimeout(timer);
                  ws.off('message', onMessage);
                  
                  try {
                    const parsed = JSON.parse(data);
                    msgResolve(parsed);
                  } catch (error) {
                    msgResolve(data.toString());
                  }
                };
                
                ws.on('message', onMessage);
              });
            },
            
            expectMessage: async (expectedData, timeout = 5000) => {
              const message = await ws.testUtils.waitForMessage(timeout);
              
              if (typeof expectedData === 'object') {
                expect(message).toMatchObject(expectedData);
              } else {
                expect(message).toEqual(expectedData);
              }
              
              return message;
            },
            
            close: () => {
              return new Promise((closeResolve) => {
                ws.on('close', closeResolve);
                ws.close();
              });
            }
          };
          
          // Capture all messages and errors for testing
          ws.on('message', (data) => {
            try {
              const parsed = JSON.parse(data);
              ws.testUtils.messages.push(parsed);
            } catch (error) {
              ws.testUtils.messages.push(data.toString());
            }
          });
          
          ws.on('error', (error) => {
            ws.testUtils.errors.push(error);
          });
          
          resolve(ws);
        });
        
        ws.on('error', reject);
      });
    },
    
    testConnection: async (url, protocols = []) => {
      const ws = await global.apiUtils.websocket.createConnection(url, protocols);
      
      // Test basic connectivity
      expect(ws.readyState).toBe(WebSocket.OPEN);
      
      return ws;
    },
    
    testSubscription: async (ws, subscriptionMessage, expectedResponseType) => {
      // Send subscription message
      ws.testUtils.sendMessage(subscriptionMessage);
      
      // Wait for subscription confirmation
      const response = await ws.testUtils.waitForMessage();
      
      expect(response).toHaveProperty('type', expectedResponseType);
      
      return response;
    },
    
    testRealTimeData: async (ws, dataCount = 5, timeout = 10000) => {
      const startTime = Date.now();
      const receivedData = [];
      
      while (receivedData.length < dataCount && (Date.now() - startTime) < timeout) {
        try {
          const data = await ws.testUtils.waitForMessage(2000);
          receivedData.push(data);
        } catch (error) {
          // Timeout waiting for message, continue
          break;
        }
      }
      
      expect(receivedData.length).toBeGreaterThanOrEqual(1);
      
      return receivedData;
    }
  },
  
  // Performance testing utilities
  performance: {
    measureResponseTime: async (testFunction) => {
      const startTime = process.hrtime.bigint();
      const result = await testFunction();
      const endTime = process.hrtime.bigint();
      
      const durationMs = Number(endTime - startTime) / 1000000;
      
      return {
        result,
        durationMs,
        durationFormatted: `${durationMs.toFixed(2)}ms`
      };
    },
    
    testConcurrentRequests: async (requestFunction, concurrency = 10) => {
      const startTime = process.hrtime.bigint();
      
      const requests = Array(concurrency).fill().map(() => requestFunction());
      const results = await Promise.allSettled(requests);
      
      const endTime = process.hrtime.bigint();
      const totalDurationMs = Number(endTime - startTime) / 1000000;
      
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      return {
        totalRequests: concurrency,
        successful,
        failed,
        successRate: (successful / concurrency) * 100,
        totalDurationMs,
        avgDurationMs: totalDurationMs / concurrency
      };
    },
    
    testThroughput: async (requestFunction, duration = 10000) => {
      const startTime = Date.now();
      const endTime = startTime + duration;
      const results = [];
      
      while (Date.now() < endTime) {
        try {
          const result = await requestFunction();
          results.push({ success: true, result });
        } catch (error) {
          results.push({ success: false, error: error.message });
        }
      }
      
      const actualDuration = Date.now() - startTime;
      const requestsPerSecond = (results.length / actualDuration) * 1000;
      const successRate = (results.filter(r => r.success).length / results.length) * 100;
      
      return {
        totalRequests: results.length,
        duration: actualDuration,
        requestsPerSecond: requestsPerSecond.toFixed(2),
        successRate: successRate.toFixed(2)
      };
    }
  }
};

// API test hooks
beforeEach(() => {
  // Clear any cached responses
  if (global.apiCache) {
    global.apiCache.clear();
  }
});

afterEach(() => {
  // Clean up any open WebSocket connections
  if (global.testConnections) {
    global.testConnections.forEach(conn => {
      if (conn.readyState === WebSocket.OPEN) {
        conn.close();
      }
    });
    global.testConnections = [];
  }
});

// Initialize global connection tracking
global.testConnections = [];

console.log('🌐 API test utilities loaded');
console.log('🔌 WebSocket support enabled');
console.log('📈 Performance testing utilities ready');
console.log('🧪 SuperTest integration configured');
