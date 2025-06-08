const WebSocket = require('ws');

const ServiceDiscoveryMiddleware = require('../../../shared/communication/service_discovery_middleware');
const Platform3MessageQueue = require('../../../shared/communication/redis_message_queue');
const HealthCheckEndpoint = require('../../../shared/communication/health_check_endpoint');
const logger = require('../../../shared/logging/platform3_logger');

class WebSocketServer {
  constructor(server, marketDataProvider) {
    this.wss = new WebSocket.Server({ server });
    this.marketDataProvider = marketDataProvider;
    this.clients = new Map();
    this.initialize();
  }

  initialize() {
    this.wss.on('connection', (ws, request) => {
      const clientId = this.generateClientId();
      this.clients.set(clientId, {
        ws,
        subscribedPairs: new Set(),
        lastHeartbeat: Date.now()
      });

      console.log(`🔗 WebSocket client connected: ${clientId}`);

      // Handle incoming messages
      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message);
          this.handleMessage(clientId, data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error.message);
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Invalid message format'
          }));
        }
      });

      // Handle client disconnect
      ws.on('close', () => {
        console.log(`🔌 WebSocket client disconnected: ${clientId}`);
        this.clients.delete(clientId);      });

      // Handle errors
      ws.on('error', (error) => {
        console.error(`WebSocket error for client ${clientId}:`, error.message);
        this.clients.delete(clientId);
      });

      // Send welcome message with available features
      ws.send(JSON.stringify({
        type: 'welcome',
        clientId,
        timestamp: new Date().toISOString(),
        availableFeatures: [
          'price_updates',
          'technical_indicators',
          'market_status',
          'news_updates',
          'trade_signals'
        ],
        supportedPairs: this.marketDataProvider.pairs,
        message: 'Connected to Platform3 Market Data Service'
      }));
    });

    // Set up market data subscription
    this.marketDataProvider.addSubscriber((prices) => {
      this.broadcastPrices(prices);
    });

    // Set up heartbeat interval
    setInterval(() => {
      this.sendHeartbeat();
    }, 30000); // 30 seconds

    console.log('🔄 WebSocket server initialized');
  }

  handleMessage(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;

    switch (data.type) {
      case 'subscribe':
        this.handleSubscribe(clientId, data.pairs);
        break;
        
      case 'unsubscribe':
        this.handleUnsubscribe(clientId, data.pairs);
        break;
        
      case 'ping':
        client.lastHeartbeat = Date.now();
        client.ws.send(JSON.stringify({ type: 'pong' }));
        break;
        
      case 'get_current_prices':
        this.sendCurrentPrices(clientId);
        break;
        
      default:
        client.ws.send(JSON.stringify({
          type: 'error',
          message: `Unknown message type: ${data.type}`
        }));
    }
  }

  handleSubscribe(clientId, pairs) {
    const client = this.clients.get(clientId);
    if (!client) return;

    if (Array.isArray(pairs)) {
      pairs.forEach(pair => client.subscribedPairs.add(pair));
    } else {
      client.subscribedPairs.add(pairs);
    }

    client.ws.send(JSON.stringify({
      type: 'subscription_confirmed',
      pairs: Array.from(client.subscribedPairs)
    }));

    console.log(`📊 Client ${clientId} subscribed to:`, Array.from(client.subscribedPairs));
  }

  handleUnsubscribe(clientId, pairs) {
    const client = this.clients.get(clientId);
    if (!client) return;

    if (Array.isArray(pairs)) {
      pairs.forEach(pair => client.subscribedPairs.delete(pair));
    } else {
      client.subscribedPairs.delete(pairs);
    }

    client.ws.send(JSON.stringify({
      type: 'unsubscription_confirmed',
      pairs: Array.from(client.subscribedPairs)
    }));
  }

  async sendCurrentPrices(clientId) {
    const client = this.clients.get(clientId);
    if (!client) return;

    try {
      const prices = await this.marketDataProvider.getCurrentPrices();
      client.ws.send(JSON.stringify({
        type: 'current_prices',
        data: prices,
        timestamp: new Date().toISOString()
      }));
    } catch (error) {
      client.ws.send(JSON.stringify({
        type: 'error',
        message: 'Failed to fetch current prices'
      }));
    }
  }

  broadcastPrices(prices) {
    const message = JSON.stringify({
      type: 'price_update',
      data: prices,
      timestamp: new Date().toISOString()
    });

    this.clients.forEach((client, clientId) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        // Only send prices for subscribed pairs
        const filteredPrices = prices.filter(price => 
          client.subscribedPairs.has(price.symbol) || client.subscribedPairs.size === 0
        );

        if (filteredPrices.length > 0) {
          client.ws.send(JSON.stringify({
            type: 'price_update',
            data: filteredPrices,
            timestamp: new Date().toISOString()
          }));
        }
      }
    });
  }

  sendHeartbeat() {
    const now = Date.now();
    
    this.clients.forEach((client, clientId) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        // Check if client is still alive (responded to ping in last 60 seconds)
        if (now - client.lastHeartbeat > 60000) {
          console.log(`💔 Client ${clientId} timeout, disconnecting`);
          client.ws.terminate();
          this.clients.delete(clientId);
        } else {
          client.ws.send(JSON.stringify({
            type: 'heartbeat',
            timestamp: new Date().toISOString()
          }));
        }
      } else {
        this.clients.delete(clientId);
      }
    });
  }

  generateClientId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }

  getConnectedClients() {
    return this.clients.size;
  }

  close() {
    this.clients.forEach((client) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.close();
      }
    });
    this.wss.close();
    console.log('🔌 WebSocket server closed');
  }
}

module.exports = WebSocketServer;


// Platform3 Microservices Integration
const serviceDiscovery = new ServiceDiscoveryMiddleware('services', PORT || 3000);
const messageQueue = new Platform3MessageQueue();
const healthCheck = new HealthCheckEndpoint('services', [
    {
        name: 'redis',
        check: async () => {
            return { healthy: true, responseTime: 0 };
        }
    }
]);

// Apply service discovery middleware
app.use(serviceDiscovery.middleware());

// Add health check endpoints
app.use('/api', healthCheck.getRouter());

// Register service with Consul on startup
serviceDiscovery.registerService().catch(err => {
    logger.error('Failed to register service', { error: err.message });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
});

process.on('SIGINT', async () => {
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
});
