const axios = require('axios');
const { createLogger } = require('./logger');

class ServiceRegistration {
  constructor(options = {}) {
    this.serviceDiscoveryUrl = options.serviceDiscoveryUrl || process.env.SERVICE_DISCOVERY_URL || 'http://localhost:3010';
    this.serviceName = options.serviceName || process.env.SERVICE_NAME;
    this.serviceHost = options.serviceHost || process.env.HOST || 'localhost';
    this.servicePort = options.servicePort || process.env.PORT;
    this.healthCheckPath = options.healthCheckPath || '/health';
    this.tags = options.tags || [];
    this.meta = options.meta || {};
    this.logger = createLogger(`ServiceRegistration-${this.serviceName}`);
    this.registrationId = null;
    this.isRegistered = false;
    this.healthCheckInterval = null;
  }

  /**
   * Register this service with service discovery
   */
  async register() {
    if (!this.serviceName || !this.servicePort) {
      throw new Error('Service name and port are required for registration');
    }

    try {
      const registration = {
        name: this.serviceName,
        address: this.serviceHost,
        port: parseInt(this.servicePort),
        tags: [
          ...this.tags,
          'http',
          process.env.NODE_ENV || 'development'
        ],
        meta: {
          version: process.env.VERSION || '1.0.0',
          environment: process.env.NODE_ENV || 'development',
          startTime: new Date().toISOString(),
          ...this.meta
        },
        check: {
          http: `http://${this.serviceHost}:${this.servicePort}${this.healthCheckPath}`,
          interval: '10s',
          timeout: '5s',
          deregisterCriticalServiceAfter: '30s'
        }
      };

      const response = await axios.post(
        `${this.serviceDiscoveryUrl}/services/register`,
        registration,
        { timeout: 10000 }
      );

      if (response.data.success) {
        this.registrationId = response.data.serviceId;
        this.isRegistered = true;
        
        this.logger.info('Service registered successfully', {
          serviceId: this.registrationId,
          serviceName: this.serviceName,
          address: `${this.serviceHost}:${this.servicePort}`
        });

        // Start periodic health check reporting
        this.startHealthCheckReporting();
        
        return this.registrationId;
      } else {
        throw new Error('Registration failed: ' + response.data.error);
      }
    } catch (error) {
      this.logger.error('Failed to register service', {
        serviceName: this.serviceName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Deregister this service from service discovery
   */
  async deregister() {
    if (!this.isRegistered || !this.registrationId) {
      this.logger.warn('Service is not registered, skipping deregistration');
      return;
    }

    try {
      await axios.delete(
        `${this.serviceDiscoveryUrl}/services/${this.registrationId}`,
        { timeout: 5000 }
      );

      this.isRegistered = false;
      this.registrationId = null;
      
      // Stop health check reporting
      this.stopHealthCheckReporting();
      
      this.logger.info('Service deregistered successfully', {
        serviceName: this.serviceName
      });
    } catch (error) {
      this.logger.error('Failed to deregister service', {
        serviceName: this.serviceName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Discover other services
   */
  async discoverService(serviceName) {
    try {
      const response = await axios.get(
        `${this.serviceDiscoveryUrl}/services/${serviceName}/instance`,
        { timeout: 5000 }
      );

      if (response.data.success) {
        return response.data.instance;
      } else {
        this.logger.warn('No healthy instances found for service', { serviceName });
        return null;
      }
    } catch (error) {
      this.logger.error('Failed to discover service', {
        serviceName,
        error: error.message
      });
      return null;
    }
  }

  /**
   * Get service URL for making HTTP requests
   */
  async getServiceUrl(serviceName, path = '') {
    try {
      const response = await axios.get(
        `${this.serviceDiscoveryUrl}/services/${serviceName}/url`,
        { 
          params: { path },
          timeout: 5000 
        }
      );

      if (response.data.success) {
        return response.data.url;
      } else {
        this.logger.warn('No healthy instances found for service URL', { serviceName });
        return null;
      }
    } catch (error) {
      this.logger.error('Failed to get service URL', {
        serviceName,
        error: error.message
      });
      return null;
    }
  }

  /**
   * Start periodic health check reporting
   */
  startHealthCheckReporting() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    // Report health status every 30 seconds
    this.healthCheckInterval = setInterval(async () => {
      try {
        // Check if service discovery is still reachable
        await axios.get(`${this.serviceDiscoveryUrl}/health`, { timeout: 3000 });
      } catch (error) {
        this.logger.warn('Service discovery health check failed', {
          error: error.message
        });
      }
    }, 30000);
  }

  /**
   * Stop health check reporting
   */
  stopHealthCheckReporting() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  /**
   * Check if service discovery is available
   */
  async isServiceDiscoveryAvailable() {
    try {
      const response = await axios.get(
        `${this.serviceDiscoveryUrl}/health`,
        { timeout: 3000 }
      );
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get registration status
   */
  getRegistrationStatus() {
    return {
      isRegistered: this.isRegistered,
      registrationId: this.registrationId,
      serviceName: this.serviceName,
      serviceAddress: `${this.serviceHost}:${this.servicePort}`,
      serviceDiscoveryUrl: this.serviceDiscoveryUrl
    };
  }

  /**
   * Setup graceful shutdown handlers
   */
  setupGracefulShutdown() {
    const gracefulShutdown = async (signal) => {
      this.logger.info(`Received ${signal}, deregistering service...`);
      
      try {
        await this.deregister();
        this.logger.info('Service deregistered successfully during shutdown');
      } catch (error) {
        this.logger.error('Failed to deregister service during shutdown', {
          error: error.message
        });
      }
      
      process.exit(0);
    };

    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    process.on('SIGUSR2', () => gracefulShutdown('SIGUSR2')); // For nodemon
  }
}

module.exports = ServiceRegistration;
