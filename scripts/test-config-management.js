#!/usr/bin/env node

/**
 * Platform3 Configuration Management Test Script
 * Tests the configuration service and client integration
 */

const axios = require('axios');
const { ConfigClient } = require('../services/shared/src/ConfigClient');

const CONFIG_SERVICE_URL = process.env.CONFIG_SERVICE_URL || 'http://localhost:3007';
const API_KEY = process.env.CONFIG_API_KEY || 'config-service-key-1';

async function testConfigurationService() {
  console.log('🧪 Testing Platform3 Configuration Management System...\n');

  try {
    // Test 1: Health Check
    console.log('1️⃣ Testing Configuration Service Health Check...');
    const healthResponse = await axios.get(`${CONFIG_SERVICE_URL}/health`);
    
    if (healthResponse.status === 200) {
      console.log('✅ Configuration Service is healthy');
      console.log(`   Status: ${healthResponse.data.status}`);
      console.log(`   Vault: ${healthResponse.data.checks.vault ? '✅' : '❌'}`);
      console.log(`   Redis: ${healthResponse.data.checks.redis ? '✅' : '❌'}`);
      console.log(`   Cache: ${healthResponse.data.checks.cache ? '✅' : '❌'}`);
    } else {
      throw new Error(`Health check failed with status: ${healthResponse.status}`);
    }

    // Test 2: Configuration Retrieval
    console.log('\n2️⃣ Testing Configuration Retrieval...');
    const configResponse = await axios.post(`${CONFIG_SERVICE_URL}/api/v1/config`, {
      service: 'test-service',
      environment: 'development'
    }, {
      headers: {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
      }
    });

    if (configResponse.status === 200 && configResponse.data.success) {
      console.log('✅ Configuration retrieval successful');
      console.log(`   Service: ${configResponse.data.data.service}`);
      console.log(`   Environment: ${configResponse.data.data.environment}`);
      console.log(`   Config Keys: ${Object.keys(configResponse.data.data.configuration).length}`);
    } else {
      throw new Error('Configuration retrieval failed');
    }

    // Test 3: Configuration Update
    console.log('\n3️⃣ Testing Configuration Update...');
    const updateResponse = await axios.put(
      `${CONFIG_SERVICE_URL}/api/v1/config/test-service/development/test-key`,
      { value: 'test-value-' + Date.now() },
      {
        headers: {
          'X-API-Key': API_KEY,
          'X-User-Id': 'test-script',
          'Content-Type': 'application/json'
        }
      }
    );

    if (updateResponse.status === 200 && updateResponse.data.success) {
      console.log('✅ Configuration update successful');
    } else {
      throw new Error('Configuration update failed');
    }

    // Test 4: Service Registration
    console.log('\n4️⃣ Testing Service Registration...');
    const registrationResponse = await axios.post(`${CONFIG_SERVICE_URL}/api/v1/register`, {
      serviceName: 'test-service',
      environment: 'development',
      configKeys: ['database', 'jwt', 'redis', 'test-key'],
      webhookUrl: 'http://test-service:3000/config-webhook'
    }, {
      headers: {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
      }
    });

    if (registrationResponse.status === 200 && registrationResponse.data.success) {
      console.log('✅ Service registration successful');
    } else {
      throw new Error('Service registration failed');
    }

    // Test 5: Configuration History
    console.log('\n5️⃣ Testing Configuration History...');
    const historyResponse = await axios.get(`${CONFIG_SERVICE_URL}/api/v1/history`, {
      params: {
        service: 'test-service',
        environment: 'development'
      },
      headers: {
        'X-API-Key': API_KEY
      }
    });

    if (historyResponse.status === 200 && historyResponse.data.success) {
      console.log('✅ Configuration history retrieval successful');
      console.log(`   History entries: ${historyResponse.data.data.length}`);
    } else {
      throw new Error('Configuration history retrieval failed');
    }

    console.log('\n🎉 All Configuration Service tests passed!');
    return true;

  } catch (error) {
    console.error('\n❌ Configuration Service test failed:', error.message);
    if (error.response) {
      console.error(`   Status: ${error.response.status}`);
      console.error(`   Data:`, error.response.data);
    }
    return false;
  }
}

async function testConfigClient() {
  console.log('\n🧪 Testing Configuration Client...\n');

  try {
    // Test ConfigClient
    console.log('1️⃣ Testing ConfigClient Initialization...');
    
    const configClient = new ConfigClient({
      serviceUrl: CONFIG_SERVICE_URL,
      apiKey: API_KEY,
      serviceName: 'test-client-service',
      environment: 'development',
      refreshInterval: 60000
    });

    // Initialize client
    await configClient.initialize();
    console.log('✅ ConfigClient initialized successfully');

    // Test configuration access
    console.log('\n2️⃣ Testing Configuration Access...');
    
    // Test getting all configuration
    const allConfig = configClient.getAll();
    console.log(`✅ Retrieved all configuration (${Object.keys(allConfig).length} keys)`);

    // Test getting specific values with defaults
    const stringValue = configClient.getString('test-key', 'default-value');
    console.log(`✅ String value: ${stringValue}`);

    const numberValue = configClient.getNumber('port', 3000);
    console.log(`✅ Number value: ${numberValue}`);

    const booleanValue = configClient.getBoolean('debug', false);
    console.log(`✅ Boolean value: ${booleanValue}`);

    // Test feature flags
    console.log('\n3️⃣ Testing Feature Flags...');
    const featureEnabled = await configClient.isFeatureEnabled('test-feature');
    console.log(`✅ Feature flag 'test-feature': ${featureEnabled ? 'enabled' : 'disabled'}`);

    // Test client status
    console.log('\n4️⃣ Testing Client Status...');
    const status = configClient.getStatus();
    console.log('✅ Client status:');
    console.log(`   Initialized: ${status.initialized}`);
    console.log(`   Config count: ${status.configCount}`);
    console.log(`   Service: ${status.serviceName}`);
    console.log(`   Environment: ${status.environment}`);

    // Clean up
    configClient.stop();
    console.log('\n🎉 All ConfigClient tests passed!');
    return true;

  } catch (error) {
    console.error('\n❌ ConfigClient test failed:', error.message);
    return false;
  }
}

async function main() {
  console.log('🚀 Platform3 Configuration Management Integration Test\n');
  console.log(`Configuration Service URL: ${CONFIG_SERVICE_URL}`);
  console.log(`API Key: ${API_KEY.substring(0, 10)}...`);
  console.log('=' * 60);

  const serviceTestPassed = await testConfigurationService();
  const clientTestPassed = await testConfigClient();

  console.log('\n' + '=' * 60);
  console.log('📊 TEST RESULTS:');
  console.log(`Configuration Service: ${serviceTestPassed ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`Configuration Client: ${clientTestPassed ? '✅ PASSED' : '❌ FAILED'}`);

  if (serviceTestPassed && clientTestPassed) {
    console.log('\n🎉 ALL TESTS PASSED! Configuration Management is working correctly.');
    process.exit(0);
  } else {
    console.log('\n❌ SOME TESTS FAILED! Please check the configuration setup.');
    process.exit(1);
  }
}

// Run the tests
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = { testConfigurationService, testConfigClient };
