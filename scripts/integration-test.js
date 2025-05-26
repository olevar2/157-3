// Service Integration Testing Script
// Tests the complete user journey and service connectivity

const axios = require('axios');
const WebSocket = require('ws');

console.log('🧪 Starting Service Integration Testing');
console.log('=====================================');

// Configuration
const services = {
  apiGateway: 'http://localhost:3001',
  userService: 'http://localhost:3002', 
  tradingService: 'http://localhost:3003',
  marketDataService: 'http://localhost:3004',
  eventSystem: 'http://localhost:3005',
  dashboard: 'http://localhost:3000'
};

let authToken = null;
let testResults = {
  serviceHealth: {},
  authentication: false,
  marketData: false,
  trading: false,
  integration: false
};

// Helper function to make authenticated requests
async function makeRequest(url, method = 'GET', data = null, useAuth = false) {
  try {
    const config = {
      method,
      url,
      timeout: 5000,
      headers: {}
    };
    
    if (useAuth && authToken) {
      config.headers.Authorization = `Bearer ${authToken}`;
    }
    
    if (data) {
      config.data = data;
      config.headers['Content-Type'] = 'application/json';
    }
    
    const response = await axios(config);
    return { success: true, data: response.data, status: response.status };
  } catch (error) {
    return { 
      success: false, 
      error: error.message, 
      status: error.response?.status,
      data: error.response?.data 
    };
  }
}

// Test 1: Service Health Checks
async function testServiceHealth() {
  console.log('\n🔍 Testing Service Health...');
  
  for (const [serviceName, baseUrl] of Object.entries(services)) {
    console.log(`Testing ${serviceName}...`);
    
    const result = await makeRequest(`${baseUrl}/health`);
    testResults.serviceHealth[serviceName] = result.success;
    
    if (result.success) {
      console.log(`✅ ${serviceName}: ${result.data?.status || 'OK'}`);
    } else {
      console.log(`❌ ${serviceName}: ${result.error}`);
    }
  }
  
  const healthyServices = Object.values(testResults.serviceHealth).filter(Boolean).length;
  console.log(`\n📊 Service Health: ${healthyServices}/${Object.keys(services).length} services healthy`);
  
  return healthyServices >= 3; // Need at least 3 core services
}

// Test 2: User Authentication Flow
async function testAuthentication() {
  console.log('\n🔐 Testing Authentication Flow...');
  
  // Test owner registration (should fail if already exists)
  console.log('Testing owner registration...');
  const registerResult = await makeRequest(`${services.userService}/api/v1/auth/register-owner`, 'POST', {
    email: 'owner@forexplatform.com',
    password: 'TestPassword123!',
    fullName: 'Platform Owner',
    phone: '+1234567890'
  });
  
  if (registerResult.success) {
    console.log('✅ Owner registration successful');
  } else if (registerResult.status === 409) {
    console.log('✅ Owner already exists (expected)');
  } else {
    console.log(`❌ Registration failed: ${registerResult.error}`);
  }
  
  // Test login
  console.log('Testing login...');
  const loginResult = await makeRequest(`${services.userService}/api/v1/auth/login`, 'POST', {
    email: 'owner@forexplatform.com',
    password: 'TestPassword123!'
  });
  
  if (loginResult.success && loginResult.data.token) {
    authToken = loginResult.data.token;
    console.log('✅ Login successful, token received');
    testResults.authentication = true;
    
    // Test profile access
    const profileResult = await makeRequest(`${services.userService}/api/v1/users/profile`, 'GET', null, true);
    if (profileResult.success) {
      console.log('✅ Profile access successful');
      console.log(`   User: ${profileResult.data.user?.email}`);
    } else {
      console.log(`❌ Profile access failed: ${profileResult.error}`);
    }
    
  } else {
    console.log(`❌ Login failed: ${loginResult.error}`);
    testResults.authentication = false;
  }
  
  return testResults.authentication;
}

// Test 3: Market Data Service
async function testMarketData() {
  console.log('\n📊 Testing Market Data Service...');
  
  // Test current prices
  const pricesResult = await makeRequest(`${services.marketDataService}/api/market-data/prices`);
  if (pricesResult.success) {
    console.log('✅ Market prices endpoint working');
    console.log(`   Received ${pricesResult.data?.prices?.length || 0} price quotes`);
    testResults.marketData = true;
  } else {
    console.log(`❌ Market prices failed: ${pricesResult.error}`);
  }
  
  // Test instruments
  const instrumentsResult = await makeRequest(`${services.marketDataService}/api/market-data/instruments`);
  if (instrumentsResult.success) {
    console.log('✅ Instruments endpoint working');
    console.log(`   Available instruments: ${instrumentsResult.data?.instruments?.length || 0}`);
  } else {
    console.log(`❌ Instruments failed: ${instrumentsResult.error}`);
  }
  
  // Test historical data
  const historyResult = await makeRequest(`${services.marketDataService}/api/market-data/history?symbol=EURUSD&timeframe=1h&limit=10`);
  if (historyResult.success) {
    console.log('✅ Historical data endpoint working');
  } else {
    console.log(`❌ Historical data failed: ${historyResult.error}`);
  }
  
  return testResults.marketData;
}

// Test 4: Trading Service
async function testTrading() {
  console.log('\n💰 Testing Trading Service...');
  
  if (!authToken) {
    console.log('❌ Cannot test trading without authentication');
    return false;
  }
  
  // Test portfolio summary
  const portfolioResult = await makeRequest(`${services.tradingService}/api/v1/portfolio/summary`, 'GET', null, true);
  if (portfolioResult.success) {
    console.log('✅ Portfolio summary endpoint working');
    console.log(`   Balance: $${portfolioResult.data?.balance || 0}`);
  } else {
    console.log(`❌ Portfolio summary failed: ${portfolioResult.error}`);
  }
  
  // Test trades list
  const tradesResult = await makeRequest(`${services.tradingService}/api/v1/trades`, 'GET', null, true);
  if (tradesResult.success) {
    console.log('✅ Trades list endpoint working');
    console.log(`   Total trades: ${tradesResult.data?.trades?.length || 0}`);
    testResults.trading = true;
  } else {
    console.log(`❌ Trades list failed: ${tradesResult.error}`);
  }
  
  // Test creating a demo trade
  const createTradeResult = await makeRequest(`${services.tradingService}/api/v1/trades`, 'POST', {
    symbol: 'EURUSD',
    type: 'market',
    side: 'buy',
    quantity: 0.01,
    stopLoss: 1.0800,
    takeProfit: 1.1000
  }, true);
  
  if (createTradeResult.success) {
    console.log('✅ Trade creation successful');
    console.log(`   Trade ID: ${createTradeResult.data?.trade?.id}`);
  } else {
    console.log(`❌ Trade creation failed: ${createTradeResult.error}`);
  }
  
  return testResults.trading;
}

// Test 5: API Gateway Integration
async function testAPIGatewayIntegration() {
  console.log('\n🌐 Testing API Gateway Integration...');
  
  if (!authToken) {
    console.log('❌ Cannot test API Gateway without authentication');
    return false;
  }
  
  // Test routing through API Gateway
  const gatewayUserResult = await makeRequest(`${services.apiGateway}/api/v1/users/profile`, 'GET', null, true);
  if (gatewayUserResult.success) {
    console.log('✅ API Gateway → User Service routing working');
  } else {
    console.log(`❌ API Gateway → User Service failed: ${gatewayUserResult.error}`);
  }
  
  const gatewayTradingResult = await makeRequest(`${services.apiGateway}/api/v1/trades`, 'GET', null, true);
  if (gatewayTradingResult.success) {
    console.log('✅ API Gateway → Trading Service routing working');
    testResults.integration = true;
  } else {
    console.log(`❌ API Gateway → Trading Service failed: ${gatewayTradingResult.error}`);
  }
  
  return testResults.integration;
}

// Test 6: Frontend Dashboard
async function testFrontendDashboard() {
  console.log('\n🖥️ Testing Frontend Dashboard...');
  
  const dashboardResult = await makeRequest(services.dashboard);
  if (dashboardResult.success) {
    console.log('✅ Frontend dashboard accessible');
    return true;
  } else {
    console.log(`❌ Frontend dashboard failed: ${dashboardResult.error}`);
    return false;
  }
}

// Main test execution
async function runIntegrationTests() {
  console.log('Starting comprehensive integration testing...\n');
  
  const results = {
    serviceHealth: await testServiceHealth(),
    authentication: await testAuthentication(),
    marketData: await testMarketData(),
    trading: await testTrading(),
    apiGateway: await testAPIGatewayIntegration(),
    frontend: await testFrontendDashboard()
  };
  
  // Summary
  console.log('\n🎯 INTEGRATION TEST RESULTS');
  console.log('============================');
  
  const passed = Object.values(results).filter(Boolean).length;
  const total = Object.keys(results).length;
  
  Object.entries(results).forEach(([test, passed]) => {
    console.log(`${passed ? '✅' : '❌'} ${test}: ${passed ? 'PASSED' : 'FAILED'}`);
  });
  
  console.log(`\n📊 Overall Result: ${passed}/${total} tests passed`);
  
  if (passed >= 4) {
    console.log('🎉 Integration testing SUCCESSFUL - Platform is functional!');
    return true;
  } else {
    console.log('⚠️ Integration testing FAILED - Some services need attention');
    return false;
  }
}

// Run the tests
runIntegrationTests()
  .then(success => {
    process.exit(success ? 0 : 1);
  })
  .catch(error => {
    console.error('❌ Test execution failed:', error);
    process.exit(1);
  });
