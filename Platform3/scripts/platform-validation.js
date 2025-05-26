// Complete Platform Validation Script
// Tests the entire user journey: Authentication → AI Analysis → Trading → Real-time Updates

const axios = require('axios');
const WebSocket = require('ws');
const { performance } = require('perf_hooks');

console.log('🚀 Starting Complete Platform Validation');
console.log('=====================================');

// Service URLs
const services = {
  apiGateway: 'http://localhost:3001',
  userService: 'http://localhost:3002',
  tradingService: 'http://localhost:3003',
  marketDataService: 'http://localhost:3004',
  eventSystem: 'http://localhost:3005',
  websocketService: 'http://localhost:3006',
  analyticsService: 'http://localhost:3007',
  dashboard: 'http://localhost:3000'
};

let authToken = null;
let testResults = {
  serviceHealth: {},
  authentication: false,
  marketData: false,
  analytics: false,
  trading: false,
  websocket: false,
  endToEnd: false
};

// Helper function for HTTP requests
async function makeRequest(url, method = 'GET', data = null, useAuth = false) {
  try {
    const config = {
      method,
      url,
      timeout: 10000,
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

// Test 1: Service Health Validation
async function validateServiceHealth() {
  console.log('\n🔍 Phase 1: Service Health Validation');
  console.log('=====================================');
  
  let healthyServices = 0;
  
  for (const [serviceName, baseUrl] of Object.entries(services)) {
    console.log(`Testing ${serviceName}...`);
    
    const result = await makeRequest(`${baseUrl}/health`);
    testResults.serviceHealth[serviceName] = result.success;
    
    if (result.success) {
      console.log(`✅ ${serviceName}: ${result.data?.status || 'OK'}`);
      healthyServices++;
    } else {
      console.log(`❌ ${serviceName}: ${result.error}`);
    }
  }
  
  console.log(`\n📊 Service Health: ${healthyServices}/${Object.keys(services).length} services healthy`);
  return healthyServices >= 6; // Need at least 6 core services
}

// Test 2: Authentication Flow Validation
async function validateAuthentication() {
  console.log('\n🔐 Phase 2: Authentication Flow Validation');
  console.log('==========================================');
  
  // Test owner registration (should work or already exist)
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
    
    // Test profile access
    const profileResult = await makeRequest(`${services.userService}/api/v1/users/profile`, 'GET', null, true);
    if (profileResult.success) {
      console.log('✅ Profile access successful');
      console.log(`   User: ${profileResult.data.user?.email}`);
      testResults.authentication = true;
    } else {
      console.log(`❌ Profile access failed: ${profileResult.error}`);
    }
  } else {
    console.log(`❌ Login failed: ${loginResult.error}`);
  }
  
  return testResults.authentication;
}

// Test 3: Market Data Service Validation
async function validateMarketData() {
  console.log('\n📊 Phase 3: Market Data Service Validation');
  console.log('==========================================');
  
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

// Test 4: Analytics Service Validation
async function validateAnalytics() {
  console.log('\n🧠 Phase 4: Analytics Service Validation');
  console.log('========================================');
  
  if (!authToken) {
    console.log('❌ Cannot test analytics without authentication');
    return false;
  }
  
  // Test technical analysis
  const technicalResult = await makeRequest(`${services.analyticsService}/api/v1/analysis/technical/EURUSD`, 'GET', null, true);
  if (technicalResult.success) {
    console.log('✅ Technical analysis endpoint working');
    console.log(`   RSI: ${technicalResult.data?.indicators?.rsi?.current || 'N/A'}`);
    console.log(`   Trend: ${technicalResult.data?.trend?.direction || 'N/A'}`);
  } else {
    console.log(`❌ Technical analysis failed: ${technicalResult.error}`);
  }
  
  // Test ML predictions
  const predictionResult = await makeRequest(`${services.analyticsService}/api/v1/predictions/EURUSD`, 'GET', null, true);
  if (predictionResult.success) {
    console.log('✅ ML predictions endpoint working');
    console.log(`   Predictions: ${predictionResult.data?.predictions?.length || 0}`);
    console.log(`   Confidence: ${(predictionResult.data?.confidence * 100).toFixed(1)}%`);
  } else {
    console.log(`❌ ML predictions failed: ${predictionResult.error}`);
  }
  
  // Test pattern recognition
  const patternResult = await makeRequest(`${services.analyticsService}/api/v1/patterns/EURUSD`, 'GET', null, true);
  if (patternResult.success) {
    console.log('✅ Pattern recognition endpoint working');
    console.log(`   Patterns detected: ${patternResult.data?.patterns?.length || 0}`);
    testResults.analytics = true;
  } else {
    console.log(`❌ Pattern recognition failed: ${patternResult.error}`);
  }
  
  // Test risk analysis
  const riskResult = await makeRequest(`${services.analyticsService}/api/v1/risk/portfolio`, 'POST', {
    positions: [
      {
        symbol: 'EURUSD',
        side: 'long',
        quantity: 10000,
        entryPrice: 1.0850,
        currentPrice: 1.0870,
        marketValue: 10870,
        unrealizedPnL: 200
      }
    ],
    accountBalance: 100000
  }, true);
  
  if (riskResult.success) {
    console.log('✅ Risk analysis endpoint working');
    console.log(`   Risk Score: ${riskResult.data?.portfolioRisk?.riskScore || 'N/A'}`);
  } else {
    console.log(`❌ Risk analysis failed: ${riskResult.error}`);
  }
  
  return testResults.analytics;
}

// Test 5: Trading Service Validation
async function validateTrading() {
  console.log('\n💰 Phase 5: Trading Service Validation');
  console.log('======================================');
  
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
    testResults.trading = true;
  } else {
    console.log(`❌ Trade creation failed: ${createTradeResult.error}`);
  }
  
  return testResults.trading;
}

// Test 6: WebSocket Service Validation
async function validateWebSocket() {
  console.log('\n🔌 Phase 6: WebSocket Service Validation');
  console.log('========================================');
  
  return new Promise((resolve) => {
    if (!authToken) {
      console.log('❌ Cannot test WebSocket without authentication');
      resolve(false);
      return;
    }
    
    const ws = new WebSocket(`ws://localhost:3006`, {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    
    let connected = false;
    let receivedMessages = 0;
    
    const timeout = setTimeout(() => {
      if (!connected) {
        console.log('❌ WebSocket connection timeout');
        ws.close();
        resolve(false);
      }
    }, 10000);
    
    ws.on('open', () => {
      console.log('✅ WebSocket connection established');
      connected = true;
      clearTimeout(timeout);
      
      // Test price subscription
      ws.send(JSON.stringify({
        type: 'subscribe:prices',
        data: { symbols: ['EURUSD', 'GBPUSD'] }
      }));
      
      // Test chat message
      ws.send(JSON.stringify({
        type: 'chat:message',
        data: { message: 'What is the current trend for EURUSD?' }
      }));
    });
    
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        receivedMessages++;
        console.log(`✅ Received WebSocket message: ${message.type || 'unknown'}`);
        
        if (receivedMessages >= 2) {
          console.log('✅ WebSocket service validation successful');
          testResults.websocket = true;
          ws.close();
          resolve(true);
        }
      } catch (error) {
        console.log(`⚠️ WebSocket message parse error: ${error.message}`);
      }
    });
    
    ws.on('error', (error) => {
      console.log(`❌ WebSocket error: ${error.message}`);
      resolve(false);
    });
    
    ws.on('close', () => {
      if (receivedMessages < 2) {
        console.log('❌ WebSocket closed before receiving expected messages');
        resolve(false);
      }
    });
    
    // Auto-resolve after 15 seconds
    setTimeout(() => {
      if (connected && receivedMessages > 0) {
        console.log('✅ WebSocket service partially validated (timeout)');
        testResults.websocket = true;
        ws.close();
        resolve(true);
      } else {
        resolve(false);
      }
    }, 15000);
  });
}

// Test 7: End-to-End User Journey
async function validateEndToEndJourney() {
  console.log('\n🎯 Phase 7: End-to-End User Journey Validation');
  console.log('===============================================');
  
  if (!authToken) {
    console.log('❌ Cannot perform end-to-end test without authentication');
    return false;
  }
  
  console.log('Testing complete user journey...');
  
  // Step 1: Get market data
  console.log('1. Fetching market data...');
  const marketData = await makeRequest(`${services.marketDataService}/api/market-data/prices`);
  if (!marketData.success) {
    console.log('❌ Failed to get market data');
    return false;
  }
  console.log('✅ Market data retrieved');
  
  // Step 2: Get AI analysis
  console.log('2. Getting AI analysis...');
  const analysis = await makeRequest(`${services.analyticsService}/api/v1/analysis/technical/EURUSD`, 'GET', null, true);
  if (!analysis.success) {
    console.log('❌ Failed to get AI analysis');
    return false;
  }
  console.log('✅ AI analysis completed');
  
  // Step 3: Execute trade based on analysis
  console.log('3. Executing trade...');
  const trade = await makeRequest(`${services.tradingService}/api/v1/trades`, 'POST', {
    symbol: 'EURUSD',
    type: 'market',
    side: 'buy',
    quantity: 0.01,
    stopLoss: 1.0800,
    takeProfit: 1.1000
  }, true);
  
  if (!trade.success) {
    console.log('❌ Failed to execute trade');
    return false;
  }
  console.log('✅ Trade executed successfully');
  
  // Step 4: Check portfolio
  console.log('4. Checking portfolio...');
  const portfolio = await makeRequest(`${services.tradingService}/api/v1/portfolio/summary`, 'GET', null, true);
  if (!portfolio.success) {
    console.log('❌ Failed to get portfolio');
    return false;
  }
  console.log('✅ Portfolio retrieved');
  
  // Step 5: Get risk analysis
  console.log('5. Performing risk analysis...');
  const risk = await makeRequest(`${services.analyticsService}/api/v1/risk/portfolio`, 'POST', {
    positions: [
      {
        symbol: 'EURUSD',
        side: 'long',
        quantity: 1000,
        entryPrice: 1.0850,
        currentPrice: 1.0870,
        marketValue: 1087,
        unrealizedPnL: 20
      }
    ],
    accountBalance: 100000
  }, true);
  
  if (!risk.success) {
    console.log('❌ Failed to get risk analysis');
    return false;
  }
  console.log('✅ Risk analysis completed');
  
  console.log('\n🎉 End-to-end user journey completed successfully!');
  testResults.endToEnd = true;
  return true;
}

// Main validation function
async function runCompleteValidation() {
  const startTime = performance.now();
  
  console.log('Starting comprehensive platform validation...\n');
  
  const results = {
    serviceHealth: await validateServiceHealth(),
    authentication: await validateAuthentication(),
    marketData: await validateMarketData(),
    analytics: await validateAnalytics(),
    trading: await validateTrading(),
    websocket: await validateWebSocket(),
    endToEnd: await validateEndToEndJourney()
  };
  
  const endTime = performance.now();
  const duration = ((endTime - startTime) / 1000).toFixed(2);
  
  // Final Summary
  console.log('\n🎯 COMPLETE PLATFORM VALIDATION RESULTS');
  console.log('=======================================');
  
  const passed = Object.values(results).filter(Boolean).length;
  const total = Object.keys(results).length;
  
  Object.entries(results).forEach(([test, passed]) => {
    console.log(`${passed ? '✅' : '❌'} ${test.replace(/([A-Z])/g, ' $1').toLowerCase()}: ${passed ? 'PASSED' : 'FAILED'}`);
  });
  
  console.log(`\n📊 Overall Result: ${passed}/${total} tests passed`);
  console.log(`⏱️ Total validation time: ${duration} seconds`);
  
  if (passed === total) {
    console.log('\n🎉 PLATFORM VALIDATION SUCCESSFUL!');
    console.log('🚀 Forex Trading Platform is fully operational and ready for production!');
    console.log('\n🌟 All systems validated:');
    console.log('   ✅ User authentication and authorization');
    console.log('   ✅ Real-time market data streaming');
    console.log('   ✅ AI-powered analytics and recommendations');
    console.log('   ✅ Trade execution and portfolio management');
    console.log('   ✅ WebSocket real-time communications');
    console.log('   ✅ End-to-end user journey');
    return true;
  } else {
    console.log('\n⚠️ PLATFORM VALIDATION INCOMPLETE');
    console.log(`${total - passed} systems need attention before production deployment`);
    return false;
  }
}

// Run the validation
runCompleteValidation()
  .then(success => {
    process.exit(success ? 0 : 1);
  })
  .catch(error => {
    console.error('❌ Validation execution failed:', error);
    process.exit(1);
  });
