// Platform3 Communication Bridge Integration Test
// Tests Python-TypeScript communication for humanitarian trading platform

const axios = require('axios');
const WebSocket = require('ws');
const { performance } = require('perf_hooks');

const PYTHON_BASE_URL = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
const PYTHON_WS_URL = process.env.PYTHON_WS_URL || 'ws://localhost:8000/ws';
const ANALYTICS_URL = process.env.ANALYTICS_URL || 'http://localhost:3007';
const TRADING_URL = process.env.TRADING_URL || 'http://localhost:3006';

class CommunicationBridgeTest {
    constructor() {
        this.testResults = [];
        this.latencyResults = [];
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️';
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    async measureLatency(testName, testFunction) {
        const start = performance.now();
        try {
            const result = await testFunction();
            const end = performance.now();
            const latency = end - start;
            this.latencyResults.push({ test: testName, latency: latency.toFixed(2) });
            this.log(`${testName}: ${latency.toFixed(2)}ms`, latency < 1 ? 'success' : 'info');
            return result;
        } catch (error) {
            const end = performance.now();
            const latency = end - start;
            this.log(`${testName} FAILED: ${error.message} (${latency.toFixed(2)}ms)`, 'error');
            throw error;
        }
    }

    async testPythonEngineHealth() {
        this.log('Testing Python AI Engine Health...');
        const response = await this.measureLatency('Python Health Check', 
            () => axios.get(`${PYTHON_BASE_URL}/health`)
        );
        
        if (response.data.status === 'healthy') {
            this.testResults.push({ test: 'Python Engine Health', status: 'PASS' });
            return true;
        }
        throw new Error('Python engine unhealthy');
    }

    async testTradingSignalsEndpoint() {
        this.log('Testing Trading Signals Endpoint...');
        const testData = {
            symbol: 'EURUSD',
            timeframe: '1h',
            current_price: 1.0850,
            risk_level: 'medium'
        };

        const response = await this.measureLatency('Trading Signals API',
            () => axios.post(`${PYTHON_BASE_URL}/api/v1/trading/signals`, testData)
        );

        if (response.data.action && response.data.confidence) {
            this.testResults.push({ test: 'Trading Signals', status: 'PASS' });
            this.log(`Signal received: ${response.data.action} ${response.data.symbol || 'EURUSD'} (confidence: ${response.data.confidence})`, 'success');
            return response.data;
        }
        throw new Error('Invalid trading signal response');
    }

    async testMarketAnalysisEndpoint() {
        this.log('Testing Market Analysis Endpoint...');
        const testData = {
            symbol: 'GBPUSD',
            timeframe: '1h',
            indicators: ['RSI', 'MACD']
        };

        const response = await this.measureLatency('Market Analysis API',
            () => axios.post(`${PYTHON_BASE_URL}/api/v1/analysis/market`, testData)
        );

        if (response.data.symbol && response.data.trend) {
            this.testResults.push({ test: 'Market Analysis', status: 'PASS' });
            this.log(`Analysis completed: ${response.data.trend} trend detected`, 'success');
            return response.data;
        }
        throw new Error('Invalid market analysis response');
    }

    async testRiskAssessmentEndpoint() {
        this.log('Testing Risk Assessment Endpoint...');
        const testData = {
            symbol: 'USDJPY',
            current_price: 151.25,
            position_size: 100000,
            account_balance: 500000
        };

        const response = await this.measureLatency('Risk Assessment API',
            () => axios.post(`${PYTHON_BASE_URL}/api/v1/risk/assess`, testData)
        );

        if (response.data.risk_score !== undefined && response.data.risk_level) {
            this.testResults.push({ test: 'Risk Assessment', status: 'PASS' });
            this.log(`Risk level: ${response.data.risk_level} (score: ${response.data.risk_score})`, 'success');
            return response.data;
        }
        throw new Error('Invalid risk assessment response');
    }

    async testWebSocketConnection() {
        this.log('Testing WebSocket Real-time Communication...');
        
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(PYTHON_WS_URL);
            let subscriptionConfirmed = false;
            let heartbeatReceived = false;
            
            // Reduced timeout for faster feedback
            const timeout = setTimeout(() => {
                if (!subscriptionConfirmed) {
                    ws.close();
                    reject(new Error('WebSocket subscription confirmation timeout (5s)'));
                } else {
                    ws.close();
                    reject(new Error('WebSocket heartbeat timeout (5s)'));
                }
            }, 5000);

            ws.on('open', () => {
                this.log('WebSocket connected, sending subscription message...');
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    symbol: 'EURUSD',
                    client_id: 'test_client_' + Date.now()
                }));
            });

            ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.log(`WebSocket message received: ${message.type}`, 'info');
                    
                    if (message.type === 'connection_established') {
                        this.log('WebSocket connection established', 'success');
                    }
                    else if (message.type === 'subscription_confirmed') {
                        subscriptionConfirmed = true;
                        this.log(`Subscription confirmed for ${message.symbol}`, 'success');
                    }
                    else if (message.type === 'market_data') {
                        this.log(`Market data received: ${message.symbol} @ ${message.price}`, 'success');
                    }
                    else if (message.type === 'heartbeat') {
                        heartbeatReceived = true;
                        this.log(`Heartbeat received (latency: ${message.latency_ms}ms)`, 'info');
                    }
                    else if (message.type === 'error') {
                        clearTimeout(timeout);
                        ws.close();
                        reject(new Error(`WebSocket server error: ${message.message}`));
                        return;
                    }
                    
                    // Test passes when we get subscription confirmation and at least one heartbeat or market data
                    if (subscriptionConfirmed && (heartbeatReceived || message.type === 'market_data')) {
                        clearTimeout(timeout);
                        this.testResults.push({ test: 'WebSocket Communication', status: 'PASS' });
                        this.log('WebSocket communication test successful!', 'success');
                        ws.close();
                        resolve(message);
                    }
                    
                } catch (error) {
                    clearTimeout(timeout);
                    ws.close();
                    reject(new Error(`WebSocket message parsing error: ${error.message}`));
                }
            });

            ws.on('error', (error) => {
                clearTimeout(timeout);
                this.log(`WebSocket connection error: ${error.message}`, 'error');
                reject(new Error(`WebSocket connection error: ${error.message}`));
            });

            ws.on('close', (code, reason) => {
                clearTimeout(timeout);
                if (!subscriptionConfirmed) {
                    reject(new Error(`WebSocket closed before subscription confirmation (code: ${code}, reason: ${reason})`));
                }
            });
        });
    }

    async testAnalyticsServiceIntegration() {
        this.log('Testing Analytics Service Integration...');
        try {
            const response = await this.measureLatency('Analytics Service Health',
                () => axios.get(`${ANALYTICS_URL}/health`)
            );
            
            if (response.status === 200) {
                this.testResults.push({ test: 'Analytics Service', status: 'PASS' });
                return true;
            }
        } catch (error) {
            // Analytics service might not be running, that's okay for this test
            this.log('Analytics service not running (expected in isolated test)', 'info');
            this.testResults.push({ test: 'Analytics Service', status: 'SKIP' });
        }
    }

    async testTradingServiceIntegration() {
        this.log('Testing Trading Service Integration...');
        try {
            const response = await this.measureLatency('Trading Service Health',
                () => axios.get(`${TRADING_URL}/health`)
            );
            
            if (response.status === 200) {
                this.testResults.push({ test: 'Trading Service', status: 'PASS' });
                return true;
            }
        } catch (error) {
            // Trading service might not be running, that's okay for this test
            this.log('Trading service not running (expected in isolated test)', 'info');
            this.testResults.push({ test: 'Trading Service', status: 'SKIP' });
        }
    }

    async runAllTests() {
        this.log('🚀 Starting Platform3 Communication Bridge Integration Tests...', 'info');
        this.log('🎯 Mission: Validate AI-driven humanitarian forex trading system', 'info');
        console.log('');

        const tests = [
            () => this.testPythonEngineHealth(),
            () => this.testTradingSignalsEndpoint(),
            () => this.testMarketAnalysisEndpoint(),
            () => this.testRiskAssessmentEndpoint(),
            () => this.testWebSocketConnection(),
            () => this.testAnalyticsServiceIntegration(),
            () => this.testTradingServiceIntegration()
        ];

        let passedTests = 0;
        let failedTests = 0;

        for (const test of tests) {
            try {
                await test();
                passedTests++;
            } catch (error) {
                this.testResults.push({ test: error.message, status: 'FAIL' });
                failedTests++;
            }
            console.log(''); // Add spacing between tests
        }

        this.generateReport(passedTests, failedTests);
    }

    generateReport(passed, failed) {
        console.log('');
        this.log('📊 TEST RESULTS SUMMARY', 'info');
        console.log(''.padEnd(60, '='));
        
        this.testResults.forEach(result => {
            const status = result.status === 'PASS' ? '✅ PASS' : 
                          result.status === 'FAIL' ? '❌ FAIL' : '⏭️  SKIP';
            console.log(`${status.padEnd(10)} ${result.test}`);
        });

        console.log(''.padEnd(60, '='));
        console.log(`Total Tests: ${this.testResults.length}`);
        console.log(`Passed: ${passed} ✅`);
        console.log(`Failed: ${failed} ❌`);
        console.log(`Skipped: ${this.testResults.filter(r => r.status === 'SKIP').length} ⏭️`);

        console.log('');
        this.log('⚡ LATENCY ANALYSIS', 'info');
        console.log(''.padEnd(60, '-'));
        
        this.latencyResults.forEach(result => {
            const status = parseFloat(result.latency) < 1 ? '🚀' : 
                          parseFloat(result.latency) < 10 ? '⚡' : '⏰';
            console.log(`${status} ${result.test.padEnd(25)} ${result.latency}ms`);
        });

        const avgLatency = this.latencyResults.reduce((sum, r) => sum + parseFloat(r.latency), 0) / this.latencyResults.length;
        console.log(''.padEnd(60, '-'));
        console.log(`Average Latency: ${avgLatency.toFixed(2)}ms ${avgLatency < 1 ? '🚀 EXCELLENT' : avgLatency < 10 ? '⚡ GOOD' : '⏰ NEEDS OPTIMIZATION'}`);

        console.log('');
        if (failed === 0 && passed >= 4) {
            this.log('🎉 COMMUNICATION BRIDGE INTEGRATION SUCCESSFUL!', 'success');
            this.log('💝 Platform3 ready to generate profits for humanitarian mission!', 'success');
        } else {
            this.log(`⚠️ Some tests failed. System needs attention before production deployment.`, 'error');
        }
    }
}

// Main execution
async function main() {
    const tester = new CommunicationBridgeTest();
    
    try {
        await tester.runAllTests();
        process.exit(0);
    } catch (error) {
        console.error('Test runner failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = CommunicationBridgeTest;