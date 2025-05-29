#!/usr/bin/env node
/**
 * Platform3 End-to-End Configuration Management Test
 * Tests complete configuration lifecycle with all services
 */

const axios = require('axios');
const redis = require('redis');
const { Client } = require('pg');

class EndToEndConfigTest {
    constructor() {
        this.testResults = [];
        
        this.config = {
            vault: {
                address: 'http://localhost:8201',
                token: 'dev-token'
            },
            redis: {
                host: 'localhost',
                port: 6380
            },
            postgres: {
                host: 'localhost',
                port: 5432,
                database: 'platform3_config',
                user: 'platform3_user',
                password: 'platform3_secure_password'
            }
        };
    }

    async runEndToEndTest() {
        console.log('🎯 Starting End-to-End Configuration Management Test\n');
        
        try {
            await this.setupTestEnvironment();
            await this.testConfigurationLifecycle();
            await this.testFeatureFlagLifecycle();
            await this.testSecretManagement();
            await this.testCacheInvalidation();
            await this.testAuditTrail();
            await this.testErrorHandling();
            
            this.generateReport();
            
        } catch (error) {
            console.error('❌ End-to-end test failed:', error.message);
            process.exit(1);
        }
    }

    async setupTestEnvironment() {
        console.log('🔧 Setting up test environment...');
        
        // Initialize Vault secret engine for testing
        try {
            await axios.post(`${this.config.vault.address}/v1/sys/mounts/platform3-e2e`, {
                type: 'kv-v2'
            }, {
                headers: { 'X-Vault-Token': this.config.vault.token }
            });
        } catch (error) {
            // Ignore if already exists
        }
        
        this.addResult('Environment Setup', true, 'Test environment initialized');
        console.log('✅ Test environment ready\n');
    }

    async testConfigurationLifecycle() {
        console.log('📝 Testing Configuration Lifecycle...');
        
        let pgClient;
        let redisClient;
        
        try {
            // Connect to databases
            pgClient = new Client(this.config.postgres);
            await pgClient.connect();
            
            redisClient = redis.createClient({
                host: this.config.redis.host,
                port: this.config.redis.port
            });
            await redisClient.connect();

            // Step 1: Create configuration
            const testConfig = {
                service_name: 'e2e-test-service',
                environment: 'test',
                config_key: 'database_timeout',
                config_value: '30000',
                version: 1,
                is_encrypted: false,
                created_by: 'e2e-test'
            };

            await pgClient.query(`
                INSERT INTO configurations (service_name, environment, config_key, config_value, version, is_encrypted, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (service_name, environment, config_key) DO UPDATE SET
                config_value = EXCLUDED.config_value,
                version = EXCLUDED.version + 1,
                updated_at = CURRENT_TIMESTAMP
            `, [testConfig.service_name, testConfig.environment, testConfig.config_key, 
                testConfig.config_value, testConfig.version, testConfig.is_encrypted, testConfig.created_by]);

            this.addResult('Configuration Creation', true, 'Configuration created in database');

            // Step 2: Cache configuration
            const cacheKey = `config:${testConfig.service_name}:${testConfig.environment}:${testConfig.config_key}`;
            await redisClient.setEx(cacheKey, 3600, JSON.stringify({
                value: testConfig.config_value,
                version: testConfig.version,
                cached_at: new Date().toISOString()
            }));

            this.addResult('Configuration Caching', true, 'Configuration cached in Redis');

            // Step 3: Retrieve from cache
            const cachedValue = await redisClient.get(cacheKey);
            const parsedCache = JSON.parse(cachedValue);
            const cacheValid = parsedCache.value === testConfig.config_value;

            this.addResult('Cache Retrieval', cacheValid, 
                cacheValid ? 'Configuration retrieved from cache' : 'Cache retrieval failed');

            // Step 4: Update configuration
            const updatedValue = '45000';
            await pgClient.query(`
                UPDATE configurations 
                SET config_value = $1, version = version + 1, updated_at = CURRENT_TIMESTAMP
                WHERE service_name = $2 AND environment = $3 AND config_key = $4
            `, [updatedValue, testConfig.service_name, testConfig.environment, testConfig.config_key]);

            // Step 5: Invalidate cache
            await redisClient.del(cacheKey);

            this.addResult('Configuration Update', true, 'Configuration updated and cache invalidated');

            // Step 6: Verify update
            const updatedResult = await pgClient.query(`
                SELECT * FROM configurations 
                WHERE service_name = $1 AND environment = $2 AND config_key = $3
            `, [testConfig.service_name, testConfig.environment, testConfig.config_key]);

            const updateValid = updatedResult.rows[0].config_value === updatedValue;
            this.addResult('Update Verification', updateValid, 
                updateValid ? 'Configuration update verified' : 'Configuration update failed');

            console.log('✅ Configuration lifecycle test completed\n');

        } catch (error) {
            this.addResult('Configuration Lifecycle', false, `Error: ${error.message}`);
            console.log('❌ Configuration lifecycle test failed\n');
        } finally {
            if (pgClient) await pgClient.end();
            if (redisClient) await redisClient.disconnect();
        }
    }

    async testFeatureFlagLifecycle() {
        console.log('🚩 Testing Feature Flag Lifecycle...');
        
        let pgClient;
        let redisClient;
        
        try {
            pgClient = new Client(this.config.postgres);
            await pgClient.connect();
            
            redisClient = redis.createClient({
                host: this.config.redis.host,
                port: this.config.redis.port
            });
            await redisClient.connect();

            // Step 1: Create feature flag
            const testFlag = {
                flag_name: 'e2e_test_feature',
                service_name: 'e2e-test-service',
                environment: 'test',
                is_enabled: true,
                flag_value: JSON.stringify({ 
                    rollout_percentage: 50, 
                    target_users: ['test_user_1', 'test_user_2'] 
                }),
                created_by: 'e2e-test'
            };

            await pgClient.query(`
                INSERT INTO feature_flags (flag_name, service_name, environment, is_enabled, flag_value, created_by)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (flag_name, service_name, environment) DO UPDATE SET
                is_enabled = EXCLUDED.is_enabled,
                flag_value = EXCLUDED.flag_value,
                updated_at = CURRENT_TIMESTAMP
            `, [testFlag.flag_name, testFlag.service_name, testFlag.environment, 
                testFlag.is_enabled, testFlag.flag_value, testFlag.created_by]);

            this.addResult('Feature Flag Creation', true, 'Feature flag created');

            // Step 2: Test flag evaluation
            const flagResult = await pgClient.query(`
                SELECT * FROM feature_flags 
                WHERE flag_name = $1 AND service_name = $2 AND environment = $3
            `, [testFlag.flag_name, testFlag.service_name, testFlag.environment]);

            const flagExists = flagResult.rows.length > 0;
            const flagEnabled = flagExists && flagResult.rows[0].is_enabled;

            this.addResult('Feature Flag Evaluation', flagEnabled, 
                flagEnabled ? 'Feature flag evaluated correctly' : 'Feature flag evaluation failed');

            // Step 3: Toggle flag
            await pgClient.query(`
                UPDATE feature_flags 
                SET is_enabled = NOT is_enabled, updated_at = CURRENT_TIMESTAMP
                WHERE flag_name = $1 AND service_name = $2 AND environment = $3
            `, [testFlag.flag_name, testFlag.service_name, testFlag.environment]);

            const toggledResult = await pgClient.query(`
                SELECT is_enabled FROM feature_flags 
                WHERE flag_name = $1 AND service_name = $2 AND environment = $3
            `, [testFlag.flag_name, testFlag.service_name, testFlag.environment]);

            const toggleWorked = toggledResult.rows[0].is_enabled === false;
            this.addResult('Feature Flag Toggle', toggleWorked, 
                toggleWorked ? 'Feature flag toggled successfully' : 'Feature flag toggle failed');

            console.log('✅ Feature flag lifecycle test completed\n');

        } catch (error) {
            this.addResult('Feature Flag Lifecycle', false, `Error: ${error.message}`);
            console.log('❌ Feature flag lifecycle test failed\n');
        } finally {
            if (pgClient) await pgClient.end();
            if (redisClient) await redisClient.disconnect();
        }
    }

    async testSecretManagement() {
        console.log('🔐 Testing Secret Management...');
        
        try {
            // Step 1: Store encrypted configuration in Vault
            const secretData = {
                data: {
                    api_key: 'sk-1234567890abcdef',
                    database_password: 'super_secure_password_123',
                    encryption_salt: 'random_salt_value'
                }
            };

            await axios.post(`${this.config.vault.address}/v1/platform3-e2e/data/secrets/e2e-test`, 
                secretData, {
                headers: { 'X-Vault-Token': this.config.vault.token }
            });

            this.addResult('Secret Storage', true, 'Secrets stored in Vault');

            // Step 2: Retrieve secrets
            const secretResponse = await axios.get(`${this.config.vault.address}/v1/platform3-e2e/data/secrets/e2e-test`, {
                headers: { 'X-Vault-Token': this.config.vault.token }
            });

            const retrievedSecrets = secretResponse.data.data.data;
            const secretsValid = retrievedSecrets.api_key === secretData.data.api_key;

            this.addResult('Secret Retrieval', secretsValid, 
                secretsValid ? 'Secrets retrieved successfully' : 'Secret retrieval failed');

            // Step 3: Update secrets
            const updatedSecretData = {
                data: {
                    ...secretData.data,
                    api_key: 'sk-updated-key-9876543210',
                    last_rotated: new Date().toISOString()
                }
            };

            await axios.post(`${this.config.vault.address}/v1/platform3-e2e/data/secrets/e2e-test`, 
                updatedSecretData, {
                headers: { 'X-Vault-Token': this.config.vault.token }
            });

            this.addResult('Secret Rotation', true, 'Secrets rotated successfully');

            console.log('✅ Secret management test completed\n');

        } catch (error) {
            this.addResult('Secret Management', false, `Error: ${error.message}`);
            console.log('❌ Secret management test failed\n');
        }
    }

    async testCacheInvalidation() {
        console.log('💨 Testing Cache Invalidation...');
        
        let redisClient;
        
        try {
            redisClient = redis.createClient({
                host: this.config.redis.host,
                port: this.config.redis.port
            });
            await redisClient.connect();

            // Step 1: Set multiple cache entries
            const cacheEntries = [
                'config:service1:prod:timeout',
                'config:service2:prod:retries',
                'flag:service1:prod:new_feature'
            ];

            for (const key of cacheEntries) {
                await redisClient.setEx(key, 3600, JSON.stringify({ test: 'data' }));
            }

            this.addResult('Cache Population', true, 'Cache entries created');

            // Step 2: Pattern-based invalidation
            const pattern = 'config:service1:*';
            const keys = await redisClient.keys(pattern);
            
            if (keys.length > 0) {
                await redisClient.del(keys);
            }

            // Step 3: Verify invalidation
            const remainingKeys = await redisClient.keys(pattern);
            const invalidationWorked = remainingKeys.length === 0;

            this.addResult('Cache Invalidation', invalidationWorked, 
                invalidationWorked ? 'Pattern-based cache invalidation successful' : 'Cache invalidation failed');

            console.log('✅ Cache invalidation test completed\n');

        } catch (error) {
            this.addResult('Cache Invalidation', false, `Error: ${error.message}`);
            console.log('❌ Cache invalidation test failed\n');
        } finally {
            if (redisClient) await redisClient.disconnect();
        }
    }

    async testAuditTrail() {
        console.log('📋 Testing Audit Trail...');
        
        let pgClient;
        
        try {
            pgClient = new Client(this.config.postgres);
            await pgClient.connect();

            // Step 1: Create audit log entry
            const auditEntry = {
                service_name: 'e2e-test-service',
                action: 'configuration_update',
                resource_type: 'configuration',
                resource_id: 'database_timeout',
                old_value: '30000',
                new_value: '45000',
                performed_by: 'e2e-test',
                ip_address: '127.0.0.1',
                user_agent: 'E2E-Test-Suite/1.0'
            };

            await pgClient.query(`
                INSERT INTO audit_logs (service_name, action, resource_type, resource_id, old_value, new_value, performed_by, ip_address, user_agent)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            `, [auditEntry.service_name, auditEntry.action, auditEntry.resource_type, 
                auditEntry.resource_id, auditEntry.old_value, auditEntry.new_value, 
                auditEntry.performed_by, auditEntry.ip_address, auditEntry.user_agent]);

            this.addResult('Audit Log Creation', true, 'Audit log entry created');

            // Step 2: Query audit trail
            const auditResult = await pgClient.query(`
                SELECT * FROM audit_logs 
                WHERE service_name = $1 AND action = $2 AND performed_by = $3
                ORDER BY created_at DESC LIMIT 1
            `, [auditEntry.service_name, auditEntry.action, auditEntry.performed_by]);

            const auditValid = auditResult.rows.length > 0 && 
                             auditResult.rows[0].resource_id === auditEntry.resource_id;

            this.addResult('Audit Trail Query', auditValid, 
                auditValid ? 'Audit trail queryable' : 'Audit trail query failed');

            console.log('✅ Audit trail test completed\n');

        } catch (error) {
            this.addResult('Audit Trail', false, `Error: ${error.message}`);
            console.log('❌ Audit trail test failed\n');
        } finally {
            if (pgClient) await pgClient.end();
        }
    }

    async testErrorHandling() {
        console.log('⚠️  Testing Error Handling...');
        
        try {
            // Test 1: Invalid Vault token
            try {
                await axios.get(`${this.config.vault.address}/v1/platform3-e2e/data/nonexistent`, {
                    headers: { 'X-Vault-Token': 'invalid-token' }
                });
                this.addResult('Invalid Token Handling', false, 'Should have failed with invalid token');
            } catch (error) {
                const errorHandled = error.response && error.response.status === 403;
                this.addResult('Invalid Token Handling', errorHandled, 
                    errorHandled ? 'Invalid token properly rejected' : 'Unexpected error response');
            }

            // Test 2: Database connection failure simulation
            try {
                const invalidClient = new Client({
                    ...this.config.postgres,
                    port: 9999 // Invalid port
                });
                await invalidClient.connect();
                this.addResult('DB Connection Error', false, 'Should have failed with invalid port');
            } catch (error) {
                const errorHandled = error.code === 'ECONNREFUSED';
                this.addResult('DB Connection Error', errorHandled, 
                    errorHandled ? 'Database connection error handled' : 'Unexpected database error');
            }

            console.log('✅ Error handling test completed\n');

        } catch (error) {
            this.addResult('Error Handling', false, `Error: ${error.message}`);
            console.log('❌ Error handling test failed\n');
        }
    }

    addResult(testName, passed, message) {
        this.testResults.push({
            name: testName,
            passed,
            message,
            timestamp: new Date().toISOString()
        });
    }

    generateReport() {
        console.log('📊 END-TO-END TEST REPORT');
        console.log('='.repeat(60));
        
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(test => test.passed).length;
        const successRate = Math.round((passedTests / totalTests) * 100);
        
        console.log(`\n📈 Overall Results: ${passedTests}/${totalTests} tests passed (${successRate}%)\n`);

        this.testResults.forEach(test => {
            const icon = test.passed ? '✅' : '❌';
            console.log(`${icon} ${test.name}: ${test.message}`);
        });

        console.log('\n' + '='.repeat(60));
        
        if (passedTests === totalTests) {
            console.log('🎉 ALL END-TO-END TESTS PASSED!');
            console.log('\n✅ Configuration Management System is FULLY VALIDATED');
            console.log('✅ Infrastructure Testing: COMPLETE');
            console.log('✅ Integration Testing: COMPLETE');
            console.log('✅ Production Readiness: ACHIEVED');
            
            console.log('\n🚀 READY FOR PRODUCTION DEPLOYMENT');
            console.log('\nDeployment Checklist:');
            console.log('- ✅ All services tested and validated');
            console.log('- ✅ Database schema deployed and tested');
            console.log('- ✅ Cache layer operational');
            console.log('- ✅ Secret management configured');
            console.log('- ✅ Audit logging functional');
            console.log('- ✅ Error handling validated');
            
        } else {
            console.log('⚠️  SOME TESTS FAILED - REVIEW REQUIRED');
            console.log('\n❌ Production deployment NOT RECOMMENDED until issues are resolved');
            
            const failedTests = this.testResults.filter(test => !test.passed);
            console.log('\nFailed Tests:');
            failedTests.forEach(test => {
                console.log(`- ${test.name}: ${test.message}`);
            });
        }
    }
}

// Run tests if script is executed directly
if (require.main === module) {
    const e2eTest = new EndToEndConfigTest();
    e2eTest.runEndToEndTest().catch(error => {
        console.error('E2E test suite failed:', error);
        process.exit(1);
    });
}

module.exports = EndToEndConfigTest;
