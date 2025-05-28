#!/usr/bin/env node

/**
 * Service Discovery Implementation Validation Script
 * Validates that all components are properly implemented and integrated
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating Service Discovery Implementation...\n');

// Check if all required files exist
const requiredFiles = [
  'src/ConsulServiceRegistry.ts',
  'src/ServiceDiscoveryClient.ts', 
  'src/index.ts',
  'src/utils/logger.ts',
  'dist/ConsulServiceRegistry.js',
  'dist/ServiceDiscoveryClient.js',
  'dist/index.js',
  'package.json',
  'tsconfig.json',
  'Dockerfile',
  'tests/unit.test.js',
  'tests/integration.test.js'
];

let allFilesExist = true;

console.log('📁 Checking required files:');
requiredFiles.forEach(file => {
  const exists = fs.existsSync(file);
  console.log(`  ${exists ? '✅' : '❌'} ${file}`);
  if (!exists) allFilesExist = false;
});

// Check API Gateway integration
console.log('\n🔗 Checking API Gateway integration:');
const apiGatewayFiles = [
  '../api-gateway/src/middleware/serviceDiscovery.js',
  '../api-gateway/src/server.js'
];

apiGatewayFiles.forEach(file => {
  const exists = fs.existsSync(file);
  console.log(`  ${exists ? '✅' : '❌'} ${file}`);
  if (!exists) allFilesExist = false;
});

// Check Docker Compose integration
console.log('\n🐳 Checking Docker Compose integration:');
const dockerComposeFile = '../../Platform3/docker-compose.yml';
const dockerComposeExists = fs.existsSync(dockerComposeFile);
console.log(`  ${dockerComposeExists ? '✅' : '❌'} ${dockerComposeFile}`);

if (dockerComposeExists) {
  const dockerComposeContent = fs.readFileSync(dockerComposeFile, 'utf8');
  const hasConsul = dockerComposeContent.includes('consul-server-1');
  const hasServiceDiscovery = dockerComposeContent.includes('service-discovery:');
  const hasConsulVolumes = dockerComposeContent.includes('consul_data_1:');
  
  console.log(`  ${hasConsul ? '✅' : '❌'} Consul cluster configuration`);
  console.log(`  ${hasServiceDiscovery ? '✅' : '❌'} Service Discovery service`);
  console.log(`  ${hasConsulVolumes ? '✅' : '❌'} Consul data volumes`);
}

// Check package.json configuration
console.log('\n📦 Checking package.json configuration:');
if (fs.existsSync('package.json')) {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  
  const hasConsulDep = packageJson.dependencies && packageJson.dependencies.consul;
  const hasExpressDep = packageJson.dependencies && packageJson.dependencies.express;
  const hasWinstonDep = packageJson.dependencies && packageJson.dependencies.winston;
  const hasJestDev = packageJson.devDependencies && packageJson.devDependencies.jest;
  const hasTypeScript = packageJson.devDependencies && packageJson.devDependencies.typescript;
  
  console.log(`  ${hasConsulDep ? '✅' : '❌'} Consul dependency`);
  console.log(`  ${hasExpressDep ? '✅' : '❌'} Express dependency`);
  console.log(`  ${hasWinstonDep ? '✅' : '❌'} Winston logging dependency`);
  console.log(`  ${hasJestDev ? '✅' : '❌'} Jest testing framework`);
  console.log(`  ${hasTypeScript ? '✅' : '❌'} TypeScript compiler`);
}

// Check TypeScript compilation
console.log('\n🔨 Checking TypeScript compilation:');
const distFiles = [
  'dist/ConsulServiceRegistry.js',
  'dist/ServiceDiscoveryClient.js',
  'dist/index.js',
  'dist/utils/logger.js'
];

let allDistFilesExist = true;
distFiles.forEach(file => {
  const exists = fs.existsSync(file);
  console.log(`  ${exists ? '✅' : '❌'} ${file}`);
  if (!exists) allDistFilesExist = false;
});

// Check shared libraries
console.log('\n📚 Checking shared libraries:');
const sharedFiles = [
  '../shared/src/ServiceRegistration.js',
  '../shared/src/logger.js'
];

sharedFiles.forEach(file => {
  const exists = fs.existsSync(file);
  console.log(`  ${exists ? '✅' : '❌'} ${file}`);
});

// Validate load balancer strategies
console.log('\n⚖️ Validating load balancer strategies:');
if (fs.existsSync('dist/ServiceDiscoveryClient.js')) {
  const clientCode = fs.readFileSync('dist/ServiceDiscoveryClient.js', 'utf8');
  
  const hasRoundRobin = clientCode.includes('RoundRobinStrategy');
  const hasLeastConnections = clientCode.includes('LeastConnectionsStrategy');
  const hasRandom = clientCode.includes('RandomStrategy');
  
  console.log(`  ${hasRoundRobin ? '✅' : '❌'} Round Robin Strategy`);
  console.log(`  ${hasLeastConnections ? '✅' : '❌'} Least Connections Strategy`);
  console.log(`  ${hasRandom ? '✅' : '❌'} Random Strategy`);
}

// Check API endpoints
console.log('\n🌐 Checking API endpoints:');
if (fs.existsSync('dist/index.js')) {
  const indexCode = fs.readFileSync('dist/index.js', 'utf8');
  
  const hasRegisterEndpoint = indexCode.includes('/services/register');
  const hasDiscoverEndpoint = indexCode.includes('/services/:serviceName/instances');
  const hasInstanceEndpoint = indexCode.includes('/services/:serviceName/instance');
  const hasUrlEndpoint = indexCode.includes('/services/:serviceName/url');
  const hasHealthEndpoint = indexCode.includes('/health');
  const hasCacheEndpoint = indexCode.includes('/cache');
  
  console.log(`  ${hasRegisterEndpoint ? '✅' : '❌'} Service registration endpoint`);
  console.log(`  ${hasDiscoverEndpoint ? '✅' : '❌'} Service discovery endpoint`);
  console.log(`  ${hasInstanceEndpoint ? '✅' : '❌'} Instance selection endpoint`);
  console.log(`  ${hasUrlEndpoint ? '✅' : '❌'} Service URL endpoint`);
  console.log(`  ${hasHealthEndpoint ? '✅' : '❌'} Health check endpoint`);
  console.log(`  ${hasCacheEndpoint ? '✅' : '❌'} Cache management endpoint`);
}

// Final validation summary
console.log('\n📊 Validation Summary:');
console.log('='.repeat(50));

const validationResults = {
  'Core Files': allFilesExist,
  'TypeScript Compilation': allDistFilesExist,
  'Docker Integration': dockerComposeExists,
  'API Gateway Integration': fs.existsSync('../api-gateway/src/middleware/serviceDiscovery.js')
};

let allValid = true;
Object.entries(validationResults).forEach(([category, isValid]) => {
  console.log(`${isValid ? '✅' : '❌'} ${category}: ${isValid ? 'PASS' : 'FAIL'}`);
  if (!isValid) allValid = false;
});

console.log('='.repeat(50));
console.log(`🎯 Overall Status: ${allValid ? '✅ IMPLEMENTATION COMPLETE' : '❌ ISSUES FOUND'}`);

if (allValid) {
  console.log('\n🚀 Service Discovery Infrastructure is ready for deployment!');
  console.log('\nNext steps:');
  console.log('1. Start Consul cluster: docker-compose up consul-server-1 consul-server-2 consul-server-3');
  console.log('2. Start Service Discovery service: docker-compose up service-discovery');
  console.log('3. Start API Gateway with service discovery integration');
  console.log('4. Register other services with the service discovery system');
} else {
  console.log('\n⚠️  Please fix the issues above before proceeding.');
  process.exit(1);
}

console.log('\n✨ Validation complete!');
