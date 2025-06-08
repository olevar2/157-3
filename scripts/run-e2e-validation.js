/**
 * Platform3 End-to-End System Validation
 * 
 * This script runs the complete Platform3 end-to-end validation suite,
 * testing the Python-TypeScript bridge and all component integrations.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Configuration
const projectRoot = path.resolve(__dirname, '..');
const tsNodePath = path.join(projectRoot, 'node_modules', '.bin', 'ts-node');
const scriptPath = path.join(projectRoot, 'src', 'testing', 'e2e', 'e2e-validation-bridge.ts');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
    typescript: false,
    python: false,
    report: true,
    help: false
};

for (const arg of args) {
    if (arg === '--ts-only') options.typescript = true;
    if (arg === '--py-only') options.python = true;
    if (arg === '--no-report') options.report = false;
    if (arg === '--help' || arg === '-h') options.help = true;
}

// Print help message
if (options.help) {
    console.log(`
Platform3 End-to-End System Validation

Usage:
  node run-e2e-validation.js [options]

Options:
  --ts-only      Run TypeScript tests only
  --py-only      Run Python tests only
  --no-report    Skip generating a detailed report
  --help, -h     Show this help message
`);
    process.exit(0);
}

// Run the validation script
console.log('🚀 Starting Platform3 End-to-End System Validation...');

const scriptArgs = [];
if (options.typescript) scriptArgs.push('--ts-only');
if (options.python) scriptArgs.push('--py-only');
if (!options.report) scriptArgs.push('--no-report');

const child = spawn(
    process.platform === 'win32' ? 'npx.cmd' : 'npx',
    ['ts-node', scriptPath, ...scriptArgs],
    { stdio: 'inherit', cwd: projectRoot }
);

child.on('exit', (code) => {
    if (code === 0) {
        console.log('✅ End-to-End Validation completed successfully');
    } else {
        console.error(`❌ End-to-End Validation failed with exit code: ${code}`);
    }
    process.exit(code);
});