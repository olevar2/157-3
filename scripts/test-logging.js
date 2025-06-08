const { execSync } = require('child_process');
const path = require('path');

// Compile and run the test
console.log('Compiling and running logging test...');

try {
    // Compile TypeScript
    execSync('npx tsc src/testing/logging-test.ts --outDir dist', { stdio: 'inherit' });

    // Run the test
    console.log('\nRunning logging test:');
    execSync('node dist/testing/logging-test.js', { stdio: 'inherit' });

    console.log('\nLogging test completed successfully.');
} catch (error) {
    console.error('Error running logging test:', error);
    process.exit(1);
}
