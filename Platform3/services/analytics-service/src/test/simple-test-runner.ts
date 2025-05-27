/**
 * Simple Test Runner for Comprehensive Validation Suite
 */

import { ComprehensiveValidationSuite } from './ComprehensiveValidationSuite_Fixed';

async function runTests() {
  console.log('🚀 Starting Platform3 67-Indicator Validation Suite...\n');

  try {
    const suite = new ComprehensiveValidationSuite();
    const results = await suite.runFullValidation();

    console.log('\n✅ Validation Suite Completed!');
    console.log(`📊 Results: ${results.passedTests}/${results.totalTests} tests passed`);
    console.log(`📁 Report saved to: ${results.reportPath}`);

    // Exit with appropriate code
    process.exit(results.failedTests === 0 ? 0 : 1);

  } catch (error) {
    console.error('❌ Critical error during validation:', error);
    process.exit(1);
  }
}

// Run the tests
runTests();
