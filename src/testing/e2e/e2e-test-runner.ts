/**
 * E2E Test Runner
 * 
 * This file provides a CLI interface for running the end-to-end tests 
 * and managing test suite execution for Platform3 system validation.
 */

import { E2ETestOrchestrator, StressTestConfig, TestScenario } from './e2e-test-orchestrator';
import * as fs from 'fs';
import * as path from 'path';

const TEST_REPORT_DIR = path.join(__dirname, '../../../reports/e2e');

/**
 * Main test runner function
 */
async function runE2ETests(options: {
    scenarios?: string[], // Specific scenarios to run
    stress?: boolean,     // Run stress test
    report?: boolean      // Generate report
}): Promise<void> {
    try {
        console.log('🚀 Starting Platform3 End-to-End Validation Suite...');

        // Ensure report directory exists
        if (!fs.existsSync(TEST_REPORT_DIR)) {
            fs.mkdirSync(TEST_REPORT_DIR, { recursive: true });
        }

        // Initialize the test orchestrator
        const orchestrator = new E2ETestOrchestrator();
        await orchestrator.initialize();

        // Subscribe to test events
        orchestrator.on('scenario:completed', ({ scenario, result }) => {
            const status = result.success ? '✅ PASSED' : '❌ FAILED';
            console.log(`${status} - ${scenario} (${result.duration.toFixed(2)}ms)`);
        });

        // Run normal test scenarios
        const results = await orchestrator.runAllScenarios();
        console.log(`\n📊 Test Scenarios Complete: ${results.filter(r => r.success).length}/${results.length} passed`);

        // Run stress test if requested
        if (options.stress) {
            console.log('\n⚡ Running stress test...');
            const stressConfig: StressTestConfig = {
                concurrentAgents: 10,
                requestsPerSecond: 100,
                durationSeconds: 10,
                rampUpSeconds: 2
            };
            
            const stressResult = await orchestrator.runStressTest(stressConfig);
            
            console.log(`Stress Test: ${stressResult.success ? '✅ PASSED' : '❌ FAILED'}`);
            console.log(`- ${stressResult.details.total_requests} total requests`);
            console.log(`- ${stressResult.details.successful_requests} successful (${(stressResult.details.success_rate * 100).toFixed(1)}%)`);
            console.log(`- Mean latency: ${stressResult.details.mean_latency.toFixed(2)}ms`);
            console.log(`- P95 latency: ${stressResult.details.p95_latency.toFixed(2)}ms`);
            
            // Add stress results to the main results
            results.push(stressResult);
        }

        // Generate final report
        const report = orchestrator.generateValidationReport();
        const validationStatus = report.validationPassed ? '✅ PASSED' : '❌ FAILED';
        
        console.log(`\n🏁 End-to-End System Validation: ${validationStatus}`);
        console.log(`Success Rate: ${(report.successRate * 100).toFixed(1)}% (${report.successfulTests}/${report.totalTests})`);
        console.log(`Average Test Duration: ${report.averageTestDuration.toFixed(2)}ms`);
        
        // Save report to file if requested
        if (options.report) {
            const reportFilePath = path.join(TEST_REPORT_DIR, `e2e-validation-report-${new Date().toISOString().replace(/:/g, '-')}.json`);
            fs.writeFileSync(reportFilePath, JSON.stringify(report, null, 2));
            console.log(`\n📄 Report saved to: ${reportFilePath}`);
        }
        
        // Stop the orchestrator
        await orchestrator.stop();
        
        // Exit with appropriate code
        process.exit(report.validationPassed ? 0 : 1);
    } catch (error) {
        console.error('❌ Error running E2E tests:', error);
        process.exit(1);
    }
}

/**
 * Parse command line arguments
 */
function parseArgs(): any {
    const args = process.argv.slice(2);
    const options: any = {
        scenarios: [],
        stress: false,
        report: true
    };
    
    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--stress':
                options.stress = true;
                break;
            case '--no-report':
                options.report = false;
                break;
            case '--scenario':
                if (i + 1 < args.length) {
                    options.scenarios.push(args[i + 1]);
                    i++;
                }
                break;
            case '--help':
                printHelp();
                process.exit(0);
                break;
        }
    }
    
    return options;
}

/**
 * Print help information
 */
function printHelp(): void {
    console.log(`
Platform3 End-to-End Validation Suite

Usage:
  node e2e-test-runner.js [options]

Options:
  --scenario <name>  Run specific test scenario
  --stress           Include stress testing
  --no-report        Skip report generation
  --help             Show this help
`);
}

// Run tests if this file is executed directly
if (require.main === module) {
    const options = parseArgs();
    runE2ETests(options).catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}