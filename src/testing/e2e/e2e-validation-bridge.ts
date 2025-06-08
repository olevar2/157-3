/**
 * E2E Validation Bridge
 * 
 * This file provides a bridge between TypeScript end-to-end testing and Python AI models,
 * allowing for comprehensive validation of the entire Platform3 workflow.
 * 
 * It launches both the TypeScript tests and Python tests, coordinating the results
 * into a unified validation report.
 */

import * as path from 'path';
import * as fs from 'fs';
import { spawn } from 'child_process';
import { E2ETestOrchestrator, ValidationResults } from './e2e-test-orchestrator';

/**
 * Configuration for the validation bridge
 */
interface BridgeConfig {
    pythonPath: string;
    pythonScript: string;
    reportDir: string;
}

/**
 * Result from running the Python validation suite
 */
interface PythonValidationResult {
    timestamp: string;
    total_tests: number;
    passed_tests: number;
    failed_tests: number;
    success_rate: number;
    total_duration_ms: number;
    tests: any[];
    validation_passed: boolean;
}

class E2EValidationBridge {
    private config: BridgeConfig;

    constructor(config: BridgeConfig) {
        this.config = config;
        
        // Ensure report directory exists
        if (!fs.existsSync(this.config.reportDir)) {
            fs.mkdirSync(this.config.reportDir, { recursive: true });
        }
    }

    /**
     * Run the TypeScript E2E test orchestrator
     */
    async runTypeScriptTests(): Promise<ValidationResults[]> {
        console.log('Running TypeScript end-to-end tests...');
        const orchestrator = new E2ETestOrchestrator();
        
        try {
            await orchestrator.initialize();
            const results = await orchestrator.runAllScenarios();
            const stressResults = await orchestrator.runStressTest({
                concurrentAgents: 5,
                requestsPerSecond: 50,
                durationSeconds: 5
            });
            results.push(stressResults);
            
            await orchestrator.stop();
            return results;
        } catch (error) {
            console.error('Error running TypeScript tests:', error);
            throw error;
        }
    }

    /**
     * Run the Python validation suite
     */
    async runPythonTests(): Promise<PythonValidationResult> {
        return new Promise((resolve, reject) => {
            console.log('Running Python validation suite...');
            
            const pythonProcess = spawn(
                this.config.pythonPath,
                [this.config.pythonScript],
                { stdio: ['ignore', 'pipe', 'pipe'] }
            );
            
            let stdout = '';
            let stderr = '';
            
            pythonProcess.stdout.on('data', (data) => {
                const output = data.toString();
                stdout += output;
                console.log(output);
            });
            
            pythonProcess.stderr.on('data', (data) => {
                const error = data.toString();
                stderr += error;
                console.error(error);
            });
            
            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`Python validation suite failed with code ${code}: ${stderr}`));
                    return;
                }
                
                // Find and parse the JSON results file
                try {
                    const reportsDir = path.join(path.dirname(this.config.pythonScript), '../../reports');
                    const files = fs.readdirSync(reportsDir);
                    
                    // Get the most recent results file (they have timestamps in the name)
                    const resultFiles = files
                        .filter(f => f.startsWith('e2e_validation_results_'))
                        .sort()
                        .reverse();
                        
                    if (resultFiles.length === 0) {
                        reject(new Error('No Python validation result file found'));
                        return;
                    }
                    
                    const resultPath = path.join(reportsDir, resultFiles[0]);
                    const resultData = fs.readFileSync(resultPath, 'utf-8');
                    const results = JSON.parse(resultData) as PythonValidationResult;
                    
                    resolve(results);
                } catch (error) {
                    reject(new Error(`Failed to parse Python validation results: ${error}`));
                }
            });
        });
    }

    /**
     * Run the complete end-to-end validation
     */
    async runFullValidation(): Promise<any> {
        const startTime = performance.now();
        
        try {
            // Run both TypeScript and Python tests (can be run in parallel)
            const [tsResults, pyResults] = await Promise.all([
                this.runTypeScriptTests(),
                this.runPythonTests()
            ]);
            
            // Generate comprehensive validation report
            const tsPassedTests = tsResults.filter(r => r.success).length;
            const tsTotalTests = tsResults.length;
            const tsSuccessRate = tsTotalTests > 0 ? tsPassedTests / tsTotalTests : 0;
            
            const pySuccessRate = pyResults.success_rate;
            
            const validationPassed = tsSuccessRate >= 0.95 && pySuccessRate >= 0.95;
            const totalDurationMs = performance.now() - startTime;
            
            // Create the combined report
            const report = {
                timestamp: new Date().toISOString(),
                typescript_tests: {
                    total_tests: tsTotalTests,
                    passed_tests: tsPassedTests,
                    failed_tests: tsTotalTests - tsPassedTests,
                    success_rate: tsSuccessRate,
                    results: tsResults
                },
                python_tests: {
                    total_tests: pyResults.total_tests,
                    passed_tests: pyResults.passed_tests,
                    failed_tests: pyResults.failed_tests,
                    success_rate: pyResults.success_rate,
                    results: pyResults.tests
                },
                overall: {
                    total_tests: tsTotalTests + pyResults.total_tests,
                    passed_tests: tsPassedTests + pyResults.passed_tests,
                    failed_tests: (tsTotalTests - tsPassedTests) + pyResults.failed_tests,
                    success_rate: (tsPassedTests + pyResults.passed_tests) / (tsTotalTests + pyResults.total_tests),
                    validation_passed: validationPassed,
                    total_duration_ms: totalDurationMs
                }
            };
            
            // Save the report
            const reportPath = this.saveReport(report);
            
            // Print summary
            this.printSummary(report, reportPath);
            
            return {
                success: validationPassed,
                reportPath
            };
        } catch (error) {
            console.error('Error running full validation:', error);
            throw error;
        }
    }

    /**
     * Save the validation report to a file
     */
    private saveReport(report: any): string {
        const timestamp = new Date().toISOString().replace(/:/g, '-').replace(/\..+/, '');
        const reportPath = path.join(this.config.reportDir, `full-validation-report-${timestamp}.json`);
        
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        
        return reportPath;
    }

    /**
     * Print a summary of the validation results
     */
    private printSummary(report: any, reportPath: string): void {
        console.log('\n');
        console.log('=' .repeat(80));
        console.log('PLATFORM3 END-TO-END VALIDATION RESULTS');
        console.log('=' .repeat(80));
        console.log();
        
        console.log('TypeScript Validation:');
        console.log(`- Tests: ${report.typescript_tests.total_tests}`);
        console.log(`- Passed: ${report.typescript_tests.passed_tests} (${(report.typescript_tests.success_rate * 100).toFixed(1)}%)`);
        console.log(`- Failed: ${report.typescript_tests.failed_tests}`);
        console.log();
        
        console.log('Python Validation:');
        console.log(`- Tests: ${report.python_tests.total_tests}`);
        console.log(`- Passed: ${report.python_tests.passed_tests} (${(report.python_tests.success_rate * 100).toFixed(1)}%)`);
        console.log(`- Failed: ${report.python_tests.failed_tests}`);
        console.log();
        
        console.log('Overall Results:');
        console.log(`- Total Tests: ${report.overall.total_tests}`);
        console.log(`- Total Passed: ${report.overall.passed_tests} (${(report.overall.success_rate * 100).toFixed(1)}%)`);
        console.log(`- Total Failed: ${report.overall.failed_tests}`);
        console.log(`- Total Duration: ${(report.overall.total_duration_ms / 1000).toFixed(2)}s`);
        console.log();
        
        const status = report.overall.validation_passed ? 
            '✅ PASSED - SYSTEM VALIDATION SUCCESSFUL' : 
            '❌ FAILED - SYSTEM VALIDATION INCOMPLETE';
        
        console.log('Final Validation Status:');
        console.log(status);
        console.log();
        
        console.log(`Detailed report saved to: ${reportPath}`);
        console.log('=' .repeat(80));
    }
}

/**
 * Run the validation bridge from the command line
 */
async function main() {
    try {
        // Configure the validation bridge
        const projectRoot = path.join(__dirname, '../../..');
        const bridge = new E2EValidationBridge({
            pythonPath: 'python',  // Use system Python
            pythonScript: path.join(projectRoot, 'tests/integration/e2e_validation_suite.py'),
            reportDir: path.join(projectRoot, 'reports/e2e')
        });
        
        // Run the validation
        const result = await bridge.runFullValidation();
        
        // Exit with appropriate code
        process.exit(result.success ? 0 : 1);
    } catch (error) {
        console.error('Validation bridge error:', error);
        process.exit(1);
    }
}

// Run the validation bridge if this file is executed directly
if (require.main === module) {
    main().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}