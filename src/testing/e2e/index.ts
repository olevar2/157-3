/**
 * End-to-End System Validation Suite for Platform3
 * 
 * This module exports the end-to-end testing components for comprehensive
 * system validation, including:
 * 
 * - E2ETestOrchestrator: Manages test scenarios execution
 * - E2EValidationBridge: Bridges TypeScript and Python tests
 * - Various test scenario and configuration interfaces
 */

export * from './e2e-test-orchestrator';
export * from './e2e-validation-bridge';

// Re-export main execution functions for ease of use
import { E2ETestOrchestrator } from './e2e-test-orchestrator';
import { E2EValidationBridge } from './e2e-validation-bridge';
import * as path from 'path';

/**
 * Run standalone TypeScript E2E tests
 */
export async function runTypeScriptE2ETests() {
    const orchestrator = new E2ETestOrchestrator();
    await orchestrator.initialize();
    const results = await orchestrator.runAllScenarios();
    await orchestrator.stop();
    return results;
}

/**
 * Run the complete end-to-end validation
 */
export async function runCompleteE2EValidation() {
    const projectRoot = path.join(__dirname, '../../..');
    const bridge = new E2EValidationBridge({
        pythonPath: 'python',  // Use system Python
        pythonScript: path.join(projectRoot, 'tests/integration/e2e_validation_suite.py'),
        reportDir: path.join(projectRoot, 'reports/e2e')
    });
    
    return await bridge.runFullValidation();
}