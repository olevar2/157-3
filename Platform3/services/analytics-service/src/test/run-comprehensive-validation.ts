#!/usr/bin/env node

/**
 * Main Test Runner for Comprehensive Validation Suite
 * Executes all 20 validation tests across 5 critical areas
 */

import ComprehensiveValidationSuite, { ValidationResults } from './ComprehensiveValidationSuite';
import fs from 'fs';
import path from 'path';

async function main() {
  console.log('🚀 Starting Platform3 67-Indicator System Validation');
  console.log('=====================================================');
  console.log('');
  console.log('🔍 Test Categories:');
  console.log('  🔬 Calculation Accuracy (4 tests)');
  console.log('  📊 Real Data Processing (4 tests)');
  console.log('  ⚡ Performance (5 tests)');
  console.log('  🛡️ Error Handling (4 tests)');
  console.log('  🔄 Integration (3 tests)');
  console.log('');
  console.log('📋 Total: 20 comprehensive validation tests');
  console.log('');

  const suite = new ComprehensiveValidationSuite();
  
  try {
    // Ensure logs directory exists
    const logsDir = path.join(process.cwd(), 'logs');
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }

    // Run the complete validation suite
    const results: ValidationResults = await suite.runFullValidation();
    
    // Display results
    console.log(results.summary);
    
    // Save detailed results to JSON
    const resultsPath = path.join(logsDir, 'comprehensive-validation-results.json');
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`📄 Detailed JSON results saved to: ${resultsPath}`);
    
    // Generate HTML report
    generateHTMLReport(results, logsDir);
    
    // Exit with appropriate code
    process.exit(results.failedTests === 0 ? 0 : 1);
    
  } catch (error) {
    console.error('❌ Validation suite failed to complete:');
    console.error(error);
    process.exit(1);
  }
}

function generateHTMLReport(results: ValidationResults, outputDir: string): void {
  const htmlPath = path.join(outputDir, 'validation-report.html');
  
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Platform3 Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #007acc; padding-bottom: 20px; margin-bottom: 30px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007acc; }
        .stat-value { font-size: 2em; font-weight: bold; color: #007acc; }
        .stat-label { color: #666; margin-top: 5px; }
        .category { margin-bottom: 30px; }
        .category-header { background: #007acc; color: white; padding: 15px; border-radius: 6px; margin-bottom: 15px; }
        .test-result { background: #f8f9fa; margin-bottom: 10px; padding: 15px; border-radius: 6px; border-left: 4px solid #28a745; }
        .test-result.failed { border-left-color: #dc3545; }
        .test-name { font-weight: bold; margin-bottom: 5px; }
        .test-duration { color: #666; font-size: 0.9em; }
        .test-details { margin-top: 10px; font-size: 0.9em; color: #555; }
        .error-message { color: #dc3545; font-style: italic; margin-top: 5px; }
        .success { color: #28a745; }
        .failed { color: #dc3545; }
        .overall-status { text-align: center; padding: 20px; font-size: 1.2em; font-weight: bold; border-radius: 8px; margin-bottom: 20px; }
        .overall-status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .overall-status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .overall-status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Platform3 67-Indicator System</h1>
            <h2>Comprehensive Validation Report</h2>
            <p>Generated on ${new Date().toLocaleString()}</p>
        </div>

        <div class="overall-status ${getOverallStatusClass(results)}">
            ${getOverallStatusMessage(results)}
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">${results.totalTests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">${results.passedTests}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failed">${results.failedTests}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${((results.passedTests / results.totalTests) * 100).toFixed(1)}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        ${generateCategoryReports(results)}
    </div>
</body>
</html>`;

  fs.writeFileSync(htmlPath, html);
  console.log(`📊 HTML report generated: ${htmlPath}`);
}

function getOverallStatusClass(results: ValidationResults): string {
  const successRate = (results.passedTests / results.totalTests) * 100;
  if (successRate === 100) return 'success';
  if (successRate >= 90) return 'warning';
  return 'error';
}

function getOverallStatusMessage(results: ValidationResults): string {
  const successRate = (results.passedTests / results.totalTests) * 100;
  if (successRate === 100) {
    return '✅ SYSTEM READY FOR PRODUCTION - All validation tests passed!';
  } else if (successRate >= 90) {
    return `⚠️ MOSTLY READY - ${results.failedTests} issues need attention before production`;
  } else {
    return `❌ NOT PRODUCTION READY - ${results.failedTests} critical issues require resolution`;
  }
}

function generateCategoryReports(results: ValidationResults): string {
  const categoryNames: { [key: string]: string } = {
    'accuracy': '🔬 Calculation Accuracy',
    'real_data': '📊 Real Data Processing',
    'performance': '⚡ Performance',
    'error_handling': '🛡️ Error Handling',
    'integration': '🔄 Integration'
  };

  return Object.entries(results.categories).map(([category, stats]) => {
    const categoryResults = results.results.filter(r => r.category === category);
    const successRate = ((stats.passed / stats.total) * 100).toFixed(1);
    
    return `
        <div class="category">
            <div class="category-header">
                <h3>${categoryNames[category] || category.toUpperCase()}</h3>
                <p>Passed: ${stats.passed}/${stats.total} (${successRate}%)</p>
            </div>
            ${categoryResults.map(result => `
                <div class="test-result ${result.passed ? 'passed' : 'failed'}">
                    <div class="test-name">
                        ${result.passed ? '✅' : '❌'} ${result.testName}
                    </div>
                    <div class="test-duration">Duration: ${result.duration}ms</div>
                    ${result.errorMessage ? `<div class="error-message">Error: ${result.errorMessage}</div>` : ''}
                    <div class="test-details">
                        <pre>${JSON.stringify(result.details, null, 2)}</pre>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
  }).join('');
}

// Run if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export default main;
