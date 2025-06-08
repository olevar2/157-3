import * as http from 'http';
import { PerformanceProfiler } from './performance-profiler';
import { BenchmarkSuite, BenchmarkResult } from './benchmark-suite';
import { RegressionDetector } from './regression-detector';

export interface DashboardConfig {
    port: number;
    updateInterval: number;
    enableRealTimeUpdates: boolean;
}

export class LatencyDashboard {
    private server: http.Server | null = null;
    private config: DashboardConfig;
    private profiler: PerformanceProfiler;
    private benchmarkSuite: BenchmarkSuite;
    private regressionDetector: RegressionDetector;
    private isRunning: boolean = false;

    constructor(
        config: DashboardConfig,
        profiler: PerformanceProfiler,
        benchmarkSuite: BenchmarkSuite,
        regressionDetector: RegressionDetector
    ) {
        this.config = config;
        this.profiler = profiler;
        this.benchmarkSuite = benchmarkSuite;
        this.regressionDetector = regressionDetector;
    }

    /**
     * Start the dashboard server
     */
    start(): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.isRunning) {
                resolve();
                return;
            }

            this.server = http.createServer((req, res) => {
                this.handleRequest(req, res);
            });

            this.server.listen(this.config.port, () => {
                this.isRunning = true;
                console.log(`Latency Dashboard running on port ${this.config.port}`);
                resolve();
            });

            this.server.on('error', reject);
        });
    }    /**
     * Stop the dashboard server
     */
    stop(): Promise<void> {
        return new Promise((resolve) => {
            if (!this.isRunning || !this.server) {
                resolve();
                return;
            }

            this.server.close(() => {
                this.isRunning = false;
                this.server = null;
                resolve();
            });
        });
    }

    /**
     * Handle HTTP requests
     */
    private handleRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
        const url = req.url || '/';
        
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Access-Control-Allow-Origin', '*');

        try {
            if (url === '/api/stats') {
                const operations = this.profiler.getOperations();
                const stats: any = {};
                operations.forEach(op => {
                    stats[op] = this.profiler.getStats(op);
                });
                res.end(JSON.stringify(stats));
            } else if (url === '/api/status') {
                const status = {
                    uptime: process.uptime(),
                    memory: process.memoryUsage(),
                    operations: this.profiler.getOperations().length,
                    isRunning: this.isRunning
                };
                res.end(JSON.stringify(status));
            } else if (url === '/') {
                res.setHeader('Content-Type', 'text/html');
                res.end(this.generateDashboardHTML());
            } else {
                res.statusCode = 404;
                res.end(JSON.stringify({ error: 'Not found' }));
            }
        } catch (error) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }));
        }
    }    /**
     * Generate basic HTML dashboard
     */
    private generateDashboardHTML(): string {
        return `
<!DOCTYPE html>
<html>
<head>
    <title>Platform3 Latency Dashboard</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
        .alert { background-color: #ffe6e6; }
        .good { background-color: #e6ffe6; }
        .button { padding: 10px 20px; margin: 5px; background: #007cba; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Platform3 Latency Dashboard</h1>
    <div id="status"></div>
    <button class="button" onclick="refreshData()">Refresh</button>
    <div id="metrics"></div>
    
    <script>
        async function refreshData() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                document.getElementById('metrics').innerHTML = '<pre>' + JSON.stringify(stats, null, 2) + '</pre>';
                
                const statusResponse = await fetch('/api/status');
                const status = await statusResponse.json();
                document.getElementById('status').innerHTML = '<pre>' + JSON.stringify(status, null, 2) + '</pre>';
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        refreshData();
        setInterval(refreshData, 5000);
    </script>
</body>
</html>`;
    }
}