import { LatencyMeasurer, LatencyStats, LatencyMeasurement } from './latency-measurement';
import { MessagePackProtocol } from '../../bridge/protocols/messagepack.protocol';

export interface BenchmarkConfig {
    iterations: number;
    warmupIterations: number;
    concurrency: number;
    timeout: number;
}

export interface BenchmarkResult {
    name: string;
    config: BenchmarkConfig;
    stats: LatencyStats;
    measurements: LatencyMeasurement[];
    timestamp: number;
}

export class BenchmarkSuite {
    private latencyMeasurer: LatencyMeasurer;
    private messagePackProtocol: MessagePackProtocol;

    constructor() {
        this.latencyMeasurer = new LatencyMeasurer();
        this.messagePackProtocol = new MessagePackProtocol({
            type: 'messagepack',
            compression: true
        });
    }

    /**
     * Run MessagePack encoding benchmark
     */
    async benchmarkMessagePackEncoding(config: BenchmarkConfig): Promise<BenchmarkResult> {
        const testMessage = {
            id: 'test-message',
            type: 'data' as const,
            payload: {
                symbol: 'EURUSD',
                timeframe: 'M1',
                indicators: ['RSI', 'MACD', 'EMA'],
                params: { period: 14, fastPeriod: 12, slowPeriod: 26 }
            },
            timestamp: Date.now()
        };

        await this.warmup(() => this.messagePackProtocol.encode(testMessage), config.warmupIterations);

        const measurements: LatencyMeasurement[] = [];
        
        for (let i = 0; i < config.iterations; i++) {
            const { measurement } = await this.latencyMeasurer.measureAsync(
                'messagepack-encode',
                () => this.messagePackProtocol.encode(testMessage)
            );
            measurements.push(measurement);
        }        const stats = this.latencyMeasurer.getStats('messagepack-encode');
        
        return {
            name: 'MessagePack Encoding',
            config,
            stats,
            measurements,
            timestamp: Date.now()
        };
    }

    /**
     * Run MessagePack decoding benchmark
     */
    async benchmarkMessagePackDecoding(config: BenchmarkConfig): Promise<BenchmarkResult> {
        const testMessage = {
            id: 'test-message',
            type: 'response' as const,
            payload: { rsi: 65.5, macd: { macd: 0.0012, signal: 0.0008, histogram: 0.0004 } }
        };

        const encodedMessage = await this.messagePackProtocol.encode(testMessage);
        await this.warmup(() => this.messagePackProtocol.decode(encodedMessage), config.warmupIterations);

        const measurements: LatencyMeasurement[] = [];
        
        for (let i = 0; i < config.iterations; i++) {
            const { measurement } = await this.latencyMeasurer.measureAsync(
                'messagepack-decode',
                () => this.messagePackProtocol.decode(encodedMessage)
            );
            measurements.push(measurement);
        }

        const stats = this.latencyMeasurer.getStats('messagepack-decode');
        
        return {
            name: 'MessagePack Decoding',
            config,
            stats,
            measurements,
            timestamp: Date.now()
        };
    }

    /**
     * Run comprehensive bridge benchmark
     */
    async benchmarkBridgeRoundTrip(config: BenchmarkConfig): Promise<BenchmarkResult> {
        const testMessage = {
            id: 'test-roundtrip',
            type: 'data' as const,
            payload: { test: 'data', timestamp: Date.now() }
        };

        await this.warmup(async () => {
            const encoded = await this.messagePackProtocol.encode(testMessage);
            await this.messagePackProtocol.decode(encoded);
        }, config.warmupIterations);

        const measurements: LatencyMeasurement[] = [];        for (let i = 0; i < config.iterations; i++) {
            const { measurement } = await this.latencyMeasurer.measureAsync(
                'bridge-roundtrip',
                async () => {
                    const encoded = await this.messagePackProtocol.encode(testMessage);
                    return await this.messagePackProtocol.decode(encoded);
                }
            );
            measurements.push(measurement);
        }

        const stats = this.latencyMeasurer.getStats('bridge-roundtrip');
        
        return {
            name: 'Bridge Round Trip',
            config,
            stats,
            measurements,
            timestamp: Date.now()
        };
    }

    /**
     * Run all benchmarks and return consolidated results
     */
    async runAllBenchmarks(config: BenchmarkConfig): Promise<BenchmarkResult[]> {
        const results: BenchmarkResult[] = [];

        console.log('Running MessagePack encoding benchmark...');
        results.push(await this.benchmarkMessagePackEncoding(config));

        console.log('Running MessagePack decoding benchmark...');
        results.push(await this.benchmarkMessagePackDecoding(config));

        console.log('Running bridge round-trip benchmark...');
        results.push(await this.benchmarkBridgeRoundTrip(config));

        return results;
    }

    /**
     * Warm up the system before benchmarking
     */
    private async warmup(operation: () => Promise<any> | any, iterations: number): Promise<void> {
        for (let i = 0; i < iterations; i++) {
            await operation();
        }
    }

    /**
     * Compare benchmark results
     */
    compareBenchmarks(baseline: BenchmarkResult[], current: BenchmarkResult[]): { improvement: number; regression: number; summary: string } {
        let totalImprovement = 0;
        let totalRegression = 0;
        const comparisons: string[] = [];

        for (const currentResult of current) {
            const baselineResult = baseline.find(b => b.name === currentResult.name);
            if (baselineResult) {
                const improvement = ((baselineResult.stats.mean - currentResult.stats.mean) / baselineResult.stats.mean) * 100;
                if (improvement > 0) {
                    totalImprovement += improvement;
                    comparisons.push(`${currentResult.name}: ${improvement.toFixed(2)}% faster`);
                } else {
                    totalRegression += Math.abs(improvement);
                    comparisons.push(`${currentResult.name}: ${Math.abs(improvement).toFixed(2)}% slower`);
                }
            }
        }

        return {
            improvement: totalImprovement,
            regression: totalRegression,
            summary: comparisons.join(', ')
        };
    }
}