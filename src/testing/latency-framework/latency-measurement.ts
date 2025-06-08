import { performance } from 'perf_hooks';
import { EventEmitter } from 'events';

export interface LatencyMeasurement {
    startTime: number;
    endTime: number;
    duration: number;
    operation: string;
    success: boolean;
    metadata?: Record<string, any>;
}

export interface LatencyStats {
    mean: number;
    median: number;
    p95: number;
    p99: number;
    min: number;
    max: number;
    count: number;
    errorRate: number;
}

export class LatencyMeasurer extends EventEmitter {
    private measurements: Map<string, LatencyMeasurement[]> = new Map();
    private ongoingOperations: Map<string, number> = new Map();
    private maxMeasurements: number = 10000;

    /**
     * Start measuring latency for an operation
     */
    startMeasurement(operationId: string, operation: string, metadata?: Record<string, any>): void {
        const startTime = performance.now();
        this.ongoingOperations.set(operationId, startTime);
        
        this.emit('measurement:started', {
            operationId,
            operation,
            startTime,
            metadata
        });
    }

    /**
     * End measurement for an operation
     */
    endMeasurement(operationId: string, operation: string, success: boolean = true, metadata?: Record<string, any>): LatencyMeasurement | null {
        const endTime = performance.now();
        const startTime = this.ongoingOperations.get(operationId);
        
        if (!startTime) {
            console.warn(`No start time found for operation ${operationId}`);
            return null;
        }

        this.ongoingOperations.delete(operationId);        
        const measurement: LatencyMeasurement = {
            startTime,
            endTime,
            duration: endTime - startTime,
            operation,
            success,
            metadata
        };

        this.recordMeasurement(operation, measurement);
        
        this.emit('measurement:completed', measurement);
        
        return measurement;
    }

    /**
     * Measure a synchronous operation
     */
    measureSync<T>(operation: string, fn: () => T, metadata?: Record<string, any>): { result: T; measurement: LatencyMeasurement } {
        const startTime = performance.now();
        let success = true;
        let result: T;
        
        try {
            result = fn();
        } catch (error) {
            success = false;
            throw error;
        } finally {
            const endTime = performance.now();
            const measurement: LatencyMeasurement = {
                startTime,
                endTime,
                duration: endTime - startTime,
                operation,
                success,
                metadata
            };
            
            this.recordMeasurement(operation, measurement);
            this.emit('measurement:completed', measurement);
        }
        
        return { result: result!, measurement: this.getMeasurements(operation).slice(-1)[0] };
    }    /**
     * Measure an asynchronous operation
     */
    async measureAsync<T>(operation: string, fn: () => Promise<T>, metadata?: Record<string, any>): Promise<{ result: T; measurement: LatencyMeasurement }> {
        const startTime = performance.now();
        let success = true;
        let result: T;
        
        try {
            result = await fn();
        } catch (error) {
            success = false;
            throw error;
        } finally {
            const endTime = performance.now();
            const measurement: LatencyMeasurement = {
                startTime,
                endTime,
                duration: endTime - startTime,
                operation,
                success,
                metadata
            };
            
            this.recordMeasurement(operation, measurement);
            this.emit('measurement:completed', measurement);
        }
        
        return { result: result!, measurement: this.getMeasurements(operation).slice(-1)[0] };
    }

    /**
     * Record a measurement for an operation
     */
    private recordMeasurement(operation: string, measurement: LatencyMeasurement): void {
        if (!this.measurements.has(operation)) {
            this.measurements.set(operation, []);
        }
        
        const measurements = this.measurements.get(operation)!;
        measurements.push(measurement);
        
        // Keep only the latest measurements to prevent memory leaks
        if (measurements.length > this.maxMeasurements) {
            measurements.splice(0, measurements.length - this.maxMeasurements);
        }
    }

    /**
     * Get all measurements for an operation
     */
    getMeasurements(operation: string): LatencyMeasurement[] {
        return this.measurements.get(operation) || [];
    }    /**
     * Calculate statistics for an operation
     */
    getStats(operation: string, timeWindowMs?: number): LatencyStats {
        const measurements = this.getMeasurements(operation);
        
        if (measurements.length === 0) {
            return {
                mean: 0, median: 0, p95: 0, p99: 0, min: 0, max: 0, count: 0, errorRate: 0
            };
        }

        // Filter by time window if specified
        let filteredMeasurements = measurements;
        if (timeWindowMs) {
            const cutoffTime = performance.now() - timeWindowMs;
            filteredMeasurements = measurements.filter(m => m.endTime >= cutoffTime);
        }

        const durations = filteredMeasurements.map(m => m.duration).sort((a, b) => a - b);
        const successfulMeasurements = filteredMeasurements.filter(m => m.success);
        
        const count = filteredMeasurements.length;
        const errorRate = count > 0 ? (count - successfulMeasurements.length) / count : 0;
        
        if (durations.length === 0) {
            return {
                mean: 0, median: 0, p95: 0, p99: 0, min: 0, max: 0, count, errorRate
            };
        }

        const mean = durations.reduce((sum, d) => sum + d, 0) / durations.length;
        const median = durations[Math.floor(durations.length / 2)];
        const p95 = durations[Math.floor(durations.length * 0.95)];
        const p99 = durations[Math.floor(durations.length * 0.99)];
        const min = durations[0];
        const max = durations[durations.length - 1];

        return { mean, median, p95, p99, min, max, count, errorRate };
    }

    getOperations(): string[] {
        return Array.from(this.measurements.keys());
    }

    clear(): void {
        this.measurements.clear();
        this.ongoingOperations.clear();
    }

    clearOperation(operation: string): void {
        this.measurements.delete(operation);
    }

    setMaxMeasurements(max: number): void {
        this.maxMeasurements = max;
    }
}