import { encode, decode } from '@msgpack/msgpack';
import { BaseProtocol } from './base.protocol';
import { BridgeMessage } from '../types';
import { performance } from 'perf_hooks';

export class MessagePackProtocol extends BaseProtocol {
    private encoder = encode;
    private decoder = decode;

    async encode(message: BridgeMessage): Promise<Buffer> {
        const start = performance.now();
        
        try {
            const encoded = this.encoder(message);
            const buffer = Buffer.from(encoded);
            
            const latency = performance.now() - start;
            this.metrics.recordLatency('encode', latency);
            
            return buffer;
        } catch (error) {
            this.metrics.recordError('encode');
            throw error;
        }
    }

    async decode(buffer: Buffer): Promise<BridgeMessage> {
        const start = performance.now();
        
        try {
            const message = this.decoder(buffer) as BridgeMessage;
            
            const latency = performance.now() - start;
            this.metrics.recordLatency('decode', latency);
            
            return message;
        } catch (error) {
            this.metrics.recordError('decode');
            throw error;
        }
    }
}
