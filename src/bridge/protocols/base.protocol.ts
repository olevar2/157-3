import { BridgeMessage, ProtocolConfig } from '../types/index';

export abstract class BaseProtocol {
  protected metrics = {
    recordLatency: (operation: string, latency: number) => {
      console.log(`[${operation}] Latency: ${latency.toFixed(3)}ms`);
      if (latency > 1) {
        console.warn(`[${operation}] High latency: ${latency.toFixed(3)}ms`);
      }
    },
    recordError: (operation: string) => {
      console.error(`[${operation}] Error occurred`);
    }
  };

  constructor(protected config: ProtocolConfig) {}

  abstract encode(message: BridgeMessage): Promise<Buffer>;
  abstract decode(buffer: Buffer): Promise<BridgeMessage>;
}
