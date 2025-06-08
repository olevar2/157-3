export interface BridgeMessage {
  id: string;
  type: 'signal' | 'data' | 'command' | 'heartbeat' | 'response' | 'execution' | 'load-test';
  payload: any;
  timestamp?: number;
  correlationId?: string;
}

export interface ProtocolConfig {
  type: 'json' | 'messagepack' | 'protobuf';
  compression?: boolean;
  encryption?: boolean;
}

export interface ChannelConfig {
  primaryUrl: string;
  secondaryUrl: string;
  protocol: ProtocolConfig;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

export interface BridgeMetrics {
  latency: Map<string, number[]>;
  errors: Map<string, number>;
  throughput: { in: number; out: number };
  uptime: number;
}
