import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { BridgeMessage, ChannelConfig } from '../types';
import { MessagePackProtocol } from '../protocols/messagepack.protocol';

export class DualChannelWebSocket extends EventEmitter {
    private primaryChannel: WebSocket | null = null;
    private secondaryChannel: WebSocket | null = null;
    private protocol: MessagePackProtocol;
    private messageQueue: Map<string, BridgeMessage> = new Map();
    private channelHealth: Map<string, boolean> = new Map();

    constructor(private config: ChannelConfig) {
        super();
        this.protocol = new MessagePackProtocol(config.protocol);
        this.setupChannels();
    }

    private async setupChannels(): Promise<void> {
        // Primary channel
        this.primaryChannel = new WebSocket(this.config.primaryUrl);
        this.setupChannelHandlers(this.primaryChannel, 'primary');

        // Secondary channel
        this.secondaryChannel = new WebSocket(this.config.secondaryUrl);
        this.setupChannelHandlers(this.secondaryChannel, 'secondary');
    }

    private setupChannelHandlers(channel: WebSocket, name: string): void {
        channel.on('open', () => {
            this.channelHealth.set(name, true);
            this.emit('channel:connected', name);
        });

        channel.on('message', async (data: Buffer) => {
            try {
                const message = await this.protocol.decode(data);
                this.emit('message', message);
            } catch (error) {
                this.emit('error', { channel: name, error });
            }
        });

        channel.on('close', () => {
            this.channelHealth.set(name, false);
            this.emit('channel:disconnected', name);
            this.reconnectChannel(name);
        });

        channel.on('error', (error) => {
            this.emit('error', { channel: name, error });
        });
    }

    async send(message: BridgeMessage): Promise<void> {
        const encoded = await this.protocol.encode(message);
        
        // Send to both channels for redundancy
        const promises: Promise<void>[] = [];
        
        if (this.primaryChannel?.readyState === WebSocket.OPEN) {
            promises.push(this.sendToChannel(this.primaryChannel, encoded));
        }
        
        if (this.secondaryChannel?.readyState === WebSocket.OPEN) {
            promises.push(this.sendToChannel(this.secondaryChannel, encoded));
        }

        if (promises.length === 0) {
            // Queue message if no channels available
            this.messageQueue.set(message.id, message);
            throw new Error('No available channels');
        }

        await Promise.race(promises); // Complete when at least one succeeds
    }

    private async sendToChannel(channel: WebSocket, data: Buffer): Promise<void> {
        return new Promise((resolve, reject) => {
            channel.send(data, (error) => {
                if (error) reject(error);
                else resolve();
            });
        });
    }

    private async reconnectChannel(name: string): Promise<void> {
        setTimeout(() => {
            if (name === 'primary') {
                this.primaryChannel = new WebSocket(this.config.primaryUrl);
                this.setupChannelHandlers(this.primaryChannel, 'primary');
            } else {
                this.secondaryChannel = new WebSocket(this.config.secondaryUrl);
                this.setupChannelHandlers(this.secondaryChannel, 'secondary');
            }
        }, this.config.reconnectDelay || 1000);
    }
}
