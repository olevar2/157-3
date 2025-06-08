import { WebSocketServer } from 'ws';
import { decode, encode } from '@msgpack/msgpack';

export function createMockWebSocketServer(port: number): WebSocketServer {
    const wss = new WebSocketServer({ port });
    
    wss.on('connection', (ws) => {
        console.log(`Mock WebSocket server connected on port ${port}`);
        
        ws.on('message', async (data: Buffer) => {
            try {
                const message = decode(data);
                console.log('Received message:', message);
                
                // Echo back with response
                const response = {
                    ...message as any,
                    type: 'response',
                    timestamp: Date.now()
                };
                
                ws.send(encode(response));
            } catch (error) {
                console.error('Mock server error:', error);
            }
        });
        
        ws.on('error', console.error);
    });
    
    return wss;
}

// Start mock servers if running directly
if (require.main === module) {
    const server1 = createMockWebSocketServer(8001);
    const server2 = createMockWebSocketServer(8002);
    
    console.log('Mock WebSocket servers running on ports 8001 and 8002');
    
    process.on('SIGINT', () => {
        server1.close();
        server2.close();
        process.exit(0);
    });
}
