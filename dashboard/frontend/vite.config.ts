import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://gateway-service:3000',
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: 'ws://dashboard-websocket:3008',
        ws: true,
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          material: ['@mui/material', '@mui/icons-material'],
          charts: ['recharts', 'lightweight-charts', '@mui/x-charts']
        }
      }
    }
  }
})
