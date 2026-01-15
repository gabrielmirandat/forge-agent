import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true, // Allow external connections
    watch: {
      usePolling: true,
      interval: 1000,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true, // Enable WebSocket proxying
        secure: false, // Allow self-signed certificates
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            // Ignore common WebSocket/proxy errors that are normal
            const errMsg = String(err.message || err);
            const errCode = (err as any).code;
            
            // Ignore EPIPE, ECONNRESET, and other common socket errors
            // These happen when clients disconnect or backend is not available
            if (
              errCode === 'EPIPE' ||
              errCode === 'ECONNRESET' ||
              errCode === 'ECONNREFUSED' ||
              errMsg.indexOf('socket') !== -1 ||
              errMsg.indexOf('EPIPE') !== -1 ||
              errMsg.indexOf('ECONNRESET') !== -1
            ) {
              return; // Silently ignore
            }
            
            // Only log unexpected errors
            console.error('Proxy error:', err);
          });
          
          // Handle WebSocket upgrade errors
          proxy.on('proxyReqWs', (proxyReq, req, socket) => {
            socket.on('error', (err) => {
              // Ignore socket errors during WebSocket upgrade
              const errCode = (err as any).code;
              if (errCode === 'EPIPE' || errCode === 'ECONNRESET') {
                return;
              }
            });
          });
        },
      },
    },
  },
})
