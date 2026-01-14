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
            // Ignore WebSocket connection errors (they're normal when client disconnects)
            const errMsg = String(err.message || err);
            if (errMsg.indexOf('socket') !== -1) {
              return;
            }
            console.log('Proxy error:', err);
          });
        },
      },
    },
  },
})
