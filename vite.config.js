import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': { // Proxy requests to /api to the Python server
        target: 'http://localhost:5000',  // Your Flask server address
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '') // Remove /api prefix
      }
    }
  },
  build: {
    assetsDir: 'static' // Serve static assets
  }
})