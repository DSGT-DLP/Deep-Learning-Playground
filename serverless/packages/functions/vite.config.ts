import { defineConfig } from 'vite'
import path from 'path';


export default defineConfig({
  resolve: {
    alias: {
      '@dlp-sst-app': path.resolve(__dirname, './Deep-Learning-Playground/serverless')
    }
  }
})