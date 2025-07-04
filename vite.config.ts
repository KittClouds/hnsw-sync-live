
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { livestoreDevtoolsPlugin } from '@livestore/devtools-vite'
import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  // Ensure Vite treats .wasm files as assets for proper serving
  assetsInclude: ['**/*.wasm'],
  plugins: [
    react(),
    livestoreDevtoolsPlugin({ schemaPath: './src/livestore/schema.ts' }),
    mode === 'development' && componentTagger(),
  ].filter(Boolean),
  worker: { format: 'es' },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}))
