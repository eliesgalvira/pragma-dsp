import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

const dspRoot = path.resolve(__dirname, '../../src')

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      'pragma-dsp/xform/fourier-fluent': path.join(dspRoot, 'xform/fourier-fluent.ts'),
      'pragma-dsp/xform/fourier': path.join(dspRoot, 'xform/fourier.ts'),
      'pragma-dsp/xform/stft': path.join(dspRoot, 'xform/stft.ts'),
      'pragma-dsp/math/complex': path.join(dspRoot, 'math/index.ts'),
      'pragma-dsp/analysis': path.join(dspRoot, 'analysis/index.ts'),
      'pragma-dsp/fluent': path.join(dspRoot, 'fluent/index.ts'),
      'pragma-dsp/core': path.join(dspRoot, 'core/index.ts'),
      'pragma-dsp': path.join(dspRoot, 'index.ts'),
    },
  },
})
