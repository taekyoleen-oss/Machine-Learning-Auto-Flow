import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const isProduction = mode === 'production';
    return {
      publicDir: 'public',
      server: {
        port: 3001,
        // 프로덕션 빌드에서는 host 설정 제거 (로컬 네트워크 권한 요청 방지)
        ...(isProduction ? {} : { host: '0.0.0.0' }),
        proxy: {
          '/api/split-data': {
            target: 'http://localhost:3002',
            changeOrigin: true,
            secure: false,
          },
          '/api/generate-ppts': {
            target: 'http://localhost:3002',
            changeOrigin: true,
            secure: false,
          }
        }
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        include: ['xlsx']
      }
    };
});
