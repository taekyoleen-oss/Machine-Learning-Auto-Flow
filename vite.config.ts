import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    // 환경 변수 로드 (.env, .env.local, .env.[mode], .env.[mode].local 모두 로드)
    const env = loadEnv(mode, process.cwd(), '');
    const isProduction = mode === 'production';
    
    // API 키 확인 (디버깅용)
    const apiKey = env.GEMINI_API_KEY || env.API_KEY;
    if (!apiKey && mode === 'development') {
      console.warn('⚠️  GEMINI_API_KEY가 설정되지 않았습니다. .env.local 파일에 GEMINI_API_KEY를 설정해주세요.');
    }
    
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
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY || env.API_KEY || ''),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY || env.API_KEY || '')
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        include: ['xlsx']
      },
      build: {
        chunkSizeWarningLimit: 1000
      }
    };
});
