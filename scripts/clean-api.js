/**
 * 빌드 전에 api 폴더를 재귀적으로 찾아 삭제하는 스크립트
 * Vercel 빌드 캐시 문제 해결을 위해 사용
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const projectRoot = path.join(__dirname, '..');

// 재귀적으로 디렉토리 삭제 함수
function deleteDirectory(dirPath) {
  if (!fs.existsSync(dirPath)) {
    return;
  }

  try {
    const files = fs.readdirSync(dirPath);
    for (const file of files) {
      const filePath = path.join(dirPath, file);
      const stat = fs.statSync(filePath);
      if (stat.isDirectory()) {
        deleteDirectory(filePath);
      } else {
        fs.unlinkSync(filePath);
      }
    }
    fs.rmdirSync(dirPath);
    console.log(`✅ Deleted: ${dirPath}`);
  } catch (error) {
    console.warn(`⚠️  Could not delete ${dirPath}:`, error.message);
  }
}

// 모든 하위 디렉토리에서 api 폴더 찾기 및 삭제
function findAndDeleteApiFolders(dir, depth = 0) {
  // 깊이 제한 (무한 루프 방지)
  if (depth > 10) {
    return;
  }

  if (!fs.existsSync(dir)) {
    return;
  }

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      // 건너뛸 디렉토리
      if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name === '.git' || entry.name === '.vercel') {
        continue;
      }

      if (entry.isDirectory()) {
        if (entry.name === 'api') {
          console.log(`🗑️  Found and removing api folder at ${fullPath}`);
          deleteDirectory(fullPath);
        } else if (entry.name === 'pages') {
          const pagesApiPath = path.join(fullPath, 'api');
          if (fs.existsSync(pagesApiPath)) {
            console.log(`🗑️  Found and removing pages/api folder at ${pagesApiPath}`);
            deleteDirectory(pagesApiPath);
          }
          // pages 폴더 내부도 재귀적으로 확인
          findAndDeleteApiFolders(fullPath, depth + 1);
        } else {
          // 다른 디렉토리도 재귀적으로 확인
          findAndDeleteApiFolders(fullPath, depth + 1);
        }
      }
    }
  } catch (error) {
    // 무시
  }
}

// 직접 경로의 api 폴더 삭제
const apiDir = path.join(projectRoot, 'api');
if (fs.existsSync(apiDir)) {
  console.log(`🗑️  Removing api/ directory at ${apiDir}`);
  deleteDirectory(apiDir);
}

// pages/api 폴더 삭제
const pagesApiDir = path.join(projectRoot, 'pages', 'api');
if (fs.existsSync(pagesApiDir)) {
  console.log(`🗑️  Removing pages/api/ directory at ${pagesApiDir}`);
  deleteDirectory(pagesApiDir);
}

// pages 폴더가 비어있으면 삭제
const pagesDir = path.join(projectRoot, 'pages');
if (fs.existsSync(pagesDir)) {
  try {
    const pagesFiles = fs.readdirSync(pagesDir);
    if (pagesFiles.length === 0) {
      fs.rmdirSync(pagesDir);
      console.log(`✅ Deleted empty pages/ directory`);
    }
  } catch (error) {
    // 무시
  }
}

// .vercel/api 폴더 삭제
const vercelApiDir = path.join(projectRoot, '.vercel', 'api');
if (fs.existsSync(vercelApiDir)) {
  console.log(`🗑️  Removing .vercel/api/ directory at ${vercelApiDir}`);
  deleteDirectory(vercelApiDir);
}

// 프로젝트 루트에서 재귀적으로 api 폴더 찾기 및 삭제
console.log('🔍 Searching for api folders recursively...');
findAndDeleteApiFolders(projectRoot);

console.log('✅ API folder cleanup completed');
