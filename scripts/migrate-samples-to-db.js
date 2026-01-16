/**
 * 기존 파일 기반 샘플을 DB로 마이그레이션하는 스크립트
 */

import db from '../server/db/samples-db.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const samplesDir = path.join(__dirname, '..', 'samples');
const metadataPath = path.join(samplesDir, 'samples-metadata.json');

console.log('Starting migration from files to database...');

// 메타데이터 로드
let metadata = {};
if (fs.existsSync(metadataPath)) {
  try {
    metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
    console.log(`Loaded metadata for ${Object.keys(metadata).length} samples`);
  } catch (error) {
    console.warn('Failed to load metadata:', error.message);
  }
}

// 샘플 파일 읽기 및 DB에 저장
if (!fs.existsSync(samplesDir)) {
  console.error(`Samples directory not found: ${samplesDir}`);
  process.exit(1);
}

const files = fs.readdirSync(samplesDir)
  .filter(f => f.endsWith('.ins') || f.endsWith('.json'))
  .filter(f => f !== 'samples-metadata.json' && f !== 'README.md' && f !== 'FILE_FORMAT_GUIDE.md');

console.log(`Found ${files.length} sample files to migrate`);

let successCount = 0;
let skipCount = 0;
let errorCount = 0;

for (const file of files) {
  const filePath = path.join(samplesDir, file);
  
  try {
    // 이미 DB에 있는지 확인
    const existing = db.prepare('SELECT id FROM samples WHERE filename = ?').get(file);
    if (existing) {
      console.log(`⏭  Skipped (already exists): ${file}`);
      skipCount++;
      continue;
    }
    
    const content = fs.readFileSync(filePath, 'utf-8');
    
    if (!content || content.trim().length === 0) {
      console.warn(`⚠  Skipped (empty): ${file}`);
      skipCount++;
      continue;
    }
    
    const data = JSON.parse(content);
    
    // 파일명에서 확장자 추출
    const name = data.name || data.projectName || file.replace(/\.(ins|json)$/i, '');
    const fileMetadata = metadata[file] || {};
    
    // DB에 저장
    const result = db.prepare(`
      INSERT INTO samples (filename, name, input_data, description, file_content)
      VALUES (?, ?, ?, ?, ?)
    `).run(
      file,
      name,
      fileMetadata.inputData || fileMetadata.input_data || '',
      fileMetadata.description || '',
      content
    );
    
    console.log(`✓ Migrated: ${file} (ID: ${result.lastInsertRowid})`);
    successCount++;
  } catch (error) {
    console.error(`✗ Failed to migrate ${file}:`, error.message);
    errorCount++;
  }
}

console.log('\nMigration completed!');
console.log(`  ✓ Success: ${successCount}`);
console.log(`  ⏭  Skipped: ${skipCount}`);
console.log(`  ✗ Errors: ${errorCount}`);
console.log(`  Total: ${files.length}`);

// 최종 통계
const totalInDb = db.prepare('SELECT COUNT(*) as count FROM samples').get();
console.log(`\nTotal samples in database: ${totalInDb.count}`);
