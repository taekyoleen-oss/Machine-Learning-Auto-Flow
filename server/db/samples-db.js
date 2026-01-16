/**
 * SQLite 데이터베이스 초기화 및 관리 모듈
 */

import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 데이터베이스 디렉토리 생성
const dbDir = path.join(__dirname, '..', '..', 'database');
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

const dbPath = path.join(dbDir, 'samples.db');
const db = new Database(dbPath);

// WAL 모드 활성화 (성능 향상)
db.pragma('journal_mode = WAL');

// 테이블 생성
db.exec(`
  CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    input_data TEXT,
    description TEXT,
    file_content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE INDEX IF NOT EXISTS idx_samples_name ON samples(name);
  CREATE INDEX IF NOT EXISTS idx_samples_filename ON samples(filename);
  CREATE INDEX IF NOT EXISTS idx_samples_created_at ON samples(created_at);
`);

// updated_at 자동 업데이트 트리거
db.exec(`
  CREATE TRIGGER IF NOT EXISTS update_samples_timestamp 
  AFTER UPDATE ON samples
  BEGIN
    UPDATE samples SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
  END;
`);

console.log(`Database initialized at: ${dbPath}`);

export default db;
