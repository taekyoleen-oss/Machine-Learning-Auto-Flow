/**
 * SQLite 데이터베이스 초기화 및 관리 모듈
 * better-sqlite3 네이티브 바인딩이 없으면 db=null로 내보내고 서버는 정상 기동 (Supabase 사용 시 로컬 DB 불필요)
 */

import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let db = null;

try {
  const require = createRequire(import.meta.url);
  const Database = require("better-sqlite3");

  const dbDir = path.join(__dirname, "..", "..", "database");
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }

  const dbPath = path.join(dbDir, "samples.db");
  db = new Database(dbPath);

  db.pragma("journal_mode = WAL");

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

  try {
    const tableInfo = db.prepare(`PRAGMA table_info(samples)`).all();
    const hasCategory = tableInfo.some((col) => col.name === "category");

    if (!hasCategory) {
      db.exec(`ALTER TABLE samples ADD COLUMN category TEXT DEFAULT '머신러닝'`);
      db.exec(`UPDATE samples SET category = '머신러닝' WHERE category IS NULL`);
      db.exec(
        `CREATE INDEX IF NOT EXISTS idx_samples_category ON samples(category)`
      );
      console.log("Category column added to existing database");
    } else {
      const updateResult = db
        .prepare(
          `UPDATE samples SET category = '머신러닝' WHERE category IS NULL OR category = ''`
        )
        .run();
      if (updateResult.changes > 0) {
        console.log(
          `Updated ${updateResult.changes} samples with default category '머신러닝'`
        );
      }
    }
  } catch (err) {
    console.warn("Error checking/adding category column:", err.message);
  }

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS update_samples_timestamp 
    AFTER UPDATE ON samples
    BEGIN
      UPDATE samples SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
  `);

  console.log(`[samples-db] Database initialized at: ${dbPath}`);
} catch (err) {
  console.warn(
    "[samples-db] better-sqlite3 not available, samples API disabled. Use Supabase for samples or run: pnpm rebuild better-sqlite3",
    err.message
  );
}

export default db;
