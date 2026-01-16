/**
 * 샘플 관리 API 라우터
 */

import express from 'express';
import multer from 'multer';
import db from '../db/samples-db.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// 업로드 디렉토리 생성
const uploadDir = path.join(__dirname, '..', '..', 'temp', 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const upload = multer({ 
  dest: uploadDir,
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB 제한
});

// 모든 샘플 조회
router.get('/', (req, res) => {
  try {
    const samples = db.prepare(`
      SELECT id, filename, name, input_data, description, 
             created_at, updated_at
      FROM samples
      ORDER BY created_at DESC
    `).all();
    
    res.json(samples);
  } catch (error) {
    console.error('Error fetching samples:', error);
    res.status(500).json({ error: error.message });
  }
});

// 특정 샘플 조회
router.get('/:id', (req, res) => {
  try {
    const sample = db.prepare(`
      SELECT * FROM samples WHERE id = ?
    `).get(parseInt(req.params.id));
    
    if (!sample) {
      return res.status(404).json({ error: 'Sample not found' });
    }
    
    // JSON 파싱
    try {
      sample.file_content = JSON.parse(sample.file_content);
    } catch (parseError) {
      console.error('Error parsing file_content:', parseError);
      sample.file_content = null;
    }
    
    res.json(sample);
  } catch (error) {
    console.error('Error fetching sample:', error);
    res.status(500).json({ error: error.message });
  }
});

// 샘플 생성 (JSON 직접 전송)
router.post('/', (req, res) => {
  try {
    const { filename, name, input_data, description, file_content } = req.body;
    
    if (!filename || !name || !file_content) {
      return res.status(400).json({ 
        error: 'filename, name, and file_content are required' 
      });
    }
    
    const result = db.prepare(`
      INSERT INTO samples (filename, name, input_data, description, file_content)
      VALUES (?, ?, ?, ?, ?)
    `).run(
      filename,
      name,
      input_data || '',
      description || '',
      JSON.stringify(file_content)
    );
    
    res.status(201).json({ 
      id: result.lastInsertRowid,
      message: 'Sample created successfully' 
    });
  } catch (error) {
    console.error('Error creating sample:', error);
    if (error.message.includes('UNIQUE constraint')) {
      return res.status(409).json({ error: 'Sample with this filename already exists' });
    }
    res.status(500).json({ error: error.message });
  }
});

// 파일에서 샘플 가져오기
router.post('/import', upload.single('file'), (req, res) => {
  let tempFilePath = null;
  
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    tempFilePath = req.file.path;
    const fileContent = fs.readFileSync(tempFilePath, 'utf-8');
    let data;
    
    try {
      data = JSON.parse(fileContent);
    } catch (parseError) {
      // 임시 파일 삭제
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      return res.status(400).json({ error: 'Invalid JSON file' });
    }
    
    // 파일명에서 확장자 추출
    const filename = req.file.originalname;
    const name = data.name || data.projectName || filename.replace(/\.(ins|json)$/i, '');
    
    // 메타데이터 읽기 (선택사항)
    const metadataPath = path.join(__dirname, '..', '..', 'samples', 'samples-metadata.json');
    let metadata = {};
    if (fs.existsSync(metadataPath)) {
      try {
        metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
      } catch (metaError) {
        console.warn('Failed to load metadata:', metaError);
      }
    }
    const fileMetadata = metadata[filename] || {};
    
    // DB에 저장
    const result = db.prepare(`
      INSERT INTO samples (filename, name, input_data, description, file_content)
      VALUES (?, ?, ?, ?, ?)
    `).run(
      filename,
      name,
      fileMetadata.inputData || fileMetadata.input_data || '',
      fileMetadata.description || '',
      fileContent
    );
    
    // 임시 파일 삭제
    if (fs.existsSync(tempFilePath)) {
      fs.unlinkSync(tempFilePath);
    }
    
    res.status(201).json({ 
      id: result.lastInsertRowid,
      message: 'Sample imported successfully' 
    });
  } catch (error) {
    console.error('Error importing sample:', error);
    
    // 임시 파일 정리
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try {
        fs.unlinkSync(tempFilePath);
      } catch (cleanupError) {
        console.error('Failed to cleanup temp file:', cleanupError);
      }
    }
    
    if (error.message.includes('UNIQUE constraint')) {
      return res.status(409).json({ error: 'Sample with this filename already exists' });
    }
    
    res.status(500).json({ error: error.message });
  }
});

// 샘플 수정
router.put('/:id', (req, res) => {
  try {
    const { name, input_data, description, file_content } = req.body;
    
    // 기존 샘플 확인
    const existing = db.prepare('SELECT id FROM samples WHERE id = ?')
      .get(parseInt(req.params.id));
    
    if (!existing) {
      return res.status(404).json({ error: 'Sample not found' });
    }
    
    const updateFields = [];
    const updateValues = [];
    
    if (name !== undefined) {
      updateFields.push('name = ?');
      updateValues.push(name);
    }
    if (input_data !== undefined) {
      updateFields.push('input_data = ?');
      updateValues.push(input_data);
    }
    if (description !== undefined) {
      updateFields.push('description = ?');
      updateValues.push(description);
    }
    if (file_content) {
      updateFields.push('file_content = ?');
      updateValues.push(JSON.stringify(file_content));
    }
    
    if (updateFields.length === 0) {
      return res.status(400).json({ error: 'No fields to update' });
    }
    
    updateValues.push(parseInt(req.params.id));
    
    const result = db.prepare(`
      UPDATE samples 
      SET ${updateFields.join(', ')}
      WHERE id = ?
    `).run(...updateValues);
    
    res.json({ message: 'Sample updated successfully' });
  } catch (error) {
    console.error('Error updating sample:', error);
    res.status(500).json({ error: error.message });
  }
});

// 샘플 삭제
router.delete('/:id', (req, res) => {
  try {
    const result = db.prepare('DELETE FROM samples WHERE id = ?')
      .run(parseInt(req.params.id));
    
    if (result.changes === 0) {
      return res.status(404).json({ error: 'Sample not found' });
    }
    
    res.json({ message: 'Sample deleted successfully' });
  } catch (error) {
    console.error('Error deleting sample:', error);
    res.status(500).json({ error: error.message });
  }
});

export default router;
