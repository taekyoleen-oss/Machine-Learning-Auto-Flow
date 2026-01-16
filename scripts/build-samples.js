/**
 * 빌드 시점에 samples와 Examples_in_Load 폴더의 파일들을 JSON으로 변환하여 public 폴더에 저장
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SAMPLES_DIR = path.join(__dirname, '..', 'samples');
const EXAMPLES_DIR = path.join(__dirname, '..', 'Examples_in_Load');
const PUBLIC_DIR = path.join(__dirname, '..', 'public');

// public 폴더가 없으면 생성
if (!fs.existsSync(PUBLIC_DIR)) {
  fs.mkdirSync(PUBLIC_DIR, { recursive: true });
}

console.log('Building samples and examples JSON files...');

// 0. 메타데이터 파일 읽기
const metadataPath = path.join(SAMPLES_DIR, 'samples-metadata.json');
let metadata = {};
if (fs.existsSync(metadataPath)) {
  try {
    const metadataContent = fs.readFileSync(metadataPath, 'utf-8');
    metadata = JSON.parse(metadataContent);
    console.log(`Loaded metadata for ${Object.keys(metadata).length} samples`);
  } catch (error) {
    console.warn(`Failed to load metadata file: ${error.message}`);
  }
}

// 1. Samples 폴더 처리
const samplesList = [];
if (fs.existsSync(SAMPLES_DIR)) {
  const files = fs.readdirSync(SAMPLES_DIR);
  console.log(`Found ${files.length} files in samples directory`);

  const sampleFiles = files
    .filter((file) => {
      // samples-metadata.json은 제외
      if (file === 'samples-metadata.json' || file === 'README.md' || file === 'FILE_FORMAT_GUIDE.md') {
        return false;
      }
      return file.endsWith('.json') || file.endsWith('.ins');
    })
    .map((file) => {
      const filePath = path.join(SAMPLES_DIR, file);
      try {
        const content = fs.readFileSync(filePath, 'utf-8');

        if (!content || content.trim().length === 0) {
          console.warn(`File ${file} is empty`);
          return null;
        }

        const data = JSON.parse(content);

        // .ins 파일 형식 변환
        if (
          file.endsWith('.ins') &&
          data.modules &&
          data.connections
        ) {
          const ext = '.ins';
          const projectName =
            data.name || data.projectName || file.replace(ext, '');

          // 메타데이터 병합
          const fileMetadata = metadata[file] || {};
          return {
            filename: file,
            name: projectName,
            inputData: fileMetadata.inputData || '',
            description: fileMetadata.description || '',
            category: fileMetadata.category || '머신러닝',
            data: {
              name: projectName,
              modules: data.modules.map((m) => ({
                type: m.type,
                position: m.position || { x: 0, y: 0 },
                name: m.name || m.type,
                parameters: m.parameters || {},
              })),
              connections: data.connections
                .map((c) => {
                  const fromIndex = data.modules.findIndex(
                    (m) => m.id === c.from.moduleId
                  );
                  const toIndex = data.modules.findIndex(
                    (m) => m.id === c.to.moduleId
                  );
                  if (fromIndex < 0 || toIndex < 0) {
                    return null;
                  }
                  return {
                    fromModuleIndex: fromIndex,
                    fromPort: c.from.portName,
                    toModuleIndex: toIndex,
                    toPort: c.to.portName,
                  };
                })
                .filter((c) => c !== null),
            },
          };
        }

        // .json 파일 형식
        // 메타데이터 병합
        const fileMetadata = metadata[file] || {};
        return {
          filename: file,
          name:
            data.name ||
            file.replace('.json', '').replace('.ins', ''),
          inputData: fileMetadata.inputData || '',
          description: fileMetadata.description || '',
          category: fileMetadata.category || '머신러닝',
          data: data,
        };
      } catch (error) {
        console.error(`Error reading file ${file}:`, error.message);
        return null;
      }
    })
    .filter((file) => file !== null);

  // 샘플 목록 정렬
  const sortOrder = [
    'Linear Regression',
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Neural Network',
    'SVM',
    'KNN',
    'K Means',
    'Naive Bayes',
    'LDA',
    'GLM Model',
    'Stat Model',
  ];

  sampleFiles.sort((a, b) => {
    const nameA = a.name || '';
    const nameB = b.name || '';

    const indexA = sortOrder.findIndex((order) => nameA.includes(order));
    const indexB = sortOrder.findIndex((order) => nameB.includes(order));

    if (indexA !== -1 && indexB !== -1) {
      return indexA - indexB;
    }
    if (indexA !== -1) {
      return -1;
    }
    if (indexB !== -1) {
      return 1;
    }
    return nameA.localeCompare(nameB);
  });

  samplesList.push(...sampleFiles);
  console.log(`Processed ${samplesList.length} sample files`);
} else {
  console.warn(`Samples directory does not exist: ${SAMPLES_DIR}`);
}

// 2. Examples_in_Load 폴더 처리
const examplesList = [];
if (fs.existsSync(EXAMPLES_DIR)) {
  const files = fs.readdirSync(EXAMPLES_DIR);
  console.log(`Found ${files.length} files in Examples_in_Load directory`);

  const csvFiles = files
    .filter((file) => file.endsWith('.csv'))
    .map((file) => {
      const filePath = path.join(EXAMPLES_DIR, file);
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        return {
          name: file,
          filename: file,
          content: content,
        };
      } catch (error) {
        console.error(`Error reading file ${file}:`, error.message);
        return null;
      }
    })
    .filter((file) => file !== null);

  examplesList.push(...csvFiles);
  console.log(`Processed ${examplesList.length} example CSV files`);
} else {
  console.warn(`Examples_in_Load directory does not exist: ${EXAMPLES_DIR}`);
}

// 3. JSON 파일로 저장
const samplesJsonPath = path.join(PUBLIC_DIR, 'samples.json');
const examplesJsonPath = path.join(PUBLIC_DIR, 'examples-in-load.json');

fs.writeFileSync(
  samplesJsonPath,
  JSON.stringify(samplesList, null, 2),
  'utf-8'
);
console.log(`Saved ${samplesList.length} samples to ${samplesJsonPath}`);

// JSON 파일로 저장 (압축 없이 저장하여 파일 크기 최소화)
const examplesJsonContent = JSON.stringify(examplesList);
fs.writeFileSync(
  examplesJsonPath,
  examplesJsonContent,
  'utf-8'
);
console.log(`Saved ${examplesList.length} examples to ${examplesJsonPath}`);

// 빌드된 파일이 제대로 생성되었는지 확인
if (fs.existsSync(examplesJsonPath)) {
  const stats = fs.statSync(examplesJsonPath);
  console.log(`✓ examples-in-load.json file created: ${stats.size} bytes`);
  const content = JSON.parse(fs.readFileSync(examplesJsonPath, 'utf-8'));
  console.log(`✓ File contains ${Array.isArray(content) ? content.length : 0} examples`);
  if (Array.isArray(content)) {
    const bostonHousing = content.find((ex) => ex.filename === 'BostonHousing.csv');
    if (bostonHousing) {
      console.log(`✓ BostonHousing.csv found in examples-in-load.json`);
    } else {
      console.warn(`⚠ BostonHousing.csv NOT found in examples-in-load.json`);
    }
    // 모든 파일 이름 출력
    console.log(`✓ Example files: ${content.map((ex) => ex.filename).join(', ')}`);
  }
} else {
  console.error(`✗ ERROR: examples-in-load.json file was not created!`);
  process.exit(1);
}

console.log('Build completed successfully!');
