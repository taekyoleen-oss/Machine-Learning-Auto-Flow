import { PCA } from 'ml-pca';

/**
 * JavaScript 기반 PCA 계산 함수 (ml-pca 라이브러리 사용)
 * Pyodide 없이 브라우저에서 직접 실행 가능
 */
export function calculatePCA(
  data: Record<string, any>[],
  featureColumns: string[],
  nComponents: number = 2
): {
  coordinates: number[][];
  explainedVariance: number[];
  validIndices: number[];
} {
  // 데이터 검증
  if (!data || !Array.isArray(data) || data.length === 0) {
    throw new Error(`PCA: Input data is invalid. Type: ${typeof data}, IsArray: ${Array.isArray(data)}, Length: ${data?.length || 0}`);
  }
  
  if (!featureColumns || !Array.isArray(featureColumns) || featureColumns.length === 0) {
    throw new Error(`PCA: Feature columns are invalid. Type: ${typeof featureColumns}, IsArray: ${Array.isArray(featureColumns)}, Length: ${featureColumns?.length || 0}`);
  }

  // 데이터를 행렬로 변환 (유효한 행만 포함)
  const matrixData: number[][] = [];
  const validIndices: number[] = [];
  
  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    const featureValues: number[] = [];
    let isValid = true;
    
    for (const col of featureColumns) {
      const value = row[col];
      // 숫자형이고 유효한 값인지 확인
      if (typeof value === 'number' && !isNaN(value) && value !== null && value !== undefined && isFinite(value)) {
        featureValues.push(value);
      } else {
        isValid = false;
        break;
      }
    }
    
    // 모든 feature 값이 유효한 경우에만 추가
    if (isValid && featureValues.length === featureColumns.length) {
      matrixData.push(featureValues);
      validIndices.push(i);
    }
  }
  
  if (matrixData.length < 2) {
    throw new Error(`Need at least 2 valid samples for PCA, got ${matrixData.length}`);
  }
  
  if (matrixData[0].length < nComponents) {
    throw new Error(`Number of features (${matrixData[0].length}) must be >= n_components (${nComponents})`);
  }
  
  // PCA 계산
  const pca = new PCA(matrixData, { nComponents });
  
  // 투영된 좌표 (2D 또는 3D) - Matrix 객체를 2D 배열로 변환
  const projected = pca.predict(matrixData);
  const projectedArray = projected.to2DArray ? projected.to2DArray() : 
    (Array.isArray(projected) ? projected : 
     Array.from({ length: matrixData.length }, (_, i) => {
       const row = projected.getRow ? projected.getRow(i) : [];
       return Array.isArray(row) ? row : Array.from(row || []);
     }));
  
  // 설명된 분산 비율
  const explainedVariance = pca.getExplainedVariance();
  
  // 전체 데이터에 대한 좌표 배열 생성 (유효하지 않은 행은 NaN으로 채움)
  const coordinates: number[][] = [];
  let validIdx = 0;
  
  for (let i = 0; i < data.length; i++) {
    if (validIndices.includes(i)) {
      // 유효한 행: PCA 결과 사용
      const coord = projectedArray[validIdx];
      coordinates.push(Array.isArray(coord) ? coord : Array.from(coord || []));
      validIdx++;
    } else {
      // 유효하지 않은 행: NaN으로 채움
      coordinates.push(Array(nComponents).fill(NaN));
    }
  }
  
  return {
    coordinates,
    explainedVariance: explainedVariance.slice(0, nComponents),
    validIndices
  };
}

