import { CanvasModule, Connection } from '../types';
import { getModuleCode } from '../codeSnippets';

/**
 * Unicode 안전 Base64 인코딩 (브라우저 btoa는 멀티바이트 문자 처리 불가)
 */
function encodeBase64(str: string): string {
  const bytes = new TextEncoder().encode(str);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * LoadData 모듈의 outputData를 Pyodide 실행용 데이터 주입 코드로 변환
 * - outputData가 있으면 Base64 인코딩하여 pd.DataFrame()으로 주입
 * - outputData가 없으면 모듈을 먼저 실행하라는 에러 코드 생성
 */
export function generateLoadDataInjectionCode(module: CanvasModule): string {
  const outputData = module.outputData as any;
  const source = String(module.parameters?.source || 'data.csv');

  if (!outputData || outputData.type !== 'DataPreview' || !outputData.rows?.length) {
    return [
      `# ⚠️  Pyodide 실행: 로컬 파일 접근 불가 (브라우저 환경 제한)`,
      `# 해결 방법: 캔버스에서 LoadData 모듈을 먼저 실행(▶)한 후 다시 시도하세요.`,
      `# 외부 Python(Jupyter, 스크립트)에서는: dataframe = pd.read_csv('${source}')`,
      `raise RuntimeError("LoadData 모듈을 먼저 실행해주세요. 모듈 선택 후 ▶ 버튼을 클릭하세요.")`,
    ].join('\n');
  }

  const rows = outputData.rows as Record<string, any>[];
  const totalCount: number = outputData.totalRowCount || rows.length;
  const columns: string[] = (outputData.columns as any[])?.map((c: any) => c.name)
    ?? Object.keys(rows[0] ?? {});
  const isPartial = rows.length < totalCount;

  // 컬럼별 배열 구조로 변환 (pd.DataFrame() 직접 주입에 최적)
  const colData: Record<string, any[]> = {};
  columns.forEach((col) => {
    colData[col] = rows.map((row) => row[col] ?? null);
  });

  const b64 = encodeBase64(JSON.stringify(colData));

  const label = isPartial
    ? `앱에서 로드된 데이터 (${rows.length}행 / 전체 ${totalCount}행 미리보기)`
    : `앱에서 로드된 데이터 (전체 ${rows.length}행)`;

  return [
    `import base64 as _b64, json as _json`,
    `# Pyodide 실행: ${label}`,
    `# 외부 Python(Jupyter, 스크립트)에서는: dataframe = pd.read_csv('${source}')`,
    `dataframe = pd.DataFrame(_json.loads(_b64.b64decode('${b64}').decode('utf-8')))`,
  ].join('\n');
}

/**
 * 모듈 간 의존성을 기반으로 실행 순서를 결정합니다 (위상 정렬)
 * 순환 의존성 감지 시 에러를 throw합니다.
 */
function getExecutionOrder(
  modules: CanvasModule[],
  connections: Connection[]
): CanvasModule[] {
  const moduleMap = new Map<string, CanvasModule>();
  modules.forEach((m) => moduleMap.set(m.id, m));

  const dependencies = new Map<string, Set<string>>();
  modules.forEach((m) => dependencies.set(m.id, new Set()));

  connections.forEach((conn) => {
    const fromModule = moduleMap.get(conn.from.moduleId);
    const toModule = moduleMap.get(conn.to.moduleId);
    if (fromModule && toModule) {
      const deps = dependencies.get(toModule.id);
      if (deps) {
        deps.add(fromModule.id);
      }
    }
  });

  const ordered: CanvasModule[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  function visit(moduleId: string, path: string[] = []) {
    if (visiting.has(moduleId)) {
      // 순환 경로를 이름으로 표시하여 에러 발생
      const cycleStart = path.indexOf(moduleId);
      const cyclePath = path.slice(cycleStart).map((id) => moduleMap.get(id)?.name ?? id);
      cyclePath.push(moduleMap.get(moduleId)?.name ?? moduleId);
      throw new Error(`순환 연결이 감지되었습니다: ${cyclePath.join(' → ')}\n파이프라인의 연결 방향을 확인해주세요.`);
    }
    if (visited.has(moduleId)) return;

    visiting.add(moduleId);
    const deps = dependencies.get(moduleId);
    if (deps) {
      deps.forEach((depId) => visit(depId, [...path, moduleId]));
    }
    visiting.delete(moduleId);
    visited.add(moduleId);

    const module = moduleMap.get(moduleId);
    if (module) {
      ordered.push(module);
    }
  }

  modules.forEach((m) => visit(m.id));
  return ordered;
}

/**
 * 모듈 타입별 내부 출력 변수명 매핑
 */
const MODULE_OUTPUT_VAR: Record<string, string> = {
  LoadData: 'dataframe',
  SelectData: 'selected_data',
  DataFiltering: 'dataframe',
  HandleMissingValues: 'cleaned_data',
  EncodeCategorical: 'encoded_data',
  ScalingTransform: 'normalized_data',
  TransformData: 'transformed_data',
  TransitionData: 'transformed_data',
  ResampleData: 'resampled_data',
  TrainModel: 'trained_model',
  ScoreModel: 'scored_data',
  EvaluateModel: 'evaluation_metrics',
  ResultModel: 'model_results',
  PredictModel: 'predicted_data',
  KNN: 'trained_model',
  OutlierDetector: 'dataframe',
  NormalityChecker: 'dataframe',
  HypothesisTesting: 'dataframe',
  Correlation: 'dataframe',
  Join: 'dataframe',
  Concat: 'dataframe',
};

/**
 * 전체 파이프라인의 Python 코드를 생성합니다
 * @param forExecution true이면 Pyodide 실행용 (LoadData에 outputData 주입)
 */
export function generateFullPipelineCode(
  modules: CanvasModule[],
  connections: Connection[],
  forExecution = false
): string {
  if (modules.length === 0) {
    return '# 파이프라인이 비어있습니다.';
  }

  // 순환 의존성은 RuntimeError로 전달하여 사용자가 인지할 수 있도록 함
  let executionOrder: CanvasModule[];
  try {
    executionOrder = getExecutionOrder(modules, connections);
  } catch (e: any) {
    throw new Error(e?.message || '파이프라인 순서 결정 중 오류가 발생했습니다.');
  }
  const moduleMap = new Map<string, CanvasModule>();
  modules.forEach((m) => moduleMap.set(m.id, m));

  const codeLines: string[] = [];
  codeLines.push('# ============================================================================');
  codeLines.push('# 전체 파이프라인 실행 코드');
  codeLines.push('# 이 코드는 Jupyter Notebook, Python 스크립트 등 외부 환경에서도 실행 가능합니다.');
  codeLines.push('# ============================================================================');
  codeLines.push('');
  codeLines.push('import pandas as pd');
  codeLines.push('import numpy as np');
  codeLines.push('');

  const variableMap = new Map<string, string>(); // moduleId -> outputVarName

  executionOrder.forEach((module, index) => {
    // ── 모듈 섹션 헤더 ──────────────────────────────────────────────────────
    codeLines.push('# ============================================================================');
    codeLines.push(`# [모듈 ${index + 1}/${executionOrder.length}] ${module.name}`);
    codeLines.push(`# 타입: ${module.type}`);
    codeLines.push('# ============================================================================');
    codeLines.push('');

    // ── 모듈 코드 생성 ──────────────────────────────────────────────────────
    let moduleCode: string;
    if (forExecution && module.type === 'LoadData') {
      // Pyodide 실행 시: 로컬 파일 대신 outputData를 직접 주입
      moduleCode = generateLoadDataInjectionCode(module);
    } else {
      try {
        moduleCode = getModuleCode(module, modules, connections);
      } catch (e: any) {
        // 코드 생성 실패 시 주석으로 숨기지 않고 Python 실행 시점에 명확한 에러 발생
        const errMsg = e?.message || String(e);
        moduleCode = [
          `# ❌ [${module.name}] 코드 생성 실패: ${errMsg}`,
          `raise RuntimeError("[${module.name}] 코드 생성에 실패했습니다: ${errMsg.replace(/"/g, "'")}")`,
        ].join('\n');
      }
    }

    // ── 출력 변수명 결정 ────────────────────────────────────────────────────
    const safeId = module.id.slice(0, 8).replace(/-/g, '_');
    let outputVarName = '';
    if (module.outputs.length > 0) {
      const outputType = module.outputs[0].type;
      if (outputType === 'data') {
        outputVarName = `data_${safeId}`;
      } else if (outputType === 'model') {
        outputVarName = `model_${safeId}`;
      } else if (outputType === 'handler') {
        outputVarName = `handler_${safeId}`;
      } else if (outputType === 'evaluation') {
        outputVarName = `eval_${safeId}`;
      } else {
        outputVarName = `result_${safeId}`;
      }
    }

    // ── 입력 연결 처리: 이전 모듈 출력을 표준 변수명으로 받음 ───────────────
    const inputConnections = connections.filter((c) => c.to.moduleId === module.id);
    const inputPrefixLines: string[] = [];

    // data_in 포트에 연결된 변수들을 순서대로 추적 (Join/Concat의 두 번째 데이터 포트 대응)
    const dataInVars: string[] = [];

    inputConnections.forEach((conn) => {
      const fromModule = moduleMap.get(conn.from.moduleId);
      const prevVarName = variableMap.get(conn.from.moduleId);
      if (!prevVarName || !fromModule) return;

      const toPort = conn.to.portName;
      const fromLabel = fromModule.name;

      if (toPort === 'data_in') {
        dataInVars.push(prevVarName);
      } else if (toPort === 'data_in2') {
        // Join/Concat의 명시적 두 번째 데이터 포트
        inputPrefixLines.push(`dataframe2 = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'model_in') {
        inputPrefixLines.push(`trained_model = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'score_data_in') {
        // ScoreModel의 데이터 포트 (모델과 분리)
        inputPrefixLines.push(`second_data = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'handler_in') {
        inputPrefixLines.push(`handler = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'dist_in') {
        inputPrefixLines.push(`dist = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'curve_in') {
        inputPrefixLines.push(`curve = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'contract_in') {
        inputPrefixLines.push(`contract = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'eval_in') {
        inputPrefixLines.push(`scored_data = ${prevVarName}  # ← [${fromLabel}] 출력`);
      }
    });

    // data_in 포트 처리: 첫 번째는 dataframe, 두 번째는 dataframe2 (Join/Concat 템플릿 기준)
    if (dataInVars.length >= 1) {
      const firstFrom = inputConnections.find((c) => c.to.portName === 'data_in');
      const firstName = firstFrom ? moduleMap.get(firstFrom.from.moduleId)?.name ?? '' : '';
      inputPrefixLines.unshift(`dataframe = ${dataInVars[0]}  # ← [${firstName}] 출력`);
    }
    if (dataInVars.length >= 2) {
      const secondFrom = inputConnections.filter((c) => c.to.portName === 'data_in')[1];
      const secondName = secondFrom ? moduleMap.get(secondFrom.from.moduleId)?.name ?? '' : '';
      inputPrefixLines.push(`dataframe2 = ${dataInVars[1]}  # ← [${secondName}] 출력`);
    }

    // 입력 연결이 있으면 모듈 코드 앞에 주입
    if (inputPrefixLines.length > 0) {
      codeLines.push('# 이전 모듈 출력을 입력으로 받음');
      inputPrefixLines.forEach((line) => codeLines.push(line));
      codeLines.push('');
    }

    codeLines.push(moduleCode);

    // ── 출력 변수 명시적 할당 ───────────────────────────────────────────────
    if (outputVarName) {
      if (module.type === 'SplitData') {
        codeLines.push(`${outputVarName}_train = train_data`);
        codeLines.push(`${outputVarName}_test = test_data`);
      } else {
        const internalVar = MODULE_OUTPUT_VAR[module.type] || '';
        if (internalVar) {
          codeLines.push(`${outputVarName} = ${internalVar}`);
        }
      }
    }

    // ── 실행 결과 확인용 print ──────────────────────────────────────────────
    if (outputVarName) {
      const outputType = module.outputs.length > 0 ? module.outputs[0].type : '';
      if (module.type === 'SplitData') {
        codeLines.push(`print(f"[${module.name}] train: {${outputVarName}_train.shape}, test: {${outputVarName}_test.shape}")`);
        codeLines.push(`print()`);
      } else if (outputType === 'data') {
        codeLines.push(`print(f"[${module.name}] shape: {${outputVarName}.shape}")`);
        codeLines.push(`print(${outputVarName}.head(5).to_string())`);
        codeLines.push(`print()`);
      } else if (outputType === 'model') {
        codeLines.push(`print(f"[${module.name}] 모델: {${outputVarName}}")`);
        codeLines.push(`print()`);
      } else {
        codeLines.push(`print(f"[${module.name}] 완료")`);
        codeLines.push(`print(${outputVarName})`);
        codeLines.push(`print()`);
      }
    }

    codeLines.push('');

    // 출력 변수명을 다음 모듈을 위해 저장
    if (outputVarName) {
      variableMap.set(module.id, outputVarName);
      // SplitData는 train/test 쌍으로 저장 (data_in 연결 시 train을 기본값으로 사용)
      if (module.type === 'SplitData') {
        variableMap.set(module.id, `${outputVarName}_train`);
      }
    }
  });

  codeLines.push('# ============================================================================');
  codeLines.push('# 파이프라인 실행 완료');
  codeLines.push('# ============================================================================');

  return codeLines.join('\n');
}
