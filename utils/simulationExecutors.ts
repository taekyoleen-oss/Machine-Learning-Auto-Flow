/**
 * runSimulation 분해: 자기완결 분기 모듈 실행기 모음.
 * - 1단계: 통계·분석 패밀리 6종 (OutlierDetector·HypothesisTesting·NormalityChecker·
 *   VIFChecker·Correlation·Statistics)
 * - 2단계: 전처리·데이터 조작 패밀리 12종 (SelectData·DataFiltering·Recommender·
 *   ColumnPlot·HandleMissingValues·EncodeCategorical·ScalingTransform·PythonScript·
 *   FeatureEngineer·TransitionData·Join·Concat)
 * App.tsx runSimulation의 분기 본문을 "문자 그대로" 이동한 것으로,
 * 유일한 기계적 변경은 ① newOutputData의 지역화+return ② 동적 import 경로
 * ("./utils/pyodideRunner" → "./pyodideRunner") 뿐이다. 로직 변경 금지.
 * 각 실행기는 (module, getSingleInputData, addLog)만 참조하는 자기완결 분기만 대상.
 */
import {
  CanvasModule,
  StatsModelFamily,
  TrainedModelOutput,
  TrainedClusteringModelOutput,
  ClusteringDataOutput,
  Connection,
  ModuleType,
  ModelDefinitionOutput,
  ColumnInfo,
  DataPreview,
  HypothesisTestType,
  NormalityTestType,
  VIFCheckerOutput,
} from "../types";
import type { SetStateAction } from "react";

export type AddLog = (
  level: "INFO" | "SUCCESS" | "ERROR" | "WARN",
  message: string
) => void;

/** runSimulation 내부 헬퍼를 그대로 주입받는다(포트 언랩 로직 재사용). */
export type GetSingleInputData = (
  moduleId: string,
  portType?: any,
  portName?: string
) => any;

/** 대형 DI 분기용 주입 타입: 최신 modules 스냅샷 getter와 상태 updater(useHistoryState). */
export type GetCurrentModules = () => CanvasModule[];
export type SetModules = (
  action: SetStateAction<CanvasModule[]> | CanvasModule[],
  overwrite?: boolean
) => void;

/**
 * 모델 종류(분류/회귀) 판정 — 지도학습 대형 분기(EvaluateModel/ScoreModel/TrainModel)에서
 * 공유. App.tsx에서 이동한 순수 함수(ModuleType만 의존).
 */
export const isClassification = (
  modelType: ModuleType,
  modelPurpose?: "classification" | "regression"
): boolean => {
  const classificationTypes = [
    ModuleType.LogisticRegression,
    ModuleType.LDA,
    ModuleType.NaiveBayes,
  ];
  const dualPurposeTypes = [
    ModuleType.KNN,
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.GradientBoosting,
    ModuleType.NeuralNetwork,
    ModuleType.SVM,
  ];

  if (classificationTypes.includes(modelType)) {
    return true;
  }
  if (
    dualPurposeTypes.includes(modelType) &&
    modelPurpose === "classification"
  ) {
    return true;
  }
  return false;
};

/**
 * OutlierDetector 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeOutlierDetector(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { columns = [] } = module.parameters;

          if (!Array.isArray(columns) || columns.length === 0) {
            throw new Error(
              "At least one column must be selected for outlier detection."
            );
          }

          if (columns.length > 5) {
            throw new Error(
              "Maximum 5 columns can be selected for outlier detection."
            );
          }

          // 컬럼 확인
          const invalidColumns: string[] = [];
          const numericColumns: string[] = [];
          columns.forEach((colName: string) => {
            const col = inputData.columns.find((c) => c.name === colName);
            if (!col) {
              invalidColumns.push(colName);
            } else if (
              !(col.type.startsWith("int") || col.type.startsWith("float"))
            ) {
              invalidColumns.push(colName);
            } else {
              numericColumns.push(colName);
            }
          });

          if (invalidColumns.length > 0) {
            throw new Error(
              `Invalid columns: ${invalidColumns.join(
                ", "
              )}. All columns must be numeric.`
            );
          }

          // Pyodide를 사용하여 Python으로 각 열에 대해 이상치 탐지
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 이상치 탐지 중... (Columns: ${columns.join(
                ", "
              )})`
            );

            const pyodideModule = await import("./pyodideRunner");
            const { detectOutliers } = pyodideModule;

            const columnResults: Array<{
              column: string;
              results: Array<{
                method: "IQR" | "ZScore" | "IsolationForest" | "Boxplot";
                outlierIndices: number[];
                outlierCount: number;
                outlierPercentage: number;
                details?: Record<string, any>;
              }>;
              totalOutliers: number;
              outlierIndices: number[];
            }> = [];

            // 모든 열에 대해 이상치 탐지 수행
            for (const column of columns) {
              const result = await detectOutliers(
                inputData.rows || [],
                column,
                ["IQR", "ZScore", "IsolationForest", "Boxplot"], // 모든 방법 사용
                1.5, // IQR multiplier
                3, // Z-score threshold
                0.1, // Isolation Forest contamination
                120000 // 타임아웃: 120초
              );

              columnResults.push({
                column,
                results: result.results,
                totalOutliers: result.totalOutliers,
                outlierIndices: result.outlierIndices,
              });
            }

            // 모든 열에서 탐지된 이상치 인덱스 합집합
            const allOutlierIndicesSet = new Set<number>();
            columnResults.forEach((cr) => {
              cr.outlierIndices.forEach((idx) => allOutlierIndicesSet.add(idx));
            });
            const allOutlierIndices = Array.from(allOutlierIndicesSet).sort(
              (a, b) => a - b
            );

            // 원본 데이터 저장 (제거 작업을 위해 필요)
            const originalRows = inputData.rows || [];

            newOutputData = {
              type: "OutlierDetectorOutput",
              columns,
              columnResults,
              totalOutliers: allOutlierIndices.length,
              allOutlierIndices,
              originalData: originalRows,
              // 초기에는 cleanedData를 생성하지 않음 (사용자가 제거할 때 생성)
            };

            addLog(
              "SUCCESS",
              `Python으로 이상치 탐지 완료: ${columns.length}개 열에서 총 ${allOutlierIndices.length}개 이상치 행 발견`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Outlier Detection 실패: ${errorMessage}`);
            throw new Error(`이상치 탐지 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * HypothesisTesting 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeHypothesisTesting(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { tests = [] } = module.parameters;

          if (!Array.isArray(tests) || tests.length === 0) {
            throw new Error(
              "At least one test must be selected for hypothesis testing."
            );
          }

          // 각 테스트에 대해 열이 선택되었는지 확인
          const invalidTests: string[] = [];
          tests.forEach((test: any, index: number) => {
            if (!test.testType) {
              invalidTests.push(`Test ${index + 1}: missing testType`);
            } else if (
              !Array.isArray(test.columns) ||
              test.columns.length === 0
            ) {
              invalidTests.push(
                `Test ${index + 1} (${test.testType}): no columns selected`
              );
            }
          });

          if (invalidTests.length > 0) {
            throw new Error(
              `Invalid test configuration:\n${invalidTests.join("\n")}`
            );
          }

          // Pyodide를 사용하여 Python으로 가설 검정 수행
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 가설 검정 수행 중... (${tests.length}개 테스트)`
            );

            const pyodideModule = await import("./pyodideRunner");
            const { performHypothesisTests } = pyodideModule;

            const results = await performHypothesisTests(
              inputData.rows || [],
              tests,
              120000 // 타임아웃: 120초
            );

            newOutputData = {
              type: "HypothesisTestingOutput",
              results: results.map((r) => ({
                testType: r.testType as HypothesisTestType,
                testName: r.testName,
                columns: r.columns,
                statistic: r.statistic,
                pValue: r.pValue,
                degreesOfFreedom: r.degreesOfFreedom,
                criticalValue: r.criticalValue,
                conclusion: r.conclusion,
                interpretation: r.interpretation,
                details: r.details,
              })),
            };

            const successCount = results.filter(
              (r) => r.pValue !== undefined && !r.testName.startsWith("Error:")
            ).length;
            addLog(
              "SUCCESS",
              `Python으로 가설 검정 완료: ${successCount}/${results.length}개 테스트 성공`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Hypothesis Testing 실패: ${errorMessage}`);
            throw new Error(`가설 검정 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * NormalityChecker 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeNormalityChecker(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { column = "", tests = [] } = module.parameters;

          if (!column) {
            throw new Error("Column must be selected for normality checking.");
          }

          if (!Array.isArray(tests) || tests.length === 0) {
            throw new Error(
              "At least one test must be selected for normality checking."
            );
          }

          // 선택된 열이 입력 데이터에 있는지 확인
          const col = inputData.columns.find((c) => c.name === column);
          if (!col) {
            throw new Error(`Column '${column}' not found in input data.`);
          }

          if (!(col.type.startsWith("int") || col.type.startsWith("float"))) {
            throw new Error(
              `Column '${column}' must be numeric for normality checking.`
            );
          }

          // Pyodide를 사용하여 Python으로 정규성 검정 수행
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 정규성 검정 수행 중... (${tests.length}개 테스트)`
            );

            const pyodideModule = await import("./pyodideRunner");
            const { performNormalityCheck } = pyodideModule;

            const result = await performNormalityCheck(
              inputData.rows || [],
              column,
              tests,
              120000 // 타임아웃: 120초
            );

            newOutputData = {
              type: "NormalityCheckerOutput",
              column: result.column,
              skewness: result.skewness,
              kurtosis: result.kurtosis,
              jarqueBera: result.jarqueBera,
              testResults: result.testResults.map((r: any) => ({
                testType: r.testType as NormalityTestType,
                testName: r.testName,
                statistic: r.statistic,
                pValue: r.pValue,
                criticalValue: r.criticalValue,
                conclusion: r.conclusion,
                interpretation: r.interpretation,
                details: r.details,
              })),
              histogramImage: result.histogramImage,
              qqPlotImage: result.qqPlotImage,
              ecdfImage: result.ecdfImage,
              boxplotImage: result.boxplotImage,
            };

            const successCount = result.testResults.filter(
              (r: any) =>
                r.statistic !== undefined && !r.testName.startsWith("Error:")
            ).length;
            addLog(
              "SUCCESS",
              `Python으로 정규성 검정 완료: ${successCount}/${result.testResults.length}개 테스트 성공`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Normality Check 실패: ${errorMessage}`);
            throw new Error(`정규성 검정 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * VIFChecker 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeVIFChecker(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { feature_columns = [] } = module.parameters;

          if (!Array.isArray(feature_columns) || feature_columns.length < 2) {
            throw new Error(
              "At least 2 feature columns must be selected for VIF calculation."
            );
          }

          // 선택된 열이 입력 데이터에 있는지 확인
          const invalidColumns: string[] = [];
          feature_columns.forEach((colName: string) => {
            const col = inputData.columns.find((c) => c.name === colName);
            if (!col) {
              invalidColumns.push(colName);
            } else {
              // 숫자형 열인지 확인
              if (
                !col.type.startsWith("int") &&
                !col.type.startsWith("float")
              ) {
                invalidColumns.push(`${colName} (non-numeric: ${col.type})`);
              }
            }
          });

          if (invalidColumns.length > 0) {
            throw new Error(
              `Invalid or non-numeric columns: ${invalidColumns.join(", ")}`
            );
          }

          // Pyodide를 사용하여 Python으로 VIF 계산 수행
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 VIF 계산 수행 중... (${feature_columns.length}개 열)`
            );

            const pyodideModule = await import("./pyodideRunner");
            const { calculateVIFPython } = pyodideModule;

            const vifResults = await calculateVIFPython(
              inputData.rows || [],
              feature_columns,
              120000 // 타임아웃: 120초
            );

            newOutputData = {
              type: "VIFCheckerOutput",
              results: vifResults,
            } as VIFCheckerOutput;

            addLog("SUCCESS", `VIF 계산 완료 (${vifResults.length}개 열 분석)`);
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `VIF 계산 실패: ${errorMessage}`);
            throw new Error(`VIF calculation failed: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * Correlation 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeCorrelation(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { columns = [] } = module.parameters;

          if (!Array.isArray(columns) || columns.length === 0) {
            throw new Error(
              "At least one column must be selected for correlation analysis."
            );
          }

          // 선택된 열이 입력 데이터에 있는지 확인
          const invalidColumns: string[] = [];
          columns.forEach((colName: string) => {
            const col = inputData.columns.find((c) => c.name === colName);
            if (!col) {
              invalidColumns.push(colName);
            }
          });

          if (invalidColumns.length > 0) {
            throw new Error(`Invalid columns: ${invalidColumns.join(", ")}`);
          }

          // 숫자형과 범주형 열 분리
          const numericColumns: string[] = [];
          const categoricalColumns: string[] = [];
          columns.forEach((colName: string) => {
            const col = inputData.columns.find((c) => c.name === colName);
            if (col) {
              if (col.type.startsWith("int") || col.type.startsWith("float")) {
                numericColumns.push(colName);
              } else if (col.type === "object") {
                categoricalColumns.push(colName);
              }
            }
          });

          // Pyodide를 사용하여 Python으로 상관분석 수행
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 상관분석 수행 중... (${columns.length}개 열)`
            );

            const pyodideModule = await import("./pyodideRunner");
            const { performCorrelationAnalysis } = pyodideModule;

            const result = await performCorrelationAnalysis(
              inputData.rows || [],
              columns,
              numericColumns,
              categoricalColumns,
              120000 // 타임아웃: 120초
            );

            // 결과 검증
            if (!result) {
              throw new Error("Correlation analysis returned no result");
            }
            if (
              !result.correlationMatrices ||
              !Array.isArray(result.correlationMatrices)
            ) {
              throw new Error(
                "Correlation analysis returned invalid correlationMatrices"
              );
            }

            newOutputData = {
              type: "CorrelationOutput",
              columns,
              numericColumns,
              categoricalColumns,
              correlationMatrices: result.correlationMatrices || [],
              heatmapImage: result.heatmapImage,
              pairplotImage: result.pairplotImage,
              summary: result.summary || {},
            };

            const methodCount = result.correlationMatrices?.length || 0;
            addLog(
              "SUCCESS",
              `Python으로 상관분석 완료: ${methodCount}개 방법, ${numericColumns.length}개 숫자형, ${categoricalColumns.length}개 범주형 열 분석`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog(
              "ERROR",
              `Python Correlation Analysis 실패: ${errorMessage}`
            );
            throw new Error(`상관분석 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * Statistics 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeStatistics(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData || !inputData.rows) {
            throw new Error(
              "Input data not available or is of the wrong type."
            );
          }

          // Pyodide를 사용하여 Python으로 통계 계산
          try {
            addLog("INFO", "Pyodide를 사용하여 Python으로 통계 계산 중...");

            const pyodideModule = await import("./pyodideRunner");
            const { calculateStatisticsPython } = pyodideModule;

            const result = await calculateStatisticsPython(
              inputData.rows,
              inputData.columns,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "StatisticsOutput",
              stats: result.stats,
              correlation: result.correlation,
              columns: inputData.columns,
            };
            addLog("SUCCESS", "Python으로 통계 계산 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python 통계 계산 실패: ${errorMessage}`);
            throw new Error(`통계 계산 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * SelectData 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeSelectData(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (inputData) {
            const selections =
              (module.parameters.columnSelections as Record<
                string,
                { selected: boolean; type: string }
              >) || {};
            const isConfigured = Object.keys(selections).length > 0;

            // 디버깅: selections 상태 확인
            console.log("SelectData Debug:", {
              isConfigured,
              selectionsKeys: Object.keys(selections),
              selections,
              inputColumns: inputData.columns.map((c) => c.name),
            });

            const newColumns: ColumnInfo[] = [];
            inputData.columns.forEach((col) => {
              const selection = selections[col.name];
              // If the module is unconfigured, default to selecting all columns. Otherwise, respect the selection.
              let shouldInclude: boolean;
              if (!isConfigured) {
                // configured가 아니면 모든 열 선택 (기본 동작)
                shouldInclude = true;
              } else {
                // configured인 경우
                // selection이 없으면 기본적으로 선택된 것으로 간주 (새로 추가된 열 등)
                // selection이 있으면 selected 값에 따라 결정
                // selected가 명시적으로 false가 아니면 선택 (true 또는 undefined도 선택으로 간주)
                shouldInclude = selection ? selection.selected !== false : true;
              }

              if (shouldInclude) {
                // selection?.type이 있고 유효한 값이면 사용, 없거나 빈 문자열이면 원본 컬럼 타입 사용
                // 원본 컬럼 타입이 이미 pandas dtype이므로 그대로 사용
                let columnType: string;
                if (selection?.type && selection.type.trim() !== "") {
                  // selection에 타입이 명시적으로 설정되어 있으면 사용
                  columnType = selection.type;
                } else {
                  // selection이 없거나 타입이 없으면 원본 컬럼 타입 사용
                  // 이는 pandas dtype (int64, float64, object 등)이어야 함
                  columnType = col.type;
                }

                // 디버깅: 컬럼 타입 확인
                if (col.name === "CHAS" || col.type === "int64") {
                  console.log("SelectData column type assignment:", {
                    colName: col.name,
                    originalType: col.type,
                    selectionType: selection?.type,
                    finalType: columnType,
                  });
                }

                newColumns.push({
                  name: col.name,
                  type: columnType,
                });
              }
            });

            // 디버깅: 선택된 열 확인
            console.log("SelectData Debug - Selected columns:", {
              newColumnsCount: newColumns.length,
              newColumnsNames: newColumns.map((c) => c.name),
            });

            if (
              isConfigured &&
              newColumns.length === 0 &&
              inputData.columns.length > 0
            ) {
              console.error("SelectData Error - No columns selected:", {
                isConfigured,
                selections,
                inputColumns: inputData.columns.map((c) => c.name),
              });
              throw new Error(
                "No columns selected. Please select at least one column in the Properties panel."
              );
            }

            // 디버깅: newColumns 타입 확인
            console.log("SelectData Debug - newColumns types:", {
              newColumns: newColumns.map((c) => ({
                name: c.name,
                type: c.type,
              })),
              inputColumns: inputData.columns.map((c) => ({
                name: c.name,
                type: c.type,
              })),
            });

            const newRows = (inputData.rows || []).map((row) => {
              const newRow: Record<string, any> = {};
              newColumns.forEach((col) => {
                const originalValue = row[col.name];
                let newValue = originalValue; // Default to original

                // pandas dtype에 따라 변환
                const dtype = col.type;

                if (dtype.startsWith("int")) {
                  // int64, int32, int16, int8
                  if (
                    originalValue === null ||
                    originalValue === undefined ||
                    String(originalValue).trim() === ""
                  ) {
                    newValue = "";
                  } else {
                    const num = Number(originalValue);
                    newValue = isNaN(num) ? "" : Math.floor(num);
                  }
                } else if (dtype.startsWith("float")) {
                  // float64, float32
                  if (
                    originalValue === null ||
                    originalValue === undefined ||
                    String(originalValue).trim() === ""
                  ) {
                    newValue = "";
                  } else {
                    const num = Number(originalValue);
                    newValue = isNaN(num) ? "" : num;
                  }
                } else if (dtype === "object") {
                  // 문자열 (object)
                  newValue =
                    originalValue === null || originalValue === undefined
                      ? ""
                      : String(originalValue);
                } else if (dtype === "bool") {
                  // 불리언
                  if (originalValue === null || originalValue === undefined) {
                    newValue = false;
                  } else {
                    const strVal = String(originalValue).toLowerCase().trim();
                    newValue =
                      strVal === "true" || strVal === "1" || strVal === "yes";
                  }
                } else if (
                  dtype === "datetime64" ||
                  dtype.startsWith("datetime")
                ) {
                  // 날짜/시간
                  newValue =
                    originalValue === null || originalValue === undefined
                      ? ""
                      : String(originalValue);
                } else if (dtype === "category") {
                  // 카테고리
                  newValue =
                    originalValue === null || originalValue === undefined
                      ? ""
                      : String(originalValue);
                }
                // 기타 타입은 원본 값 유지

                newRow[col.name] = newValue;
              });
              return newRow;
            });
            newOutputData = {
              type: "DataPreview",
              columns: newColumns,
              totalRowCount: inputData.totalRowCount,
              rows: newRows,
            };
          } else {
            throw new Error(
              "Input data not available or is of the wrong type."
            );
          }
  return newOutputData;
}

/**
 * DataFiltering 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeDataFiltering(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const {
            filter_type = "row",
            conditions = [],
            logical_operator = "AND",
          } = module.parameters;

          if (!conditions || conditions.length === 0) {
            throw new Error(
              "At least one condition is required for filtering."
            );
          }

          // Pyodide를 사용하여 Python으로 필터링 수행
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 데이터 필터링 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { filterDataPython } = pyodideModule;

            const result = await filterDataPython(
              inputData.rows || [],
              filter_type,
              conditions,
              logical_operator,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            addLog("SUCCESS", "Python으로 데이터 필터링 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python DataFiltering 실패: ${errorMessage}`);
            throw new Error(`데이터 필터링 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * Recommender 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeRecommender(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const {
            user_col = "",
            item_col = "",
            rating_col = "",
            n_components = 2,
            top_n = 5,
          } = module.parameters;

          if (!user_col || !item_col || !rating_col) {
            throw new Error(
              "User, Item, and Rating columns must all be selected."
            );
          }

          // Pyodide를 사용하여 Python으로 협업 필터링 추천 수행 (NMF, random_state=42)
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 협업 필터링 추천 수행 중... (user=${user_col}, item=${item_col}, rating=${rating_col})`
            );

            const pyodideModule = await import("./pyodideRunner");
            const { runRecommenderPython } = pyodideModule;

            const result = await runRecommenderPython(
              inputData.rows || [],
              user_col,
              item_col,
              rating_col,
              Number(n_components) || 2,
              Number(top_n) || 5,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            addLog(
              "SUCCESS",
              `Python으로 추천 완료: ${result.rows.length}건 추천 생성`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Recommender 실패: ${errorMessage}`);
            throw new Error(`추천 수행 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * ColumnPlot 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeColumnPlot(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const {
            plot_type = "single",
            column1 = "",
            column2 = "",
          } = module.parameters;

          if (!column1) {
            throw new Error("Column 1 must be selected.");
          }

          if (plot_type === "double" && !column2) {
            throw new Error("Column 2 must be selected for two-column plots.");
          }

          // 컬럼 타입 확인
          const col1 = inputData.columns.find((c) => c.name === column1);
          const col2 = column2
            ? inputData.columns.find((c) => c.name === column2)
            : undefined;

          if (!col1) {
            throw new Error(`Column '${column1}' not found in input data.`);
          }

          const column1Type: "number" | "string" =
            col1.type === "number" ? "number" : "string";
          const column2Type: "number" | "string" | undefined = col2
            ? col2.type === "number"
              ? "number"
              : "string"
            : undefined;

          // 사용 가능한 차트 타입 결정
          const getAvailableCharts = (
            plot_type: "single" | "double",
            column1Type: "number" | "string",
            column2Type?: "number" | "string"
          ): string[] => {
            if (plot_type === "single") {
              if (column1Type === "number") {
                return [
                  "Histogram",
                  "KDE Plot",
                  "Boxplot",
                  "Violin Plot",
                  "ECDF Plot",
                  "QQ-Plot",
                  "Line Plot",
                  "Area Plot",
                ];
              } else {
                return [
                  "Bar Plot",
                  "Count Plot",
                  "Pie Chart",
                  "Frequency Table",
                ];
              }
            } else {
              if (column1Type === "number" && column2Type === "number") {
                return [
                  "Scatter Plot",
                  "Hexbin Plot",
                  "Joint Plot",
                  "Line Plot",
                  "Regression Plot",
                  "Heatmap",
                ];
              } else if (
                (column1Type === "number" && column2Type === "string") ||
                (column1Type === "string" && column2Type === "number")
              ) {
                return [
                  "Box Plot",
                  "Violin Plot",
                  "Bar Plot",
                  "Strip Plot",
                  "Swarm Plot",
                ];
              } else {
                return ["Grouped Bar Plot", "Heatmap", "Mosaic Plot"];
              }
            }
          };

          const availableCharts = getAvailableCharts(
            plot_type as "single" | "double",
            column1Type,
            column2Type
          );

          // ColumnPlotOutput 생성 (실제 차트는 View Details에서 생성)
          newOutputData = {
            type: "ColumnPlotOutput",
            plot_type: plot_type as "single" | "double",
            column1,
            column2: column2 || undefined,
            column1Type,
            column2Type,
            availableCharts,
          };

          addLog("SUCCESS", "Column Plot 설정 완료");
  return newOutputData;
}

/**
 * HandleMissingValues 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeHandleMissingValues(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(
            module.id,
            "data",
            "data_in"
          ) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          // 두 번째 입력 확인
          const inputData2 = getSingleInputData(
            module.id,
            "data",
            "data_in2"
          ) as DataPreview | null;

          const { method, strategy, n_neighbors } = module.parameters;
          const columnSelections = module.parameters.columnSelections || {};

          // columnSelections에서 선택된 열만 추출 (기본값: 모든 열 선택)
          const selectedColumns = inputData.columns
            .filter((col) => {
              const selection = columnSelections[col.name];
              return selection?.selected !== false; // 기본값은 true (선택됨)
            })
            .map((col) => col.name);

          const columns =
            selectedColumns.length > 0 &&
            selectedColumns.length < inputData.columns.length
              ? selectedColumns
              : null; // 모든 열이 선택된 경우 null (전체 처리)

          // Pyodide를 사용하여 Python으로 결측치 처리 수행
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 결측치 처리 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { handleMissingValuesPython } = pyodideModule;

            const result = await handleMissingValuesPython(
              inputData.rows || [],
              method || "impute",
              strategy || "mean",
              columns || null,
              parseInt(n_neighbors) || 5,
              60000, // 타임아웃: 60초
              inputData2 ? inputData2.rows || [] : null,
              inputData.columns // 입력 컬럼 정보 전달 (원본 dtype 유지용)
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            // 두 번째 출력이 있으면 별도로 저장 (모듈의 outputData2 속성에 저장)
            if (result.rows2 && result.columns2) {
              (module as any).outputData2 = {
                type: "DataPreview",
                columns: result.columns2,
                totalRowCount: result.rows2.length,
                rows: result.rows2,
              };
            }

            addLog("SUCCESS", "Python으로 결측치 처리 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python HandleMissingValues 실패: ${errorMessage}`);
            throw new Error(`결측치 처리 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * EncodeCategorical 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeEncodeCategorical(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(
            module.id,
            "data",
            "data_in"
          ) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          // 두 번째 입력 확인
          const inputData2 = getSingleInputData(
            module.id,
            "data",
            "data_in2"
          ) as DataPreview | null;

          const {
            method,
            columns: targetColumns,
            ordinal_mapping: ordinalMappingStr,
            drop,
            handle_unknown,
          } = module.parameters;

          const columnsToEncode =
            targetColumns && targetColumns.length > 0
              ? targetColumns
              : inputData.columns
                  .filter((c) => c.type === "string")
                  .map((c) => c.name);

          // Pyodide를 사용하여 Python으로 인코딩 수행
          try {
            addLog("INFO", "Pyodide를 사용하여 Python으로 인코딩 수행 중...");

            const pyodideModule = await import("./pyodideRunner");
            const { encodeCategoricalPython } = pyodideModule;

            let ordinalMapping: Record<string, string[]> | null = null;
            if (ordinalMappingStr) {
              try {
                ordinalMapping = JSON.parse(ordinalMappingStr);
              } catch (e) {
                ordinalMapping = null;
              }
            }

            const result = await encodeCategoricalPython(
              inputData.rows || [],
              method || "label",
              columnsToEncode.length > 0 ? columnsToEncode : null,
              ordinalMapping,
              drop || "first",
              handle_unknown || "ignore",
              60000, // 타임아웃: 60초
              inputData2 ? inputData2.rows || [] : null
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            // 두 번째 출력이 있으면 별도로 저장
            if (result.rows2 && result.columns2) {
              (module as any).outputData2 = {
                type: "DataPreview",
                columns: result.columns2,
                totalRowCount: result.rows2.length,
                rows: result.rows2,
              };
            }

            addLog("SUCCESS", "Python으로 인코딩 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python EncodeCategorical 실패: ${errorMessage}`);
            throw new Error(`인코딩 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * ScalingTransform 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeScalingTransform(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(
            module.id,
            "data",
            "data_in"
          ) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          // 두 번째 입력 확인
          const inputData2 = getSingleInputData(
            module.id,
            "data",
            "data_in2"
          ) as DataPreview | null;

          const selections =
            (module.parameters.columnSelections as Record<
              string,
              { selected: boolean }
            >) || {};
          const method = (module.parameters.method as string) || "MinMax";

          // columnSelections가 없거나 비어있으면 모든 숫자형 열 선택
          const hasSelections =
            selections && Object.keys(selections).length > 0;
          const columnsToNormalize = inputData.columns
            .filter((col) => {
              // pandas dtype이 숫자형인지 확인 (int64, float64 등)
              if (
                !col ||
                !(col.type.startsWith("int") || col.type.startsWith("float"))
              )
                return false;
              // columnSelections가 없거나 비어있으면 모든 숫자형 열 선택
              if (!hasSelections) return true;
              // 해당 열이 selections에 없으면 선택된 것으로 간주 (기본값)
              if (!selections[col.name]) return true;
              return selections[col.name]?.selected !== false;
            })
            .map((col) => col.name);

          // Pyodide를 사용하여 Python으로 정규화 수행
          try {
            addLog("INFO", "Pyodide를 사용하여 Python으로 정규화 수행 중...");

            const pyodideModule = await import("./pyodideRunner");
            const { normalizeDataPython } = pyodideModule;

            const result = await normalizeDataPython(
              inputData.rows || [],
              method || "MinMax",
              columnsToNormalize,
              60000, // 타임아웃: 60초
              inputData2 ? inputData2.rows || [] : null,
              inputData.columns // 입력 컬럼 정보 전달 (원본 dtype 유지용)
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            // 두 번째 출력이 있으면 별도로 저장
            if (result.rows2 && result.columns2) {
              (module as any).outputData2 = {
                type: "DataPreview",
                columns: result.columns2,
                totalRowCount: result.rows2.length,
                rows: result.rows2,
              };
            }

            addLog("SUCCESS", "Python으로 정규화 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python NormalizeData 실패: ${errorMessage}`);
            throw new Error(`정규화 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * PythonScript 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executePythonScript(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const code =
            (module.parameters.code as string) || "scripted_data = dataframe";

          try {
            addLog("INFO", "Pyodide 샌드박스에서 사용자 Python 스크립트 실행 중...");
            const pyodideModule = await import("./pyodideRunner");
            const { runUserScriptPython } = pyodideModule;
            const result = await runUserScriptPython(
              inputData.rows || [],
              code,
              60000
            );
            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };
            addLog("SUCCESS", "사용자 Python 스크립트 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Script 실패: ${errorMessage}`);
            throw new Error(`Python Script 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * FeatureEngineer 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeFeatureEngineer(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const operations = (module.parameters.operations as any[]) || [];

          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 특징 공학 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { featureEngineerPython } = pyodideModule;

            const result = await featureEngineerPython(
              inputData.rows || [],
              operations,
              60000
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            addLog("SUCCESS", "Python으로 특징 공학 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python FeatureEngineer 실패: ${errorMessage}`);
            throw new Error(`특징 공학 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * TransitionData 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeTransitionData(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const transformations =
            (module.parameters.transformations as Record<string, string>) || {};

          // Pyodide를 사용하여 Python으로 수학적 변환 수행
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 데이터 변환 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { transformDataPython } = pyodideModule;

            const result = await transformDataPython(
              inputData.rows || [],
              transformations,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            addLog("SUCCESS", "Python으로 데이터 변환 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python TransitionData 실패: ${errorMessage}`);
            throw new Error(`데이터 변환 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * Join 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeJoin(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData1 = getSingleInputData(
            module.id,
            "data",
            "data_in"
          ) as DataPreview | null;
          const inputData2 = getSingleInputData(
            module.id,
            "data",
            "data_in2"
          ) as DataPreview | null;

          if (!inputData1 || !inputData2) {
            throw new Error("Both input data sources must be connected.");
          }

          const { join_type, left_on, right_on, on, how, suffixes } =
            module.parameters;

          // 조인 키 검증
          if (!on && (!left_on || !right_on)) {
            throw new Error(
              "Join key must be specified (on or left_on/right_on)."
            );
          }

          // Pyodide를 사용하여 Python으로 조인 수행
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 데이터 조인 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { joinDataPython } = pyodideModule;

            const result = await joinDataPython(
              inputData1.rows || [],
              inputData2.rows || [],
              join_type || "inner",
              left_on || null,
              right_on || null,
              on || null,
              how || join_type || "inner",
              suffixes || ["_x", "_y"],
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "JoinOutput",
              rows: result.rows,
              columns: result.columns,
            };

            addLog(
              "SUCCESS",
              `Python으로 데이터 조인 완료: ${result.rows.length}행 × ${result.columns.length}열`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Join 실패: ${errorMessage}`);
            throw new Error(`데이터 조인 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * Concat 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeConcat(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData1 = getSingleInputData(
            module.id,
            "data",
            "data_in"
          ) as DataPreview | null;
          const inputData2 = getSingleInputData(
            module.id,
            "data",
            "data_in2"
          ) as DataPreview | null;

          if (!inputData1 || !inputData2) {
            throw new Error("Both input data sources must be connected.");
          }

          const { axis, ignore_index, sort } = module.parameters;
          const rows1 = inputData1.rows?.length || 0;
          const rows2 = inputData2.rows?.length || 0;
          const cols1 = inputData1.columns?.length || 0;
          const cols2 = inputData2.columns?.length || 0;

          // 요구사항 검증
          const isVertical = axis === "vertical";
          if (isVertical && cols1 !== cols2) {
            throw new Error(
              `Column count mismatch: Input 1 has ${cols1} columns, Input 2 has ${cols2} columns. For vertical concatenation, both inputs must have the same number of columns.`
            );
          }
          if (!isVertical && rows1 !== rows2) {
            throw new Error(
              `Row count mismatch: Input 1 has ${rows1} rows, Input 2 has ${rows2} rows. For horizontal concatenation, both inputs must have the same number of rows.`
            );
          }

          // Pyodide를 사용하여 Python으로 연결 수행
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 데이터 연결 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { concatDataPython } = pyodideModule;

            const result = await concatDataPython(
              inputData1.rows || [],
              inputData2.rows || [],
              axis || "vertical",
              ignore_index || false,
              sort || false,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "ConcatOutput",
              rows: result.rows,
              columns: result.columns,
            };

            addLog(
              "SUCCESS",
              `Python으로 데이터 연결 완료: ${result.rows.length}행 × ${result.columns.length}열`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python Concat 실패: ${errorMessage}`);
            throw new Error(`데이터 연결 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * KMeans 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeKMeans(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // K-Means 모델 정의만 생성 (LinearRegression처럼)
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "sklearn",
            modelType: "KMeans" as any,
            parameters: {
              n_clusters: module.parameters.n_clusters || 3,
              init: module.parameters.init || "k-means++",
              n_init: module.parameters.n_init || 10,
              max_iter: module.parameters.max_iter || 300,
              random_state: module.parameters.random_state || 42,
            },
          } as ModelDefinitionOutput;
          addLog(
            "INFO",
            `K-Means 모델 정의 모듈 '${module.name}'이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * PrincipalComponentAnalysis 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executePrincipalComponentAnalysis(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // PCA 모델 정의만 생성
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "sklearn",
            modelType: "PCA" as any,
            parameters: {
              n_components: module.parameters.n_components || 2,
            },
          } as ModelDefinitionOutput;
          addLog(
            "INFO",
            `PCA 모델 정의 모듈 '${module.name}'이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * DBSCAN 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeDBSCAN(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // DBSCAN 모델 정의만 생성
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "sklearn",
            modelType: "DBSCAN" as any,
            parameters: {
              eps: module.parameters.eps ?? 0.5,
              min_samples: module.parameters.min_samples ?? 5,
            },
          } as ModelDefinitionOutput;
          addLog(
            "INFO",
            `DBSCAN 모델 정의 모듈 '${module.name}'이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * HierarchicalClustering 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeHierarchicalClustering(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // 계층적(Agglomerative) 모델 정의만 생성
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "sklearn",
            modelType: "HierarchicalClustering" as any,
            parameters: {
              n_clusters: module.parameters.n_clusters ?? 3,
              linkage: module.parameters.linkage || "ward",
              metric: module.parameters.metric || "euclidean",
            },
          } as ModelDefinitionOutput;
          addLog(
            "INFO",
            `계층적 클러스터링 모델 정의 모듈 '${module.name}'이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * FeatureImportance 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeFeatureImportance(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // 순열 특징중요도는 학습된 모델 객체가 필요하지만, 인앱 실행기는 (지도학습의 경우)
          // 선형 계수만 보관하므로 트리 등 일반 모델을 인앱에서 재예측할 수 없다.
          // 따라서 인앱에서는 안내만 제공하고, 실제 계산은 '전체 코드 보기'로 내보낸
          // 결정적 Python(permutation_importance, random_state=42)에서 수행한다(정직한 한계).
          newOutputData = {
            type: "DataPreview",
            columns: [{ name: "info", type: "string" }],
            totalRowCount: 1,
            rows: [
              {
                info:
                  "순열 특징중요도는 '전체 코드 보기'로 내보낸 Python에서 결정적으로 계산됩니다(random_state=42).",
              },
            ],
          };
          addLog(
            "INFO",
            "FeatureImportance: 인앱은 안내만 제공 — 전체코드 내보내기에서 순열 중요도를 계산합니다(정직한 한계)."
          );
  return newOutputData;
}

/**
 * OLSModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeOLSModel(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "OLS",
            parameters: {},
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (OLS)이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * LogisticModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeLogisticModel(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "Logit",
            parameters: {},
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Logistic)이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * PoissonModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executePoissonModel(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "Poisson",
            parameters: {
              max_iter: module.parameters.max_iter || 100,
            },
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Poisson)이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * QuasiPoissonModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeQuasiPoissonModel(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "QuasiPoisson",
            parameters: {
              max_iter: module.parameters.max_iter || 100,
            },
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Quasi-Poisson)이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * NegativeBinomialModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeNegativeBinomialModel(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "NegativeBinomial",
            parameters: {
              max_iter: module.parameters.max_iter || 100,
              disp: module.parameters.disp || 1.0,
            },
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Negative Binomial)이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * StatModels 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeStatModels(
  module: CanvasModule,
  getSingleInputData: GetSingleInputData,
  addLog: AddLog
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: module.parameters.model,
            parameters: {},
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (${module.parameters.model})이 생성되었습니다.`
          );
  return newOutputData;
}

/**
 * TrainClusteringModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 의존성 주입: connections·currentModules를 파라미터로 받는다(setModules/getSingleInputData 미사용).
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeTrainClusteringModel(
  module: CanvasModule,
  addLog: AddLog,
  connections: Connection[],
  currentModules: CanvasModule[]
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // TrainClusteringModel: 모델 + 데이터로 학습
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const modelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (!modelSourceModule)
            throw new Error("Model source module not found.");

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data not available or is of the wrong type."
            );

          // 모듈을 미리 선택(PropertiesPanel 초기화)하지 않아도 바로 실행되도록,
          // feature_columns가 비어 있으면 즉시 수치형 컬럼으로 기본값을 채운다.
          // (이전 버그: module.parameters만 갱신하고 아래 필터에서 쓰는 지역 변수는
          //  빈 배열 그대로라 'No valid feature columns' 오류가 났음.)
          let { feature_columns = [] } = module.parameters;
          if (!feature_columns || feature_columns.length === 0) {
            // 기본값: 모든 숫자형 컬럼 사용
            const numericColumns = inputData.columns
              .filter(
                (c) => c.type.startsWith("int") || c.type.startsWith("float")
              )
              .map((c) => c.name);
            if (numericColumns.length === 0) {
              throw new Error("No numeric columns found in the data.");
            }
            feature_columns = numericColumns;
            module.parameters.feature_columns = numericColumns;
          }

          const ordered_feature_columns = inputData.columns
            .map((c) => c.name)
            .filter((name) => feature_columns.includes(name));

          if (ordered_feature_columns.length === 0) {
            throw new Error("No valid feature columns found in the data.");
          }

          // 모델 타입 확인
          if (
            modelSourceModule.type !== ModuleType.KMeans &&
            modelSourceModule.type !== ModuleType.PrincipalComponentAnalysis &&
            modelSourceModule.type !== ModuleType.DBSCAN &&
            modelSourceModule.type !== ModuleType.HierarchicalClustering
          ) {
            throw new Error(
              "TrainClusteringModel supports K-Means, PCA, DBSCAN, and Hierarchical models."
            );
          }

          // Python으로 클러스터링 모델 학습
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 클러스터링 모델 학습 중...`
            );

            const pyodideModule = await import("./pyodideRunner");
            const rows = inputData.rows || [];
            if (rows.length === 0) {
              throw new Error("No data rows available for training.");
            }

            // Extract feature matrix X
            const X: number[][] = [];
            for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
              const row = rows[rowIndex];
              const featureRow: number[] = [];
              for (const col of ordered_feature_columns) {
                let value = row[col];

                // 값이 null이나 undefined인 경우 처리
                if (value === null || value === undefined) {
                  throw new Error(
                    `Invalid value in column '${col}' at row ${
                      rowIndex + 1
                    }: null or undefined value found. Please handle missing values first.`
                  );
                }

                // 값이 객체인 경우 처리 (예: {value: 25} 같은 형태)
                if (typeof value === "object" && value !== null) {
                  // 객체에 value 속성이 있으면 사용
                  if ("value" in value && typeof value.value === "number") {
                    value = value.value;
                  } else if (
                    "Value" in value &&
                    typeof value.Value === "number"
                  ) {
                    value = value.Value;
                  } else {
                    // 객체를 JSON 문자열로 변환 후 숫자로 파싱 시도
                    const stringValue = JSON.stringify(value);
                    const parsed = parseFloat(stringValue);
                    if (isNaN(parsed)) {
                      throw new Error(
                        `Invalid value in column '${col}' at row ${
                          rowIndex + 1
                        }: expected number, got object ${JSON.stringify(value)}`
                      );
                    }
                    value = parsed;
                  }
                }

                // 값이 문자열인 경우 숫자로 변환 시도
                if (typeof value === "string") {
                  const trimmed = value.trim();
                  if (
                    trimmed === "" ||
                    trimmed.toLowerCase() === "null" ||
                    trimmed.toLowerCase() === "nan"
                  ) {
                    throw new Error(
                      `Invalid value in column '${col}' at row ${
                        rowIndex + 1
                      }: empty string or null value found. Please handle missing values first.`
                    );
                  }
                  const parsed = parseFloat(trimmed);
                  if (isNaN(parsed)) {
                    throw new Error(
                      `Invalid value in column '${col}' at row ${
                        rowIndex + 1
                      }: cannot convert string "${trimmed}" to number`
                    );
                  }
                  value = parsed;
                }

                // 최종 검증: 숫자인지 확인
                if (typeof value !== "number" || isNaN(value)) {
                  throw new Error(
                    `Invalid value in column '${col}' at row ${
                      rowIndex + 1
                    }: expected number, got ${typeof value} (value: ${JSON.stringify(
                      value
                    )})`
                  );
                }

                featureRow.push(value);
              }
              X.push(featureRow);
            }

            if (modelSourceModule.type === ModuleType.KMeans) {
              const { fitKMeansPython } = pyodideModule;
              const modelParams =
                modelSourceModule.outputData?.type === "ModelDefinitionOutput"
                  ? modelSourceModule.outputData.parameters
                  : modelSourceModule.parameters;

              const fitResult = await fitKMeansPython(
                X,
                modelParams.n_clusters || 3,
                modelParams.init || "k-means++",
                modelParams.n_init || 10,
                modelParams.max_iter || 300,
                modelParams.random_state || 42,
                ordered_feature_columns,
                60000
              );

              newOutputData = {
                type: "TrainedClusteringModelOutput",
                modelType: ModuleType.KMeans,
                featureColumns: ordered_feature_columns,
                model: fitResult.model,
                centroids: fitResult.centroids,
                inertia: fitResult.inertia,
              } as TrainedClusteringModelOutput;
            } else if (
              modelSourceModule.type === ModuleType.PrincipalComponentAnalysis
            ) {
              const { fitPCAPython } = pyodideModule;
              const modelParams =
                modelSourceModule.outputData?.type === "ModelDefinitionOutput"
                  ? modelSourceModule.outputData.parameters
                  : modelSourceModule.parameters;

              const fitResult = await fitPCAPython(
                X,
                modelParams.n_components || 2,
                ordered_feature_columns,
                60000
              );

              newOutputData = {
                type: "TrainedClusteringModelOutput",
                modelType: ModuleType.PrincipalComponentAnalysis,
                featureColumns: ordered_feature_columns,
                model: fitResult.model,
                components: fitResult.components,
                explainedVarianceRatio: fitResult.explainedVarianceRatio,
                mean: fitResult.mean,
              } as TrainedClusteringModelOutput;
            } else if (
              modelSourceModule.type === ModuleType.DBSCAN ||
              modelSourceModule.type === ModuleType.HierarchicalClustering
            ) {
              // Transductive(.predict 없음): fit_predict로 라벨을 계산해 저장한다.
              const { fitTransductiveClusteringPython } = pyodideModule;
              const modelParams =
                modelSourceModule.outputData?.type === "ModelDefinitionOutput"
                  ? modelSourceModule.outputData.parameters
                  : modelSourceModule.parameters;
              const algorithm =
                modelSourceModule.type === ModuleType.DBSCAN
                  ? "dbscan"
                  : "agglomerative";

              const fitResult = await fitTransductiveClusteringPython(
                X,
                algorithm,
                modelParams,
                ordered_feature_columns,
                60000
              );

              newOutputData = {
                type: "TrainedClusteringModelOutput",
                modelType: modelSourceModule.type,
                featureColumns: ordered_feature_columns,
                model: fitResult.model,
                labels: fitResult.labels,
                nClusters: fitResult.nClusters,
                nNoise: fitResult.nNoise,
              } as TrainedClusteringModelOutput;
            }

            addLog("SUCCESS", "클러스터링 모델 학습 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `클러스터링 모델 학습 실패: ${errorMessage}`);
            throw new Error(`클러스터링 모델 학습 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * ClusteringData 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 의존성 주입: connections·currentModules를 파라미터로 받는다(setModules/getSingleInputData 미사용).
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeClusteringData(
  module: CanvasModule,
  addLog: AddLog,
  connections: Connection[],
  currentModules: CanvasModule[]
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          // ClusteringData: 학습된 모델로 새 데이터에 클러스터 할당 또는 PCA 변환
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const trainedModelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (
            !trainedModelSourceModule ||
            !trainedModelSourceModule.outputData ||
            trainedModelSourceModule.outputData.type !==
              "TrainedClusteringModelOutput"
          ) {
            throw new Error(
              "A successfully trained clustering model must be connected to 'model_in'."
            );
          }

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data for clustering not available or is of the wrong type."
            );

          const trainedModel =
            trainedModelSourceModule.outputData as TrainedClusteringModelOutput;

          // Python으로 클러스터 할당 또는 PCA 변환 수행
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 클러스터링 데이터 처리 중...`
            );

            const pyodideModule = await import("./pyodideRunner");
            const rows = inputData.rows || [];
            if (rows.length === 0) {
              throw new Error("No data rows available for clustering.");
            }

            // Extract feature matrix X
            const X: number[][] = [];
            for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
              const row = rows[rowIndex];
              const featureRow: number[] = [];
              for (const col of trainedModel.featureColumns) {
                let value = row[col];

                // 값이 null이나 undefined인 경우 처리
                if (value === null || value === undefined) {
                  throw new Error(
                    `Invalid value in column '${col}' at row ${
                      rowIndex + 1
                    }: null or undefined value found. Please handle missing values first.`
                  );
                }

                // 값이 객체인 경우 처리 (예: {value: 25} 같은 형태)
                if (typeof value === "object" && value !== null) {
                  // 객체에 value 속성이 있으면 사용
                  if ("value" in value && typeof value.value === "number") {
                    value = value.value;
                  } else if (
                    "Value" in value &&
                    typeof value.Value === "number"
                  ) {
                    value = value.Value;
                  } else {
                    // 객체를 JSON 문자열로 변환 후 숫자로 파싱 시도
                    const stringValue = JSON.stringify(value);
                    const parsed = parseFloat(stringValue);
                    if (isNaN(parsed)) {
                      throw new Error(
                        `Invalid value in column '${col}' at row ${
                          rowIndex + 1
                        }: expected number, got object ${JSON.stringify(value)}`
                      );
                    }
                    value = parsed;
                  }
                }

                // 값이 문자열인 경우 숫자로 변환 시도
                if (typeof value === "string") {
                  const trimmed = value.trim();
                  if (
                    trimmed === "" ||
                    trimmed.toLowerCase() === "null" ||
                    trimmed.toLowerCase() === "nan"
                  ) {
                    throw new Error(
                      `Invalid value in column '${col}' at row ${
                        rowIndex + 1
                      }: empty string or null value found. Please handle missing values first.`
                    );
                  }
                  const parsed = parseFloat(trimmed);
                  if (isNaN(parsed)) {
                    throw new Error(
                      `Invalid value in column '${col}' at row ${
                        rowIndex + 1
                      }: cannot convert string "${trimmed}" to number`
                    );
                  }
                  value = parsed;
                }

                // 최종 검증: 숫자인지 확인
                if (typeof value !== "number" || isNaN(value)) {
                  throw new Error(
                    `Invalid value in column '${col}' at row ${
                      rowIndex + 1
                    }: expected number, got ${typeof value} (value: ${JSON.stringify(
                      value
                    )})`
                  );
                }

                featureRow.push(value);
              }
              X.push(featureRow);
            }

            if (trainedModel.modelType === ModuleType.KMeans) {
              const { predictKMeansPython } = pyodideModule;
              const result = await predictKMeansPython(
                X,
                trainedModel.model,
                trainedModel.featureColumns,
                60000
              );

              // 클러스터 할당이 추가된 데이터 생성
              const newRows = rows.map((row, idx) => ({
                ...row,
                cluster: result.clusters[idx],
              }));
              const newColumns = [
                ...inputData.columns,
                { name: "cluster", type: "number" },
              ];

              const clusteredData: DataPreview = {
                type: "DataPreview",
                columns: newColumns,
                totalRowCount: inputData.totalRowCount,
                rows: newRows,
              };

              newOutputData = {
                type: "ClusteringDataOutput",
                clusteredData,
                modelType: ModuleType.KMeans,
              } as ClusteringDataOutput;
            } else if (
              trainedModel.modelType === ModuleType.PrincipalComponentAnalysis
            ) {
              const { transformPCAPython } = pyodideModule;
              const result = await transformPCAPython(
                X,
                trainedModel.model,
                trainedModel.featureColumns,
                60000
              );

              // PCA 변환된 데이터 생성
              const n_components = result.transformedData[0]?.length || 2;
              const newColumns: ColumnInfo[] = Array.from(
                { length: n_components },
                (_, i) => ({
                  name: `PC${i + 1}`,
                  type: "number",
                })
              );

              const newRows = result.transformedData.map((row) => {
                const newRow: Record<string, number> = {};
                for (let i = 0; i < n_components; i++) {
                  newRow[`PC${i + 1}`] = row[i];
                }
                return newRow;
              });

              const transformedData: DataPreview = {
                type: "DataPreview",
                columns: newColumns,
                totalRowCount: inputData.totalRowCount,
                rows: newRows,
              };

              newOutputData = {
                type: "ClusteringDataOutput",
                clusteredData: transformedData,
                modelType: ModuleType.PrincipalComponentAnalysis,
              } as ClusteringDataOutput;
            } else if (
              trainedModel.modelType === ModuleType.DBSCAN ||
              trainedModel.modelType === ModuleType.HierarchicalClustering
            ) {
              // Transductive: 학습 시 계산한 라벨을 동일 데이터에 그대로 부여한다.
              // (이들은 새 데이터를 예측할 수 없으므로 행 수가 일치해야 한다.)
              const labels = trainedModel.labels || [];
              if (labels.length !== rows.length) {
                throw new Error(
                  `Transductive 클러스터링(DBSCAN/Hierarchical)은 새 데이터에 예측할 수 없습니다. 학습 ${labels.length}행 ≠ 입력 ${rows.length}행. 학습에 사용한 동일 데이터를 연결하세요.`
                );
              }
              const newRows = rows.map((row, idx) => ({
                ...row,
                cluster: labels[idx],
              }));
              const newColumns = [
                ...inputData.columns,
                { name: "cluster", type: "number" },
              ];

              const clusteredData: DataPreview = {
                type: "DataPreview",
                columns: newColumns,
                totalRowCount: inputData.totalRowCount,
                rows: newRows,
              };

              newOutputData = {
                type: "ClusteringDataOutput",
                clusteredData,
                modelType: trainedModel.modelType,
              } as ClusteringDataOutput;
            }

            addLog("SUCCESS", "클러스터링 데이터 처리 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `클러스터링 데이터 처리 실패: ${errorMessage}`);
            throw new Error(`클러스터링 데이터 처리 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * EvaluateModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 의존성 주입(대형): connections·currentModules·getSingleInputData·getCurrentModules·setModules.
 * currentModules는 호출 시점 값을 그대로 전달받는다(runSimulation 내 let 재할당 반영).
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeEvaluateModel(
  module: CanvasModule,
  addLog: AddLog,
  connections: Connection[],
  currentModules: CanvasModule[],
  getSingleInputData: GetSingleInputData,
  getCurrentModules: GetCurrentModules,
  setModules: SetModules
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const inputData = getSingleInputData(
            module.id,
            "data",
            "data_in"
          ) as DataPreview;
          if (!inputData)
            throw new Error("Input data for evaluation not available.");

          // 최신 모듈 상태에서 파라미터 가져오기 (threshold 변경 반영)
          // getCurrentModules()를 통해 항상 최신 상태를 가져옴
          const latestModules = getCurrentModules();
          const latestModule =
            latestModules.find((m) => m.id === module.id) || module;
          let { label_column, prediction_column, model_type, threshold } =
            latestModule.parameters;

          // threshold가 설정되어 있으면 로그 출력 (디버깅용)
          if (threshold !== undefined && threshold !== null) {
            addLog(
              "INFO",
              `Evaluate Model [${module.name}] 실행 시 threshold: ${threshold} (최신 상태에서 가져옴)`
            );
          } else {
            addLog(
              "INFO",
              `Evaluate Model [${module.name}] threshold가 설정되지 않음`
            );
          }

          // 연결된 Train Model을 찾아서 modelPurpose를 자동으로 감지 및 기본값 설정
          let detectedModelType: "classification" | "regression" =
            model_type === "regression" ? "regression" : "classification";
          let trainModelLabelColumn: string | null = null;

          // Evaluate Model의 입력 연결 찾기 (보통 Score Model)
          const inputConnection = connections.find(
            (c) => c.to.moduleId === module.id
          );
          if (inputConnection) {
            const sourceModule = currentModules.find(
              (m) => m.id === inputConnection.from.moduleId
            );

            // Score Model인 경우, 그 Score Model이 연결된 Train Model 찾기
            if (sourceModule?.type === ModuleType.ScoreModel) {
              const modelInputConnection = connections.find(
                (c) =>
                  c.to.moduleId === sourceModule.id &&
                  c.to.portName === "model_in"
              );
              if (modelInputConnection) {
                const trainModelModule = currentModules.find(
                  (m) =>
                    m.id === modelInputConnection.from.moduleId &&
                    m.outputData?.type === "TrainedModelOutput"
                );
                if (
                  trainModelModule?.outputData?.type === "TrainedModelOutput"
                ) {
                  const trainedModel = trainModelModule.outputData;
                  trainModelLabelColumn = trainedModel.labelColumn;

                  // modelPurpose가 있으면 사용, 없으면 modelType으로 추론
                  if (trainedModel.modelPurpose) {
                    detectedModelType = trainedModel.modelPurpose;
                    addLog(
                      "INFO",
                      `연결된 모델 타입 자동 감지: ${detectedModelType} (${trainModelModule.name})`
                    );
                  } else {
                    // modelType으로 분류 모델인지 확인
                    const isClassModel = isClassification(
                      trainedModel.modelType,
                      trainedModel.modelPurpose
                    );
                    detectedModelType = isClassModel
                      ? "classification"
                      : "regression";
                    addLog(
                      "INFO",
                      `모델 타입 자동 감지: ${detectedModelType} (${trainModelModule.name})`
                    );
                  }
                }
              }
            }
          }

          // 자동 기본값 설정
          const inputColumns = inputData.columns.map((c) => c.name);
          const paramUpdates: Record<string, any> = {};

          // label_column 자동 설정
          if (!label_column) {
            if (
              trainModelLabelColumn &&
              inputColumns.includes(trainModelLabelColumn)
            ) {
              label_column = trainModelLabelColumn;
              paramUpdates.label_column = label_column;
              addLog("INFO", `Label column 자동 설정: ${label_column}`);
            } else if (inputColumns.length > 0) {
              label_column = inputColumns[0];
              paramUpdates.label_column = label_column;
              addLog("INFO", `Label column 자동 설정: ${label_column}`);
            }
          }

          // prediction_column 자동 설정
          if (!prediction_column) {
            if (
              detectedModelType === "classification" &&
              trainModelLabelColumn
            ) {
              // 분류 모델: {label_column}_Predict_Proba_1 찾기
              const probaColumn = `${trainModelLabelColumn}_Predict_Proba_1`;
              if (inputColumns.includes(probaColumn)) {
                prediction_column = probaColumn;
                paramUpdates.prediction_column = prediction_column;
                addLog(
                  "INFO",
                  `Prediction column 자동 설정: ${prediction_column} (확률값)`
                );
              } else if (inputColumns.includes("Predict")) {
                prediction_column = "Predict";
                paramUpdates.prediction_column = prediction_column;
                addLog(
                  "INFO",
                  `Prediction column 자동 설정: ${prediction_column}`
                );
              }
            } else {
              // 회귀 모델: Predict 사용
              if (inputColumns.includes("Predict")) {
                prediction_column = "Predict";
                paramUpdates.prediction_column = prediction_column;
                addLog(
                  "INFO",
                  `Prediction column 자동 설정: ${prediction_column}`
                );
              }
            }
          }

          // model_type 자동 설정
          if (model_type !== detectedModelType) {
            paramUpdates.model_type = detectedModelType;
          }

          // threshold 기본값 설정 (분류 모델인 경우, 값이 없을 때만)
          // threshold가 이미 설정되어 있으면 절대 덮어쓰지 않음
          if (detectedModelType === "classification") {
            if (threshold === undefined || threshold === null) {
              // threshold가 없을 때만 기본값 설정
              threshold = 0.5;
              // paramUpdates에 추가하지 않음 (사용자가 변경한 값이 있을 수 있으므로)
              // 대신 threshold 변수만 업데이트하여 평가에 사용
              addLog(
                "INFO",
                `Evaluate Model [${module.name}] threshold 기본값 사용: ${threshold} (파라미터에는 저장하지 않음)`
              );
            } else {
              // threshold가 이미 설정되어 있으면 그 값을 사용
              addLog(
                "INFO",
                `Evaluate Model [${module.name}] threshold 사용: ${threshold}`
              );
            }
          }

          // 자동으로 설정한 파라미터들을 모듈에 저장
          // threshold는 절대 paramUpdates에 추가하지 않음 (사용자가 변경한 값 유지)
          if (Object.keys(paramUpdates).length > 0) {
            setModules(
              (prev) =>
                prev.map((m) => {
                  if (m.id === module.id) {
                    // threshold를 제외한 파라미터만 업데이트
                    const finalParamUpdates = { ...paramUpdates };
                    // threshold가 paramUpdates에 있으면 제거
                    delete finalParamUpdates.threshold;

                    // 기존 threshold 값 확인 (절대 변경하지 않음)
                    const existingThreshold = m.parameters?.threshold;

                    // threshold는 기존 값을 명시적으로 유지 (절대 변경하지 않음)
                    const updatedParameters = {
                      ...m.parameters,
                      ...finalParamUpdates,
                      // threshold는 기존 값 유지 (변경하지 않음)
                      threshold:
                        existingThreshold !== undefined &&
                        existingThreshold !== null
                          ? existingThreshold
                          : threshold !== undefined && threshold !== null
                          ? threshold
                          : 0.5,
                    };

                    addLog(
                      "INFO",
                      `Evaluate Model [${module.name}] 파라미터 업데이트 후 threshold: ${updatedParameters.threshold} (기존: ${existingThreshold})`
                    );

                    return { ...m, parameters: updatedParameters };
                  }
                  return m;
                }),
              true
            );
          }

          if (!label_column || !prediction_column) {
            throw new Error(
              "Label and prediction columns must be configured for evaluation."
            );
          }

          const rows = inputData.rows || [];
          if (rows.length === 0)
            throw new Error("No rows in input data to evaluate.");

          // Pyodide를 사용하여 Python으로 평가 메트릭 계산
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 모델 평가 수행 중..."
            );

            const pyodideModule = await import("./pyodideRunner");
            const { evaluateModelPython } = pyodideModule;

            // 분류 모델인 경우 여러 threshold에 대한 precision/recall도 계산
            const calculateThresholdMetrics =
              detectedModelType === "classification";

            const result = await evaluateModelPython(
              rows,
              label_column,
              prediction_column,
              detectedModelType, // 자동 감지된 모델 타입 사용
              threshold, // threshold 전달 (분류 모델인 경우)
              120000, // 타임아웃: 120초 (여러 threshold 계산 시 시간이 더 걸림)
              calculateThresholdMetrics // 여러 threshold에 대한 precision/recall 계산
            );

            const { thresholdMetrics, ...metrics } = result;

            addLog("SUCCESS", "Python으로 모델 평가 완료");

            // 혼동행렬 추출
            const confusionMatrix =
              detectedModelType === "classification" &&
              typeof metrics["TP"] === "number" &&
              typeof metrics["FP"] === "number" &&
              typeof metrics["TN"] === "number" &&
              typeof metrics["FN"] === "number"
                ? {
                    tp: metrics["TP"] as number,
                    fp: metrics["FP"] as number,
                    tn: metrics["TN"] as number,
                    fn: metrics["FN"] as number,
                  }
                : undefined;

            newOutputData = {
              type: "EvaluationOutput",
              modelType: detectedModelType, // 자동 감지된 모델 타입 사용
              metrics,
              confusionMatrix,
              threshold:
                detectedModelType === "classification" ? threshold : undefined,
              thresholdMetrics: thresholdMetrics, // 여러 threshold에 대한 precision/recall
            };
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python EvaluateModel 실패: ${errorMessage}`);
            throw new Error(`모델 평가 실패: ${errorMessage}`);
          }
  return newOutputData;
}

/**
 * ScoreModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 의존성 주입(대형): connections·currentModules·getSingleInputData·getCurrentModules·setModules.
 * currentModules는 호출 시점 값을 그대로 전달받는다(runSimulation 내 let 재할당 반영).
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeScoreModel(
  module: CanvasModule,
  addLog: AddLog,
  connections: Connection[],
  currentModules: CanvasModule[],
  getSingleInputData: GetSingleInputData,
  getCurrentModules: GetCurrentModules,
  setModules: SetModules
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const trainedModelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (
            !trainedModelSourceModule ||
            !trainedModelSourceModule.outputData ||
            trainedModelSourceModule.outputData.type !== "TrainedModelOutput"
          ) {
            throw new Error(
              "A successfully trained model must be connected to 'model_in'."
            );
          }

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data for scoring not available or is of the wrong type."
            );

          const trainedModel = trainedModelSourceModule.outputData;
          const modelIsClassification = isClassification(
            trainedModel.modelType,
            trainedModel.modelPurpose
          );
          const labelColumn = trainedModel.labelColumn;

          // KNN, Decision Tree, SVM, LDA, NaiveBayes, RandomForest, GradientBoosting 모델의 경우 별도 처리 (coefficients/intercept가 없는 모델 → 훈련 데이터로 재적합 후 예측)
          if (
            trainedModel.modelType === ModuleType.KNN ||
            trainedModel.modelType === ModuleType.DecisionTree ||
            trainedModel.modelType === ModuleType.NeuralNetwork ||
            trainedModel.modelType === ModuleType.SVM ||
            trainedModel.modelType === ModuleType.LDA ||
            trainedModel.modelType === ModuleType.NaiveBayes ||
            trainedModel.modelType === ModuleType.RandomForest ||
            trainedModel.modelType === ModuleType.GradientBoosting
          ) {
            // Train Model 모듈에서 훈련 데이터 가져오기
            const trainModelModule = currentModules.find(
              (m) => m.id === trainedModelSourceModule.id
            );

            if (!trainModelModule) {
              throw new Error("Train Model module not found.");
            }

            // Train Model의 입력 데이터 찾기
            const trainDataInputConnection = connections.find(
              (c) =>
                c.to.moduleId === trainModelModule.id &&
                c.to.portName === "data_in"
            );

            if (!trainDataInputConnection) {
              throw new Error(
                `Training data connection not found for ${trainedModel.modelType} model.`
              );
            }

            const trainDataSourceModule = currentModules.find(
              (m) => m.id === trainDataInputConnection.from.moduleId
            );

            if (!trainDataSourceModule || !trainDataSourceModule.outputData) {
              throw new Error("Training data source module not found.");
            }

            let trainingData: DataPreview | null = null;
            if (trainDataSourceModule.outputData.type === "DataPreview") {
              trainingData = trainDataSourceModule.outputData;
            } else if (
              trainDataSourceModule.outputData.type === "SplitDataOutput"
            ) {
              const portName = trainDataInputConnection.from.portName;
              if (portName === "train_data_out") {
                trainingData = trainDataSourceModule.outputData.train;
              } else if (portName === "test_data_out") {
                trainingData = trainDataSourceModule.outputData.test;
              }
            }

            if (!trainingData) {
              throw new Error(
                `Training data not available for ${trainedModel.modelType} model.`
              );
            }

            // 모델 정의 모듈 찾기
            const modelDefConnection = connections.find(
              (c) =>
                c.to.moduleId === trainModelModule.id &&
                c.to.portName === "model_in"
            );

            if (!modelDefConnection) {
              throw new Error(
                `${trainedModel.modelType} model definition connection not found.`
              );
            }

            const modelDefModule = currentModules.find(
              (m) => m.id === modelDefConnection.from.moduleId
            );

            if (!modelDefModule) {
              throw new Error(
                `${trainedModel.modelType} model definition module not found.`
              );
            }

            try {
              const pyodideModule = await import("./pyodideRunner");
              let result: {
                rows: any[];
                columns: Array<{ name: string; type: string }>;
              };

              if (trainedModel.modelType === ModuleType.KNN) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 KNN 모델 예측 수행 중..."
                );

                const modelPurpose =
                  modelDefModule.parameters.model_purpose || "classification";
                const nNeighbors =
                  parseInt(modelDefModule.parameters.n_neighbors, 10) || 3;
                const weights = modelDefModule.parameters.weights || "uniform";
                const algorithm = modelDefModule.parameters.algorithm || "auto";
                const metric = modelDefModule.parameters.metric || "minkowski";

                const { scoreKNNPython } = pyodideModule;
                result = await scoreKNNPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelIsClassification ? "classification" : "regression",
                  nNeighbors,
                  weights,
                  algorithm,
                  metric,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 KNN 모델 예측 완료");
              } else if (trainedModel.modelType === ModuleType.DecisionTree) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 Decision Tree 모델 예측 수행 중..."
                );

                const modelPurpose =
                  modelDefModule.parameters.model_purpose || "classification";
                const criterion = modelDefModule.parameters.criterion || "gini";
                const maxDepth =
                  modelDefModule.parameters.max_depth === "" ||
                  modelDefModule.parameters.max_depth === null ||
                  modelDefModule.parameters.max_depth === undefined
                    ? null
                    : parseInt(modelDefModule.parameters.max_depth, 10);
                const minSamplesSplit =
                  parseInt(modelDefModule.parameters.min_samples_split, 10) ||
                  2;
                const minSamplesLeaf =
                  parseInt(modelDefModule.parameters.min_samples_leaf, 10) || 1;

                const { scoreDecisionTreePython } = pyodideModule;
                result = await scoreDecisionTreePython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelIsClassification ? "classification" : "regression",
                  criterion,
                  maxDepth,
                  minSamplesSplit,
                  minSamplesLeaf,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 Decision Tree 모델 예측 완료");
              } else if (trainedModel.modelType === ModuleType.NeuralNetwork) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 Neural Network 모델 예측 수행 중..."
                );

                const modelPurpose =
                  modelDefModule.parameters.model_purpose || "classification";
                const hiddenLayerSizes =
                  modelDefModule.parameters.hidden_layer_sizes || "100";
                const activation =
                  modelDefModule.parameters.activation || "relu";
                const maxIter =
                  parseInt(modelDefModule.parameters.max_iter, 10) || 200;
                const randomState =
                  parseInt(modelDefModule.parameters.random_state, 10) || 2022;

                const { scoreNeuralNetworkPython } = pyodideModule;
                result = await scoreNeuralNetworkPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelIsClassification ? "classification" : "regression",
                  hiddenLayerSizes,
                  activation,
                  maxIter,
                  randomState,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 Neural Network 모델 예측 완료");
              } else if (trainedModel.modelType === ModuleType.SVM) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 SVM 모델 예측 수행 중..."
                );

                const modelPurpose =
                  modelDefModule.parameters.model_purpose || "classification";
                const kernel = modelDefModule.parameters.kernel || "rbf";
                const C = parseFloat(modelDefModule.parameters.C) || 1.0;
                const gamma =
                  modelDefModule.parameters.gamma === "" ||
                  modelDefModule.parameters.gamma === null ||
                  modelDefModule.parameters.gamma === undefined
                    ? "scale"
                    : modelDefModule.parameters.gamma;
                const degree =
                  parseInt(modelDefModule.parameters.degree, 10) || 3;

                const gammaValue =
                  typeof gamma === "string" ? gamma : parseFloat(gamma);

                const { scoreSVMPython } = pyodideModule;
                result = await scoreSVMPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelIsClassification ? "classification" : "regression",
                  kernel,
                  C,
                  gammaValue,
                  degree,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 SVM 모델 예측 완료");
              } else if (
                trainedModel.modelType === ModuleType.LDA
              ) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 LDA 모델 예측 수행 중..."
                );

                const solver = modelDefModule.parameters.solver || "svd";
                const shrinkage =
                  modelDefModule.parameters.shrinkage === "" ||
                  modelDefModule.parameters.shrinkage === null ||
                  modelDefModule.parameters.shrinkage === undefined
                    ? null
                    : parseFloat(modelDefModule.parameters.shrinkage);
                const nComponents =
                  modelDefModule.parameters.n_components === "" ||
                  modelDefModule.parameters.n_components === null ||
                  modelDefModule.parameters.n_components === undefined
                    ? null
                    : parseInt(modelDefModule.parameters.n_components, 10);

                const { scoreLDAPython } = pyodideModule;
                result = await scoreLDAPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  solver,
                  shrinkage,
                  nComponents,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 LDA 모델 예측 완료");
              } else if (trainedModel.modelType === ModuleType.NaiveBayes) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 Naive Bayes 모델 예측 수행 중..."
                );

                const modelType =
                  modelDefModule.parameters.model_type || "Gaussian";
                const alpha =
                  parseFloat(modelDefModule.parameters.alpha) || 1.0;

                const { scoreNaiveBayesPython } = pyodideModule;
                result = await scoreNaiveBayesPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelType,
                  alpha,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 Naive Bayes 모델 예측 완료");
              } else if (
                trainedModel.modelType === ModuleType.RandomForest
              ) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 Random Forest 모델 예측 수행 중..."
                );

                const nEstimators =
                  parseInt(modelDefModule.parameters.n_estimators, 10) || 100;
                const criterion =
                  modelDefModule.parameters.criterion ||
                  (modelIsClassification ? "gini" : "mse");
                const maxDepth =
                  modelDefModule.parameters.max_depth === "" ||
                  modelDefModule.parameters.max_depth === null ||
                  modelDefModule.parameters.max_depth === undefined
                    ? null
                    : parseInt(modelDefModule.parameters.max_depth, 10);
                const maxFeatures =
                  modelDefModule.parameters.max_features === "" ||
                  modelDefModule.parameters.max_features === null ||
                  modelDefModule.parameters.max_features === undefined
                    ? null
                    : modelDefModule.parameters.max_features;

                const { scoreRandomForestPython } = pyodideModule;
                result = await scoreRandomForestPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelIsClassification ? "classification" : "regression",
                  nEstimators,
                  criterion,
                  maxDepth,
                  maxFeatures,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 Random Forest 모델 예측 완료");
              } else if (
                trainedModel.modelType === ModuleType.GradientBoosting
              ) {
                addLog(
                  "INFO",
                  "Pyodide를 사용하여 Python으로 Gradient Boosting 모델 예측 수행 중..."
                );

                const nEstimators =
                  parseInt(modelDefModule.parameters.n_estimators, 10) || 100;
                const learningRate =
                  parseFloat(modelDefModule.parameters.learning_rate) || 0.1;
                const maxDepth =
                  parseInt(modelDefModule.parameters.max_depth, 10) || 3;

                const { scoreGradientBoostingPython } = pyodideModule;
                result = await scoreGradientBoostingPython(
                  inputData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  modelIsClassification ? "classification" : "regression",
                  nEstimators,
                  learningRate,
                  maxDepth,
                  trainingData.rows || [],
                  trainedModel.featureColumns,
                  labelColumn,
                  60000
                );

                addLog("SUCCESS", "Python으로 Gradient Boosting 모델 예측 완료");
              } else {
                throw new Error(
                  `Unsupported model type for ScoreModel: ${trainedModel.modelType}`
                );
              }

              newOutputData = {
                type: "DataPreview",
                columns: result.columns,
                totalRowCount: inputData.totalRowCount,
                rows: result.rows,
              };
            } catch (error: any) {
              const errorMessage = error.message || String(error);
              addLog(
                "ERROR",
                `Python ${trainedModel.modelType} ScoreModel 실패: ${errorMessage}`
              );
              throw new Error(`모델 예측 실패: ${errorMessage}`);
            }
          } else {
            // 기존 방식 (coefficients/intercept 사용)
            // Pyodide를 사용하여 Python으로 예측 수행
            try {
              addLog(
                "INFO",
                "Pyodide를 사용하여 Python으로 모델 예측 수행 중..."
              );

              const pyodideModule = await import("./pyodideRunner");
              const { scoreModelPython } = pyodideModule;

              const result = await scoreModelPython(
                inputData.rows || [],
                trainedModel.featureColumns,
                trainedModel.coefficients,
                trainedModel.intercept,
                labelColumn,
                modelIsClassification ? "classification" : "regression",
                60000 // 타임아웃: 60초
              );

              // 분류: Predict는 학습 시 클래스 코드(0..k-1)이므로 저장된
              // classLabels로 원 라벨 복원(문자열은 문자열로, 수치는 수치로).
              // 이전엔 코드가 그대로 노출되어 문자열/비{0,1} 라벨에서
              // EvaluateModel 비교가 어긋났다(예: y_true '>50K' vs Predict 1).
              const storedClassLabels = trainedModel.classLabels;
              if (
                modelIsClassification &&
                storedClassLabels &&
                storedClassLabels.length >= 2
              ) {
                const allNumericClasses = storedClassLabels.every(
                  (k) => !isNaN(Number(k))
                );
                result.rows = result.rows.map((r: any) => {
                  const code = r["Predict"];
                  if (
                    typeof code === "number" &&
                    Number.isInteger(code) &&
                    code >= 0 &&
                    code < storedClassLabels.length
                  ) {
                    return {
                      ...r,
                      Predict: allNumericClasses
                        ? Number(storedClassLabels[code])
                        : storedClassLabels[code],
                    };
                  }
                  return r;
                });
                if (!allNumericClasses) {
                  result.columns = result.columns.map((c: any) =>
                    c.name === "Predict" ? { ...c, type: "string" } : c
                  );
                }
              }

              newOutputData = {
                type: "DataPreview",
                columns: result.columns,
                totalRowCount: inputData.totalRowCount,
                rows: result.rows,
              };

              addLog("SUCCESS", "Python으로 모델 예측 완료");

              // 연결된 Evaluate Model의 파라미터 자동 설정
              const evaluateModelConnections = connections.filter(
                (c) =>
                  c.from.moduleId === module.id &&
                  currentModules.find((m) => m.id === c.to.moduleId)?.type ===
                    ModuleType.EvaluateModel
              );

              for (const evalConn of evaluateModelConnections) {
                const evalModule = currentModules.find(
                  (m) => m.id === evalConn.to.moduleId
                );
                if (evalModule) {
                  const evalParams = evalModule.parameters || {};
                  const updates: Record<string, any> = {};

                  const inputColumns = result.columns.map((c) => c.name);

                  // label_column 자동 설정 (항상 업데이트)
                  if (inputColumns.includes(labelColumn)) {
                    updates.label_column = labelColumn;
                  } else if (inputColumns.length > 0) {
                    updates.label_column = inputColumns[0];
                  }

                  // prediction_column 자동 설정 (항상 업데이트)
                  if (modelIsClassification) {
                    const probaColumn = `${labelColumn}_Predict_Proba_1`;
                    if (inputColumns.includes(probaColumn)) {
                      updates.prediction_column = probaColumn;
                    } else if (inputColumns.includes("Predict")) {
                      updates.prediction_column = "Predict";
                    }
                  } else {
                    if (inputColumns.includes("Predict")) {
                      updates.prediction_column = "Predict";
                    }
                  }

                  // model_type 자동 설정 (항상 업데이트)
                  const detectedModelType = modelIsClassification
                    ? "classification"
                    : "regression";
                  updates.model_type = detectedModelType;

                  // threshold 기본값 설정 (분류 모델인 경우, 값이 없을 때만)
                  // threshold가 이미 설정되어 있으면 절대 변경하지 않음
                  if (
                    modelIsClassification &&
                    (evalParams.threshold === undefined ||
                      evalParams.threshold === null)
                  ) {
                    // threshold가 없을 때만 기본값 설정
                    updates.threshold = 0.5;
                  }
                  // threshold가 이미 설정되어 있으면 updates에 추가하지 않음

                  // 파라미터 업데이트 (threshold는 절대 덮어쓰지 않음)
                  if (Object.keys(updates).length > 0) {
                    setModules(
                      (prev) =>
                        prev.map((m) => {
                          if (m.id === evalModule.id) {
                            // threshold를 제외한 파라미터만 업데이트
                            const finalUpdates = { ...updates };
                            const existingThreshold = m.parameters?.threshold;

                            // threshold가 이미 있으면 절대 덮어쓰지 않음
                            if (
                              existingThreshold !== undefined &&
                              existingThreshold !== null
                            ) {
                              delete finalUpdates.threshold;
                            }

                            // threshold를 제외한 파라미터만 업데이트하고, threshold는 기존 값 유지
                            return {
                              ...m,
                              parameters: {
                                ...m.parameters,
                                ...finalUpdates,
                                // threshold는 기존 값 명시적으로 유지
                                threshold:
                                  existingThreshold !== undefined &&
                                  existingThreshold !== null
                                    ? existingThreshold
                                    : finalUpdates.threshold !== undefined
                                    ? finalUpdates.threshold
                                    : m.parameters?.threshold,
                              },
                            };
                          }
                          return m;
                        }),
                      true
                    );

                    // 파라미터 업데이트만 하고 자동 재실행은 하지 않음
                    // 사용자가 수동으로 실행하거나, Score Model이 완료된 후에 실행되도록 함
                    addLog(
                      "INFO",
                      `Evaluate Model [${evalModule.name}] 파라미터가 자동으로 설정되었습니다. 실행하려면 모듈을 클릭하세요.`
                    );
                  }
                }
              }
            } catch (error: any) {
              const errorMessage = error.message || String(error);
              addLog("ERROR", `Python ScoreModel 실패: ${errorMessage}`);
              throw new Error(`모델 예측 실패: ${errorMessage}`);
            }
          }
  return newOutputData;
}

/**
 * TrainModel 모듈 실행기 — App.tsx runSimulation의 해당 분기 본문을 문자 그대로 이동.
 * 의존성 주입: connections·currentModules를 파라미터로 받는다(setModules/getSingleInputData 미사용).
 * 동작 불변: 에러는 그대로 throw되어 호출부(runSimulation)의 catch가 처리한다.
 */
export async function executeTrainModel(
  module: CanvasModule,
  addLog: AddLog,
  connections: Connection[],
  currentModules: CanvasModule[]
): Promise<CanvasModule["outputData"]> {
  let newOutputData: CanvasModule["outputData"] | undefined = undefined;
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const modelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (!modelSourceModule)
            throw new Error("Model source module not found.");

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data not available or is of the wrong type."
            );

          const { feature_columns, label_column } = module.parameters;
          if (
            !feature_columns ||
            feature_columns.length === 0 ||
            !label_column
          ) {
            throw new Error("Feature and label columns are not configured.");
          }

          const ordered_feature_columns = inputData.columns
            .map((c) => c.name)
            .filter((name) => feature_columns.includes(name));

          if (ordered_feature_columns.length === 0) {
            throw new Error("No valid feature columns found in the data.");
          }

          let trainedModelOutput: TrainedModelOutput | undefined = undefined;
          let intercept = 0;
          const coefficients: Record<string, number> = {};
          const metrics: Record<string, number> = {};

          const modelIsClassification = isClassification(
            modelSourceModule.type,
            modelSourceModule.parameters.model_purpose
          );
          const modelIsRegression = !modelIsClassification;

          // Prepare data for training
          const rows = inputData.rows || [];
          if (rows.length === 0) {
            throw new Error("No data rows available for training.");
          }

          // Extract feature matrix X and target vector y
          const X: number[][] = [];
          const y: number[] = [];

          if (!rows || rows.length === 0) {
            throw new Error("Input data has no rows.");
          }

          if (
            !ordered_feature_columns ||
            ordered_feature_columns.length === 0
          ) {
            throw new Error("No feature columns specified.");
          }

          // 분류 모델은 범주형(문자열) 라벨을 정수 코드로 인코딩한다.
          // (이전 버그: 라벨이 숫자가 아니면 모든 행을 버려 'No valid data rows' 발생 —
          //  Adult Income의 income(>50K/<=50K) 같은 문자열 라벨에서 분류 학습 불가했음.)
          // 클래스 코드는 사전 스캔 후 정렬 순서로 부여한다(결정적·sklearn LabelEncoder 정합,
          // 수치 라벨은 수치 정렬). 이전에는 등장 순서로 부여해 훈련 데이터 첫 행의 클래스에
          // 따라 이진 라벨 0/1이 뒤집힐 수 있었고(예: 첫 행 라벨=1 → {1:0, 0:1}), 그 코드로
          // 학습된 모델의 ScoreModel 확률(Proba_1)·EvaluateModel 혼동행렬이 반전되었다.
          const labelClassMap: Record<string, number> = {};
          if (modelIsClassification) {
            const uniqueLabelKeys = Array.from(
              new Set(
                rows
                  .filter((r: any) => r)
                  .map((r: any) => r[label_column])
                  .filter(
                    (v: any) =>
                      v !== null && v !== undefined && String(v).trim() !== ""
                  )
                  .map((v: any) => String(v))
              )
            );
            const allNumericLabels = uniqueLabelKeys.every(
              (k) => !isNaN(Number(k))
            );
            uniqueLabelKeys.sort((a, b) =>
              allNumericLabels
                ? Number(a) - Number(b)
                : a < b
                  ? -1
                  : a > b
                    ? 1
                    : 0
            );
            uniqueLabelKeys.forEach((k, i) => {
              labelClassMap[k] = i;
            });
          }

          for (let rowIdx = 0; rowIdx < rows.length; rowIdx++) {
            const row = rows[rowIdx];
            if (!row) {
              continue; // Skip null/undefined rows
            }

            const featureRow: number[] = [];
            let hasValidFeatures = true;

            for (
              let colIdx = 0;
              colIdx < ordered_feature_columns.length;
              colIdx++
            ) {
              const col = ordered_feature_columns[colIdx];
              if (!col) {
                hasValidFeatures = false;
                break;
              }
              const value = row[col];
              if (
                typeof value === "number" &&
                !isNaN(value) &&
                value !== null &&
                value !== undefined
              ) {
                featureRow.push(value);
              } else {
                hasValidFeatures = false;
                break;
              }
            }

            if (!hasValidFeatures) {
              continue; // Skip rows with invalid features
            }

            if (featureRow.length !== ordered_feature_columns.length) {
              continue; // Skip rows with incomplete features
            }

            const labelValue = row[label_column];
            if (modelIsClassification) {
              // 분류: 비어있지 않은 라벨이면 클래스 코드로 인코딩(문자열/숫자/불리언 허용).
              if (
                labelValue !== null &&
                labelValue !== undefined &&
                String(labelValue).trim() !== ""
              ) {
                const key = String(labelValue);
                if (!(key in labelClassMap)) continue; // 사전 스캔에 없던 값(방어)
                X.push(featureRow);
                y.push(labelClassMap[key]);
              }
            } else {
              // 회귀: 숫자 라벨만(숫자 형태의 문자열도 허용).
              const num =
                typeof labelValue === "number" ? labelValue : Number(labelValue);
              if (
                labelValue !== null &&
                labelValue !== undefined &&
                String(labelValue).trim() !== "" &&
                !isNaN(num)
              ) {
                X.push(featureRow);
                y.push(num);
              }
            }
          }

          if (X.length === 0) {
            throw new Error(
              `No valid data rows found after filtering. Checked ${
                rows.length
              } rows. Ensure feature columns (${ordered_feature_columns.join(
                ", "
              )}) and label column (${label_column}) contain valid numeric values.`
            );
          }

          if (X.length < ordered_feature_columns.length) {
            throw new Error(
              `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but only found ${X.length} valid rows.`
            );
          }

          // tuningSummary를 초기화 (모든 모델 타입에서 사용 가능하도록)
          let tuningSummary: TrainedModelOutput["tuningSummary"] = undefined;

          if (modelIsRegression) {
            // Pyodide를 사용하여 Python으로 Linear Regression 훈련
            if (modelSourceModule.type === ModuleType.LinearRegression) {
              const fitIntercept =
                modelSourceModule.parameters.fit_intercept === "True";
              const modelType =
                modelSourceModule.parameters.model_type || "LinearRegression";
              const alpha =
                parseFloat(modelSourceModule.parameters.alpha) || 1.0;
              const l1_ratio =
                parseFloat(modelSourceModule.parameters.l1_ratio) || 0.5;
              const parseCandidates = (
                raw: any,
                fallback: number[]
              ): number[] => {
                if (Array.isArray(raw)) {
                  const parsed = raw
                    .map((val) => {
                      const num =
                        typeof val === "number" ? val : parseFloat(val);
                      return isNaN(num) ? null : num;
                    })
                    .filter((num): num is number => num !== null);
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "string") {
                  const parsed = raw
                    .split(",")
                    .map((part) => parseFloat(part.trim()))
                    .filter((num) => !isNaN(num));
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "number" && !isNaN(raw)) {
                  return [raw];
                }
                return fallback;
              };
              const tuningEnabled =
                modelSourceModule.parameters.tuning_enabled === "True";
              const tuningOptions = tuningEnabled
                ? {
                    enabled: true,
                    strategy: "GridSearch" as const,
                    alphaCandidates: parseCandidates(
                      modelSourceModule.parameters.alpha_candidates,
                      [alpha]
                    ),
                    l1RatioCandidates:
                      modelType === "ElasticNet"
                        ? parseCandidates(
                            modelSourceModule.parameters.l1_ratio_candidates,
                            [l1_ratio]
                          )
                        : undefined,
                    cvFolds: Math.max(
                      2,
                      parseInt(modelSourceModule.parameters.cv_folds, 10) || 5
                    ),
                    scoringMetric:
                      modelSourceModule.parameters.scoring_metric ||
                      "neg_mean_squared_error",
                  }
                : undefined;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 ${modelType} 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitLinearRegressionPython } = pyodideModule;

                const fitResult = await fitLinearRegressionPython(
                  X,
                  y,
                  modelType,
                  fitIntercept,
                  alpha,
                  l1_ratio,
                  ordered_feature_columns, // feature columns 전달
                  60000, // 타임아웃: 60초
                  tuningOptions
                );

                if (
                  !fitResult.coefficients ||
                  fitResult.coefficients.length !==
                    ordered_feature_columns.length
                ) {
                  throw new Error(
                    `Coefficient count mismatch: expected ${
                      ordered_feature_columns.length
                    }, got ${fitResult.coefficients?.length || 0}.`
                  );
                }

                intercept = fitResult.intercept;
                ordered_feature_columns.forEach((col, idx) => {
                  if (fitResult.coefficients[idx] !== undefined) {
                    coefficients[col] = fitResult.coefficients[idx];
                  } else {
                    throw new Error(
                      `Missing coefficient for feature ${col} at index ${idx}.`
                    );
                  }
                });
                tuningSummary = fitResult.tuning
                  ? {
                      enabled: Boolean(fitResult.tuning.enabled),
                      strategy: fitResult.tuning.strategy,
                      bestParams: fitResult.tuning.bestParams,
                      bestScore:
                        typeof fitResult.tuning.bestScore === "number"
                          ? fitResult.tuning.bestScore
                          : undefined,
                      scoringMetric: fitResult.tuning.scoringMetric,
                      candidates: Array.isArray(fitResult.tuning.candidates)
                        ? fitResult.tuning.candidates
                        : undefined,
                    }
                  : undefined;
                if (tuningSummary?.enabled && tuningSummary.bestParams) {
                  addLog(
                    "INFO",
                    `Hyperparameter tuning selected params: ${Object.entries(
                      tuningSummary.bestParams
                    )
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}.`
                  );
                }

                // Python에서 계산된 메트릭 사용
                const r2Value =
                  typeof fitResult.metrics["R-squared"] === "number"
                    ? fitResult.metrics["R-squared"]
                    : parseFloat(fitResult.metrics["R-squared"]);
                const mseValue =
                  typeof fitResult.metrics["Mean Squared Error"] === "number"
                    ? fitResult.metrics["Mean Squared Error"]
                    : parseFloat(fitResult.metrics["Mean Squared Error"]);
                const rmseValue =
                  typeof fitResult.metrics["Root Mean Squared Error"] ===
                  "number"
                    ? fitResult.metrics["Root Mean Squared Error"]
                    : parseFloat(fitResult.metrics["Root Mean Squared Error"]);

                metrics["R-squared"] = parseFloat(r2Value.toFixed(4));
                metrics["Mean Squared Error"] = parseFloat(mseValue.toFixed(4));
                metrics["Root Mean Squared Error"] = parseFloat(
                  rmseValue.toFixed(4)
                );

                addLog("SUCCESS", `Python으로 ${modelType} 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python LinearRegression 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (
              modelSourceModule.type === ModuleType.PoissonModel ||
              modelSourceModule.type === ModuleType.QuasiPoissonModel ||
              modelSourceModule.type === ModuleType.NegativeBinomialModel ||
              modelSourceModule.type === ModuleType.PoissonRegression ||
              modelSourceModule.type === ModuleType.NegativeBinomialRegression
            ) {
              // statsmodels를 사용한 포아송/음이항/Quasi-Poisson 회귀
              let distributionType: string;
              let maxIter: number;
              let disp: number;

              if (modelSourceModule.type === ModuleType.PoissonModel) {
                distributionType = "Poisson";
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = 1.0;
              } else if (
                modelSourceModule.type === ModuleType.QuasiPoissonModel
              ) {
                distributionType = "QuasiPoisson";
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = 1.0;
              } else if (
                modelSourceModule.type === ModuleType.NegativeBinomialModel
              ) {
                distributionType = "NegativeBinomial";
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = parseFloat(modelSourceModule.parameters.disp) || 1.0;
              } else {
                // 기존 모듈 (deprecated)
                distributionType =
                  modelSourceModule.parameters.distribution_type ||
                  (modelSourceModule.type === ModuleType.PoissonRegression
                    ? "Poisson"
                    : "NegativeBinomial");
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = parseFloat(modelSourceModule.parameters.disp) || 1.0;
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 ${distributionType} 회귀 모델 훈련 중 (statsmodels)...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitCountRegressionStatsmodels } = pyodideModule;

                const fitResult = await fitCountRegressionStatsmodels(
                  X,
                  y,
                  distributionType,
                  ordered_feature_columns,
                  maxIter,
                  disp,
                  60000 // 타임아웃: 60초
                );

                intercept = fitResult.intercept;
                Object.entries(fitResult.coefficients).forEach(
                  ([col, coef]) => {
                    coefficients[col] = coef;
                  }
                );

                // 통계량 설정
                Object.entries(fitResult.metrics).forEach(([key, value]) => {
                  if (typeof value === "number") {
                    metrics[key] = parseFloat(value.toFixed(4));
                  } else {
                    metrics[key] = value;
                  }
                });

                // TrainedModelOutput에 summary 정보 추가 (StatsModelsResultOutput 형식으로)
                trainedModelOutput = {
                  type: "TrainedModelOutput",
                  modelType: modelSourceModule.type,
                  modelPurpose: "regression",
                  coefficients,
                  intercept,
                  metrics,
                  featureColumns: ordered_feature_columns,
                  labelColumn: label_column,
                  tuningSummary: undefined,
                  // statsmodels 결과를 StatsModelsResultOutput 형식으로 저장
                  statsModelsResult: {
                    type: "StatsModelsResultOutput",
                    summary: fitResult.summary,
                    modelType: distributionType as StatsModelFamily,
                    labelColumn: label_column,
                    featureColumns: ordered_feature_columns,
                  },
                };

                addLog(
                  "SUCCESS",
                  `Python으로 ${distributionType} 회귀 모델 훈련 완료 (statsmodels)`
                );
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python ${distributionType} 회귀 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.DecisionTree) {
              // Pyodide를 사용하여 Python으로 Decision Tree 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              // modelPurpose에 따라 기본 criterion 설정
              const defaultCriterion =
                modelPurpose === "classification" ? "gini" : "mse";
              const criterion =
                modelSourceModule.parameters.criterion || defaultCriterion;
              const maxDepth =
                modelSourceModule.parameters.max_depth === "" ||
                modelSourceModule.parameters.max_depth === null ||
                modelSourceModule.parameters.max_depth === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.max_depth, 10);
              const minSamplesSplit =
                parseInt(modelSourceModule.parameters.min_samples_split, 10) ||
                2;
              const minSamplesLeaf =
                parseInt(modelSourceModule.parameters.min_samples_leaf, 10) ||
                1;
              const classWeight =
                modelPurpose === "classification"
                  ? modelSourceModule.parameters.class_weight || null
                  : null;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Decision Tree 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitDecisionTreePython } = pyodideModule;

                const fitResult = await fitDecisionTreePython(
                  X,
                  y,
                  modelPurpose,
                  criterion,
                  maxDepth,
                  minSamplesSplit,
                  minSamplesLeaf,
                  classWeight,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Decision Tree는 coefficients와 intercept가 없으므로 Feature Importance 사용
                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  // Feature Importance를 coefficients로 사용
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  // Feature Importance가 없는 경우 0으로 설정
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                // Python에서 계산된 메트릭 사용
                metrics["R-squared"] = parseFloat(
                  (fitResult.metrics["R-squared"] || 0).toFixed(4)
                );
                metrics["Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Root Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Mean Absolute Error"] = parseFloat(
                  (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                );

                addLog("SUCCESS", `Python으로 Decision Tree 모델 훈련 완료`);

                // Decision Tree plot_tree 생성을 위한 훈련 데이터와 모델 파라미터 저장
                const trainingDataForPlot = rows.map((row) => {
                  const dataRow: any = {};
                  ordered_feature_columns.forEach((col) => {
                    dataRow[col] = row[col];
                  });
                  dataRow[label_column] = row[label_column];
                  return dataRow;
                });

                // trainedModelOutput에 훈련 데이터와 모델 파라미터 추가
                if (!trainedModelOutput) {
                  trainedModelOutput = {
                    type: "TrainedModelOutput",
                    modelType: modelSourceModule.type,
                    modelPurpose: modelPurpose,
                    coefficients,
                    intercept,
                    metrics,
                    featureColumns: ordered_feature_columns,
                    labelColumn: label_column,
                    tuningSummary: undefined,
                    trainingData: trainingDataForPlot,
                    modelParameters: {
                      criterion,
                      maxDepth,
                      minSamplesSplit,
                      minSamplesLeaf,
                      classWeight,
                    },
                  };
                } else {
                  trainedModelOutput.trainingData = trainingDataForPlot;
                  trainedModelOutput.modelParameters = {
                    criterion,
                    maxDepth,
                    minSamplesSplit,
                    minSamplesLeaf,
                    classWeight,
                  };
                }
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Decision Tree 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.NeuralNetwork) {
              // Pyodide를 사용하여 Python으로 Neural Network 훈련 (회귀)
              const modelPurpose = "regression";
              const hiddenLayerSizes =
                modelSourceModule.parameters.hidden_layer_sizes || "100";
              const activation =
                modelSourceModule.parameters.activation || "relu";
              const maxIter =
                parseInt(modelSourceModule.parameters.max_iter, 10) || 200;
              const randomState =
                parseInt(modelSourceModule.parameters.random_state, 10) || 2022;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Neural Network 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitNeuralNetworkPython } = pyodideModule;

                const fitResult = await fitNeuralNetworkPython(
                  X,
                  y,
                  modelPurpose,
                  hiddenLayerSizes,
                  activation,
                  maxIter,
                  randomState,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Neural Network는 coefficients와 intercept가 없으므로 Feature Importance 사용
                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  // Feature Importance를 coefficients로 사용
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  // Feature Importance가 없는 경우 0으로 설정
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                // Python에서 계산된 메트릭 사용
                metrics["R-squared"] = parseFloat(
                  (fitResult.metrics["R-squared"] || 0).toFixed(4)
                );
                metrics["MSE"] = parseFloat(
                  (fitResult.metrics["MSE"] || 0).toFixed(4)
                );
                metrics["RMSE"] = parseFloat(
                  (fitResult.metrics["RMSE"] || 0).toFixed(4)
                );
                metrics["MAE"] = parseFloat(
                  (fitResult.metrics["MAE"] || 0).toFixed(4)
                );

                addLog("SUCCESS", `Python으로 Neural Network 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Neural Network 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.SVM) {
              // Pyodide를 사용하여 Python으로 SVM 훈련 (회귀)
              const modelPurpose = "regression";
              const kernel = modelSourceModule.parameters.kernel || "rbf";
              const C = parseFloat(modelSourceModule.parameters.C) || 1.0;
              const gamma =
                modelSourceModule.parameters.gamma === "" ||
                modelSourceModule.parameters.gamma === null ||
                modelSourceModule.parameters.gamma === undefined
                  ? "scale"
                  : modelSourceModule.parameters.gamma;
              const degree =
                parseInt(modelSourceModule.parameters.degree, 10) || 3;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 SVM 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitSVMPython } = pyodideModule;

                const gammaValue =
                  typeof gamma === "string" ? gamma : parseFloat(gamma);

                const fitResult = await fitSVMPython(
                  X,
                  y,
                  modelPurpose,
                  kernel,
                  C,
                  gammaValue,
                  degree,
                  false, // probability는 SVC(분류) 전용 옵션 — SVR(회귀)에서는 미사용
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // SVM은 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용
                metrics["R-squared"] = parseFloat(
                  (fitResult.metrics["R-squared"] || 0).toFixed(4)
                );
                metrics["Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Root Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Mean Absolute Error"] = parseFloat(
                  (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                );

                addLog("SUCCESS", `Python으로 SVM 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog("ERROR", `Python SVM 훈련 실패: ${errorMessage}`);
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.RandomForest) {
              // Pyodide를 사용하여 Python으로 Random Forest 훈련 (회귀, 결정적 random_state=42)
              const modelPurpose = "regression";
              const nEstimators =
                parseInt(modelSourceModule.parameters.n_estimators, 10) || 100;
              const criterion =
                modelSourceModule.parameters.criterion || "mse";
              const maxDepth =
                modelSourceModule.parameters.max_depth === "" ||
                modelSourceModule.parameters.max_depth === null ||
                modelSourceModule.parameters.max_depth === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.max_depth, 10);
              const maxFeatures =
                modelSourceModule.parameters.max_features === "" ||
                modelSourceModule.parameters.max_features === null ||
                modelSourceModule.parameters.max_features === undefined
                  ? null
                  : modelSourceModule.parameters.max_features;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Random Forest 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitRandomForestPython } = pyodideModule;

                const fitResult = await fitRandomForestPython(
                  X,
                  y,
                  modelPurpose,
                  nEstimators,
                  criterion,
                  maxDepth,
                  maxFeatures,
                  ordered_feature_columns,
                  60000
                );

                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                metrics["R-squared"] = parseFloat(
                  (fitResult.metrics["R-squared"] || 0).toFixed(4)
                );
                metrics["Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Root Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Mean Absolute Error"] = parseFloat(
                  (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                );

                addLog("SUCCESS", `Python으로 Random Forest 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Random Forest 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.GradientBoosting) {
              // Pyodide를 사용하여 Python으로 Gradient Boosting 훈련 (회귀, 결정적 random_state=42)
              const modelPurpose = "regression";
              const nEstimators =
                parseInt(modelSourceModule.parameters.n_estimators, 10) || 100;
              const learningRate =
                parseFloat(modelSourceModule.parameters.learning_rate) || 0.1;
              const maxDepth =
                parseInt(modelSourceModule.parameters.max_depth, 10) || 3;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Gradient Boosting 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitGradientBoostingPython } = pyodideModule;

                const fitResult = await fitGradientBoostingPython(
                  X,
                  y,
                  modelPurpose,
                  nEstimators,
                  learningRate,
                  maxDepth,
                  ordered_feature_columns,
                  60000
                );

                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                metrics["R-squared"] = parseFloat(
                  (fitResult.metrics["R-squared"] || 0).toFixed(4)
                );
                metrics["Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Root Mean Squared Error"] = parseFloat(
                  (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(4)
                );
                metrics["Mean Absolute Error"] = parseFloat(
                  (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                );

                addLog("SUCCESS", `Python으로 Gradient Boosting 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Gradient Boosting 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else {
              // For other regression models, use simulation for now
              intercept = Math.random() * 10;
              ordered_feature_columns.forEach((col) => {
                coefficients[col] = Math.random() * 5 - 2.5;
              });
              metrics["R-squared"] = 0.65 + Math.random() * 0.25;
              metrics["Mean Squared Error"] = 150 - Math.random() * 100;
              metrics["Root Mean Squared Error"] = Math.sqrt(
                metrics["Mean Squared Error"]
              );
            }
          } else if (modelIsClassification) {
            // Pyodide를 사용하여 Python으로 Logistic Regression 훈련
            if (modelSourceModule.type === ModuleType.LogisticRegression) {
              const penalty = modelSourceModule.parameters.penalty || "l2";
              const C = parseFloat(modelSourceModule.parameters.C) || 1.0;
              const solver = modelSourceModule.parameters.solver || "lbfgs";
              const maxIter =
                parseInt(modelSourceModule.parameters.max_iter, 10) || 100;

              const parseCandidates = (
                raw: any,
                fallback: number[]
              ): number[] => {
                if (Array.isArray(raw)) {
                  const parsed = raw
                    .map((val) => {
                      const num =
                        typeof val === "number" ? val : parseFloat(val);
                      return isNaN(num) ? null : num;
                    })
                    .filter((num): num is number => num !== null);
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "string") {
                  const parsed = raw
                    .split(",")
                    .map((part) => parseFloat(part.trim()))
                    .filter((num) => !isNaN(num));
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "number" && !isNaN(raw)) {
                  return [raw];
                }
                return fallback;
              };
              const tuningEnabled =
                modelSourceModule.parameters.tuning_enabled === "True";
              const tuningOptions = tuningEnabled
                ? {
                    enabled: true,
                    strategy: "GridSearch" as const,
                    cCandidates: parseCandidates(
                      modelSourceModule.parameters.c_candidates,
                      [C]
                    ),
                    l1RatioCandidates:
                      penalty === "elasticnet"
                        ? parseCandidates(
                            modelSourceModule.parameters.l1_ratio_candidates,
                            [0.5]
                          )
                        : undefined,
                    cvFolds: Math.max(
                      2,
                      parseInt(modelSourceModule.parameters.cv_folds, 10) || 5
                    ),
                    scoringMetric:
                      modelSourceModule.parameters.scoring_metric || "accuracy",
                  }
                : undefined;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Logistic Regression 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitLogisticRegressionPython } = pyodideModule;

                const fitResult = await fitLogisticRegressionPython(
                  X,
                  y,
                  penalty,
                  C,
                  solver,
                  maxIter,
                  ordered_feature_columns,
                  60000, // 타임아웃: 60초
                  tuningOptions
                );

                // Logistic Regression은 다중 클래스를 지원하므로 coefficients가 2D 배열일 수 있음
                if (
                  !fitResult.coefficients ||
                  !Array.isArray(fitResult.coefficients)
                ) {
                  throw new Error(
                    `Invalid coefficients: expected array, got ${typeof fitResult.coefficients}.`
                  );
                }

                // 이진 분류인 경우
                if (
                  fitResult.coefficients.length === 1 &&
                  fitResult.coefficients[0].length ===
                    ordered_feature_columns.length
                ) {
                  intercept = fitResult.intercept[0];
                  ordered_feature_columns.forEach((col, idx) => {
                    if (fitResult.coefficients[0][idx] !== undefined) {
                      coefficients[col] = fitResult.coefficients[0][idx];
                    } else {
                      throw new Error(
                        `Missing coefficient for feature ${col} at index ${idx}.`
                      );
                    }
                  });
                } else {
                  // 다중 클래스인 경우 첫 번째 클래스의 계수 사용
                  intercept = fitResult.intercept[0] || 0;
                  ordered_feature_columns.forEach((col, idx) => {
                    if (
                      fitResult.coefficients[0] &&
                      fitResult.coefficients[0][idx] !== undefined
                    ) {
                      coefficients[col] = fitResult.coefficients[0][idx];
                    } else {
                      coefficients[col] = 0;
                    }
                  });
                }

                tuningSummary = fitResult.tuning
                  ? {
                      enabled: Boolean(fitResult.tuning.enabled),
                      strategy: fitResult.tuning.strategy,
                      bestParams: fitResult.tuning.bestParams,
                      bestScore:
                        typeof fitResult.tuning.bestScore === "number"
                          ? fitResult.tuning.bestScore
                          : undefined,
                      scoringMetric: fitResult.tuning.scoringMetric,
                      candidates: Array.isArray(fitResult.tuning.candidates)
                        ? fitResult.tuning.candidates
                        : undefined,
                    }
                  : undefined;
                if (tuningSummary?.enabled && tuningSummary.bestParams) {
                  addLog(
                    "INFO",
                    `Hyperparameter tuning selected params: ${Object.entries(
                      tuningSummary.bestParams
                    )
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}.`
                  );
                }

                // Python에서 계산된 메트릭 사용
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog(
                  "SUCCESS",
                  `Python으로 Logistic Regression 모델 훈련 완료`
                );
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python LogisticRegression 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.KNN) {
              // Pyodide를 사용하여 Python으로 KNN 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              const nNeighbors =
                parseInt(modelSourceModule.parameters.n_neighbors, 10) || 3;
              const weights = modelSourceModule.parameters.weights || "uniform";
              const algorithm =
                modelSourceModule.parameters.algorithm || "auto";
              const metric = modelSourceModule.parameters.metric || "minkowski";

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 KNN 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitKNNPython } = pyodideModule;

                const fitResult = await fitKNNPython(
                  X,
                  y,
                  modelPurpose,
                  nNeighbors,
                  weights,
                  algorithm,
                  metric,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // KNN은 coefficients와 intercept가 없으므로 메트릭만 사용
                // coefficients와 intercept는 빈 값으로 설정
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용
                if (modelPurpose === "classification") {
                  metrics["Accuracy"] = parseFloat(
                    (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                  );
                  metrics["Precision"] = parseFloat(
                    (fitResult.metrics["Precision"] || 0).toFixed(4)
                  );
                  metrics["Recall"] = parseFloat(
                    (fitResult.metrics["Recall"] || 0).toFixed(4)
                  );
                  metrics["F1-Score"] = parseFloat(
                    (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                  );
                  if (fitResult.metrics["ROC-AUC"] !== undefined) {
                    metrics["ROC-AUC"] = parseFloat(
                      fitResult.metrics["ROC-AUC"].toFixed(4)
                    );
                  }
                } else {
                  metrics["R-squared"] = parseFloat(
                    (fitResult.metrics["R-squared"] || 0).toFixed(4)
                  );
                  metrics["Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Root Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(
                      4
                    )
                  );
                  metrics["Mean Absolute Error"] = parseFloat(
                    (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 KNN 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog("ERROR", `Python KNN 훈련 실패: ${errorMessage}`);
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.DecisionTree) {
              // Pyodide를 사용하여 Python으로 Decision Tree 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              const criterion =
                modelSourceModule.parameters.criterion || "gini";
              const maxDepth =
                modelSourceModule.parameters.max_depth === "" ||
                modelSourceModule.parameters.max_depth === null ||
                modelSourceModule.parameters.max_depth === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.max_depth, 10);
              const minSamplesSplit =
                parseInt(modelSourceModule.parameters.min_samples_split, 10) ||
                2;
              const minSamplesLeaf =
                parseInt(modelSourceModule.parameters.min_samples_leaf, 10) ||
                1;
              const classWeight =
                modelPurpose === "classification"
                  ? modelSourceModule.parameters.class_weight || null
                  : null;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Decision Tree 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitDecisionTreePython } = pyodideModule;

                const fitResult = await fitDecisionTreePython(
                  X,
                  y,
                  modelPurpose,
                  criterion,
                  maxDepth,
                  minSamplesSplit,
                  minSamplesLeaf,
                  classWeight,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Decision Tree는 coefficients와 intercept가 없으므로 Feature Importance 사용
                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  // Feature Importance를 coefficients로 사용
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  // Feature Importance가 없는 경우 0으로 설정
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                // Python에서 계산된 메트릭 사용
                if (modelPurpose === "classification") {
                  metrics["Accuracy"] = parseFloat(
                    (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                  );
                  metrics["Precision"] = parseFloat(
                    (fitResult.metrics["Precision"] || 0).toFixed(4)
                  );
                  metrics["Recall"] = parseFloat(
                    (fitResult.metrics["Recall"] || 0).toFixed(4)
                  );
                  metrics["F1-Score"] = parseFloat(
                    (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                  );
                  if (fitResult.metrics["ROC-AUC"] !== undefined) {
                    metrics["ROC-AUC"] = parseFloat(
                      fitResult.metrics["ROC-AUC"].toFixed(4)
                    );
                  }
                } else {
                  metrics["R-squared"] = parseFloat(
                    (fitResult.metrics["R-squared"] || 0).toFixed(4)
                  );
                  metrics["Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Root Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(
                      4
                    )
                  );
                  metrics["Mean Absolute Error"] = parseFloat(
                    (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Decision Tree 모델 훈련 완료`);

                // Decision Tree plot_tree 생성을 위한 훈련 데이터와 모델 파라미터 저장
                const trainingDataForPlot = rows.map((row) => {
                  const dataRow: any = {};
                  ordered_feature_columns.forEach((col) => {
                    dataRow[col] = row[col];
                  });
                  dataRow[label_column] = row[label_column];
                  return dataRow;
                });

                // trainedModelOutput에 훈련 데이터와 모델 파라미터 추가
                if (!trainedModelOutput) {
                  trainedModelOutput = {
                    type: "TrainedModelOutput",
                    modelType: modelSourceModule.type,
                    modelPurpose: modelPurpose,
                    coefficients,
                    intercept,
                    metrics,
                    featureColumns: ordered_feature_columns,
                    labelColumn: label_column,
                    tuningSummary: undefined,
                    trainingData: trainingDataForPlot,
                    modelParameters: {
                      criterion,
                      maxDepth,
                      minSamplesSplit,
                      minSamplesLeaf,
                      classWeight,
                    },
                  };
                } else {
                  trainedModelOutput.trainingData = trainingDataForPlot;
                  trainedModelOutput.modelParameters = {
                    criterion,
                    maxDepth,
                    minSamplesSplit,
                    minSamplesLeaf,
                    classWeight,
                  };
                }
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Decision Tree 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if ((modelSourceModule.type as ModuleType) === ModuleType.SVM) {
              // Pyodide를 사용하여 Python으로 SVM 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              const kernel = modelSourceModule.parameters.kernel || "rbf";
              const C = parseFloat(modelSourceModule.parameters.C) || 1.0;
              const gamma =
                modelSourceModule.parameters.gamma === "" ||
                modelSourceModule.parameters.gamma === null ||
                modelSourceModule.parameters.gamma === undefined
                  ? "scale"
                  : modelSourceModule.parameters.gamma;
              const degree =
                parseInt(modelSourceModule.parameters.degree, 10) || 3;
              const probability =
                modelSourceModule.parameters.probability === "True" ||
                modelSourceModule.parameters.probability === true;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 SVM 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitSVMPython } = pyodideModule;

                const gammaValue =
                  typeof gamma === "string" ? gamma : parseFloat(gamma);

                const fitResult = await fitSVMPython(
                  X,
                  y,
                  modelPurpose,
                  kernel,
                  C,
                  gammaValue,
                  degree,
                  probability,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // SVM은 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용
                if (modelPurpose === "classification") {
                  metrics["Accuracy"] = parseFloat(
                    (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                  );
                  metrics["Precision"] = parseFloat(
                    (fitResult.metrics["Precision"] || 0).toFixed(4)
                  );
                  metrics["Recall"] = parseFloat(
                    (fitResult.metrics["Recall"] || 0).toFixed(4)
                  );
                  metrics["F1-Score"] = parseFloat(
                    (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                  );
                  if (fitResult.metrics["ROC-AUC"] !== undefined) {
                    metrics["ROC-AUC"] = parseFloat(
                      fitResult.metrics["ROC-AUC"].toFixed(4)
                    );
                  }
                } else {
                  metrics["R-squared"] = parseFloat(
                    (fitResult.metrics["R-squared"] || 0).toFixed(4)
                  );
                  metrics["Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Root Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(
                      4
                    )
                  );
                  metrics["Mean Absolute Error"] = parseFloat(
                    (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 SVM 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog("ERROR", `Python SVM 훈련 실패: ${errorMessage}`);
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (
              (modelSourceModule.type as ModuleType) === ModuleType.LDA
            ) {
              // Pyodide를 사용하여 Python으로 LDA 훈련
              const solver = modelSourceModule.parameters.solver || "svd";
              const shrinkage =
                modelSourceModule.parameters.shrinkage === "" ||
                modelSourceModule.parameters.shrinkage === null ||
                modelSourceModule.parameters.shrinkage === undefined
                  ? null
                  : parseFloat(modelSourceModule.parameters.shrinkage);
              const nComponents =
                modelSourceModule.parameters.n_components === "" ||
                modelSourceModule.parameters.n_components === null ||
                modelSourceModule.parameters.n_components === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.n_components, 10);

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 LDA 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitLDAPython } = pyodideModule;

                const fitResult = await fitLDAPython(
                  X,
                  y,
                  solver,
                  shrinkage,
                  nComponents,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // LDA는 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용 (LDA는 분류만 지원)
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 LDA 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog("ERROR", `Python LDA 훈련 실패: ${errorMessage}`);
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if ((modelSourceModule.type as ModuleType) === ModuleType.NaiveBayes) {
              // Pyodide를 사용하여 Python으로 Naive Bayes 훈련
              const modelType =
                modelSourceModule.parameters.model_type || "GaussianNB";
              // model_type에서 "NB" 제거 (예: "GaussianNB" -> "Gaussian")
              const modelTypeShort = modelType.replace("NB", "");
              const alpha =
                parseFloat(modelSourceModule.parameters.alpha) || 1.0;
              const fitPrior =
                modelSourceModule.parameters.fit_prior === "True" ||
                modelSourceModule.parameters.fit_prior === true;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Naive Bayes 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitNaiveBayesPython } = pyodideModule;

                const fitResult = await fitNaiveBayesPython(
                  X,
                  y,
                  modelTypeShort,
                  alpha,
                  fitPrior,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Naive Bayes는 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용 (Naive Bayes는 분류만 지원)
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Naive Bayes 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Naive Bayes 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if ((modelSourceModule.type as ModuleType) === ModuleType.DecisionTree) {
              // Pyodide를 사용하여 Python으로 Decision Tree 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              const criterion =
                modelSourceModule.parameters.criterion || "gini";
              const maxDepth =
                modelSourceModule.parameters.max_depth === "" ||
                modelSourceModule.parameters.max_depth === null ||
                modelSourceModule.parameters.max_depth === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.max_depth, 10);
              const minSamplesSplit =
                parseInt(modelSourceModule.parameters.min_samples_split, 10) ||
                2;
              const minSamplesLeaf =
                parseInt(modelSourceModule.parameters.min_samples_leaf, 10) ||
                1;
              const classWeight =
                modelPurpose === "classification"
                  ? modelSourceModule.parameters.class_weight || null
                  : null;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Decision Tree 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitDecisionTreePython } = pyodideModule;

                const fitResult = await fitDecisionTreePython(
                  X,
                  y,
                  modelPurpose,
                  criterion,
                  maxDepth,
                  minSamplesSplit,
                  minSamplesLeaf,
                  classWeight,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Decision Tree는 coefficients와 intercept가 없으므로 Feature Importance 사용
                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  // Feature Importance를 coefficients로 사용
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  // Feature Importance가 없는 경우 0으로 설정
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                // Python에서 계산된 메트릭 사용
                if (modelPurpose === "classification") {
                  metrics["Accuracy"] = parseFloat(
                    (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                  );
                  metrics["Precision"] = parseFloat(
                    (fitResult.metrics["Precision"] || 0).toFixed(4)
                  );
                  metrics["Recall"] = parseFloat(
                    (fitResult.metrics["Recall"] || 0).toFixed(4)
                  );
                  metrics["F1-Score"] = parseFloat(
                    (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                  );
                  if (fitResult.metrics["ROC-AUC"] !== undefined) {
                    metrics["ROC-AUC"] = parseFloat(
                      fitResult.metrics["ROC-AUC"].toFixed(4)
                    );
                  }
                } else {
                  metrics["R-squared"] = parseFloat(
                    (fitResult.metrics["R-squared"] || 0).toFixed(4)
                  );
                  metrics["Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Root Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(
                      4
                    )
                  );
                  metrics["Mean Absolute Error"] = parseFloat(
                    (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Decision Tree 모델 훈련 완료`);

                // Decision Tree plot_tree 생성을 위한 훈련 데이터와 모델 파라미터 저장
                const trainingDataForPlot = rows.map((row) => {
                  const dataRow: any = {};
                  ordered_feature_columns.forEach((col) => {
                    dataRow[col] = row[col];
                  });
                  dataRow[label_column] = row[label_column];
                  return dataRow;
                });

                // trainedModelOutput에 훈련 데이터와 모델 파라미터 추가
                if (!trainedModelOutput) {
                  trainedModelOutput = {
                    type: "TrainedModelOutput",
                    modelType: modelSourceModule.type,
                    modelPurpose: modelPurpose,
                    coefficients,
                    intercept,
                    metrics,
                    featureColumns: ordered_feature_columns,
                    labelColumn: label_column,
                    tuningSummary: undefined,
                    trainingData: trainingDataForPlot,
                    modelParameters: {
                      criterion,
                      maxDepth,
                      minSamplesSplit,
                      minSamplesLeaf,
                      classWeight,
                    },
                  };
                } else {
                  trainedModelOutput.trainingData = trainingDataForPlot;
                  trainedModelOutput.modelParameters = {
                    criterion,
                    maxDepth,
                    minSamplesSplit,
                    minSamplesLeaf,
                    classWeight,
                  };
                }
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Decision Tree 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if ((modelSourceModule.type as ModuleType) === ModuleType.SVM) {
              // Pyodide를 사용하여 Python으로 SVM 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              const kernel = modelSourceModule.parameters.kernel || "rbf";
              const C = parseFloat(modelSourceModule.parameters.C) || 1.0;
              const gamma =
                modelSourceModule.parameters.gamma === "" ||
                modelSourceModule.parameters.gamma === null ||
                modelSourceModule.parameters.gamma === undefined
                  ? "scale"
                  : modelSourceModule.parameters.gamma;
              const degree =
                parseInt(modelSourceModule.parameters.degree, 10) || 3;
              const probability =
                modelSourceModule.parameters.probability === "True" ||
                modelSourceModule.parameters.probability === true;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 SVM 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitSVMPython } = pyodideModule;

                const gammaValue =
                  typeof gamma === "string" ? gamma : parseFloat(gamma);

                const fitResult = await fitSVMPython(
                  X,
                  y,
                  modelPurpose,
                  kernel,
                  C,
                  gammaValue,
                  degree,
                  probability,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // SVM은 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용
                if (modelPurpose === "classification") {
                  metrics["Accuracy"] = parseFloat(
                    (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                  );
                  metrics["Precision"] = parseFloat(
                    (fitResult.metrics["Precision"] || 0).toFixed(4)
                  );
                  metrics["Recall"] = parseFloat(
                    (fitResult.metrics["Recall"] || 0).toFixed(4)
                  );
                  metrics["F1-Score"] = parseFloat(
                    (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                  );
                  if (fitResult.metrics["ROC-AUC"] !== undefined) {
                    metrics["ROC-AUC"] = parseFloat(
                      fitResult.metrics["ROC-AUC"].toFixed(4)
                    );
                  }
                } else {
                  metrics["R-squared"] = parseFloat(
                    (fitResult.metrics["R-squared"] || 0).toFixed(4)
                  );
                  metrics["Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Root Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(
                      4
                    )
                  );
                  metrics["Mean Absolute Error"] = parseFloat(
                    (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 SVM 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog("ERROR", `Python SVM 훈련 실패: ${errorMessage}`);
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (
              (modelSourceModule.type as ModuleType) === ModuleType.LDA
            ) {
              // Pyodide를 사용하여 Python으로 LDA 훈련
              const solver = modelSourceModule.parameters.solver || "svd";
              const shrinkage =
                modelSourceModule.parameters.shrinkage === "" ||
                modelSourceModule.parameters.shrinkage === null ||
                modelSourceModule.parameters.shrinkage === undefined
                  ? null
                  : parseFloat(modelSourceModule.parameters.shrinkage);
              const nComponents =
                modelSourceModule.parameters.n_components === "" ||
                modelSourceModule.parameters.n_components === null ||
                modelSourceModule.parameters.n_components === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.n_components, 10);

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 LDA 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitLDAPython } = pyodideModule;

                const fitResult = await fitLDAPython(
                  X,
                  y,
                  solver,
                  shrinkage,
                  nComponents,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // LDA는 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용 (LDA는 분류만 지원)
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 LDA 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog("ERROR", `Python LDA 훈련 실패: ${errorMessage}`);
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if ((modelSourceModule.type as ModuleType) === ModuleType.NaiveBayes) {
              // Pyodide를 사용하여 Python으로 Naive Bayes 훈련
              const modelType =
                modelSourceModule.parameters.model_type || "GaussianNB";
              // model_type에서 "NB" 제거 (예: "GaussianNB" -> "Gaussian")
              const modelTypeShort = modelType.replace("NB", "");
              const alpha =
                parseFloat(modelSourceModule.parameters.alpha) || 1.0;
              const fitPrior =
                modelSourceModule.parameters.fit_prior === "True" ||
                modelSourceModule.parameters.fit_prior === true;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Naive Bayes 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitNaiveBayesPython } = pyodideModule;

                const fitResult = await fitNaiveBayesPython(
                  X,
                  y,
                  modelTypeShort,
                  alpha,
                  fitPrior,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Naive Bayes는 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용 (Naive Bayes는 분류만 지원)
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Naive Bayes 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Naive Bayes 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.NeuralNetwork) {
              // Pyodide를 사용하여 Python으로 Neural Network 훈련 (분류)
              const modelPurpose = "classification";
              const hiddenLayerSizes =
                modelSourceModule.parameters.hidden_layer_sizes || "100";
              const activation =
                modelSourceModule.parameters.activation || "relu";
              const maxIter =
                parseInt(modelSourceModule.parameters.max_iter, 10) || 200;
              const randomState =
                parseInt(modelSourceModule.parameters.random_state, 10) || 2022;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Neural Network 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitNeuralNetworkPython } = pyodideModule;

                const fitResult = await fitNeuralNetworkPython(
                  X,
                  y,
                  modelPurpose,
                  hiddenLayerSizes,
                  activation,
                  maxIter,
                  randomState,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // Neural Network는 coefficients와 intercept가 없으므로 메트릭만 사용
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Neural Network 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Neural Network 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.RandomForest) {
              // Pyodide를 사용하여 Python으로 Random Forest 훈련 (분류, 결정적 random_state=42)
              const modelPurpose = "classification";
              const nEstimators =
                parseInt(modelSourceModule.parameters.n_estimators, 10) || 100;
              const criterion =
                modelSourceModule.parameters.criterion || "gini";
              const maxDepth =
                modelSourceModule.parameters.max_depth === "" ||
                modelSourceModule.parameters.max_depth === null ||
                modelSourceModule.parameters.max_depth === undefined
                  ? null
                  : parseInt(modelSourceModule.parameters.max_depth, 10);
              const maxFeatures =
                modelSourceModule.parameters.max_features === "" ||
                modelSourceModule.parameters.max_features === null ||
                modelSourceModule.parameters.max_features === undefined
                  ? null
                  : modelSourceModule.parameters.max_features;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Random Forest 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitRandomForestPython } = pyodideModule;

                const fitResult = await fitRandomForestPython(
                  X,
                  y,
                  modelPurpose,
                  nEstimators,
                  criterion,
                  maxDepth,
                  maxFeatures,
                  ordered_feature_columns,
                  60000
                );

                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    (fitResult.metrics["ROC-AUC"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Random Forest 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Random Forest 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.GradientBoosting) {
              // Pyodide를 사용하여 Python으로 Gradient Boosting 훈련 (분류, 결정적 random_state=42)
              const modelPurpose = "classification";
              const nEstimators =
                parseInt(modelSourceModule.parameters.n_estimators, 10) || 100;
              const learningRate =
                parseFloat(modelSourceModule.parameters.learning_rate) || 0.1;
              const maxDepth =
                parseInt(modelSourceModule.parameters.max_depth, 10) || 3;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Gradient Boosting 모델 훈련 중...`
                );

                const pyodideModule = await import("./pyodideRunner");
                const { fitGradientBoostingPython } = pyodideModule;

                const fitResult = await fitGradientBoostingPython(
                  X,
                  y,
                  modelPurpose,
                  nEstimators,
                  learningRate,
                  maxDepth,
                  ordered_feature_columns,
                  60000
                );

                intercept = 0;
                if (
                  fitResult.featureImportances &&
                  Object.keys(fitResult.featureImportances).length > 0
                ) {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = fitResult.featureImportances[col] || 0;
                  });
                } else {
                  ordered_feature_columns.forEach((col) => {
                    coefficients[col] = 0;
                  });
                }

                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    (fitResult.metrics["ROC-AUC"] || 0).toFixed(4)
                  );
                }

                addLog("SUCCESS", `Python으로 Gradient Boosting 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python Gradient Boosting 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else {
              // For other classification models, use simulation for now
              intercept = Math.random() - 0.5;
              ordered_feature_columns.forEach((col) => {
                coefficients[col] = Math.random() * 2 - 1;
              });
              metrics["Accuracy"] = 0.75 + Math.random() * 0.2;
              metrics["Precision"] = 0.7 + Math.random() * 0.25;
              metrics["Recall"] = 0.7 + Math.random() * 0.25;
              metrics["F1-Score"] =
                (2 * (metrics["Precision"] * metrics["Recall"])) /
                (metrics["Precision"] + metrics["Recall"]);
            }
          } else {
            throw new Error(
              `Training simulation for model type '${modelSourceModule.type}' is not implemented, or its 'model_purpose' parameter is not set correctly.`
            );
          }

          // trainedModelOutput이 이미 설정되지 않은 경우에만 기본값으로 생성
          if (!trainedModelOutput) {
            trainedModelOutput = {
              type: "TrainedModelOutput",
              modelType: modelSourceModule.type,
              modelPurpose: modelIsClassification
                ? "classification"
                : "regression",
              coefficients,
              intercept,
              metrics,
              featureColumns: ordered_feature_columns,
              labelColumn: label_column,
              // 분류: 클래스 목록(코드 순서=정렬 순서) 저장 — ScoreModel이
              // 코드 예측(0..k-1)을 원 라벨로 복원할 수 있게 한다.
              ...(modelIsClassification &&
              Object.keys(labelClassMap).length >= 2
                ? {
                    classLabels: Object.keys(labelClassMap).sort(
                      (a, b) => labelClassMap[a] - labelClassMap[b]
                    ),
                  }
                : {}),
              tuningSummary,
            };
          }

          newOutputData = trainedModelOutput;
  return newOutputData;
}
