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
