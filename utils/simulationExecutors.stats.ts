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
import {
  AddLog,
  GetSingleInputData,
  GetCurrentModules,
  SetModules,
  isClassification,
} from "./simulationExecutors.shared";

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
