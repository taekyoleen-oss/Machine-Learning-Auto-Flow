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

          let ordered_feature_columns = inputData.columns
            .map((c) => c.name)
            .filter((name) => feature_columns.includes(name));

          if (ordered_feature_columns.length === 0) {
            // 저장된 feature_columns가 현재 데이터의 어떤 컬럼과도 일치하지 않음
            // (예: 인코딩이 깨진 채 저장된 이전 파이프라인, 업스트림 컬럼명 변경 등).
            // 바로 실패시키는 대신 빈 선택과 동일하게 수치형 컬럼으로 자동 대체한다.
            const numericColumns = inputData.columns
              .filter(
                (c) => c.type.startsWith("int") || c.type.startsWith("float")
              )
              .map((c) => c.name);
            if (numericColumns.length === 0) {
              throw new Error("No valid feature columns found in the data.");
            }
            addLog(
              "WARN",
              `저장된 feature_columns가 현재 데이터와 일치하지 않아 수치형 컬럼(${numericColumns.join(", ")})으로 자동 대체합니다.`
            );
            ordered_feature_columns = numericColumns;
            module.parameters.feature_columns = numericColumns;
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
