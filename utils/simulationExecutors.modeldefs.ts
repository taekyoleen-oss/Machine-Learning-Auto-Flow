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
