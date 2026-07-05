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
