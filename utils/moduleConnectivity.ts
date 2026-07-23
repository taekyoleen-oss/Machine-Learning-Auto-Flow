import { ModuleType } from "../types";
import { DEFAULT_MODULES } from "../constants";

// 클러스터링 트레이너(TrainClusteringModel)가 받는 모델 계열.
// simulationExecutors.clustering.ts 의 런타임 허용 목록과 동일하게 유지한다.
export const CLUSTERING_MODEL_TYPES: ReadonlySet<ModuleType> = new Set([
  ModuleType.KMeans,
  ModuleType.PCA, // = PrincipalComponentAnalysis (동일 값 "PCA")
  ModuleType.DBSCAN,
  ModuleType.HierarchicalClustering,
]);

// 설명의 '연결 가능한 주요 모듈'에서 제외할 기초 분석 모듈(기술통계·상관 등).
// 연결 자체를 막는 게 아니라 설명 목록만 덜 어수선하게 한다.
const BASIC_MODULE_TYPES: ReadonlySet<ModuleType> = new Set([
  ModuleType.Statistics,
  ModuleType.NormalityChecker,
  ModuleType.ColumnPlot,
  ModuleType.OutlierDetector,
  ModuleType.HypothesisTesting,
  ModuleType.Correlation,
  ModuleType.VIFChecker,
]);

/**
 * 두 포트 사이 연결 허용 여부.
 * 기본은 코스 포트 타입 일치(data/model/evaluation/handler).
 * 추가로 model→model 연결에서 트레이너의 모델 계열을 강제한다(실행기가 런타임에
 * 이미 거부하는 규칙을 연결 시점으로 앞당겨 '부적절한 연결'을 막는다):
 *  - TrainClusteringModel: 클러스터링 모델(KMeans·PCA·DBSCAN·계층형)만
 *  - TrainModel(지도학습 트레이너): 클러스터링 모델은 불가
 */
export function isConnectionAllowed(
  fromType: ModuleType, // 출력(생산) 모듈
  fromPortType: string,
  toType: ModuleType, // 입력(소비) 모듈
  toPortType: string
): boolean {
  if (fromPortType !== toPortType) return false;
  if (fromPortType === "model") {
    if (toType === ModuleType.TrainClusteringModel) {
      return CLUSTERING_MODEL_TYPES.has(fromType);
    }
    if (toType === ModuleType.TrainModel) {
      return !CLUSTERING_MODEL_TYPES.has(fromType);
    }
  }
  return true;
}

/**
 * 이 모듈이 연결 가능한 '주요' 모듈(기초 분석 제외) — 포트 호환성으로 계산.
 * upstream = 이 모듈의 입력으로 들어올 수 있는 모듈,
 * downstream = 이 모듈의 출력이 갈 수 있는 모듈.
 */
export function getConnectableModules(moduleType: ModuleType): {
  upstream: ModuleType[];
  downstream: ModuleType[];
} {
  const self = DEFAULT_MODULES.find((m) => m.type === moduleType);
  const upstream = new Set<ModuleType>();
  const downstream = new Set<ModuleType>();
  if (self) {
    for (const other of DEFAULT_MODULES) {
      if (other.type === moduleType || BASIC_MODULE_TYPES.has(other.type)) {
        continue;
      }
      const goesTo = self.outputs.some((op) =>
        other.inputs.some((ip) =>
          isConnectionAllowed(moduleType, op.type, other.type, ip.type)
        )
      );
      if (goesTo) downstream.add(other.type);
      const comesFrom = other.outputs.some((op) =>
        self.inputs.some((ip) =>
          isConnectionAllowed(other.type, op.type, moduleType, ip.type)
        )
      );
      if (comesFrom) upstream.add(other.type);
    }
  }
  return { upstream: [...upstream], downstream: [...downstream] };
}
