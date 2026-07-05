/**
 * runSimulation 실행기 배럴 — 도메인 파일 재수출(App.tsx 임포트 경로 불변).
 * 분할: shared(타입·isClassification) + stats·preprocessing·modeldefs·clustering·supervised.
 */
export * from "./simulationExecutors.shared";
export * from "./simulationExecutors.stats";
export * from "./simulationExecutors.preprocessing";
export * from "./simulationExecutors.modeldefs";
export * from "./simulationExecutors.clustering";
export * from "./simulationExecutors.supervised";
