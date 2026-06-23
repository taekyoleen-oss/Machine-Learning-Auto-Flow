# 3-7 모델 재학습/지속학습 — 모델 버전 스냅샷 (frontend)

날짜: 2026-06-22 / 적용: ML Auto Flow + JMDC (공통, byte-identical)

## 변경/추가 파일 (양쪽 동일)
- `utils/retrainExport.ts` (신규) — 버전 스냅샷 코드 생성기. `inspectScoringPipeline` 재사용 + LoadData 데이터 소스 추론. `inspectRetrainPipeline`, `sanitizeVersionLabel`, `generateRetrainSnapshotCode`.
- `components/PipelineCodePanel.tsx` (수정) — 헤더에 🗂️ "모델 버전" 버튼(AdvancedOnly 게이트, 스코어링과 상호배타), 버전 라벨/데이터 소스 입력 + 지속학습 워크플로 안내 패널 + 스냅샷 코드 미리보기.

## 스냅샷 코드가 만드는 것
- 메타 헤더: VERSION / 모델타입 / 라벨컬럼 / 피처컬럼 / 데이터소스 / 워크플로 안내.
- `joblib.dump(trained_model, 'model_<ver>.joblib')` + 버전 메타 JSON 사이드카(`model_<ver>.meta.json`: version/estimator/feature_columns/label_column/data_source/metrics{}).
- 로드/버전비교 가이드 블록.

## 재현성/안전
- 버전 라벨은 UI 입력값(기본 'v1')만 사용 — Date.now() 등 자동 타임스탬프 없음 → 동일 입력=동일 출력(결정적).
- 실행엔진/연결/codeSnippets 실행템플릿/generatePipelineCode/verify 경로 미변경. 읽기 전용 메타 inspect만.
- 고급기능 비밀번호 게이트(AdvancedOnly + AdvancedLockBadge + ADVANCED_BTN_DIM) — 스코어링과 동일.

## 검증
- vite build: 양쪽 성공.
- npm run verify:pipelines: 양쪽 15/15 PASS.
- 변경 코드 양쪽 byte-identical 확인(diff 무차이).
