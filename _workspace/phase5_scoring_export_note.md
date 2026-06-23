# Phase 5 (작업 6) — 스코어링/결과 내보내기 변경 노트

## 추가/변경 파일 (양쪽 앱 동일)
- utils/scoringExport.ts (신규): 스코어링 스니펫 빌더
- components/PipelineCodePanel.tsx (수정): 고급기능 게이트 '스코어링' 토글 + 패널

## 헬퍼 API (utils/scoringExport.ts)
- inspectScoringPipeline(modules, connections): ScoringExportInfo
  -> TrainModel 탐색, feature_columns/label_column/추정기 라벨 추론, available 플래그
- buildScoringJsonSamples(info): { request, response }  (결정적 해시 기반 샘플값)
- generateScoringCode(modules, connections, framework='fastapi'|'flask'): string
  -> [1] joblib 저장/로드 + FEATURE_COLUMNS + [2] FastAPI/Flask /predict + [3] 요청/응답 JSON 샘플 + curl

## 게이트
- PipelineCodePanel 헤더의 '🚀 스코어링' 버튼은 <AdvancedOnly>로 감쌈 (ADVANCED_BTN_DIM + AdvancedLockBadge)
- 스코어링 패널 본문도 <AdvancedOnly>로 이중 보호. PipelineCodePanel 자체도 App.tsx에서 isAdvancedUnlocked일 때만 열림.

## 불변식 가드레일 준수
- generateFullPipelineCode / codeSnippets 실행 템플릿 미변경 (read-only 사용만).
- 모듈 실행/연결/data_analysis_modules.py/시각화 미변경.
- verify:pipelines 12/12 PASS (양쪽), vite build 성공 (양쪽).
- 두 파일 byte-identical (diff 확인).
