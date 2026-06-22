# 횡단 공통 개선 계획 — 4개 노드기반 앱 입출력(I/O) 일괄 적용

> 출처 근거: `docs/azure_ml_book/01_book_based_improvements.md` (Jeff Barnes, *Microsoft Azure Essentials: Azure Machine Learning*, 2015 기반).
> 본 문서는 **실행 가능한 체크리스트형 SSOT**다. 01의 개선안 중 **도메인과 무관하게 공통으로 쓰이는 입출력(I/O) 영역**을 추려, 유사 구조의 4개 앱에 **한 번에** 적용할 수 있도록 항목별 적용 매트릭스 · 앱별 영향 파일 경로 · 진행 순서를 담는다.
> **계획서**이며 실제 코드 구현은 사용자 검토·승인 후 별도 진행한다.

---

## 0. 목적 · 범위

`01_book_based_improvements.md`는 ML Auto Flow(베이스)를 위한 개선안이지만, 네 앱 모두 **같은 패러다임**을 공유한다:

> 캔버스에 모듈(노드)을 배치 → 포트로 연결 → 실행 → 결과 미리보기 → 내보내기

따라서 **데이터를 어떻게 들여오고(입력) 결과를 어떻게 내보내는가(출력)**, 그리고 **재현성·샘플·메타데이터** 영역은 도메인(ML 교과서 / 헬스케어 / 재보험 계리 / 생명보험)과 무관하게 공통이다. 반면 **sklearn 특화 모듈**(부스팅 트리·평가지표·스윕·추천)은 Python 런타임이 있는 앱에만 해당한다.

### 대상 4개 앱 구조 요약

| 앱 | 아키텍처 | Python/Pyodide | verify/ | docs/ | 도메인 | dev 포트 |
|---|---|---|---|---|---|---|
| **ML Auto Flow** (원본) | 캔버스 + `types.ts`/`constants.ts`/`codeSnippets.ts` | ✅ Pyodide + sklearn | ✅ (`verify/run-verification.mjs`) | ✅ `azure_ml_book/` | ML 교과서 | 3003 |
| **ML_Auto_Flow-JMDC** | **동일 코드베이스(상위집합)** | ✅ | ✅ (`datasets/` 부재) | ❌ → 신설 | ML + JMDC 헬스케어(J1~J7) | 3003 / 백엔드 3002 |
| **DFA-Auto-Flow** | **동일 패턴**(Canvas, `codeSnippets.ts`, `generatePipelineCode.ts`, `data_analysis_modules.py`) | ✅ Pyodide + sklearn | △ (`.claude` 하네스만, 정식 verify 미정) | ❌ → 신설 | 보험계리 DFA / XoL 재보험 | 3001 |
| **life matrix flow new** | **동일 패턴**(캔버스, `types.ts`/`constants.ts`), `codeSnippets.ts`는 **의사코드 전용** | ❌ **순수 TypeScript** | ❌ → TS 신설 | ✅ (`docs/ClaimsReserveModules.md` 등) | 생명보험 보험료/준비금 | 3005 |

> **핵심 분기:** ML Auto Flow · JMDC · DFA는 Pyodide로 실제 Python을 실행하므로 sklearn 모듈·byte-identical 재현이 가능하다. **life matrix flow new는 순수 TS**라 sklearn 계열 항목이 **해당 없음**이고, 대신 동일 개념을 **TS 결정성**으로 재현·검증한다.

---

## 1. 공통 적용 매트릭스 (01 항목 × 4개 앱)

| 01 항목 | ML Auto Flow | JMDC | DFA | life matrix | 성격 |
|---|:---:|:---:|:---:|:---:|---|
| **2-3 데이터 개요/요약 패널** | ✅ | ✅ | ✅ | ✅ | **공통 I/O(입력)** |
| **3-2 URL/원격 데이터 로더** | ✅ | ✅ | ✅(claim) | ✅(TS `fetch`) | **공통 I/O(입력)** |
| **3-3 번들 샘플/레퍼런스 파이프라인 확장** | ✅ | ✅ | ✅ | ✅ | **공통 I/O** |
| **샘플 메타데이터 스키마 강화** | ✅ | ✅ | ✅ | ✅ | **공통 I/O** |
| **재현성 verify 하네스** | 기존 | 기존 | △→정식화 | ❌→**TS 신설** | **공통(고가치)** |
| **3-6 스코어링·결과 내보내기** | ✅ | ✅ | ✅ | △(이미 Excel/PPT/.lifx) | **공통 I/O(출력)** |
| 2-1 Evaluate ROC/AUC·혼동행렬 | ✅ | ✅ | ✅ | ❌ | sklearn 계열 |
| 2-2 회귀지표 정합(RMSE/MAE 등) | ✅ | ✅ | ✅ | ❌ | sklearn 계열 |
| 3-1 그래디언트 부스팅 모듈 | ✅ | ✅ | ✅(지도학습군) | ❌ | sklearn 계열 |
| 3-4 하이퍼파라미터 스윕/CV | ✅ | ✅ | △ | ❌ | sklearn 계열 |
| 3-5 추천 모듈 | ✅ | ✅(헬스케어 교차판매) | ❌ | ❌ | sklearn 계열 |
| 3-7 재학습/지속학습 워크플로 | ✅ | ✅ | ✅ | △(시나리오 재실행) | 모델 계열 |

> ✅=직접 적용, △=부분/대체 적용, ❌=해당 없음. 본 공통 문서는 **상단 6개(공통 I/O + verify)**를 다룬다. sklearn 계열·재학습은 각 앱의 `01_improvements_*.md`에서 도메인 맥락으로 다룬다.

---

## 2. 공통 작업 단위 (체크리스트)

각 작업 단위는 **한 번에(앱 4개 동시) 적용 가능한 최소 단위**다. 앱별 영향 파일은 탐색으로 확인된 실제 경로다.

---

### ☐ 작업 1 — 데이터 개요/요약 패널 (01의 2-3)

데이터를 들여온 직후 **열별 타입·고유값·결측치 수·간단 분포**를 한 화면에 요약. 입력 단계의 공통 UX.

| 앱 | 영향 파일 | 비고 |
|---|---|---|
| ML / JMDC | `components/StatisticsPreviewModal.tsx`, 신규 개요 뷰; (Python측 결측 카운트 `data_analysis_modules.py`) | 결측치 수 강조(책 workclass 1,836 결측 예시) |
| DFA | `components/PropertiesPanel.tsx`(데이터 미리보기 탭), `components/DataPreviewModal.tsx` | claim 데이터 열 요약 |
| life matrix | `components/ParameterInputModal.tsx`, `components/DataPreviewModal.tsx` | 위험률/요율 입력 요약(TS 집계) |

- **재현성 영향:** 없음(읽기 전용 요약).
- **동기화:** ML↔JMDC 동일 적용.

---

### ☐ 작업 2 — URL/원격 데이터 로더 (01의 3-2) ★입출력 핵심

로컬 파일 전용인 데이터 입력에 **URL 소스 옵션** 추가.

| 앱 | 영향 파일 | 구현 메모 |
|---|---|---|
| ML / JMDC | `codeSnippets.ts`(LoadData 템플릿 `pd.read_csv({source})`는 URL도 동작), `components/PropertiesPanel.tsx`(URL 입력 필드), `server/`(CORS 프록시 엔드포인트), `data_analysis_modules.py` | Pyodide CORS는 서버 프록시 경유 |
| DFA | `codeSnippets.ts`(LoadClaimData/LoadData), `components/PropertiesPanel.tsx`, `server/split-data-server.js`(프록시) | claim CSV URL |
| life matrix | `codeSnippets.ts`(의사코드만 갱신), 입력 로직 `components/ParameterInputModal.tsx`에서 **TS `fetch()`** 로 CSV 수신 | 표준생명표/요율 URL. Python 아님 |

- **재현성 영향:** URL 데이터는 외부 가변 → 검증용은 **로컬 스냅샷 권장**(문서 명시).
- **동기화:** ML↔JMDC 동일. life matrix는 fetch 기반 대체 구현.

---

### ☐ 작업 3 — 번들 샘플/레퍼런스 파이프라인 확장 (01의 3-3) ★난이도 낮음·고효용

각 앱에 도메인 대표 예제를 **로드 즉시 실행 가능한 레퍼런스 파이프라인**으로 추가.

| 앱 | 영향 파일 | 추가할 레퍼런스(예) |
|---|---|---|
| ML / JMDC | `samples/*.json`, `samples/samples-metadata.json`, (ML) `verify/datasets/` | 이미 `Book_*`(자동차/성인/도매) 존재 → 확장 |
| JMDC 전용 | 위 + `data/jmdc_synthetic/` 코호트 레퍼런스 | J1~J7 합성코호트 데모 |
| DFA | `sampleData.ts`, `savedSamples.ts`, `samples/`, `localSamples.json` | DFA 집계분포·XoL 프라이싱 레퍼런스 |
| life matrix | `samples/*.lifx`, `samples/samples-metadata.json`, `sampleData.ts` | 종신/정기/양로 + 표준위험률 |

- **재현성 영향:** 데이터 고정으로 **강화**.
- **동기화:** ML↔JMDC 공통 자산.

---

### ☐ 작업 4 — 샘플 메타데이터 스키마 강화

`samples-metadata.json`을 풍부하게(설명·태그·카테고리·데이터파일·기대 출력·도메인 속성).

| 앱 | 영향 파일 | 도메인 속성(예) |
|---|---|---|
| ML / JMDC | `samples/samples-metadata.json` | 모델 종류·데이터 출처·기대 지표 |
| DFA | `localSamples.json`, `utils/samples.ts`, (있으면) `samples-metadata.json` | 분포가정·재보험 레이어 |
| life matrix | `samples/samples-metadata.json` | 보험종류(종신/정기/양로)·계리기초·규제구분·기대 보험료 |

- **재현성 영향:** 없음(메타데이터).
- **동기화:** 스키마 형태 4개 앱 통일 권장(횡단 이식성).

---

### ☐ 작업 5 — 재현성 verify 하네스 (정식화 / TS 신설) ★고가치

내보낸 코드가 외부에서 **동일 결과로 재현**되는지 자동 회귀 검증.

| 앱 | 현황 | 작업 |
|---|---|---|
| ML | ✅ `verify/run-verification.mjs` + `verify/pipelines/*.json` | 신규 모듈 픽스처 추가만 |
| JMDC | ✅ 동일 하네스(`verify/datasets/` 부재) | `datasets/` 보강 + 픽스처 |
| DFA | △ `.claude` 하네스(python-parity-qa)만 | `verify/run-verification.mjs` 패턴 **정식 이식**(Python 2회 byte-identical) |
| life matrix | ❌ 없음 | **TS 재현성 하네스 신설** — `verify/`에 파이프라인 JSON + 기대 보험료/준비금 출력, TS 실행엔진 2회 실행 동일성 단언(`vitest` 활용) |

- **재현성 영향:** 이 작업이 곧 재현성 보증 장치.
- **동기화:** ML↔JMDC 픽스처 공유 가능.

---

### ☐ 작업 6 — 스코어링·결과 내보내기 (01의 3-6) ★입출력(출력)

학습/계산 결과를 외부에서 쓸 수 있게 내보내기.

| 앱 | 영향 파일 | 형태 |
|---|---|---|
| ML / JMDC | `utils/`(신규 내보내기), `codeSnippets.ts`, `components/PipelineCodeModal.tsx` | `joblib` 저장 + FastAPI/Flask 스코어링 스니펫 + 요청/응답 JSON. **고급기능 게이트 대상** |
| DFA | `utils/generatePipelineCode.ts`, `components/PipelineCode*` 상당물 | 모델/프라이싱 결과 스코어링 스니펫 |
| life matrix | `utils/buildSlideReport.ts`, `DataPreviewModal.tsx`(xlsx), `utils/fileOperations.ts`(.lifx) | **이미 Excel/PPT/.lifx 보유** → 보험료 산출 결과 "재계산 함수" 내보내기로 보강 |

- **재현성 영향:** 직렬화 결정성 유지.
- **동기화:** ML↔JMDC 공통. 모두 **고급기능 비밀번호 게이트**(API·코드 영역) 정책과 정합.

---

## 3. 권장 진행 순서 & 의존성

난이도 낮고 효용 높은 입출력부터, 검증을 마지막에 두어 앞 작업을 한꺼번에 확정한다.

```
작업4(메타 스키마) → 작업3(샘플 확장) → 작업1(데이터 개요)
     → 작업2(URL 로더) → 작업6(내보내기) → 작업5(verify 정식화/신설)
```

| 순서 | 작업 | 난이도 | 의존성 |
|---|---|---|---|
| 1 | 4. 샘플 메타 스키마 | 낮음 | — |
| 2 | 3. 샘플/레퍼런스 확장 | 낮음 | 작업4 스키마 |
| 3 | 1. 데이터 개요 패널 | 낮음~중 | — |
| 4 | 2. URL 로더 | 중(CORS) | — |
| 5 | 6. 결과/스코어링 내보내기 | 중 | — |
| 6 | 5. verify 하네스 | 중 | 작업3 샘플·작업2/6 산출 검증 |

> sklearn 계열(2-1·2-2·3-1·3-4·3-5)·재학습(3-7)은 **각 앱 `01_improvements_*.md`**에서 도메인 우선순위로 별도 계획.

---

## 4. 불변식 가드레일 (모든 공통 작업이 지킬 것)

1. **Python 재현성(ML·JMDC·DFA):** `data_analysis_modules.py` ↔ `codeSnippets.ts` 정합 유지, 무작위 단계 `random_state=42` 고정, `verify` 픽스처로 byte-identical 확인.
2. **두 앱 동기화(ML↔JMDC):** 공통 변경은 양쪽 동일 적용, JMDC 전용(헬스케어)만 차이로 남김.
3. **life matrix(순수 TS) 결정성:** 무작위·부동소수 연산 순서 고정으로 2회 실행 동일 출력 보장(sklearn 미적용).
4. **고급기능 게이트:** URL 로더·코드/스코어링 내보내기 등 API·코드 영역은 각 앱의 비밀번호 게이트(Advanced*Context) 정책에 편입.

---

## 5. 연계 문서

| 문서 | 위치 | 내용 |
|---|---|---|
| 원본 개선안 | `ML Auto Flow/docs/azure_ml_book/01_book_based_improvements.md` | 본 공통 항목의 근거 |
| 책자 방향(원본) | `ML Auto Flow/docs/azure_ml_book/02_app_booklet_direction.md` | 각 앱 `02_app_booklet_direction.md`의 원형 |
| JMDC 앱별 계획 | `ML_Auto_Flow-JMDC/docs/book_based_plan/01_improvements_jmdc.md` | sklearn 계열 + 헬스케어 |
| DFA 앱별 계획 | `DFA-Auto-Flow/docs/book_based_plan/01_improvements_dfa.md` | 계리 도메인 재해석 |
| life matrix 앱별 계획 | `life matrix flow new/docs/book_based_plan/01_improvements_life.md` | I/O·검증·책자 중심(sklearn 해당 없음) |
