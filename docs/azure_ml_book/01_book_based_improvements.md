# 산출물 1 — 책 기반 개선 사항 및 추가 기능

> 출처: Jeff Barnes, *Microsoft Azure Essentials: Azure Machine Learning* (Microsoft Press, 2015).
> 대상: **ML Auto Flow (베이스)** 기준. 향후 구현 시 **JMDC 동기화 필요 지점**을 각 항목에 표기.
> 본 문서는 **계획·제안서**이며, 실제 코드 구현은 사용자 검토·승인 후 별도 진행한다.

---

## 0. 한눈에 보기

이 책은 Azure ML Studio의 **시각적 파이프라인** 패러다임 —
`데이터셋 → 시각화 → Split → Train → Score → Evaluate → 웹서비스 배포 → 재학습` — 을 따른다.
ML Auto Flow는 이 흐름의 **모델링 부분(데이터셋~Evaluate)**을 이미 대부분 충실히 구현했고,
오히려 **브라우저 내 Python(Pyodide) 실행 + byte-identical 재현 검증 + AI 해설**이라는 책에 없는 강점을 갖췄다.

반면 책에는 있으나 앱에 **없는 영역**은 주로 파이프라인의 **앞단(데이터 수급)과 뒷단(배포·재학습)**, 그리고
책의 핵심 알고리즘인 **부스팅 트리**·**추천기**다. 아래 개선/추가 기능은 이 격차를 메우는 것을 우선한다.

---

## 1. 책 워크플로 vs 앱 현황 비교표

| 책(Azure ML Studio) 단계 | 책 모듈 | ML Auto Flow 대응 | 상태 |
|---|---|---|---|
| 데이터 업로드 | Upload Dataset | `LoadData` (CSV/Excel, Pyodide) | ✅ |
| 외부 저장소 직접 로드 | **Reader (URL)** | — | ❌ 없음 |
| 데이터 시각화/통계 | Visualize dataset | `Statistics`, `Correlation`, `ColumnPlot` | △ 분산되어 있음 |
| 결측치 처리 | Clean Missing Data | `HandleMissingValues` (mean/median/mode/KNN) | ✅ |
| 열 선택/제외 | Project Columns | `SelectData`, `DataFiltering` | ✅ |
| 데이터 분할 | Split | `SplitData` (75/25, stratify, seed=42) | ✅ |
| 모델 정의 | Linear/Logistic Regression, Decision Tree 등 | 11종 지도학습 모듈 | ✅ |
| **부스팅 트리** | **(Two-Class) Boosted Decision Tree** | — | ❌ 없음 |
| 학습 | Train Model | `TrainModel` | ✅ |
| 예측 | Score Model | `ScoreModel` | ✅ |
| 평가 | Evaluate Model (ROC/AUC/혼동행렬) | `EvaluateModel` | △ 지표 일부만 |
| 군집 | K-Means + Train Clustering + Assign to Clusters | `KMeans`+`TrainClusteringModel`+`ClusteringData` | ✅ |
| **추천** | **Matchbox Recommender** | — | ❌ 없음 |
| 하이퍼파라미터 탐색 | Sweep Parameters / Cross-Validate | — | ❌ 없음 |
| 웹서비스 배포 | Publish as Web Service (요청/응답 JSON) | — | ❌ Python 내보내기만 |
| 재학습 | Retrain / Batch Execution | — | ❌ 없음 |

---

## 2. 개선 항목 (기존 기능 강화)

### 2-1. EvaluateModel을 "책 수준"으로 강화 — *우선순위 높음*
책 Ch3는 평가 시 **ROC 곡선·AUC·혼동행렬(Confusion Matrix)·정밀도/재현율·임계값(threshold) 조정**을 핵심으로 다룬다.
- 현행 `EvaluateModel`은 분류 지표(accuracy/precision/recall/F1)와 혼동행렬, 회귀 지표(MSE/R²)를 제공.
- **보강**: ROC/AUC 곡선 시각화, 임계값 슬라이더에 따른 지표 변화, PR 곡선 추가.
- 영향 파일: `data_analysis_modules.py`(평가 로직), `codeSnippets.ts`(Python 템플릿), `components/EvaluationPreviewModal.tsx`(시각화).
- **재현성 영향**: 지표 계산은 결정적 → 불변식 영향 없음. 템플릿 정합만 유지(`data_analysis_modules.py` ↔ `codeSnippets.ts`).
- **두 앱 동기화**: 공통(양쪽 동일 적용).

### 2-2. 회귀 평가지표 용어 정합
책 Ch5는 회귀 평가에서 **MAE, RMSE, Relative Squared Error, Relative Absolute Error, Coefficient of Determination(R²)** 5종을 표준으로 제시.
- **보강**: `EvaluateModel(regression)`에 RMSE·MAE·상대오차 항목을 명시적으로 노출(현재 MSE/R² 중심).
- 영향: `data_analysis_modules.py`, `codeSnippets.ts`, `EvaluationPreviewModal.tsx`.
- 재현성: 결정적. 두 앱: 공통.

### 2-3. 통합 "데이터 시각화/요약" 패널
책 Ch3는 데이터셋을 올리면 열별 **고유값 수·결측치 수·히스토그램**을 즉시 보여준다.
- **보강**: `Statistics`/`Correlation`/`ColumnPlot`로 흩어진 탐색 기능을 LoadData 직후 "데이터 개요" 한 화면으로 묶기(결측치 수 강조 — 책의 workclass 1,836개 결측 예시처럼).
- 영향: `components/StatisticsPreviewModal.tsx`, 신규 개요 뷰. 재현성 영향 없음. 두 앱: 공통.

---

## 3. 추가 기능 (신규 모듈/기능)

각 항목: **출처 / 근거 / 영향 파일 / 난이도 / 우선순위 / 재현성·동기화 영향**.

### 3-1. 그래디언트 부스팅 트리 모듈 (분류·회귀) — ★최우선
- **출처**: Ch3(Two-Class Boosted Decision Tree로 성인 소득 예측), 책 전반의 대표 알고리즘.
- **근거**: 책의 1순위 분류기인데 앱엔 부재. DecisionTree/RandomForest만 있어 책 예제를 "동일 알고리즘"으로 재현 불가.
- **구현**: `sklearn.ensemble.GradientBoostingClassifier/Regressor` 또는 `HistGradientBoosting*`. 기존 `DecisionTree`/`RandomForest` 모듈 패턴을 그대로 복제.
- **영향 파일**: `types.ts`(`GradientBoosting` enum), `constants.ts`(팔레트+DEFAULT_MODULES), `codeSnippets.ts`(템플릿, `random_state=42`), `data_analysis_modules.py`, `components/PropertiesPanel.tsx`(n_estimators·learning_rate·max_depth 편집기), `components/ComponentRenderer.tsx`.
- **난이도**: 중(기존 트리 모듈 미러링이라 낮은 편).
- **재현성**: `random_state=42` 고정 필수. 신규 `verify/pipelines/` 픽스처로 byte-identical 검증.
- **두 앱 동기화**: 공통(양쪽 동일).

### 3-2. URL/공개 저장소 데이터 로더 — ★높음
- **출처**: Ch6 — Reader 모듈로 UCI URL(`Wholesale customers`)을 인터넷에서 직접 로드.
- **근거**: 현 `LoadData`는 로컬 파일 전용. 책 예제 데이터(UCI)를 바로 끌어오는 경험 부재.
- **구현**: `LoadData`에 "URL" 소스 옵션 추가 → `pd.read_csv(url)`. Pyodide에서는 CORS/프록시 고려(서버 경유 fetch 가능, Express 백엔드에 프록시 엔드포인트).
- **영향**: `codeSnippets.ts`(LoadData 템플릿 `pd.read_csv({source})`는 URL도 그대로 동작), `components/PropertiesPanel.tsx`, `server/`(프록시), `data_analysis_modules.py`.
- **난이도**: 중(브라우저 CORS가 변수).
- **재현성**: URL 데이터는 외부 가변성 있음 → 재현 검증용으론 로컬 스냅샷 권장(문서에 명시).
- **두 앱 동기화**: 공통.

### 3-3. 번들 샘플 데이터셋 확장 (책과 1:1) — ★높음·난이도 낮음
- **출처**: 책 전 챕터(Adult, Automobile, Wholesale, Restaurant).
- **근거**: 현재 번들은 사실상 iris + `Examples_in_Load/`의 몇 종. 책과 동일 데이터로 학습 경험 제공.
- **구현**: `Examples_in_Load/`(및 `verify/datasets/`)에 `imports-85`, `adult`, `wholesale` 추가 + `samples-metadata.json` 등록. (산출물 3에서 실제 수행)
- **영향**: `Examples_in_Load/`, `sampleData.ts`, `samples/`, `samples-metadata.json`.
- **난이도**: 낮음. **재현성**: 데이터 고정으로 오히려 강화. **두 앱**: 공통.

### 3-4. 하이퍼파라미터 스윕 / 교차검증 — ★중
- **출처**: Azure ML "Sweep Parameters", Cross-Validate Model(책 워크플로 언급).
- **근거**: 앱은 단일 파라미터 학습만. 모델 튜닝 자동화 부재.
- **구현**: 신규 `SweepParameters` 모듈 — `GridSearchCV`/`RandomizedSearchCV` 래핑. Train 계열과 연결.
- **영향**: `types.ts`, `constants.ts`, `codeSnippets.ts`, `data_analysis_modules.py`, `PropertiesPanel.tsx`.
- **난이도**: 중~상. **재현성**: `random_state`+`cv` 고정 필요. **두 앱**: 공통.

### 3-5. 추천 시스템 모듈 (협업 필터링) — ★중
- **출처**: Ch7 — Matchbox Recommender(레스토랑 평점).
- **근거**: 책의 독립 챕터 주제이나 앱 미지원. 추천은 보험/헬스케어 교차판매에도 활용 가치.
- **구현**: `surprise`(SVD/KNN) 또는 `implicit`/행렬분해. Pyodide 패키지 가용성 사전 확인 필요(없으면 외부 Python 전용으로 한정).
- **영향**: `types.ts`, `constants.ts`, `codeSnippets.ts`, `data_analysis_modules.py`, 신규 미리보기 모달.
- **난이도**: 상(Pyodide 패키지 제약). **재현성**: seed 고정. **두 앱**: 공통(단, JMDC는 헬스케어 추천 시나리오 추가 고려).

### 3-6. 모델 배포 / 스코어링 내보내기 — ★중
- **출처**: Ch3(웹서비스 배포·요청/응답 JSON), Ch4(C#/R/Python/ASP.NET 클라이언트, CORS).
- **근거**: 앱은 "분석용 Python 코드" 내보내기까지만. 책의 핵심 가치인 "모델을 서비스로"가 빠짐.
- **구현**: 학습된 모델을 `joblib`로 저장 + **FastAPI/Flask 스코어링 스니펫** 내보내기 + 요청/응답 JSON 샘플(책 형식). 배치 스코어링 스크립트.
- **영향**: 신규 내보내기 옵션(`utils/`), `codeSnippets.ts`, `components/PipelineCodeModal.tsx`. **고급기능 비밀번호 게이트** 대상에 부합(API·코드 영역).
- **난이도**: 중. **재현성**: 모델 직렬화 결정성 유지. **두 앱**: 공통.

### 3-7. 모델 재학습 / 지속학습 워크플로 — ★낮음(장기)
- **출처**: Ch8 — Retraining, Batch Execution, 지속학습 피드백 루프.
- **근거**: 앱은 저장된 파이프라인을 "다시 실행"할 뿐 신규 데이터로 재학습하는 정형 흐름 없음.
- **구현**: 저장 파이프라인에 "새 데이터 주입 → 재학습 → 모델 버전 저장" 단계화. 모델 버전 관리 UI.
- **영향**: `samples/` 포맷 확장, `App.tsx`, 신규 모델 저장소.
- **난이도**: 상. **재현성**: 버전별 시드/데이터 스냅샷 기록. **두 앱**: 공통.

---

## 4. 우선순위 요약

| 우선순위 | 항목 | 난이도 | 비고 |
|---|---|---|---|
| 1 | 3-1 그래디언트 부스팅 모듈 | 중 | 책 핵심 알고리즘, 트리 모듈 미러링 |
| 2 | 3-3 번들 데이터셋 확장 | 낮음 | 산출물 3에서 일부 선행 |
| 3 | 2-1 Evaluate 강화(ROC/AUC) | 중 | 책 평가 화면 재현 |
| 4 | 3-2 URL 데이터 로더 | 중 | CORS 검토 |
| 5 | 2-2 회귀지표 정합 | 낮음 | 용어 일치 |
| 6 | 3-6 배포/스코어링 내보내기 | 중 | 책 Ch3/4 핵심 가치 |
| 7 | 3-4 하이퍼파라미터 스윕 | 중상 | |
| 8 | 3-5 추천 모듈 | 상 | Pyodide 제약 |
| 9 | 3-7 재학습 워크플로 | 상 | 장기 |

> 모든 신규/강화 항목은 두 가지 **절대 불변식**을 지킨다: ①Python 재현성(`data_analysis_modules.py` ↔ `codeSnippets.ts` 정합 + 시드 고정 + verify 픽스처), ②두 앱 동기화(공통 변경은 양쪽 동일, JMDC 전용만 차이).


---

## 부록: 구현 결과 (2026-06-22)

> 본 개선안의 **공통 I/O 6종 + 모델 개선 3종**이 실제 코드로 구현·검증되었습니다(에이전트 하네스 + Python 재현성 verify). 추천·재학습은 후속 진행 예정.

### 항목별 구현 상태
| 항목 | 상태 | 비고 |
|---|---|---|
| 2-1 Evaluate ROC/AUC·혼동행렬·임계값·PR | ✅ 구현 | EvaluateModel 분류에 ROC-AUC·Average Precision, 임계값 슬라이더·PR 차트(앱), 결정적 |
| 2-2 회귀지표 정합(RMSE/MAE/상대오차) | ✅ 구현 | RSE·RAE 추가(기존 MSE/R² 보존) |
| 2-3 데이터 개요/요약 패널 | ✅ 구현 | utils/dataOverview.ts + DataOverviewPanel(읽기전용·클라이언트 계산) |
| 3-1 그래디언트 부스팅 모듈 | ✅ 구현 | GradientBoosting{Classifier,Regressor}, random_state=42, 픽스처 14 |
| 3-2 URL 데이터 로더 | ✅ 구현 | 입력층 fetch+/api/proxy-csv, sourceType=url 가산 분기(LoadData 실행 불변) |
| 3-3 번들 샘플 확장 | ✅ 구현 | Book_* 3종 + 메타 스키마 |
| 3-4 하이퍼파라미터 스윕 | ✅ 구현 | SweepParameters(GridSearchCV, 정수 cv→결정적), 픽스처 15 |
| 3-6 배포/스코어링 내보내기 | ✅ 구현 | utils/scoringExport.ts(joblib+FastAPI/Flask), 고급기능 게이트 |
| 3-5 추천 시스템 모듈 | ⏳ 후속 | NMF/TruncatedSVD(Pyodide 호환) 기반 예정 — 에이전트 API 복구 후 |
| 3-7 재학습/지속학습 | ⏳ 후속 | 장기 항목 — 후속 진행 |

### 검증 (Python 재현성)
- `npm run verify:pipelines` → **14/14 PASS** (신규 픽스처 14_gradient_boosting, 15_sweep_gridsearch 포함, 외부 Python 2회 byte-identical).
- `vite build` 성공. 모든 변경은 가산적·하위호환(LoadData/실행/연결/시각화 불변식 불훼손).
- ML Auto Flow ↔ JMDC 공통 코드 동기화(byte-identical), JMDC 전용 차이만 유지.
