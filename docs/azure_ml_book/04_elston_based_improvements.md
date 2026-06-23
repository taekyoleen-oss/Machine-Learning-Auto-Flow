# 산출물 4 — Elston 책 기반 공통 개선 계획 (4개 앱) + ML/JMDC 예제 추가

> 출처: Stephen F. Elston, *Data Science in the Cloud with Microsoft Azure Machine Learning and Python* (O'Reilly, 2016).
> 대상: **Auto Flow 4개 앱** — `ML Auto Flow`(베이스), `ML_Auto_Flow-JMDC`(JMDC 상위집합), `DFA-Auto-Flow`(유사 구조), `life matrix flow`(독자 구조).
> 본 문서는 **계획·제안서**다. 실제 코드 구현은 사용자 검토·승인 후 별도 진행하며, 두 가지 절대 불변식
> (①Python 재현성 `data_analysis_modules.py` ↔ `codeSnippets.ts` 정합 + 시드 고정 + verify 픽스처,
> ②두 앱 동기화: 공통 변경은 양쪽 동일, JMDC 전용만 차이)을 모든 항목에서 지킨다.

---

## 0. 한눈에 보기 — 이 책이 우리에게 주는 것

Elston의 책은 앞선 두 책(*Azure ML Essentials* 2015, *Mastering Azure ML* 2020)과 결이 다르다.
하나의 **회귀 사례(자전거 대여 시간별 수요예측, UCI Bike Sharing, 17,379행)** 를 처음부터 끝까지
**반복적으로 다듬는** 실전 워크플로를 보여 준다. 그리고 그 과정의 무게중심이 우리 앱의 철학과 정확히 겹친다.

1. **클라우드 캔버스 ↔ 로컬 Python IDE(Spyder) 왕복** — `Execute Python Script` 모듈로 같은 코드를
   캔버스 안에서도, 로컬에서도 돌린다. → 이것이 곧 ML Auto Flow의 **"브라우저 Pyodide 실행 + 코드 내보내기"** 다.
2. **반복적 특징공학** — 정규화·추세(trend) 카운터·**순환 시간변환**·**상호작용 특징**·de-trending.
3. **순열 특징중요도(Permutation Feature Importance) → 특징 가지치기**(17개 → 3개로 감축).
4. **Decision Forest 회귀**(= RandomForest, 배깅)와 **잔차 진단**(잔차-vs-실제, 잔차 시계열, 시각별 박스플롯).
5. **분위수 기반 이상치 필터**(SQL/Apply SQL Transformation, 0.20 분위수).
6. **Sweep Parameters(랜덤 스윕) + 교차검증** — 폴드별 RSE·R²의 **평균·표준편차**로 일반화 신뢰도를 읽음.
7. **비대칭 비용의 업무 프레이밍** — "재고 부족이 과잉보다 더 나쁘다" → 지표·임계값 선택의 근거.

우리 앱들은 이미 `LoadData·SplitData·RandomForest·SweepParameters(GridSearchCV)·EvaluateModel·
TransformData·DataFiltering·Correlation·ColumnPlot`을 갖췄다. **격차는 주로 진단(잔차)·교차검증 리포팅·
시계열 특징공학·특징중요도** 영역에 있다. 아래 공통 개선은 이 격차를 메우고, 신규 예제는 책의 사례를
4개 앱 도메인으로 이식한다.

---

## 1. Elston 워크플로 vs 4개 앱 현황 비교표

| Elston 단계 | 책 모듈(Azure ML) | Auto Flow 대응 | 상태(ML/JMDC 기준) |
|---|---|---|---|
| 데이터 적재 | Dataset / Reader | `LoadData`(CSV/URL/예제) | ✅ |
| 로컬 Python 왕복 | Execute Python Script + Spyder | 코드 내보내기 + Pyodide 실행 | ✅(철학 일치) |
| 특징공학(정규화·신규열) | Execute Python Script | `ScalingTransform`·`TransformData`·`EncodeCategorical` | △ 신규열/순환/상호작용은 수작업 |
| 시계열 추세/순환 특징 | (Python: monthCount·xformHr) | — | ❌ 전용 지원 없음 |
| 데이터 시각화/EDA | Execute Python Script(matplotlib) | `Correlation`·`ColumnPlot`·`Statistics` | △ 잔차·시계열 플롯 부재 |
| 분위수 이상치 필터 | Apply SQL Transformation(SQLite) | `DataFiltering`·`OutlierDetector` | △ 분위수 기준 직접 지정 보강 여지 |
| 분할 | Split(70/30, 60/40) | `SplitData` | ✅ |
| 비선형 회귀 | **Decision Forest Regression** | `RandomForest`(+`GradientBoosting`) | ✅ |
| 특징 중요도/가지치기 | **Permutation Feature Importance** | — | ❌ 없음 |
| 예측 | Score Model | `ScoreModel` | ✅ |
| 평가 | Evaluate Model(RSE·R²) | `EvaluateModel`(R²·RMSE·MAE·RSE·RAE) | ✅ |
| **잔차 진단** | Execute Python Script(잔차 플롯) | — | ❌ 없음(핵심 격차) |
| 파라미터 탐색 | **Sweep Parameters(랜덤)** | `SweepParameters`(GridSearchCV) | △ 격자만(랜덤·CV리포트 보강) |
| **교차검증 리포트** | **Cross Validate Model(폴드별 평균·표준편차)** | (GridSearchCV 내부 cv) | ❌ 폴드별 표 노출 없음 |
| 배포 | Publish as Web Service | 스코어링 코드 내보내기 | ✅ |

---

## 2. 공통 개선 (4개 앱) — 기존 기능 강화

각 항목: **출처 / 근거 / 구현 / 영향 파일 / 난이도 / 재현성 영향 / 앱 적용성**.
앱 적용성 표기: **ML·JMDC**(공통, byte-identical), **DFA**(유사 구조 — 동등 적용 가능, 모듈명 확인 필요),
**life matrix**(독자 구조 — 개념 동일, 어댑터 필요).

### 2-1. 회귀 잔차 진단 패널 — ★최우선
- **출처**: Elston "The Decision Forest Model" / "visualizeresids.py" — 잔차-vs-실제, **잔차 시계열**, 시각별 **박스플롯**.
- **근거**: 책이 가장 공들이는 단계인데 4개 앱 모두 부재. R²/RMSE 숫자만으로는 "어디서·언제 틀리나"를 못 본다
  (Elston: "결과가 흥미롭지만 다소 추상적이다" → 잔차 플롯으로 구체화). 비대칭 비용·피크 수요 과소예측을 드러내는 핵심.
- **구현**: `EvaluateModel(regression)` 결과에 잔차 진단 추가 — ①잔차 vs 예측(또는 실제) 산점도, ②잔차 분포(히스토그램/Q-Q),
  ③(시계열/그룹 컬럼이 있으면)잔차 시계열·범주별 박스플롯. 전부 결정적(고정 예측 → 고정 잔차).
- **영향**: `data_analysis_modules.py`(evaluate_model에 잔차 산출 가산), `codeSnippets.ts`(템플릿 정합),
  `components/EvaluationPreviewModal.tsx`(차트). 앱 전용 차트는 내보낸 코드와 무관(재현성 무영향) — 5장 임계값 슬라이더와 동일 패턴.
- **난이도**: 중. **재현성**: 잔차=예측−실제는 결정적 → 불변식 영향 없음(템플릿 정합만 유지).
- **앱 적용성**: ML·JMDC 공통, DFA 동등, life matrix 어댑터.

### 2-2. 교차검증 폴드별 리포팅 — ★높음
- **출처**: Elston "Cross Validation" — 폴드별 RSE·R²와 **평균·표준편차**로 일반화 신뢰도 판단("std ≪ mean → 잘 일반화").
- **근거**: `SweepParameters`는 내부적으로 `GridSearchCV(cv=정수)`를 쓰지만(9·24장, 결정적), **폴드별 점수·평균·표준편차**를
  사용자에게 보여 주지 않는다. 책은 이 분산을 모델 신뢰의 핵심 근거로 삼는다.
- **구현**: `SweepParameters` 결과에 `cv_results_`의 폴드별 점수 + mean/std 표를 노출(이미 계산되어 있음 — 표출만).
  정수 cv → 완전 재현(byte-identical 유지).
- **영향**: `data_analysis_modules.py`/`codeSnippets.ts`(이미 GridSearchCV; 출력 포맷만), 결과 미리보기 모달.
- **난이도**: 낮음~중. **재현성**: 정수 cv 고정 → 결정적. **앱 적용성**: ML·JMDC 공통, DFA 동등, life matrix 어댑터.

### 2-3. 시계열·순환 특징공학 지원 — ★높음
- **출처**: Elston "A First Set of Transformations" — `monthCount`(추세), `dayCount`, `xformHr`(시각을 24h에 걸쳐 "부드럽게 봉우리"로),
  `isWorking`, `xformWorkHr`(근무일×시각 **상호작용**), 그리고 시각화에서의 **선형 추세 제거(de-trending)**.
- **근거**: 시계열·주기 데이터(시간/요일/월)는 보험·헬스케어·DFA에 흔한데, 현 `TransformData`로는 순환 인코딩·상호작용·추세 카운터를
  매번 수작업해야 한다. 21장(특징공학)·24장(튜닝)의 자연스러운 확장.
- **구현**: `TransformData`(또는 신규 `FeatureEngineer`)에 옵션 추가 — ①**순환 인코딩**(`sin/cos(2π·t/T)`로 시각/요일/월),
  ②**추세 카운터**(시작점부터의 경과 단위), ③**상호작용 항**(두 열 곱/조합), ④**다항/구간화**(21장). 전부 결정적.
- **영향**: `types.ts`/`constants.ts`(옵션), `codeSnippets.ts`+`data_analysis_modules.py`(변환 함수 1:1), `PropertiesPanel.tsx`.
- **난이도**: 중. **재현성**: 순수 결정적 변환 → 강화. **앱 적용성**: ML·JMDC 공통, DFA 동등, life matrix 어댑터(연령·기간 축에 순환/추세 적용).

### 2-4. 순열 특징중요도 + 특징 가지치기 — ★중
- **출처**: Elston "Pruning features" — Permutation Feature Importance로 중요도 순 정렬 → 낮은 것부터 제거(17→3개), 성능 거의 불변.
- **근거**: 4개 앱에 특징 중요도 도구가 없다. 과적합 방지·해석가능성(22장 SHAP 논의와 연결)·모형 문서화(거버넌스)에 직접 기여.
- **구현**: 학습 모델에 `sklearn.inspection.permutation_importance`(고정 `random_state=42`, `n_repeats` 고정 → 결정적) 적용,
  중요도 막대 + 권고 가지치기 목록. 트리 모델은 `feature_importances_`도 함께.
- **영향**: 신규 경량 모듈 또는 `EvaluateModel` 확장, `codeSnippets.ts`+`data_analysis_modules.py`, 미리보기 차트.
- **난이도**: 중. **재현성**: `random_state`+`n_repeats` 고정 필수 → verify 픽스처. **앱 적용성**: ML·JMDC 공통, DFA 동등, life matrix 어댑터.

### 2-5. 분위수 기반 이상치 필터 — ★중
- **출처**: Elston "Filtering with SQLite" / 0.20 분위수로 잔차·응답 이상치 제거(훈련셋 한정).
- **근거**: 현 `OutlierDetector`(IQR/z-score)·`DataFiltering`은 있으나, **사용자 지정 분위수 컷오프**로 응답/잔차를 자르는 흐름이 약하다.
  Elston은 분위수 선택 자체를 튜닝 손잡이로 본다("0.20은 임의값 — 바꿔 보라").
- **구현**: `DataFiltering`/`OutlierDetector`에 "분위수 컷오프(하/상)" 옵션 + **훈련셋에만 적용**(누수 방지, 2장) 가이드.
- **영향**: `data_analysis_modules.py`/`codeSnippets.ts`, `PropertiesPanel.tsx`.
- **난이도**: 낮음~중. **재현성**: 결정적. **앱 적용성**: ML·JMDC 공통, DFA 동등, life matrix 어댑터.

### 2-6. In-pipeline EDA 시각화 강화 — ★중
- **출처**: Elston "Exploring the data" — 상관행렬 히트맵, 산점도, 시계열, (잔차)박스플롯을 파이프라인 안에서 즉시.
- **근거**: `Correlation`·`ColumnPlot`은 있으나 **산점도·시계열·범주별 박스플롯**과 **de-trended 상관**이 부족.
- **구현**: `ColumnPlot`에 산점도/시계열/박스플롯 타입 추가, `Correlation`에 de-trend 옵션. 시각화는 앱 표시용(재현성 무영향).
- **영향**: `components/*Plot*`·`CorrelationPreviewModal.tsx`. **난이도**: 중. **앱 적용성**: ML·JMDC 공통, DFA 동등, life matrix 어댑터.

### 2-7. (고급) 커스텀 Python 코드 모듈 — ★낮음(장기·고급 게이트)
- **출처**: Elston 전반의 `Execute Python Script` 모듈 — 사용자가 임의 변환/시각화 코드를 파이프라인에 삽입.
- **근거**: Elston 워크플로의 심장. 단, **임의 코드 실행은 재현성·안전성 리스크**가 크다.
- **구현(신중)**: Pyodide 샌드박스에서 `df_in → df_out` 시그니처의 제한된 사용자 코드 셀. **고급기능 비밀번호 게이트** 필수.
  내보낸 코드에 그대로 포함되므로 재현성은 사용자 코드에 의존 → 시드 고정 가이드·경고 표기. 기본 비활성, 옵트인.
- **영향**: 신규 모듈, 실행기, 보안 검토. **난이도**: 상. **재현성**: 사용자 책임(명시). **앱 적용성**: ML·JMDC 우선(공통), DFA/life matrix 후순위.

---

## 3. 신규 예제 추가 (ML·JMDC 중심) — 책 사례의 도메인 이식

### 3-1. 자전거 대여 수요예측 회귀 예제 (UCI Bike Sharing) — ★최우선
- **출처**: Elston 본문 전체 사례. UCI Bike Sharing(시간별 17,379행, 9특징, 2년).
- **근거**: 책을 "동일 데이터·동일 흐름"으로 재현. 시계열 특징공학·잔차진단·CV를 한 예제로 묶는 **종합 실습**.
  기존 Book_* 예제(자동차/성인소득/도매고객/추천)와 동일 등록 패턴.
- **구현**(ML·JMDC 공통):
  - 데이터: `verify/datasets/bike_sharing_hourly.csv`(UCI 공개, 헤더 정리) + `Examples_in_Load/` 등록.
  - 샘플: `samples/Book_BikeSharing_DemandForecast.json` — `LoadData → TransformData(순환/추세/상호작용) →
    SplitData(0.7) → RandomForest(회귀) → TrainModel → ScoreModel → EvaluateModel(+잔차진단)`.
  - 검증 픽스처: `verify/pipelines/17_bike_demand_rf.json` — 외부 Python 2회 byte-identical(시드 42 고정).
  - Supabase `autoflow_samples`(`app_section=ML`, 두 앱 공유) 등록.
- **난이도**: 중. **재현성**: 데이터 스냅샷 고정 + `random_state=42` → byte-identical. **앱 적용성**: ML·JMDC 공통.

### 3-2. (JMDC 전용) 헬스케어 이용/청구 수요예측 예제 — ★높음
- **근거**: 자전거 수요 ↔ **헬스케어 자원·청구 수요**는 같은 구조(시간 축 + 주기성 + 추세 + 비대칭 비용).
  JMDC 코호트/이용 데이터에 동일 파이프라인(순환 특징·잔차진단·CV)을 적용해 **월별 외래/입원·청구 건수 예측**을 시연.
- **구현**: JMDC 전용 샘플 `samples/JMDC_Utilization_DemandForecast.json` + 합성/공개 대체 데이터(개인정보 무포함).
  비대칭 비용(자원 부족 위험)을 평가 해설(✨AI)과 연결.
- **난이도**: 중. **재현성**: 시드·데이터 고정. **앱 적용성**: **JMDC 전용**(ML에는 미적용 — 의도된 차이).

### 3-3. (DFA-Auto-Flow) 현금흐름·지급준비금 시계열 예측 적용 노트 — ★중
- **근거**: DFA(동적 재무분석)는 시계열·시나리오가 핵심. Elston의 추세·순환·잔차진단·CV는 **현금흐름/지급준비금 진전 예측**에 직접 이식 가능.
- **구현(계획)**: DFA 모듈 인벤토리 확인 후, 공통 개선 2-1~2-5를 DFA 캔버스에 동등 적용 + DFA 도메인 예제 1종.
  *(DFA 코드베이스 미정밀조사 — 구현 단계에서 모듈명/구조 확인 필요. 정직한 한계.)*
- **난이도**: 중. **앱 적용성**: DFA(유사 구조).

### 3-4. (life matrix flow) 사망률·생존 시계열 예측 적용 노트 — ★중
- **근거**: life matrix는 사망률(Lee–Carter/CBD/APC 등)·생존 분석 중심(독자 구조). Elston의 시계열 특징·잔차진단·CV는
  **연령–기간 사망률 표면**의 예측·진단에 응용 가능(추세=기간 효과, 순환=계절성, 잔차진단=적합도).
- **구현(계획)**: life matrix 모듈 인벤토리 확인 후, 잔차진단(2-1)·CV리포트(2-2)·특징중요도(2-4)를 **어댑터**로 적용.
  *(life matrix는 독자 구조 — 직접 byte-identical 동기화 대상 아님. 개념 이식.)*
- **난이도**: 중~상. **앱 적용성**: life matrix(독자 — 어댑터).

---

## 4. 앱별 적용 매트릭스 (불변식 영향)

| 개선/예제 | ML | JMDC | DFA | life matrix | 재현성 영향 |
|---|---|---|---|---|---|
| 2-1 잔차 진단 | ● 공통 | ● 공통 | ○ 동등 | △ 어댑터 | 무(앱 표시) |
| 2-2 CV 폴드 리포트 | ● 공통 | ● 공통 | ○ 동등 | △ 어댑터 | 무(정수 cv 결정적) |
| 2-3 시계열·순환 특징 | ● 공통 | ● 공통 | ○ 동등 | △ 어댑터 | 강화(결정적 변환) |
| 2-4 순열 특징중요도 | ● 공통 | ● 공통 | ○ 동등 | △ 어댑터 | 시드 고정 필요 |
| 2-5 분위수 이상치 필터 | ● 공통 | ● 공통 | ○ 동등 | △ 어댑터 | 무 |
| 2-6 EDA 시각화 강화 | ● 공통 | ● 공통 | ○ 동등 | △ 어댑터 | 무 |
| 2-7 커스텀 코드 모듈(고급) | ● 우선 | ● 우선 | ◌ 후순위 | ◌ 후순위 | 사용자 책임(경고) |
| 3-1 자전거 수요예측 예제 | ● 공통 | ● 공통 | — | — | 시드·데이터 고정 |
| 3-2 헬스케어 수요예측 예제 | — | ● 전용 | — | — | 시드·데이터 고정 |
| 3-3 DFA 시계열 예제 | — | — | ○ | — | 시드 고정 |
| 3-4 life matrix 적용 | — | — | — | △ | 시드 고정 |

● 공통(byte-identical) · ○ 동등(구조 유사, 확인 필요) · △ 어댑터(독자 구조) · ◌ 후순위 · — 비대상

---

## 5. 우선순위 요약

| 우선순위 | 항목 | 난이도 | 비고 |
|---|---|---|---|
| 1 | 3-1 자전거 수요예측 예제(ML·JMDC) | 중 | 책 사례 1:1 재현 + 종합 실습, 기존 Book_* 패턴 |
| 2 | 2-1 회귀 잔차 진단 | 중 | 책의 핵심 단계, 결정적, 공통 |
| 3 | 2-2 교차검증 폴드 리포트 | 낮음~중 | 이미 계산됨 — 표출만 |
| 4 | 2-3 시계열·순환 특징공학 | 중 | 보험/헬스케어/DFA 공통 수요 |
| 5 | 2-4 순열 특징중요도·가지치기 | 중 | 해석가능성·거버넌스 |
| 6 | 3-2 헬스케어 수요예측 예제(JMDC) | 중 | JMDC 전용 |
| 7 | 2-5 분위수 이상치 필터 | 낮음~중 | 누수 방지 가이드 동반 |
| 8 | 2-6 EDA 시각화 강화 | 중 | |
| 9 | 3-3 / 3-4 DFA·life matrix 이식 | 중~상 | 코드베이스 확인 선행 |
| 10 | 2-7 커스텀 Python 코드 모듈 | 상 | 고급 게이트·보안 검토(장기) |

> **권고 실행 순서.** 가장 안전하고 가치 높은 #1~#3(예제 + 잔차진단 + CV 리포트)을 **ML·JMDC 공통**으로 먼저 구현·검증
> (`npm run verify:pipelines`로 byte-identical 확인) 후, #4~#7을 추가, DFA·life matrix는 코드베이스 확인 뒤 동등/어댑터 적용한다.
> 본 계획서는 booklet `제9부(28장 — 자전거 수요예측 종합 사례)`와 짝을 이룬다 — 책자는 *방법론·개념*을, 본 문서는 *구현 경로*를 담는다.

---

## 6. 부록: 구현 상태 (2026-06-23 — 1차 구현 완료)

> 1차 **RandomForest 갭 + #3-1 예제 + #2-1 잔차요약 + #2-2 CV리포트**, 2차 **#2-5 분위수 필터**, 3차 **#2-3 FeatureEngineer**, 4차 **#2-4 FeatureImportance**를
> **ML·JMDC 공통**으로 구현·검증했다(`npm run verify:pipelines` 양쪽 **19/19 PASS**, build 성공, 공통 코드 byte-identical).

| 항목 | 상태 | 비고 |
|---|---|---|
| (선결) RandomForest 전체코드 export 템플릿 | ✅ 구현 | codeSnippets.ts에 누락돼 있던 RandomForest 템플릿 신설(재현성 갭 해소). DecisionTree/GradientBoosting 미러 |
| 3-1 자전거 수요예측 예제 | ✅ 구현 | UCI Bike Sharing(17,379행) + `17_bike_demand_rf` 픽스처(RandomForest 회귀, R²≈0.9422) + 샘플·메타 등록 |
| 2-1 잔차 진단 (요약 + 시각 차트) | ✅ 구현 | EvaluateModel 회귀에 잔차 mean/std/5수요약/3σ이상치 **결정적 출력**(export) + **EvaluationPreviewModal에 잔차 진단 차트**(잔차 vs 실제 산점도·잔차 분포 히스토그램, `generateResidualPlotPython`, 앱 표시용) |
| 2-2 교차검증 폴드 리포트 | ✅ 구현 | SweepParameters에 폴드별 mean±std·split 점수 출력(정수 cv→결정적) |
| 2-3 시계열·순환 특징공학 | ✅ 구현 | **신규 `FeatureEngineer` 모듈**(cyclical sin/cos · interaction a*b · trend 순번). export+인앱 모두 지원, 픽스처 `19_feature_engineer`. 자전거 R² 0.9422→0.9493 개선 |
| 2-4 순열 특징중요도·가지치기 | ✅ 구현(export) | **신규 `FeatureImportance` 모듈**(sklearn permutation_importance, random_state=42). 픽스처 `20`. 자전거: hr≫workingday≫temp…(holiday·windspeed=가지치기 후보). 인앱은 안내(트리 모델 재예측 불가=정직한 한계), 실제 계산은 내보낸 Python |
| 2-5 분위수 이상치 필터 | ✅ 구현 | DataFiltering에 `quantile_above`/`quantile_below` 연산자 + UI + 픽스처 `18_datafilter_quantile`(자전거 5~95% 트리밍). 결정적·훈련셋 한정 권장 |
| 2-6 EDA 시각화 강화 | ✅ 이미 충족 | 기존 `ColumnPlot`가 산점도·박스·라인(시계열)·바이올린·KDE·ECDF·QQ·헥스빈·조인트·회귀·히트맵 등 다수 제공(단일/이중 × 수치/범주). Elston EDA 요구를 이미 포괄 — 중복 추가 불필요(de-trend 상관만 niche 후속) |
| 2-7 커스텀 Python 코드 모듈 | ⏳ 계획(장기) | 고급 게이트·보안 검토 |
| 3-2 헬스케어 수요예측 예제(JMDC) | ✅ 구현(JMDC 전용) | 합성 일별 외래 수요(730행, `jmdc_monthly_utilization.csv`) + 샘플 `JMDC_Utilization_DemandForecast` + 픽스처 `21`(FeatureEngineer 순환→RF). R²≈0.9676. 자전거 구조의 헬스케어 이전(28.9). JMDC verify 20/20 |
| 3-3 / 3-4 DFA·life matrix 이식 | ⏳ 계획 | 코드베이스 확인 선행 |
| booklet 제9부(28장) | ✅ 작성 완료 | 28장 "구현 상태" 표에 위 1차 결과 반영 |

### 1차 구현 검증 (Python 재현성·두 앱 동기화)
- `npm run verify:pipelines` → **19/19 PASS** (양쪽, 신규 `17_bike_demand_rf`·`18_datafilter_quantile`·`19_feature_engineer`·`20_feature_importance` 포함, 외부 Python 2회 byte-identical).
- `npm run build` 성공(양쪽). 변경은 전부 가산적·결정적(`.6f`, `np.quantile`), 기존 지표·출력 보존.
- 공통 변경(RF 템플릿·잔차 요약·CV 리포트·픽스처·샘플·데이터) **양쪽 byte-identical**, JMDC 전용 차이(cohort)만 유지.
- 2-1 잔차 시각 차트(EvaluationPreviewModal)·2-3 FeatureEngineer·2-4 FeatureImportance·2-5 분위수 필터까지 구현 완료(빌드 성공·양쪽 byte-identical).
- 잔여(2-6 EDA 시각화·2-7 커스텀 코드 모듈·3-2 JMDC 예제·3-3/3-4 DFA·life matrix)는 다음 트랜치.

> 본 문서는 산출물 01(Essentials)·02(책자 방향)·03(모델 재현)의 후속이며, Elston 사례의 **공통 개선 + 예제 이식**을 정리했다.
> 구현은 `ml-flow-orchestrator` 하네스로 두 불변식을 지키며 단계 진행한다.
