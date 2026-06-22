# ML Auto Flow — 시각적 머신러닝 파이프라인 (책자)

> 출처: 본 책자는 docs/azure_ml_book 계획·검증 자료로부터 작성됨 (MVP).

> 본 섹션은 위 1~6절의 제작 방향에 따라 **실제로 작성한 책자 본문(MVP)**이다(2026-06-22).
> 출처 자료: `01_book_based_improvements.md`, `03_book_models_reproduction.md`, `cross_app_io_improvements.md`, 앱 소스(`moduleDescriptions.ts`, `codeSnippets.ts`, `verify/`).
> 모든 수치·코드는 `npm run verify:pipelines` (**14/14 PASS**, 외부 Python 2회 byte-identical)로 검증된 것만 사용했다.
> Markdown 단일 진실원천(SSOT)이며, `make-pdf`로 PDF 책자, `app-doc-ppt`로 요약 PPT 파생 가능.

## 표지 / 머리말

**ML Auto Flow — 브라우저에서 즉시 실행되는 시각적 머신러닝 파이프라인**

이 책자는 Jeff Barnes의 *Microsoft Azure Essentials: Azure Machine Learning*의 학습 흐름(데이터셋 → 시각화 → 분할 → 학습 → 채점 → 평가 → 배포)을 **클라우드 가입·과금 없이** 브라우저만으로 재현한다. Azure ML Studio가 클라우드 캔버스라면, ML Auto Flow는 **로컬 캔버스 + 브라우저 내 Python(Pyodide)** 이다.

**책 대비 4대 차별점**
1. **클라우드·과금 불필요** — 설치·가입 없이 브라우저에서 즉시 시작.
2. **브라우저 내 Python 실행(Pyodide)** — numpy/pandas/scikit-learn/statsmodels가 WASM으로 동작.
3. **재현성 보증** — "전체 코드 보기"로 내보낸 Python이 외부에서 **동일 결과(byte-identical)**, 자동 회귀 검증(`verify/`).
4. **AI 한국어 해설** — 코드·결과·오류를 그 자리에서 설명.

---

## 1장 — 데이터 과학과 ML 입문 (책 Ch1)

머신러닝은 데이터에서 규칙을 학습해 예측·분류·군집을 수행한다.
- **지도학습(Supervised):** 정답(레이블)이 있는 데이터로 학습 — 회귀(연속값), 분류(범주).
- **비지도학습(Unsupervised):** 레이블 없이 구조를 찾음 — 군집(K-Means), 차원축소(PCA).
- **예측 분석 워크플로:** 데이터 적재 → 전처리 → 분할(train/test) → 모델 학습 → 평가 → (배포).

ML Auto Flow는 이 흐름을 **노드(모듈)를 캔버스에 놓고 포트로 연결**하는 방식으로 표현한다. 각 모듈은 하나의 분석 단계이며, 연결선은 데이터·모델의 흐름이다.

---

## 2장 — ML Auto Flow 시작하기 (책 Ch3 전반)

**구성요소**
- **모듈 팔레트:** 데이터 적재·전처리·분할·지도/비지도 학습·통계·평가 모듈.
- **캔버스:** 모듈을 드래그 배치하고 포트(입력/출력)를 연결해 파이프라인 구성.
- **실행:** 모듈별 ▶ 실행 또는 전체 실행. 계산은 브라우저 Pyodide에서 수행.
- **미리보기:** 각 결과는 표·차트·요약 모달로 확인.

**첫 단계 — 데이터 들여오기(`LoadData`)**
로컬 CSV 업로드 외에 **URL 직접 로드**도 지원한다(입력층에서 받아 동일 파서로 처리하므로 실행·재현성에는 영향이 없다). 데이터를 올리면 **데이터 개요 패널**이 행·열 수, 열별 타입(수치형/범주형), 결측치 수를 즉시 요약한다(책의 workclass 1,836개 결측 예시처럼 결측 열을 강조).

> 클라우드 가입 절차가 없으므로, 책의 Ch2~3 가입·과금 설명은 "브라우저 열기" 한 줄로 대체된다.

---

## 3장 — 첫 파이프라인: 회귀 end-to-end (책 Ch5, 자동차 가격)

**목표:** UCI *Automobile* 데이터로 자동차 `price` 예측.
**파이프라인:** `LoadData → SplitData(0.75, seed 42) → LinearRegression → TrainModel → ScoreModel → EvaluateModel(regression)`
**특성(결측 없는 수치형 10개):** symboling, wheel-base, length, width, height, curb-weight, engine-size, compression-ratio, city-mpg, highway-mpg.

**검증된 결과 (test n=51):** R² **0.7561** · RMSE 5,129.9 · MAE 3,487.7.
> 평가 모듈은 이제 회귀에서 RMSE·MAE에 더해 **상대제곱오차(RSE)·상대절대오차(RAE)**를 함께 제시한다(책의 회귀 5종 지표와 정합).

**내보낸 동등 Python (검증됨):**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("imports-85-hdrs.csv")
features = ["symboling","wheel-base","length","width","height",
            "curb-weight","engine-size","compression-ratio","city-mpg","highway-mpg"]
X, y = df[features], df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.75, random_state=42, shuffle=True)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
print("R2 =", r2_score(y_te, pred))
print("RMSE =", mean_squared_error(y_te, pred) ** 0.5)
```
앱 샘플 `Book_Automobile_LinearRegression.json` 로드 → 실행으로 동일 재현(픽스처 `11_book_automobile_linreg`).

---

## 4장 — 분류 모델 (책 Ch3, 성인 소득)

**목표:** UCI *Adult* 데이터로 `income`(>50K) 분류.
**파이프라인:** `LoadData → SplitData(0.8, seed 42) → DecisionTree(max_depth=8) → TrainModel → ScoreModel → EvaluateModel(classification)`
**검증된 결과 (test n=6,513):** Accuracy **0.8322** · Precision(>50K) 0.8136 · Recall 0.3947 · F1 0.5315.

**평가 강화(신규):** 분류 평가에 **ROC 곡선·ROC-AUC·정밀도-재현율(PR) 곡선·Average Precision**과 **임계값 슬라이더**(앱 내에서 임계값을 움직이며 정밀도/재현율/F1 변화를 즉시 확인)가 추가되었다. 책 Ch3의 평가 화면(ROC/AUC·혼동행렬·임계값 조정)을 그대로 재현한다.

**그래디언트 부스팅(신규 모듈):** 책의 1순위 분류기인 **Two-Class Boosted Decision Tree**에 대응하는 `GradientBoosting` 모듈을 추가했다(`sklearn.ensemble.GradientBoostingClassifier/Regressor`, `random_state=42`). 의사결정나무를 GradientBoosting으로 교체해 동일 데이터에서 책의 부스팅 트리 예제를 재현할 수 있다(파라미터: n_estimators·learning_rate·max_depth).

앱 샘플 `Book_AdultIncome_DecisionTree.json`(픽스처 `13_book_adult_clf`), GBM 픽스처 `14_gradient_boosting`.

---

## 5장 — 군집 분석 (책 Ch6, 도매고객)

**목표:** UCI *Wholesale customers*(440행, 8열) 세분화.
**파이프라인:** `LoadData → KMeans(n_clusters=4, seed 42) + TrainClusteringModel → ClusteringData`
**검증된 결과:** 군집 0/1/2/3 = **277 / 95 / 58 / 10**명, inertia ≈ 6.49e+10.
**해석:** 다수(277)는 일반 소비, 소수(10)는 특정 카테고리 고액 지출(대형 거래처). `StandardScaler` 선행 시 금액 단위 큰 Fresh/Grocery 지배를 줄여 더 균형 잡힌 세분화 가능.

군집 패밀리는 K-Means 외 **DBSCAN·계층적(Agglomerative)·PCA**도 전체코드 내보내기를 지원한다. 앱 샘플 `Book_Wholesale_KMeans.json`(픽스처 `12_book_wholesale_kmeans`).

---

## 6장 — 통계·검정 모듈 (책에 없는 확장 영역)

ML Auto Flow는 책을 넘어 **통계 분석·가설검정** 모듈을 제공한다: 기술통계·상관분석·정규성 검정·가설검정·이상치 탐지·VIF(다중공선성)·OLS/로지스틱/포아송(statsmodels) 등. 각 결과 모달은 표·차트와 함께 **✨AI 한국어 해설**을 제공해 해석을 돕는다.

---

## 7장 — Python 코드 내보내기와 재현성 (앱 차별화 ★)

ML Auto Flow의 핵심 가치다. "전체 코드 보기"는 캔버스 파이프라인을 **외부에서 그대로 실행 가능한 standalone Python**으로 내보낸다.
- **재현성 불변식:** 모든 무작위 단계 `random_state=42` 고정 → 외부 Python에서도 동일 결과.
- **자동 회귀 검증:** `npm run verify:pipelines` 가 각 픽스처의 내보낸 코드를 외부 Python으로 **2회 실행**해 출력이 **byte-identical**인지 단언. 현재 **14/14 PASS**(회귀·분류·전처리·statsmodels·신경망·군집 4종·PCA·Book 3종·**그래디언트 부스팅**·**하이퍼파라미터 스윕** 포함).
- **하이퍼파라미터 스윕(신규):** `SweepParameters` 모듈은 `GridSearchCV`(정수 cv → 완전 결정적)로 최적 추정기를 찾아 Train/Score/Evaluate에 그대로 연결한다.
- **스코어링 배포 내보내기(신규):** 학습 모델을 `joblib` 저장 + FastAPI/Flask 스코어링 엔드포인트 + 요청/응답 JSON 샘플로 내보낸다(책 Ch3/4의 "모델을 서비스로").

> 이 장이 "책에 적힌 대로 실행하면 동일 결과"를 보증한다.

---

## 8장 — AI 보조 기능 (앱 차별화 ★)

사용자별 로컬 API 키(localStorage)로 동작하는 AI 헬퍼:
- **코드 해설:** 내보낸 Python을 한국어로 설명.
- **결과 해석:** 통계·평가 결과 모달에서 ✨ 해설.
- **오류 진단:** 모듈 실행 오류의 원인·수정 제안.
- **AI 파이프라인 생성:** 분석 목표나 데이터로부터 파이프라인 초안 생성.

> AI·코드·API 기능은 **고급기능 비밀번호 게이트**로 보호된다(일반 사용자는 모듈 배치·연결·실행·미리보기만 사용).

---

## 9장 — 고급 모델 & 신규 개선 (책 너머)

- **그래디언트 부스팅**(분류·회귀, seed 42) — 책의 부스팅 트리 대응, verify 픽스처 보유.
- **평가 강화** — ROC/AUC·PR 곡선·임계값 조정, 회귀 RSE/RAE.
- **하이퍼파라미터 스윕** — GridSearchCV 기반 자동 튜닝.
- (보험·계리 확장) 사망률·빈도-심도 등 도메인 모듈 — `moduleDescriptions.ts` 참조.
- **후속 예정:** 추천 시스템(협업 필터링, Pyodide 호환 행렬분해), 재학습/지속학습 워크플로.

---

## 부록 A — 모듈 레퍼런스 (요약)

각 모듈 카드는 앱의 `moduleDescriptions.ts`에 **제목·역할·입출력·사용 시점·연결·흔한 오류·주의**가 정의되어 있어 그대로 발췌 가능하다. 대표 카테고리:
- **데이터 I/O:** LoadData(파일·URL), 데이터 개요 패널.
- **전처리:** 결측치 처리, 열 선택/필터, 인코딩, 스케일링, 분할(SplitData).
- **지도학습:** LinearRegression, LogisticRegression, DecisionTree, RandomForest, **GradientBoosting**, SVM, KNN, NeuralNetwork.
- **모델 연산:** TrainModel, **SweepParameters**, ScoreModel, EvaluateModel(ROC/AUC·회귀지표).
- **비지도:** KMeans, DBSCAN, Hierarchical, PCA, (군집 할당) ClusteringData.
- **통계·검정:** 기술통계·상관·정규성·가설검정·VIF·OLS/Logistic/Poisson.

## 부록 B — 변경 이력

기능 연혁은 `CLAUDE.md`(변경 이력 표)와 `HISTORY.md`에 누적된다. 2026-06-22 기준 최신: 횡단 공통 I/O 6종, 그래디언트 부스팅, Evaluate ROC/AUC 강화, 하이퍼파라미터 스윕 구현(verify 14/14).

---

## 제작 메모 (판단 사항)

- **범위:** 책의 8장 구조를 미러링하되 앱 고유 강점(7·8장)과 신규 개선(9장)을 부각. MVP는 1~9장 + 부록 A/B.
- **코드·수치:** 전부 `verify`로 검증된 것만 사용(미검증 추정치 배제).
- **스크린샷:** 추후 Playwright(MCP)로 `npm run dev`(127.0.0.1:3003) 캔버스·결과 모달을 자동 캡처해 각 장에 삽입 가능(샘플 로드 시 동일 화면 재현).
- **PDF/PPT:** 본 Markdown을 `make-pdf`로 출판 품질 PDF, 필요 시 `app-doc-ppt`로 발표용 PPT 파생.
- **두 앱:** 본문 1~8장은 JMDC와 공통, JMDC 헬스케어(J1~J7) 장은 JMDC 책자에 별도 부록으로 분리 권장.
