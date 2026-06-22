# 산출물 3 — 책의 데이터·모델을 ML Auto Flow로 재현하기

> 출처: Jeff Barnes, *Microsoft Azure Essentials: Azure Machine Learning* (Microsoft Press, 2015).
> 본 문서는 책의 대표 예제(회귀·분류·군집)를 **ML Auto Flow의 실제 모듈로 재현**하는 전체 개발 과정과,
> 그 결과(**검증 통과 + 실제 지표**)를 담는다. 추천(Ch7)은 앱 미지원이라 "확장 제안"으로 분리한다.

## 0. 이번 작업으로 실제 생성·검증된 산출물

| 종류 | 경로 | 상태 |
|---|---|---|
| 데이터셋(헤더 정리본) | `verify/datasets/imports-85-hdrs.csv` (자동차, 201행) | ✅ 생성 |
| 데이터셋 | `verify/datasets/adult.csv` (성인소득, 32,561행) | ✅ 생성 |
| 데이터셋 | `verify/datasets/wholesale_customers.csv` (도매고객, 440행) | ✅ 생성 |
| 검증 픽스처(회귀) | `verify/pipelines/11_book_automobile_linreg.json` | ✅ verify PASS |
| 검증 픽스처(분류) | `verify/pipelines/13_book_adult_clf.json` | ✅ verify PASS |
| 검증 픽스처(군집) | `verify/pipelines/12_book_wholesale_kmeans.json` | ✅ verify PASS |
| 앱 로드용 샘플 | `samples/Book_Automobile_LinearRegression.json` | ✅ 생성 |
| 앱 로드용 샘플 | `samples/Book_AdultIncome_DecisionTree.json` | ✅ 생성 |
| 앱 로드용 샘플 | `samples/Book_Wholesale_KMeans.json` | ✅ 생성 |

> **`npm run verify:pipelines` 결과: 12/12 PASS** (신규 3종 포함, 외부 Python 2회 실행 byte-identical 재현 확인).

### 데이터 출처/라이선스
- 자동차: UCI *Automobile* (`imports-85.data`, 1987). `?`→결측 처리, price 결측 4행 제거, 26개 컬럼 헤더 부여.
- 성인소득: UCI *Adult / Census Income*. 15개 컬럼 헤더 부여, 값 앞뒤 공백 제거, `?`→결측.
- 도매고객: UCI *Wholesale customers* (`00292`). 원본 헤더 그대로(8열).
- 모두 UCI Machine Learning Repository 공개 데이터. 상업적 재배포 시 원 출처 표기 권장.

---

## 1. 공통 개발 과정 (책 단계 ↔ 앱 모듈 1:1)

책의 Azure ML Studio 워크플로와 앱 모듈의 대응:

| 책 단계 | 앱 모듈 | 비고 |
|---|---|---|
| 데이터셋 업로드 | `LoadData` | `pd.read_csv(source)` |
| (Clean Missing Data) | `HandleMissingValues` | 본 재현은 결측 없는 수치형 특성 선택으로 대체(아래 주석) |
| (Project Columns) | `TrainModel`의 `feature_columns` | 사용할 특성만 지정 |
| Split | `SplitData` | `train_size`, `random_state=42`, `shuffle` |
| 모델 정의 | `LinearRegression`/`DecisionTree`/`KMeans` | 시드 42 고정 |
| Train Model | `TrainModel` (지도) / `TrainClusteringModel` (군집) | |
| Score Model | `ScoreModel` (지도) / `ClusteringData` (군집) | 예측/군집 할당 |
| Evaluate Model | `EvaluateModel` | 회귀/분류 지표 |

**재현성 설계 원칙**: 모든 무작위 단계에 `random_state=42` 고정 → 외부 Python에서도 동일 결과.
**데이터 정합 주의**: 앱의 `LinearRegression`/`DecisionTree`는 순수 sklearn이라 입력 특성이 수치형이어야 한다.
책의 Azure는 범주형을 자동 인코딩하지만, 본 재현은 **결측 없는 수치형 특성만 선택**해 깔끔한 byte-identical 재현을 보장했다.
범주형까지 쓰려면 `EncodeCategorical` + `HandleMissingValues`를 체인 앞에 추가하면 된다(향후 확장, 산출물 1의 3-1/3-3 참조).

---

## 2. 주제 A — 회귀: 자동차 가격 예측 (책 Ch5)

- **데이터**: `imports-85-hdrs.csv` (201행). **목표**: `price` 예측.
- **모델**: `LinearRegression` (책의 Linear Regression과 동일 계열).
- **특성(10개, 결측 없는 수치형)**: symboling, wheel-base, length, width, height, curb-weight, engine-size, compression-ratio, city-mpg, highway-mpg.
  - 책은 `bore`, `stroke`를 제외(Project Columns) — 본 재현도 두 컬럼 미사용.
- **파이프라인**: `LoadData → SplitData(0.75, seed 42) → LinearRegression → TrainModel → ScoreModel → EvaluateModel(regression)`.
- **앱 사용**: 샘플 `Book_Automobile_LinearRegression.json` 로드 → 실행. (픽스처 `11_book_automobile_linreg.json`)

### 실제 결과 (test n=51)
| 지표 | 값 |
|---|---|
| 결정계수 R² | **0.7561** |
| RMSE | 5,129.9 |
| MAE | 3,487.7 |

**해석**: 10개 수치형 특성만으로 가격 분산의 약 76%를 설명. 엔진 크기·차폭·공차중량이 가격과 강하게 연동.
책처럼 범주형(make, body-style 등)을 인코딩해 추가하면 R² 추가 향상 여지가 있다.

### 동등 Python (앱 "전체 코드 보기"가 내보내는 코드의 핵심 — 검증됨)
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

---

## 3. 주제 B — 분류: 성인 소득 예측 (책 Ch3)

- **데이터**: `adult.csv` (32,561행). **목표**: `income` (`<=50K` vs `>50K`) 분류.
- **모델**: `DecisionTree` (classification, `max_depth=8`).
  - 책은 **Two-Class Boosted Decision Tree** 사용. 앱엔 부스팅 트리가 없어 의사결정나무로 대체(산출물 1의 3-1에서 그래디언트 부스팅 추가를 최우선 제안).
- **특성(6개, 결측 없는 수치형)**: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
- **파이프라인**: `LoadData → SplitData(0.8, seed 42) → DecisionTree → TrainModel → ScoreModel → EvaluateModel(classification)`.
- **앱 사용**: 샘플 `Book_AdultIncome_DecisionTree.json`. (픽스처 `13_book_adult_clf.json`)

### 실제 결과 (test n=6,513)
| 지표 | 값 |
|---|---|
| Accuracy | **0.8322** |
| Precision (>50K) | 0.8136 |
| Recall (>50K) | 0.3947 |
| F1 (>50K) | 0.5315 |

**해석**: 수치형 6특성만으로 정확도 83%. 다만 고소득(>50K) 재현율이 낮음(0.39) — 클래스 불균형 + 범주형(직업·학력·결혼상태) 미사용 탓.
책 수준으로 끌어올리려면 ①`EncodeCategorical`로 14개 특성 전부 사용, ②그래디언트 부스팅, ③`ResampleData`(SMOTE) 또는 임계값 조정이 효과적(산출물 1 항목들과 직접 연결).

---

## 4. 주제 C — 군집: 도매고객 세분화 (책 Ch6)

- **데이터**: `wholesale_customers.csv` (440행, 8열: Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen).
- **모델**: `KMeans` (n_clusters=**4**, seed 42) — 책의 Number of Centroids=4, Euclidean과 동일.
- **파이프라인**: `LoadData → KMeans + TrainClusteringModel → ClusteringData`. (책의 K-Means Clustering + Train Clustering Model + Assign to Clusters와 1:1)
- **앱 사용**: 샘플 `Book_Wholesale_KMeans.json`. (픽스처 `12_book_wholesale_kmeans.json`)

### 실제 결과
| 군집 | 고객 수 |
|---|---|
| 0 | 277 |
| 1 | 95 |
| 2 | 58 |
| 3 | 10 |

- inertia(군집 내 제곱합) ≈ 6.49e+10.
- **해석**: 대다수(277명)는 일반 소비 패턴, 소수 군집(10명)은 특정 카테고리 고액 지출(대형 거래처) 등으로 해석 가능.
  책처럼 스케일링(StandardScaler) 후 군집화하면 금액 단위가 큰 Fresh/Grocery의 지배를 줄여 더 균형 잡힌 세분화가 된다(앱 `ScalingTransform` 추가로 실험 가능).

---

## 5. 주제 D — 추천: 레스토랑 평점 (책 Ch7) — 확장 제안

- 책은 **Matchbox Recommender**로 사용자×아이템 평점에서 추천을 생성한다.
- **앱 현황**: 추천(협업 필터링) 모듈 **없음** → 현재 재현 불가.
- **제안**: 산출물 1의 **3-5 추천 시스템 모듈**(행렬분해/`surprise` 등) 신설 후 재현. Pyodide 패키지 가용성 확인 필요(미지원 시 외부 Python 전용).
- 보험/헬스케어 맥락 확장: 상품 교차판매 추천 등 JMDC 시나리오와 연계 여지.

---

## 6. 재현 방법 (사용자 검증)

1. **자동 회귀 검증**: 프로젝트 루트에서
   ```bash
   npm run verify:pipelines
   ```
   → `11/12/13_book_*` 포함 **12/12 PASS** 확인(외부 Python 2회 byte-identical).
2. **앱에서 직접 실행**: 앱 실행(`npm run dev`, http://127.0.0.1:3003) → 샘플 불러오기 →
   `Book_Automobile_LinearRegression` / `Book_AdultIncome_DecisionTree` / `Book_Wholesale_KMeans` 로드 →
   `LoadData`의 데이터로 `verify/datasets/`의 해당 CSV 지정 후 실행 → 결과 미리보기/평가 확인.
3. **코드 내보내기**: 각 파이프라인 "전체 코드 보기" → 외부 Jupyter/Python에 붙여넣어 동일 결과 재현.

---

## 7. 두 앱(베이스/JMDC) 동기화 메모

- 본 산출물의 **데이터셋·픽스처·샘플·문서는 공통 자산** → 양쪽 앱에 동일 적용 가능.
- 단, 향후 추가 모듈(그래디언트 부스팅·추천 등)은 **공통**으로 양쪽에 동일 반영하고, JMDC 전용 헬스케어 시나리오만 차이로 남긴다(프로젝트 불변식).
