# 모듈 이름 및 실행 파일 매핑

이 문서는 각 모듈의 표시 이름과 실제 실행되는 파일/함수를 매핑합니다.

## Data Preprocess

| 모듈 이름 (표시)      | ModuleType          | 실행 파일/함수                                         | 설명                                                 |
| --------------------- | ------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| Load Data             | LoadData            | `App.tsx` - `runSimulation` (인라인 CSV 파싱)          | CSV 파일을 로드하고 파싱합니다                       |
| Statistics            | Statistics          | `utils/pyodideRunner.ts` - `calculateStatisticsPython` | 데이터의 기술 통계를 계산합니다                      |
| Select Data           | SelectData          | `App.tsx` - `runSimulation` (인라인 로직)              | 선택된 열만 필터링합니다                             |
| Data Filtering        | DataFiltering       | `utils/pyodideRunner.ts` - `filterDataPython`          | 조건에 따라 행/열을 필터링합니다                     |
| Handle Missing Values | HandleMissingValues | `utils/pyodideRunner.ts` - `handleMissingValuesPython` | 결측치를 처리합니다 (제거 또는 대체)                 |
| Encode Categorical    | EncodeCategorical   | `utils/pyodideRunner.ts` - `encodeCategoricalPython`   | 범주형 변수를 인코딩합니다 (Label, One-Hot, Ordinal) |
| Scaling Transform     | ScalingTransform    | `utils/pyodideRunner.ts` - `normalizeDataPython`       | 수치형 변수를 정규화/표준화합니다                    |
| Transition Data       | TransitionData      | `utils/pyodideRunner.ts` - `transformDataPython`       | 수치형 열에 수학적 변환을 적용합니다                 |
| Resample Data         | ResampleData        | `utils/pyodideRunner.ts` - `resampleDataPython`        | 클래스 불균형을 처리합니다 (SMOTE, NearMiss 등)      |

## Stat Lab

| 모듈 이름 (표시)   | ModuleType        | 실행 파일/함수                                          | 설명                                                                   |
| ------------------ | ----------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- |
| Column Plot        | ColumnPlot        | `utils/pyodideRunner.ts` - `createColumnPlotPython`     | 단일/이중 열에 대한 다양한 플롯을 생성합니다                           |
| Outlier Detector   | OutlierDetector   | `utils/pyodideRunner.ts` - `detectOutliers`             | 이상치를 탐지합니다 (IQR, Z-score, Isolation Forest, Boxplot)          |
| Hypothesis Testing | HypothesisTesting | `utils/pyodideRunner.ts` - `performHypothesisTests`     | 다양한 가설 검정을 수행합니다 (t-test, chi-square, ANOVA 등)           |
| Normality Checker  | NormalityChecker  | `utils/pyodideRunner.ts` - `performNormalityCheck`      | 정규성을 검정합니다 (Shapiro-Wilk, KS-test 등)                         |
| Correlation        | Correlation       | `utils/pyodideRunner.ts` - `performCorrelationAnalysis` | 변수 간 상관관계를 분석합니다 (Pearson, Spearman, Kendall, Cramér's V) |

## Data Analysis - Operations

| 모듈 이름 (표시)       | ModuleType           | 실행 파일/함수                                               | 설명                                   |
| ---------------------- | -------------------- | ------------------------------------------------------------ | -------------------------------------- |
| Split Data             | SplitData            | `utils/pyodideRunner.ts` - `splitDataPython`                 | 데이터를 훈련/테스트 세트로 분할합니다 |
| Train Model            | TrainModel           | `App.tsx` - `runSimulation` (모델 타입별 함수 호출)          | 머신러닝 모델을 학습시킵니다           |
| Score Model            | ScoreModel           | `App.tsx` - `runSimulation` (모델 타입별 함수 호출)          | 학습된 모델로 예측을 수행합니다        |
| Evaluate Model         | EvaluateModel        | `utils/pyodideRunner.ts` - `evaluateModelPython`             | 모델의 성능을 평가합니다               |
| Train Clustering Model | TrainClusteringModel | `App.tsx` - `runSimulation` (KMeans/PCA 함수 호출)           | 비지도 학습 모델을 학습시킵니다        |
| Clustering Data        | ClusteringData       | `App.tsx` - `runSimulation` (KMeans/PCA transform 함수 호출) | 학습된 클러스터링 모델을 적용합니다    |

## Data Analysis - Supervised Learning

| 모듈 이름 (표시)             | ModuleType         | 실행 파일/함수                                                         | 설명                             |
| ---------------------------- | ------------------ | ---------------------------------------------------------------------- | -------------------------------- |
| Linear Regression            | LinearRegression   | `utils/pyodideRunner.ts` - `fitLinearRegressionPython`                 | 선형 회귀 모델을 정의합니다      |
| Logistic Regression          | LogisticRegression | `utils/pyodideRunner.ts` - `fitLogisticRegressionPython`               | 로지스틱 회귀 모델을 정의합니다  |
| Decision Tree                | DecisionTree       | `utils/pyodideRunner.ts` - `fitDecisionTreePython`                     | 결정 나무 모델을 정의합니다      |
| Random Forest                | RandomForest       | `utils/pyodideRunner.ts` - `fitDecisionTreePython` (RandomForest 모드) | 랜덤 포레스트 모델을 정의합니다  |
| Neural Network               | NeuralNetwork      | `utils/pyodideRunner.ts` - `fitNeuralNetworkPython`                    | 신경망 모델을 정의합니다         |
| Support Vector Machine       | SVM                | `utils/pyodideRunner.ts` - `fitSVMPython`                              | SVM 모델을 정의합니다            |
| Linear Discriminant Analysis | LDA                | `utils/pyodideRunner.ts` - `fitLDAPython`                              | 선형 판별 분석 모델을 정의합니다 |
| Naive Bayes                  | NaiveBayes         | `utils/pyodideRunner.ts` - `fitNaiveBayesPython`                       | 나이브 베이즈 모델을 정의합니다  |
| K-Nearest Neighbors          | KNN                | `utils/pyodideRunner.ts` - `fitKNNPython`                              | KNN 모델을 정의합니다            |

## Data Analysis - Unsupervised Learning

| 모듈 이름 (표시)             | ModuleType | 실행 파일/함수                               | 설명                                 |
| ---------------------------- | ---------- | -------------------------------------------- | ------------------------------------ |
| K-Means Clustering           | KMeans     | `utils/pyodideRunner.ts` - `fitKMeansPython` | K-Means 클러스터링 모델을 정의합니다 |
| Principal Component Analysis | PCA        | `utils/pyodideRunner.ts` - `fitPCAPython`    | 주성분 분석 모델을 정의합니다        |

## Tradition Analysis - Operations

| 모듈 이름 (표시)  | ModuleType       | 실행 파일/함수                                       | 설명                                            |
| ----------------- | ---------------- | ---------------------------------------------------- | ----------------------------------------------- |
| Stat Models       | StatModels       | `App.tsx` - `runSimulation` (모델 정의만 생성)       | 통계 모델을 정의합니다 (Gamma, Tweedie 등)      |
| Result Model      | ResultModel      | `utils/pyodideRunner.ts` - `runStatsModel`           | 통계 모델을 피팅하고 결과를 보여줍니다          |
| Predict Model     | PredictModel     | `utils/pyodideRunner.ts` - `predictWithStatsmodel`   | 피팅된 통계 모델로 예측을 수행합니다            |
| Diversion Checker | DiversionChecker | `utils/pyodideRunner.ts` - `dispersionCheckerPython` | 과분산을 확인하고 적절한 회귀 모델을 추천합니다 |
| Evaluate Stat     | EvaluateStat     | `utils/pyodideRunner.ts` - `evaluateStatsPython`     | 통계 모델의 성능을 평가합니다                   |

## Tradition Analysis - Statistical Model

| 모듈 이름 (표시)        | ModuleType            | 실행 파일/함수                                 | 설명                            |
| ----------------------- | --------------------- | ---------------------------------------------- | ------------------------------- |
| OLS Model               | OLSModel              | `App.tsx` - `runSimulation` (모델 정의만 생성) | OLS 회귀 모델을 정의합니다      |
| Logistic Model          | LogisticModel         | `App.tsx` - `runSimulation` (모델 정의만 생성) | 로지스틱 회귀 모델을 정의합니다 |
| Poisson Model           | PoissonModel          | `App.tsx` - `runSimulation` (모델 정의만 생성) | 포아송 회귀 모델을 정의합니다   |
| Quasi-Poisson Model     | QuasiPoissonModel     | `App.tsx` - `runSimulation` (모델 정의만 생성) | 준포아송 회귀 모델을 정의합니다 |
| Negative Binomial Model | NegativeBinomialModel | `App.tsx` - `runSimulation` (모델 정의만 생성) | 음이항 회귀 모델을 정의합니다   |

## 기타 모듈

| 모듈 이름 (표시) | ModuleType | 실행 파일/함수      | 설명                      |
| ---------------- | ---------- | ------------------- | ------------------------- |
| TextBox          | TextBox    | 실행 없음 (UI 전용) | 텍스트 상자 (시각적 요소) |
| GroupBox         | GroupBox   | 실행 없음 (UI 전용) | 그룹 상자 (시각적 요소)   |

## 참고사항

- **인라인 로직**: `App.tsx`의 `runSimulation` 함수 내에서 직접 구현된 로직
- **Python 함수**: `utils/pyodideRunner.ts`에 정의된 Pyodide를 사용한 Python 실행 함수
- **모델 정의 모듈**: 모델 정의만 생성하고, 실제 학습은 `TrainModel` 또는 `ResultModel`에서 수행
- **Transform 모듈**: Missing Transform, Encoding Transform, Scaling Transform은 fit-transform 패턴을 지원하며 두 개의 입력/출력을 가집니다
