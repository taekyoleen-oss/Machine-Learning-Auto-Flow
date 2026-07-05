// fix: Removed erroneous import of ModuleType from './App' to resolve circular dependency and declaration merge errors.
export enum ModuleType {
  LoadData = "LoadData",
  Statistics = "Statistics", // New module type
  SelectData = "SelectData",
  DataFiltering = "DataFiltering",
  ColumnPlot = "ColumnPlot",
  OutlierDetector = "OutlierDetector",
  HypothesisTesting = "HypothesisTesting",
  NormalityChecker = "NormalityChecker",
  Correlation = "Correlation",
  HandleMissingValues = "HandleMissingValues",
  TransformData = "TransformData",
  EncodeCategorical = "EncodeCategorical",
  ScalingTransform = "ScalingTransform",
  FeatureEngineer = "FeatureEngineer",
  PythonScript = "PythonScript",
  TransitionData = "TransitionData",
  ResampleData = "ResampleData",
  SplitData = "SplitData",
  Join = "Join",
  Concat = "Concat",

  // Supervised Learning Models
  LinearRegression = "LinearRegression",
  LogisticRegression = "LogisticRegression",
  PoissonRegression = "PoissonRegression",
  NegativeBinomialRegression = "NegativeBinomialRegression",
  DecisionTree = "DecisionTree",
  RandomForest = "RandomForest",
  GradientBoosting = "GradientBoosting",
  NeuralNetwork = "NeuralNetwork",
  SVM = "SVM",
  LDA = "LDA",
  NaiveBayes = "NaiveBayes",
  KNN = "KNN",

  // Model Operations
  TrainModel = "TrainModel",
  SweepParameters = "SweepParameters",
  ScoreModel = "ScoreModel",
  EvaluateModel = "EvaluateModel",
  FeatureImportance = "FeatureImportance",

  // Documentation (meta) module — 파이프라인 말단. 분석 모듈이 아님(export/verify 무관).
  ModelAnalysisReport = "ModelAnalysisReport",

  // Unsupervised Learning
  KMeans = "KMeans",
  PCA = "PCA",
  // 별칭: App.tsx/PropertiesPanel/ComponentRenderer/constants(DEFAULT_MODULES)가
  // PrincipalComponentAnalysis를 참조한다. 동일 값("PCA") 별칭 멤버로 정합성을 맞춰
  // 죽어있던 PCA 분기가 실제 PCA 모듈(type "PCA")과 매칭되도록 한다.
  PrincipalComponentAnalysis = "PCA",
  DBSCAN = "DBSCAN",
  HierarchicalClustering = "HierarchicalClustering",

  // Clustering Operations
  TrainClusteringModel = "TrainClusteringModel",
  ClusteringData = "ClusteringData",

  // Recommendation (Collaborative Filtering)
  Recommender = "Recommender",

  // Legacy/StatModels - Keeping for now
  StatModels = "StatModels",
  ResultModel = "ResultModel",
  PredictModel = "PredictModel",

  // Traditional Analysis - Statistical Models
  OLSModel = "OLSModel",
  LogisticModel = "LogisticModel",
  PoissonModel = "PoissonModel",
  QuasiPoissonModel = "QuasiPoissonModel",
  NegativeBinomialModel = "NegativeBinomialModel",
  DiversionChecker = "DiversionChecker",
  EvaluateStat = "EvaluateStat",
  VIFChecker = "VIFChecker",

  // Advanced Models - Mortality Models
  MortalityResult = "MortalityResult",
  LeeCarterModel = "LeeCarterModel",
  CBDModel = "CBDModel",
  APCModel = "APCModel",
  RHModel = "RHModel",
  PlatModel = "PlatModel",
  PSplineModel = "PSplineModel",

  // Frequency-Severity Simulation Models
  SimulateFreqSevTable = "SimulateFreqSevTable",
  CombineLossModel = "CombineLossModel",

  // Deprecating these
  LogisticTradition = "LogisticTradition",

  // Shape Types
  TextBox = "TextBox",
  GroupBox = "GroupBox",
}

export enum ModuleStatus {
  Pending = "Pending",
  Running = "Running",
  Success = "Success",
  Error = "Error",
}

export interface Port {
  name: string;
  type: "data" | "model" | "evaluation" | "handler";
}

export interface ColumnInfo {
  name: string;
  type: string;
}

export interface DataPreview {
  type: "DataPreview"; // Differentiator
  columns: ColumnInfo[];
  totalRowCount: number;
  rows?: Record<string, any>[];
}

// Types for the new Statistics module output
export interface DescriptiveStats {
  [columnName: string]: {
    count: number;
    mean: number;
    std: number;
    min: number;
    "25%": number;
    "50%": number; // median
    "75%": number;
    max: number;
    variance: number;
    nulls: number;
    mode: number | string;
    skewness: number;
    kurtosis: number;
    nonNullCount: number; // Non-Null count (same as count, for info() compatibility)
    dtype: string; // Data type (e.g., 'int64', 'float64', 'object')
  };
}

export interface CorrelationMatrix {
  [column1: string]: {
    [column2: string]: number;
  };
}

export interface StatisticsOutput {
  type: "StatisticsOutput"; // Differentiator
  stats: DescriptiveStats;
  correlation: CorrelationMatrix;
  columns: ColumnInfo[]; // Keep original column info
}

export interface SplitDataOutput {
  type: "SplitDataOutput";
  train: DataPreview;
  test: DataPreview;
}

export interface JoinOutput {
  type: "JoinOutput";
  rows: any[];
  columns: Array<{ name: string; type: string }>;
}

export interface ConcatOutput {
  type: "ConcatOutput";
  rows: any[];
  columns: Array<{ name: string; type: string }>;
}

export interface TuningCandidateScore {
  params: Record<string, number>;
  score: number;
}

export interface TuningSummary {
  enabled: boolean;
  strategy?: "grid";
  bestParams?: Record<string, number>;
  bestScore?: number;
  scoringMetric?: string;
  candidates?: TuningCandidateScore[];
}

export interface TrainedModelOutput {
  type: "TrainedModelOutput";
  modelType: ModuleType;
  modelPurpose?: "classification" | "regression";
  coefficients: Record<string, number>;
  intercept: number;
  metrics: Record<string, number>;
  featureColumns: string[];
  labelColumn: string;
  // 분류 학습 시 클래스 목록(코드 순서 = 정렬 순서). ScoreModel이 코드 예측을
  // 원 라벨로 복원하는 데 사용(문자열/비{0,1} 수치 라벨 정합).
  classLabels?: string[];
  tuningSummary?: TuningSummary;
  statsModelsResult?: StatsModelsResultOutput; // statsmodels 결과 (포아송/음이항 회귀용)
  trainingData?: Record<string, any>[]; // Decision Tree plot 생성을 위한 훈련 데이터
  modelParameters?: {
    // Decision Tree plot 생성을 위한 모델 파라미터
    criterion?: string;
    maxDepth?: number | null;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    classWeight?: string | null;
  };
}

export type StatsModelFamily =
  | "OLS"
  | "Logistic"
  | "Logit"
  | "Poisson"
  | "QuasiPoisson"
  | "NegativeBinomial"
  | "Gamma"
  | "Tweedie";

export interface ModelDefinitionOutput {
  type: "ModelDefinitionOutput";
  // "sklearn": 클러스터링/차원축소(K-Means·PCA·DBSCAN·계층형) 모델 정의도 이 타입으로 표현된다.
  modelFamily: "statsmodels" | "sklearn";
  modelType: StatsModelFamily;
  parameters: Record<string, any>;
}

export interface StatsModelsResultOutput {
  type: "StatsModelsResultOutput";
  modelType: StatsModelFamily;
  summary: {
    coefficients: Record<
      string,
      {
        coef: number;
        "std err": number;
        t?: number;
        z?: number;
        "P>|t|"?: number;
        "P>|z|"?: number;
        "[0.025": number;
        "0.975]": number;
      }
    >;
    metrics: Record<string, string | number>;
  };
  featureColumns: string[];
  labelColumn: string;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ThresholdMetric {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface EvaluationOutput {
  type: "EvaluationOutput";
  modelType: "classification" | "regression";
  metrics: Record<string, number | string>;
  confusionMatrix?: ConfusionMatrix;
  threshold?: number;
  thresholdMetrics?: ThresholdMetric[]; // 여러 threshold에 대한 precision/recall
}

// --- New Unsupervised Learning Outputs ---
export interface KMeansOutput {
  type: "KMeansOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
  centroids: Record<string, number>[];
  model: any; // To hold inertia_ or other model properties
}

export interface PCAOutput {
  type: "PCAOutput";
  transformedData: DataPreview;
  explainedVarianceRatio: number[];
}

// --- Clustering Model Outputs ---
export interface TrainedClusteringModelOutput {
  type: "TrainedClusteringModelOutput";
  modelType: ModuleType; // KMeans or PrincipalComponentAnalysis
  featureColumns: string[];
  model: any; // The trained clustering model
  // For K-Means
  centroids?: Record<string, number>[];
  inertia?: number;
  // For PCA
  components?: number[][];
  explainedVarianceRatio?: number[];
  mean?: number[];
  // For transductive clusterers (DBSCAN / Hierarchical): labels computed at fit time.
  labels?: number[];
  nClusters?: number;
  nNoise?: number;
}

export interface ClusteringDataOutput {
  type: "ClusteringDataOutput";
  clusteredData: DataPreview; // Data with cluster assignments or PCA transformed data
  modelType: ModuleType;
}

export interface MissingHandlerOutput {
  type: "MissingHandlerOutput";
  method: "remove_row" | "impute" | "knn";
  // For impute
  strategy?: "mean" | "median" | "mode";
  // For KNN
  n_neighbors?: number;
  metric?: string;
  // For all methods that are not row removal, we need the values computed from the training set
  imputation_values: Record<string, number | string>; // e.g. { 'Age': 29.5, 'Embarked': 'S' }
}

export interface EncoderOutput {
  type: "EncoderOutput";
  method: "label" | "one_hot" | "ordinal";
  mappings: Record<string, Record<string, number> | string[]>;
  columns_to_encode: string[];
  // one-hot params that are passed through
  drop?: "first" | "if_binary" | null;
  handle_unknown?: "error" | "ignore";
}

export interface NormalizerOutput {
  type: "NormalizerOutput";
  method: "MinMax" | "StandardScaler" | "RobustScaler";
  stats: Record<
    string,
    {
      min?: number;
      max?: number;
      mean?: number;
      stdDev?: number;
      median?: number;
      iqr?: number;
    }
  >;
}

export interface ColumnPlotOutput {
  type: "ColumnPlotOutput";
  plot_type: "single" | "double";
  column1: string;
  column2?: string;
  column1Type: "number" | "string";
  column2Type?: "number" | "string";
  availableCharts: string[];
  selectedChart?: string;
  imageBase64?: string; // 차트 이미지 (base64)
}

export interface OutlierResult {
  method: "IQR" | "ZScore" | "IsolationForest" | "Boxplot";
  outlierIndices: number[]; // 이상치로 탐지된 행 인덱스
  outlierCount: number;
  outlierPercentage: number;
  details?: Record<string, any>; // 방법별 상세 정보
}

export interface ColumnOutlierResult {
  column: string;
  results: OutlierResult[]; // 각 방법별 결과
  totalOutliers: number; // 모든 방법에서 탐지된 총 이상치 수 (중복 제거)
  outlierIndices: number[]; // 모든 방법에서 탐지된 이상치 인덱스 (중복 제거)
}

export interface OutlierDetectorOutput {
  type: "OutlierDetectorOutput";
  columns: string[]; // 분석된 열 목록
  columnResults: ColumnOutlierResult[]; // 각 열별 결과
  totalOutliers: number; // 모든 열에서 탐지된 총 이상치 수 (중복 제거)
  allOutlierIndices: number[]; // 모든 열에서 탐지된 이상치 인덱스 (중복 제거)
  cleanedData?: Record<string, any>[]; // 이상치 제거된 데이터 (선택적)
  originalData?: Record<string, any>[]; // 원본 데이터 (제거 작업을 위해 필요)
}

export type HypothesisTestType =
  | "t_test_one_sample"
  | "t_test_independent"
  | "t_test_paired"
  | "chi_square"
  | "anova"
  | "ks_test"
  | "shapiro_wilk"
  | "levene";

export interface HypothesisTestResult {
  testType: HypothesisTestType;
  testName: string;
  columns: string[]; // 사용된 열
  statistic?: number; // 검정 통계량
  pValue?: number; // p-value
  degreesOfFreedom?: number | number[]; // 자유도
  criticalValue?: number; // 임계값
  conclusion?: string; // 결론 (예: "Reject H0", "Fail to reject H0")
  interpretation?: string; // 해석
  details?: Record<string, any>; // 검정별 상세 정보
}

export interface HypothesisTestingOutput {
  type: "HypothesisTestingOutput";
  results: HypothesisTestResult[]; // 각 검정별 결과
}

// 상관분석(Correlation) 모듈의 방법별 결과. 위의 index-signature형 CorrelationMatrix와
// 별개 타입이다(이전에는 같은 이름으로 선언 병합되어 타입 오류를 유발했다).
export interface CorrelationMethodResult {
  method: "pearson" | "spearman" | "kendall" | "cramers_v";
  matrix: Record<string, Record<string, number>>; // 상관계수 행렬
  columns: string[]; // 분석된 열 목록
}

export interface CorrelationOutput {
  type: "CorrelationOutput";
  columns: string[]; // 분석된 열 목록
  numericColumns: string[]; // 숫자형 열
  categoricalColumns: string[]; // 범주형 열
  correlationMatrices: CorrelationMethodResult[]; // 각 방법별 상관계수 행렬
  heatmapImage?: string; // Heatmap 이미지 (base64)
  pairplotImage?: string; // Pairplot 이미지 (base64)
  summary?: Record<string, any>; // 요약 통계
}

export interface VIFCheckerOutput {
  type: "VIFCheckerOutput";
  results: Array<{
    column: string;
    vif: number;
  }>;
}

export type NormalityTestType =
  | "shapiro_wilk"
  | "kolmogorov_smirnov"
  | "anderson_darling"
  | "dagostino_k2";

export interface NormalityTestResult {
  testType: NormalityTestType;
  testName: string;
  statistic?: number;
  pValue?: number;
  criticalValue?: number;
  conclusion?: string; // "Reject H0" or "Fail to reject H0"
  interpretation?: string;
  details?: Record<string, any>;
}

export interface NormalityCheckerOutput {
  type: "NormalityCheckerOutput";
  column: string; // 분석된 열
  skewness: number;
  kurtosis: number;
  jarqueBera: {
    statistic: number;
    pValue: number;
    conclusion: string;
  };
  testResults: NormalityTestResult[]; // 선택된 검정 방법별 결과
  histogramImage?: string; // Histogram + Normal Curve Overlay (base64)
  qqPlotImage?: string; // QQ-Plot (base64)
  ecdfImage?: string; // ECDF vs Normal CDF (base64)
  boxplotImage?: string; // Boxplot (base64)
}

export interface MortalityModelOutput {
  type: "MortalityModelOutput";
  modelType: "LeeCarter" | "CBD" | "APC" | "RH" | "Plat" | "PSpline";
  a_x?: Record<string, number>; // 연령별 파라미터
  b_x?: Record<string, number>; // 연령별 파라미터
  k_t?: Record<string, number>; // 연도별 파라미터
  beta?: Record<string, number>; // CBD 모델용
  kappa_1?: Record<string, number>; // CBD 모델용
  kappa_2?: Record<string, number>; // CBD 모델용
  gamma_c?: Record<string, number>; // APC/RH 모델용
  b_x_1?: Record<string, number>; // RH 모델용
  b_x_2?: Record<string, number>; // RH 모델용
  k_t_1?: Record<string, number>; // RH 모델용
  k_t_2?: Record<string, number>; // RH 모델용
  mortality_matrix: Record<string, Record<string, number>>;
  predicted_mortality: Record<string, Record<string, number>>;
  mse: number;
  mae: number;
  ages: number[];
  years: number[];
  forecast_years?: number[];
  forecast_mortality?: Record<string, Record<string, number>>;
}

export interface MortalityResultOutput {
  type: "MortalityResultOutput";
  models: Array<{
    modelType: string;
    mse: number;
    mae: number;
    aic?: number;
    bic?: number;
  }>;
  comparison: {
    best_model: string;
    metrics_comparison: Record<string, Record<string, number>>;
  };
  visualizations: {
    mortality_curves?: string; // base64 이미지
    forecast_comparison?: string; // base64 이미지
    parameter_comparison?: string; // base64 이미지
  };
}

// --- Model Analysis Report (documentation / meta module) ---
// 파이프라인 말단에 두는 문서화 모듈. AI(Claude) 또는 결정적 폴백으로 자기완결 HTML 보고서를
// 만들어 모듈 결과로 저장한다. 데이터 분석이 아니므로 export/verify 대상이 아니다.
export interface ReportContext {
  modelType?: string; // 예: "이진 분류", "회귀", "군집(KMeans)"
  datasetName?: string;
  dataSource?: string; // 파일명/경로
  rowCount?: number;
  columnCount?: number;
  columns?: Array<{ name: string; type: string; sample?: string }>;
  sampleRows?: Record<string, any>[]; // 원본 표본(처음 N행)
  classDistribution?: Array<{ label: string; count: number; ratio?: number }>;
  split?: {
    train_size?: number;
    random_state?: number | null;
    shuffle?: boolean;
    stratify?: boolean;
  };
  modelDefinition?: {
    kind?: string; // 예: "DecisionTree"
    params?: Record<string, any>;
  };
  features?: string[]; // 학습에 사용된 특성
  labelColumn?: string;
  metrics?: Record<string, number | string>;
  confusionMatrix?: { tp: number; fp: number; tn: number; fn: number };
  thresholdMetrics?: Array<Record<string, number>>; // 임계값 스윕(있으면)
  steps?: Array<{ type: string; name: string; params?: Record<string, any> }>;
  clustering?: {
    k?: number;
    inertia?: number;
    nClusters?: number;
    nNoise?: number;
    distribution?: Array<{ cluster: string; count: number }>;
  };
  extraInfo?: string; // 사용자 추가정보(텍스트 + PDF 추출 텍스트 병합)
  title?: string;
  generatedAt?: string;
}

export interface ModelReportOutput {
  type: "ModelReportOutput";
  html: string;
  generatedAt: string;
  source: "ai" | "fallback";
  context: ReportContext;
}

// --- JMDC 계열(재보험/XoL·분포적합·통계평가) 모듈 출력 ---
// 이들 모듈은 파이프라인에서 discriminated union(`type`)으로만 취급되고 미리보기 모달은
// 필드를 동적으로 읽는다. 각 필드 스키마가 Python 산출물(snake_case 원본 포함)에 따라
// 유동적이라 인덱스 시그니처로 허용한다. `type` 판별자는 문자열 리터럴로 고정되어 있어
// discriminated-union 좁히기는 정상 동작한다. (값·런타임 영향 없는 타입 선언 전용.)
export interface FittedDistributionOutput {
  type: "FittedDistributionOutput";
  [key: string]: any;
}

export interface ExposureCurveOutput {
  type: "ExposureCurveOutput";
  [key: string]: any;
}

export interface XoLPriceOutput {
  type: "XoLPriceOutput";
  [key: string]: any;
}

export interface XolContractOutput {
  type: "XolContractOutput";
  [key: string]: any;
}

export interface FinalXolPriceOutput {
  type: "FinalXolPriceOutput";
  [key: string]: any;
}

export interface DiversionCheckerOutput {
  type: "DiversionCheckerOutput";
  [key: string]: any;
}

export interface EvaluateStatOutput {
  type: "EvaluateStatOutput";
  [key: string]: any;
}

export interface CanvasModule {
  id: string;
  name: string;
  type: ModuleType;
  position: { x: number; y: number };
  status: ModuleStatus;
  parameters: Record<string, any>;
  inputs: Port[];
  outputs: Port[];
  outputData?:
    | DataPreview
    | StatisticsOutput
    | SplitDataOutput
    | JoinOutput
    | ConcatOutput
    | TrainedModelOutput
    | ModelDefinitionOutput
    | StatsModelsResultOutput
    | EvaluationOutput
    | KMeansOutput
    | PCAOutput
    | TrainedClusteringModelOutput
    | ClusteringDataOutput
    | MissingHandlerOutput
    | EncoderOutput
    | NormalizerOutput
    | ColumnPlotOutput
    | OutlierDetectorOutput
    | HypothesisTestingOutput
    | NormalityCheckerOutput
    | CorrelationOutput
    | VIFCheckerOutput
    | MortalityModelOutput
    | MortalityResultOutput
    | ModelReportOutput
    | FittedDistributionOutput
    | ExposureCurveOutput
    | XoLPriceOutput
    | XolContractOutput
    | FinalXolPriceOutput
    | DiversionCheckerOutput
    | EvaluateStatOutput;
  executionTime?: number; // milliseconds, set after successful run
  notes?: string; // user-written memo shown on the module card
  // Shape-specific properties
  shapeData?: {
    // For TextBox
    text?: string;
    // TextBox 크기·글자 크기(런타임에서 사용)
    width?: number;
    height?: number;
    fontSize?: number;
    // For GroupBox
    moduleIds?: string[]; // IDs of modules in this group
    bounds?: { x: number; y: number; width: number; height: number }; // Bounding box of the group
  };
}

export interface Connection {
  id: string;
  from: { moduleId: string; portName: string };
  to: { moduleId: string; portName: string };
}
