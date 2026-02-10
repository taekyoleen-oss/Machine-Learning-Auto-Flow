# ML Auto Flow - 앱 분석 및 사용 가이드

## 📋 목차
1. [앱 개요](#앱-개요)
2. [주요 기능](#주요-기능)
3. [사용 방법](#사용-방법)
4. [모듈 카테고리](#모듈-카테고리)
5. [워크플로우 예제](#워크플로우-예제)
6. [특별 기능](#특별-기능)
7. [기술 스택](#기술-스택)

---

## 🎯 앱 개요

**ML Auto Flow**는 머신러닝 워크플로우를 시각적으로 구성하고 실행할 수 있는 웹 기반 애플리케이션입니다. 드래그 앤 드롭 방식으로 데이터 분석 파이프라인을 구축하고, 브라우저에서 직접 Python 기반 머신러닝 모델을 실행할 수 있습니다.

### 핵심 특징
- ✅ **시각적 파이프라인 빌더**: 노드 기반 인터페이스로 복잡한 ML 워크플로우 구성
- ✅ **브라우저 내 실행**: Pyodide를 활용하여 서버 없이 브라우저에서 Python 코드 실행
- ✅ **풍부한 모듈 라이브러리**: 70개 이상의 데이터 분석 및 머신러닝 모듈 제공
- ✅ **AI 기반 파이프라인 생성**: Gemini API를 활용한 자동 파이프라인 생성
- ✅ **실시간 결과 시각화**: 각 모듈의 실행 결과를 상세하게 확인
- ✅ **샘플 관리 시스템**: 재사용 가능한 파이프라인 샘플 저장 및 공유

---

## 🚀 주요 기능

### 1. 시각적 파이프라인 구성
- **드래그 앤 드롭**: Toolbox에서 모듈을 Canvas로 드래그하여 배치
- **연결 시스템**: 모듈 간 데이터 흐름을 시각적으로 연결
- **자동 레이아웃**: 모듈 배치를 자동으로 정렬하여 가독성 향상
- **실행 순서 자동 결정**: 의존성에 따라 모듈 실행 순서 자동 계산

### 2. 데이터 처리
- **다양한 데이터 소스**: CSV 파일 로드, Excel 파일 지원
- **데이터 전처리**: 결측치 처리, 정규화, 인코딩, 변환 등
- **데이터 분석**: 기술 통계량, 상관관계 분석, 이상치 탐지
- **데이터 분할**: 훈련/테스트 데이터 분할

### 3. 머신러닝 모델
- **지도 학습**: 선형 회귀, 로지스틱 회귀, 의사결정나무, 랜덤 포레스트, 신경망, SVM, KNN 등
- **비지도 학습**: K-Means 클러스터링, PCA 차원 축소
- **통계 모델**: OLS, 로지스틱, 포아송, 음이항 회귀 등 (statsmodels 기반)
- **고급 모델**: 사망률 예측 모델 (Lee-Carter, CBD, APC, RH, Plat, P-Spline)

### 4. 모델 평가 및 예측
- **성능 지표**: 분류(정확도, 정밀도, 재현율, F1) 및 회귀(MSE, RMSE, R²) 지표
- **혼동 행렬**: 분류 모델의 성능 시각화
- **예측 생성**: 훈련된 모델로 새 데이터 예측

### 5. AI 기반 기능
- **AI로 파이프라인 생성**: 목표나 데이터를 설명하면 자동으로 파이프라인 생성
- **AI 데이터 분석**: 데이터를 업로드하면 자동으로 분석 계획 수립 및 실행

### 6. 결과 시각화
- **상세 통계**: 각 모듈의 실행 결과를 표, 차트, 그래프로 표시
- **데이터 미리보기**: 처리된 데이터를 테이블 형식으로 확인
- **모델 성능 시각화**: 학습 곡선, 혼동 행렬, ROC 곡선 등

### 7. 샘플 관리
- **샘플 저장**: 현재 파이프라인을 샘플로 저장
- **샘플 로드**: 저장된 샘플을 불러와서 재사용
- **샘플 공유**: 샘플 파일을 다른 사용자와 공유 가능

### 8. PPT 자동 생성
- **파이프라인 문서화**: 각 모듈별 PowerPoint 프레젠테이션 자동 생성
- **AI 기반 설명**: 각 모듈의 기능과 결과를 AI가 자동으로 설명

---

## 📖 사용 방법

### 기본 워크플로우

#### 1단계: 데이터 로드
1. Toolbox에서 **"Load Data"** 모듈을 Canvas로 드래그
2. 모듈을 클릭하여 Properties Panel에서 CSV 파일 선택
3. "Run" 버튼 클릭하여 데이터 로드

#### 2단계: 데이터 전처리
1. 필요한 전처리 모듈 추가:
   - **Handle Missing Values**: 결측치 처리
   - **Encode Categorical**: 범주형 변수 인코딩
   - **Scaling Transform**: 데이터 정규화
2. Load Data 모듈의 출력을 전처리 모듈의 입력에 연결
3. 각 모듈의 파라미터 설정 후 실행

#### 3단계: 모델 정의 및 훈련
1. **모델 정의 모듈** 추가 (예: Linear Regression, Decision Tree)
2. **Train Model** 모듈 추가
3. 전처리된 데이터와 모델 정의를 Train Model에 연결
4. Feature Columns와 Label Column 설정
5. "Run" 버튼으로 모델 훈련

#### 4단계: 모델 평가
1. **Split Data** 모듈로 훈련/테스트 데이터 분할
2. **Score Model** 모듈로 예측 생성
3. **Evaluate Model** 모듈로 성능 평가
4. "View Details" 버튼으로 상세 결과 확인

### 고급 기능 사용

#### AI로 파이프라인 생성하기
1. 상단 툴바의 **"AI로 파이프라인 생성하기"** 버튼 클릭
2. 목표를 입력 (예: "주택 가격 예측 모델 만들기")
3. AI가 자동으로 파이프라인 생성
4. 생성된 파이프라인을 Canvas에 추가

#### 샘플 사용하기
1. 상단 툴바의 **"Samples"** 메뉴 클릭
2. 원하는 샘플 선택
3. 샘플이 Canvas에 로드됨
4. 필요에 따라 수정하여 사용

#### 파이프라인 저장/불러오기
- **저장**: 상단 툴바의 "Save" 버튼 클릭
- **불러오기**: "Load" 버튼으로 저장된 파이프라인 불러오기
- **샘플로 저장**: "Sample로 저장" 버튼으로 재사용 가능한 샘플 생성

---

## 📦 모듈 카테고리

### 데이터 로드 및 조작
- **Load Data**: CSV/Excel 파일 로드
- **Data Joiner**: 두 데이터셋 조인 (inner, outer, left, right)
- **Data Concatenator**: 데이터셋 결합 (행/열 방향)
- **Select Data**: 컬럼 선택/제거
- **Data Filtering**: 조건에 따른 데이터 필터링

### 데이터 분석
- **Statistics**: 기술 통계량 및 상관관계 분석
- **Column Plot**: 다양한 차트 시각화 (히스토그램, 산점도, 박스플롯 등)
- **Outlier Detector**: 이상치 탐지 (IQR, Z-score, Isolation Forest)
- **Hypothesis Testing**: 가설 검정 (t-test, chi-square, ANOVA 등)
- **Normality Checker**: 정규성 검정
- **Correlation**: 상관관계 분석 (Pearson, Spearman, Kendall, Cramér's V)
- **VIF Checker**: 다중공선성 검사

### 데이터 전처리
- **Handle Missing Values**: 결측치 처리 (제거, 대체, KNN)
- **Encode Categorical**: 범주형 변수 인코딩 (Label, One-Hot, Ordinal)
- **Scaling Transform**: 데이터 정규화 (MinMax, StandardScaler, RobustScaler)
- **Transition Data**: 수학적 변환 (로그, 제곱근, 제곱 등)
- **Resample Data**: 클래스 불균형 처리 (SMOTE, NearMiss)

### 데이터 분할
- **Split Data**: 훈련/테스트 데이터 분할

### 지도 학습 모델
- **Linear Regression**: 선형 회귀
- **Logistic Regression**: 로지스틱 회귀
- **Decision Tree**: 의사결정나무
- **Random Forest**: 랜덤 포레스트
- **Neural Network**: 신경망
- **SVM**: 서포트 벡터 머신
- **KNN**: K-최근접 이웃
- **LDA**: 선형 판별 분석
- **Naive Bayes**: 나이브 베이즈

### 비지도 학습
- **K-Means Clustering**: K-Means 클러스터링
- **Principal Component Analysis**: 주성분 분석

### 모델 운영
- **Train Model**: 모델 훈련
- **Score Model**: 예측 생성
- **Evaluate Model**: 모델 성능 평가

### 통계 모델 (statsmodels)
- **OLS Model**: 최소제곱 회귀
- **Logistic Model**: 로지스틱 회귀
- **Poisson Model**: 포아송 회귀
- **Quasi-Poisson Model**: 준포아송 회귀
- **Negative Binomial Model**: 음이항 회귀
- **Diversion Checker**: 모델 이탈 검사
- **Evaluate Stat**: 통계 모델 평가

### 고급 모델
- **Mortality Models**: 사망률 예측 모델
  - Lee-Carter Model
  - CBD Model
  - APC Model
  - RH Model
  - Plat Model
  - P-Spline Model
- **Mortality Result**: 여러 사망률 모델 비교

---

## 🔄 워크플로우 예제

### 예제 1: 회귀 문제 (주택 가격 예측)

```
1. Load Data → 주택 데이터 CSV 로드
2. Statistics → 데이터 기본 통계 확인
3. Handle Missing Values → 결측치 처리
4. Encode Categorical → 범주형 변수 인코딩
5. Scaling Transform → 특징 정규화
6. Split Data → 훈련/테스트 분할
7. Linear Regression → 모델 정의
8. Train Model → 모델 훈련
9. Score Model → 예측 생성
10. Evaluate Model → 성능 평가
```

### 예제 2: 분류 문제 (고객 이탈 예측)

```
1. Load Data → 고객 데이터 로드
2. Statistics → 데이터 탐색
3. Outlier Detector → 이상치 탐지 및 제거
4. Correlation → 변수 간 상관관계 확인
5. Handle Missing Values → 결측치 처리
6. Encode Categorical → 범주형 변수 인코딩
7. Resample Data → 클래스 불균형 처리 (SMOTE)
8. Split Data → 훈련/테스트 분할
9. Random Forest → 모델 정의
10. Train Model → 모델 훈련
11. Score Model → 예측 생성
12. Evaluate Model → 성능 평가 (혼동 행렬 확인)
```

### 예제 3: 클러스터링 (고객 세분화)

```
1. Load Data → 고객 데이터 로드
2. Statistics → 데이터 탐색
3. Scaling Transform → 특징 정규화
4. K-Means → 클러스터링 모델 정의
5. Train Clustering Model → 모델 훈련
6. Clustering Data → 클러스터 할당
7. (View Details로 클러스터 중심점 및 품질 지표 확인)
```

---

## ✨ 특별 기능

### 1. 자동 레이아웃
- **Auto Layout** 버튼으로 모듈을 자동으로 정렬
- 데이터 흐름에 따라 왼쪽에서 오른쪽으로 배치
- 입력이 2개인 모듈의 경우 위아래로 정렬

### 2. 히스토리 관리
- **Undo/Redo**: 작업 이력 관리
- **실행 취소/재실행**: 실수로 삭제한 모듈 복구 가능

### 3. 코드 생성
- **Code 탭**: 각 모듈의 Python 코드 확인
- **전체 파이프라인 코드**: 전체 워크플로우의 Python 코드 생성

### 4. 터미널 로그
- **Terminal 탭**: 각 모듈 실행 시 로그 확인
- **에러 추적**: 문제 발생 시 상세한 에러 메시지 확인

### 5. 다크 모드
- **테마 전환**: 라이트/다크 모드 지원
- **사용자 선호도 저장**: 브라우저에 테마 설정 저장

### 6. 확대/축소 및 팬
- **줌 인/아웃**: 마우스 휠로 캔버스 확대/축소
- **팬**: 드래그로 캔버스 이동
- **전체 보기**: 모든 모듈이 보이도록 자동 조정

---

## 🛠 기술 스택

### 프론트엔드
- **React 19**: UI 프레임워크
- **TypeScript**: 타입 안정성
- **Vite**: 빌드 도구
- **React Flow**: 노드 기반 UI 라이브러리

### 백엔드
- **Express.js**: API 서버
- **SQLite**: 샘플 데이터 저장
- **Python**: 데이터 분석 스크립트 실행

### 데이터 분석
- **Pyodide**: 브라우저에서 Python 실행
- **scikit-learn**: 머신러닝 알고리즘
- **pandas**: 데이터 처리
- **statsmodels**: 통계 모델
- **matplotlib**: 시각화

### AI 기능
- **Google Gemini API**: 파이프라인 자동 생성

### 배포
- **Vercel**: 웹 호스팅
- **Node.js**: 서버 환경

---

## 📝 사용 팁

### 1. 모듈 실행 순서
- 모듈은 의존성에 따라 자동으로 실행됩니다
- 모든 입력이 준비되면 자동으로 실행됩니다
- 수동 실행도 가능합니다 (각 모듈의 "Run" 버튼)

### 2. 데이터 흐름 확인
- 연결선의 색상으로 데이터 타입을 구분할 수 있습니다
- "View Details" 버튼으로 각 모듈의 출력 데이터를 확인하세요

### 3. 파라미터 튜닝
- 각 모듈의 Properties Panel에서 파라미터를 조정할 수 있습니다
- 변경 후 "Run" 버튼으로 다시 실행하세요

### 4. 에러 처리
- 모듈 실행 실패 시 빨간색으로 표시됩니다
- Terminal 탭에서 상세한 에러 메시지를 확인하세요
- Properties Panel에서 입력 데이터를 확인하세요

### 5. 성능 최적화
- 큰 데이터셋의 경우 샘플링을 고려하세요
- 불필요한 모듈은 제거하여 파이프라인을 단순화하세요

---

## 🎓 학습 리소스

### 내장 가이드
- **PYTHON_ANALYSIS_README.md**: Python 분석 모듈 사용법
- **PPT_생성_가이드.md**: PPT 자동 생성 기능 사용법
- **SAMPLES_README.md**: 샘플 관리 가이드

### 샘플 데이터
- **Examples_in_Load**: Load Data 모듈에서 사용할 수 있는 예제 데이터
- **Samples**: 미리 구성된 파이프라인 샘플

---

## 🔗 관련 링크

- **GitHub**: 프로젝트 저장소
- **배포 사이트**: https://www.insureautoflow.com/
- **문서**: 프로젝트 내 README 및 가이드 문서

---

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. **Terminal 탭**에서 에러 로그 확인
2. **브라우저 개발자 도구** (F12)에서 네트워크 및 콘솔 확인
3. **프로젝트 이슈 트래커**에 문제 보고

---

**ML Auto Flow**로 더 쉽고 빠르게 머신러닝 파이프라인을 구축하세요! 🚀
