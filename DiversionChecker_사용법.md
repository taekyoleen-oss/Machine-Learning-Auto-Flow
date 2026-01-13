# Diversion Checker 모듈 사용법

## 개요

Diversion Checker는 카운트 데이터(count data)의 과대산포(overdispersion)를 측정하고, 적합한 회귀 모델을 추천하는 모듈입니다.

## 연결 방법

### 입력 연결

- **입력 포트**: `data_in` (type: "data")
- **연결 가능한 모듈**:
  - `Load Data`: CSV 파일에서 데이터를 로드한 경우
  - `Select Data`: 데이터를 선택/필터링한 경우
  - `Statistics`: 통계 분석 후 데이터
  - 기타 데이터를 출력하는 모든 모듈

### 출력 연결

- **출력 포트**: `result_out` (type: "evaluation")
- **출력 내용**: 과대산포 검사 결과 및 모델 추천

## 사용 단계

### 1. 데이터 준비

1. `Load Data` 모듈을 사용하여 CSV 파일을 로드합니다.
2. (선택) `Select Data` 모듈로 필요한 컬럼만 선택합니다.
3. (선택) `Handle Missing Values` 모듈로 결측치를 처리합니다.

### 2. Diversion Checker 모듈 추가

1. 왼쪽 Toolbox에서 "Tradition Analysis" 카테고리를 확장합니다.
2. "Diversion Checker" 모듈을 더블클릭하거나 캔버스로 드래그합니다.

### 3. 연결 설정

1. 데이터 소스 모듈의 `data_out` 포트를 Diversion Checker의 `data_in` 포트에 연결합니다.
   ```
   [데이터 소스 모듈] --data_out--> data_in [Diversion Checker]
   ```

### 4. 파라미터 설정

Diversion Checker 모듈을 선택하고 오른쪽 Properties 패널에서 다음을 설정합니다:

- **Feature Columns** (특성 컬럼):
  - 독립 변수(설명 변수)로 사용할 컬럼들을 선택합니다.
  - 예: `["age", "gender", "exposure"]`
- **Label Column** (종속 변수):
  - 종속 변수(목표 변수)로 사용할 컬럼을 선택합니다.
  - 카운트 데이터여야 합니다 (0 이상의 정수).
  - 예: `"claim_count"`, `"accident_count"`
- **Max Iterations** (최대 반복 횟수):
  - 모델 피팅 시 최대 반복 횟수입니다.
  - 기본값: 100

### 5. 실행

1. Diversion Checker 모듈을 선택합니다.
2. 상단 툴바의 "Run" 버튼을 클릭하거나 모듈의 실행 버튼을 클릭합니다.

## 결과 해석

Diversion Checker는 다음 5단계 분석을 수행합니다:

### 1. 포아송 모델 적합

- 기본 포아송 회귀 모델을 피팅합니다.

### 2. Dispersion φ (phi) 계산

- 과대산포 정도를 나타내는 지표입니다.
- 계산 공식: `φ = Σ(pearson_residual²) / (n - p - 1)`
  - `n`: 샘플 수
  - `p`: 특성 컬럼 수

### 3. 모델 추천 (φ 기준)

- **φ < 1.2**: **Poisson 모델 추천**
  - 과대산포가 거의 없으므로 포아송 모델이 적합합니다.
- **1.2 ≤ φ < 2**: **Quasi-Poisson 모델 추천**
  - 약간의 과대산포가 있으므로 Quasi-Poisson 모델을 사용합니다.
- **φ ≥ 2**: **Negative Binomial 모델 추천**
  - 심각한 과대산포가 있으므로 음이항 모델을 사용합니다.

### 4. AIC 비교 (보조 기준)

- Poisson 모델과 Negative Binomial 모델의 AIC를 비교합니다.
- AIC가 낮을수록 더 나은 모델 적합도를 의미합니다.

### 5. Cameron–Trivedi Test (최종 확인)

- 과대산포를 통계적으로 검증하는 테스트입니다.
- 유의미한 결과가 나오면 과대산포가 존재함을 의미합니다.

## 출력 예시

```
=== 과대산포 검사 (Diversion Checker) ===

1. 포아송 모델 적합 중...
--- Poisson 회귀 모델 결과 ---
[모델 요약 통계]

2. Dispersion φ 계산 중...
Dispersion φ = 2.456789

3. 모델 추천:
φ ≥ 2 → Negative Binomial 모델 추천

4. 포아송 vs 음이항 AIC 비교 (보조 기준):
Poisson AIC: 1234.567
음이항 모델 적합 중...
Negative Binomial AIC: 1200.123
AIC 비교: Negative Binomial이 더 낮은 AIC를 가짐 (더 나은 적합도)

5. Cameron–Trivedi test (최종 확인):
[테스트 결과]
```

## 주의사항

1. **데이터 타입**: Label Column은 반드시 카운트 데이터(0 이상의 정수)여야 합니다.
2. **결측치 처리**: 결측치가 있는 경우 먼저 처리해야 합니다.
3. **특성 컬럼**: 숫자형 데이터여야 합니다. 범주형 데이터는 인코딩이 필요합니다.
4. **샘플 크기**: 충분한 샘플 수가 필요합니다 (최소 30개 이상 권장).

## 다음 단계

Diversion Checker의 추천을 바탕으로:

1. **Poisson 추천**: `Poisson Regression` 모듈 사용
2. **Quasi-Poisson 추천**: `Stat Models` 모듈에서 "QuasiPoisson" 선택
3. **Negative Binomial 추천**: `Negative Binomial Regression` 모듈 사용

## 관련 모듈

- `Poisson Regression`: 포아송 회귀 모델
- `Negative Binomial Regression`: 음이항 회귀 모델
- `Stat Models`: 통계 모델 (QuasiPoisson 포함)
- `Result Model`: 모델 결과 분석
