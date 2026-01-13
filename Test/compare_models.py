"""
compactiv.csv 파일을 사용하여 Linear Regression, Lasso, Ridge, ElasticNet 모델을 실행하고
앱의 결과와 비교하는 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

def load_data(file_path: str):
    """CSV 파일에서 데이터를 로드합니다."""
    print(f"데이터 로드 중: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료. Shape: {df.shape}")
        print("\n데이터 미리보기:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def split_data(df: pd.DataFrame, train_size: float = 0.7, random_state: int = 42):
    """데이터를 훈련/테스트 세트로 분할합니다."""
    train_df, test_df = train_test_split(df, train_size=train_size, random_state=random_state, shuffle=True)
    return train_df, test_df

def create_linear_model(model_type: str = 'LinearRegression', fit_intercept: bool = True,
                        alpha: float = 1.0, l1_ratio: float = 0.5):
    """선형 회귀 모델을 생성합니다."""
    print(f"{model_type} 모델 생성 중...")
    
    if model_type == 'LinearRegression':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_type == 'Lasso':
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif model_type == 'Ridge':
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=42)
    else:
        print(f"알 수 없는 모델 타입: {model_type}. LinearRegression을 사용합니다.")
        model = LinearRegression(fit_intercept=fit_intercept)
    
    print("모델 생성 완료.")
    return model

def train_model(model, df: pd.DataFrame, feature_columns: list, label_column: str):
    """모델을 훈련시킵니다."""
    print("모델 훈련 시작...")
    
    X = df[feature_columns]
    y = df[label_column]
    
    print(f"훈련 데이터: {X.shape[0]} 샘플, {X.shape[1]} 특성")
    
    model.fit(X, y)
    
    print("모델 훈련 완료.")
    return model

def score_model(model, df: pd.DataFrame, feature_columns: list):
    """모델로 예측을 수행합니다."""
    print("모델 예측 수행 중...")
    
    features_to_use = [col for col in feature_columns if col in df.columns]
    X_score = df[features_to_use]
    
    predictions = model.predict(X_score)
    
    scored_df = df.copy()
    scored_df['Predict'] = predictions
    
    print("예측 완료.")
    return scored_df

def evaluate_model(model, df: pd.DataFrame, label_column: str, prediction_column: str = 'Predict',
                   model_type: str = 'regression'):
    """모델의 성능을 평가합니다."""
    print("모델 평가 중...")
    
    y_true = df[label_column]
    y_pred = df[prediction_column]
    
    metrics = {}
    
    if model_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        metrics['r2'] = r2
        metrics['mae'] = mae
        
        print(f"평균 제곱 오차 (MSE): {mse:.4f}")
        print(f"평균 제곱근 오차 (RMSE): {rmse:.4f}")
        print(f"결정 계수 (R²): {r2:.4f}")
        print(f"평균 절대 오차 (MAE): {mae:.4f}")
    
    return metrics

def format_coefficients(model, feature_names):
    """모델의 계수를 딕셔너리 형태로 변환"""
    coefficients = {}
    if hasattr(model, 'coef_'):
        for i, feature in enumerate(feature_names):
            coefficients[feature] = float(model.coef_[i])
    return coefficients

def run_model_comparison(csv_path: str, test_size: float = 0.3, random_state: int = 42):
    """
    CSV 파일을 사용하여 4가지 모델을 실행하고 결과를 비교합니다.
    
    Parameters:
    -----------
    csv_path : str
        CSV 파일 경로
    test_size : float
        테스트 데이터 비율
    random_state : int
        랜덤 시드
    """
    print("=" * 80)
    print("머신러닝 모델 비교 분석")
    print("=" * 80)
    print(f"\n데이터 파일: {csv_path}")
    print(f"테스트 데이터 비율: {test_size}")
    print(f"랜덤 시드: {random_state}\n")
    
    # 1. 데이터 로드
    print("\n" + "=" * 80)
    print("1. 데이터 로드")
    print("=" * 80)
    df = load_data(csv_path)
    
    if df is None:
        print("데이터 로드 실패!")
        return
    
    print(f"\n데이터 Shape: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    
    # 타겟 변수는 마지막 컬럼 (y)
    target_column = df.columns[-1]
    feature_columns = [col for col in df.columns if col != target_column]
    
    print(f"\n타겟 변수: {target_column}")
    print(f"특성 변수 개수: {len(feature_columns)}")
    print(f"특성 변수: {feature_columns[:5]}..." if len(feature_columns) > 5 else f"특성 변수: {feature_columns}")
    
    # 결측치 확인
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n경고: 결측치 발견")
        print(missing_values[missing_values > 0])
        # 결측치 제거
        df = df.dropna()
        print(f"결측치 제거 후 Shape: {df.shape}")
    
    # 2. 데이터 분할
    print("\n" + "=" * 80)
    print("2. 데이터 분할")
    print("=" * 80)
    train_df, test_df = split_data(df, train_size=1-test_size, random_state=random_state)
    
    print(f"훈련 데이터: {train_df.shape[0]} 샘플")
    print(f"테스트 데이터: {test_df.shape[0]} 샘플")
    
    # 3. 모델 실행 및 결과 저장
    models_to_test = [
        ('LinearRegression', {'model_type': 'LinearRegression', 'fit_intercept': True}),
        ('Lasso', {'model_type': 'Lasso', 'fit_intercept': True, 'alpha': 1.0}),
        ('Ridge', {'model_type': 'Ridge', 'fit_intercept': True, 'alpha': 1.0}),
        ('ElasticNet', {'model_type': 'ElasticNet', 'fit_intercept': True, 'alpha': 1.0, 'l1_ratio': 0.5}),
    ]
    
    results = {}
    
    print("\n" + "=" * 80)
    print("3. 모델 훈련 및 평가")
    print("=" * 80)
    
    for model_name, model_params in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"모델: {model_name}")
        print(f"{'=' * 80}")
        
        # 모델 생성
        model = create_linear_model(**model_params)
        
        # 모델 훈련
        trained_model = train_model(model, train_df, feature_columns, target_column)
        
        # 예측
        train_scored = score_model(trained_model, train_df, feature_columns)
        test_scored = score_model(trained_model, test_df, feature_columns)
        
        # 평가
        train_metrics = evaluate_model(trained_model, train_scored, target_column, 
                                       prediction_column='Predict', model_type='regression')
        test_metrics = evaluate_model(trained_model, test_scored, target_column, 
                                     prediction_column='Predict', model_type='regression')
        
        # 계수 및 절편 추출
        intercept = float(trained_model.intercept_) if hasattr(trained_model, 'intercept_') else 0.0
        coefficients = format_coefficients(trained_model, feature_columns)
        
        # 결과 저장
        results[model_name] = {
            'intercept': intercept,
            'coefficients': coefficients,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'num_features': len(feature_columns),
            'train_samples': len(train_df),
            'test_samples': len(test_df)
        }
        
        # 결과 출력
        print(f"\n{model_name} 결과 요약:")
        print(f"  절편 (Intercept): {intercept:.6f}")
        print(f"  계수 개수: {len(coefficients)}")
        print(f"  훈련 데이터 R²: {train_metrics.get('r2', 0):.6f}")
        print(f"  테스트 데이터 R²: {test_metrics.get('r2', 0):.6f}")
        print(f"  훈련 데이터 RMSE: {train_metrics.get('rmse', 0):.6f}")
        print(f"  테스트 데이터 RMSE: {test_metrics.get('rmse', 0):.6f}")
        
        # 계수 출력 (처음 5개만)
        print(f"\n  주요 계수 (처음 5개):")
        for i, (feature, coef) in enumerate(list(coefficients.items())[:5]):
            print(f"    {feature}: {coef:.6f}")
    
    # 4. 결과 비교 요약
    print("\n" + "=" * 80)
    print("4. 모델 비교 요약")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train_R2': [results[m]['train_metrics'].get('r2', 0) for m in results.keys()],
        'Test_R2': [results[m]['test_metrics'].get('r2', 0) for m in results.keys()],
        'Train_RMSE': [results[m]['train_metrics'].get('rmse', 0) for m in results.keys()],
        'Test_RMSE': [results[m]['test_metrics'].get('rmse', 0) for m in results.keys()],
        'Train_MSE': [results[m]['train_metrics'].get('mse', 0) for m in results.keys()],
        'Test_MSE': [results[m]['test_metrics'].get('mse', 0) for m in results.keys()],
        'Intercept': [results[m]['intercept'] for m in results.keys()],
    })
    
    print("\n성능 지표 비교:")
    print(comparison_df.to_string(index=False))
    
    # 최고 성능 모델 찾기
    best_test_r2 = comparison_df.loc[comparison_df['Test_R2'].idxmax()]
    best_test_rmse = comparison_df.loc[comparison_df['Test_RMSE'].idxmin()]
    
    print(f"\n최고 테스트 R²: {best_test_r2['Model']} (R² = {best_test_r2['Test_R2']:.6f})")
    print(f"최저 테스트 RMSE: {best_test_rmse['Model']} (RMSE = {best_test_rmse['Test_RMSE']:.6f})")
    
    # 5. 결과를 JSON 파일로 저장
    output_file = os.path.join(os.path.dirname(csv_path), 'model_comparison_results.json')
    
    # JSON 직렬화 가능한 형태로 변환
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'intercept': result['intercept'],
            'coefficients': result['coefficients'],
            'train_metrics': result['train_metrics'],
            'test_metrics': result['test_metrics'],
            'num_features': result['num_features'],
            'train_samples': result['train_samples'],
            'test_samples': result['test_samples']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 저장되었습니다: {output_file}")
    
    # 6. 앱과의 비교를 위한 정보 출력
    print("\n" + "=" * 80)
    print("5. 앱과의 비교를 위한 참고 정보")
    print("=" * 80)
    print("\n앱에서 사용하는 파라미터:")
    print("  - fit_intercept: True")
    print("  - Lasso alpha: 1.0")
    print("  - Ridge alpha: 1.0")
    print("  - ElasticNet alpha: 1.0, l1_ratio: 0.5")
    print("\n앱의 결과 형식:")
    print("  - intercept: 숫자")
    print("  - coefficients: {feature_name: coefficient_value}")
    print("  - metrics: {r2, mse, rmse 등}")
    
    return results, comparison_df

if __name__ == "__main__":
    # CSV 파일 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    
    # 모델 비교 실행 (random_state=43로 테스트)
    print("\n" + "=" * 80)
    print("random_state=43로 Linear Regression 실행")
    print("=" * 80)
    results_43, comparison_df_43 = run_model_comparison(csv_path, test_size=0.3, random_state=43)
    
    # random_state=42와 비교
    print("\n" + "=" * 80)
    print("random_state=42로 Linear Regression 실행 (비교용)")
    print("=" * 80)
    results_42, comparison_df_42 = run_model_comparison(csv_path, test_size=0.3, random_state=42)
    
    # 결과 비교
    print("\n" + "=" * 80)
    print("random_state 비교 (Linear Regression)")
    print("=" * 80)
    lr_43 = results_43['LinearRegression']
    lr_42 = results_42['LinearRegression']
    
    print(f"\nrandom_state=43:")
    print(f"  테스트 R²: {lr_43['test_metrics']['r2']:.6f}")
    print(f"  테스트 RMSE: {lr_43['test_metrics']['rmse']:.6f}")
    print(f"  절편: {lr_43['intercept']:.6f}")
    
    print(f"\nrandom_state=42:")
    print(f"  테스트 R²: {lr_42['test_metrics']['r2']:.6f}")
    print(f"  테스트 RMSE: {lr_42['test_metrics']['rmse']:.6f}")
    print(f"  절편: {lr_42['intercept']:.6f}")
    
    print(f"\n차이:")
    print(f"  R² 차이: {lr_43['test_metrics']['r2'] - lr_42['test_metrics']['r2']:.6f}")
    print(f"  RMSE 차이: {lr_43['test_metrics']['rmse'] - lr_42['test_metrics']['rmse']:.6f}")
    
    # 앱에서 보고한 값과 비교
    app_r2 = 0.6857
    print(f"\n앱에서 보고한 테스트 R²: {app_r2}")
    print(f"Python 스크립트 결과 (random_state=43): {lr_43['test_metrics']['r2']:.6f}")
    print(f"차이: {abs(lr_43['test_metrics']['r2'] - app_r2):.6f}")
    
    if abs(lr_43['test_metrics']['r2'] - app_r2) < 0.01:
        print("\n[OK] 결과가 거의 일치합니다!")
    else:
        print("\n[WARNING] 결과에 차이가 있습니다. 가능한 원인:")
        print("  1. 데이터 분할 방식의 차이 (train_size, shuffle 등)")
        print("  2. 데이터 전처리 차이 (정규화, 결측치 처리 등)")
        print("  3. 수치 연산 정밀도 차이 (JavaScript vs Python)")
        print("  4. 앱에서 다른 데이터셋을 사용했을 가능성")
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

