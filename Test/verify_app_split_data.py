"""
SplitData 모듈 검증 스크립트

이 스크립트는 data_analysis_modules.py의 split_data 함수를 사용합니다.
현재는 sklearn의 train_test_split 함수를 사용합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def app_split_data(df: pd.DataFrame, train_size: float = 0.7, random_state: int = 42, shuffle: bool = True):
    """
    sklearn의 train_test_split을 사용하여 데이터를 분할합니다.
    data_analysis_modules.py의 split_data 함수와 동일한 로직입니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        분할할 데이터프레임
    train_size : float
        훈련 세트 비율 (0.0 ~ 1.0)
    random_state : int
        랜덤 시드
    shuffle : bool
        셔플 여부
    
    Returns:
    --------
    tuple
        (훈련 데이터프레임, 테스트 데이터프레임)
    """
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=None
    )
    return train_df, test_df


def verify_split_data(csv_path: str, train_size: float = 0.7, random_state: int = 42):
    """
    SplitData 로직을 검증하고 결과를 출력합니다.
    
    Parameters:
    -----------
    csv_path : str
        CSV 파일 경로
    train_size : float
        훈련 세트 비율
    random_state : int
        랜덤 시드
    """
    print("=" * 80)
    print("SplitData 모듈 검증 (sklearn train_test_split 사용)")
    print("=" * 80)
    print(f"\n파일: {csv_path}")
    print(f"훈련 세트 비율: {train_size}")
    print(f"랜덤 시드: {random_state}")
    print(f"셔플: True")
    
    # 데이터 로드
    print("\n" + "-" * 80)
    print("1. 데이터 로드")
    print("-" * 80)
    df = pd.read_csv(csv_path)
    print(f"데이터 Shape: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    
    # 결측치 확인
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n경고: 결측치 발견")
        print(missing_values[missing_values > 0])
        df = df.dropna()
        print(f"결측치 제거 후 Shape: {df.shape}")
    
    # 데이터 분할
    print("\n" + "-" * 80)
    print("2. 데이터 분할 (sklearn train_test_split 사용)")
    print("-" * 80)
    train_df, test_df = app_split_data(df, train_size=train_size, random_state=random_state, shuffle=True)
    
    print(f"훈련 데이터: {len(train_df)} 행 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"테스트 데이터: {len(test_df)} 행 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 분할 결과 검증
    print("\n" + "-" * 80)
    print("3. 분할 결과 검증")
    print("-" * 80)
    print(f"총 행 수: {len(train_df) + len(test_df)}")
    print(f"원본 행 수: {len(df)}")
    assert len(train_df) + len(test_df) == len(df), "분할 후 행 수가 일치하지 않습니다!"
    print("[OK] 행 수 검증 통과")
    
    # 샘플 데이터 확인
    print("\n훈련 데이터 샘플 (처음 5행):")
    print(train_df.head())
    print("\n테스트 데이터 샘플 (처음 5행):")
    print(test_df.head())
    
    # 통계 정보
    print("\n" + "-" * 80)
    print("4. 통계 정보")
    print("-" * 80)
    if len(df.columns) > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\n훈련 데이터 통계:")
            print(train_df[numeric_cols].describe())
            print("\n테스트 데이터 통계:")
            print(test_df[numeric_cols].describe())
    
    return train_df, test_df


def test_linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          feature_columns: list, target_column: str):
    """
    Linear Regression 모델을 훈련하고 평가합니다.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        훈련 데이터
    test_df : pd.DataFrame
        테스트 데이터
    feature_columns : list
        특성 컬럼 리스트
    target_column : str
        타겟 컬럼 이름
    """
    print("\n" + "=" * 80)
    print("5. Linear Regression 모델 훈련 및 평가")
    print("=" * 80)
    
    # 데이터 준비
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    
    print(f"\n특성 개수: {len(feature_columns)}")
    print(f"훈련 샘플 수: {len(X_train)}")
    print(f"테스트 샘플 수: {len(X_test)}")
    
    # 모델 훈련
    print("\n모델 훈련 중...")
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    print("[OK] 모델 훈련 완료")
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 평가 지표 계산
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # 결과 출력
    print("\n" + "-" * 80)
    print("모델 성능 지표")
    print("-" * 80)
    print(f"\n절편 (Intercept): {model.intercept_:.6f}")
    print(f"\n훈련 데이터:")
    print(f"  R²: {train_r2:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    print(f"\n테스트 데이터:")
    print(f"  R²: {test_r2:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    
    # 계수 출력 (처음 10개)
    print(f"\n계수 (Coefficients) - 처음 10개:")
    for i, feature in enumerate(feature_columns[:10]):
        print(f"  {feature}: {model.coef_[i]:.6f}")
    if len(feature_columns) > 10:
        print(f"  ... (총 {len(feature_columns)}개 특성)")
    
    return {
        'intercept': model.intercept_,
        'coefficients': dict(zip(feature_columns, model.coef_)),
        'train_metrics': {
            'r2': train_r2,
            'rmse': train_rmse,
            'mae': train_mae
        },
        'test_metrics': {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        }
    }


def main():
    """메인 실행 함수"""
    import os
    
    # CSV 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    # utils/Test/compactiv.csv도 확인
    if not os.path.exists(csv_path):
        alt_path = os.path.join(script_dir, '..', 'utils', 'Test', 'compactiv.csv')
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            print(f"오류: 파일을 찾을 수 없습니다.")
            print(f"  시도한 경로:")
            print(f"    1. {csv_path}")
            print(f"    2. {alt_path}")
            return
    
    # 파라미터 설정 (앱에서 사용하는 값으로 변경 가능)
    train_size = 0.7
    random_state = 42
    
    # 데이터 분할 검증
    train_df, test_df = verify_split_data(csv_path, train_size=train_size, random_state=random_state)
    
    # 타겟 변수와 특성 변수 설정
    target_column = train_df.columns[-1]  # 마지막 컬럼이 타겟
    feature_columns = [col for col in train_df.columns if col != target_column]
    
    # Linear Regression 테스트
    results = test_linear_regression(train_df, test_df, feature_columns, target_column)
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("검증 완료")
    print("=" * 80)
    print(f"\n이 스크립트는 data_analysis_modules.py의 split_data 함수를 사용합니다.")
    print(f"현재는 sklearn의 train_test_split 함수를 사용합니다.")
    print(f"\n주요 파라미터:")
    print(f"  train_size: {train_size}")
    print(f"  random_state: {random_state}")
    print(f"  shuffle: True")
    print(f"\n테스트 R²: {results['test_metrics']['r2']:.6f}")
    print(f"\n참고: sklearn의 train_test_split은 Mersenne Twister 랜덤 생성기를 사용하므로,")
    print(f"앱의 JavaScript LCG와는 다른 결과가 나올 수 있습니다.")


if __name__ == "__main__":
    main()

