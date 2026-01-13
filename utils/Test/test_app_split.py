"""
앱의 SplitData 로직을 재현하여 Python의 sklearn과 비교하는 스크립트
앱은 JavaScript의 자체 셔플 알고리즘을 사용하므로, 이를 Python으로 재현합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

def app_split_data(df, train_size=0.7, random_state=42, shuffle=True):
    """
    앱의 SplitData 로직을 정확히 재현합니다.
    앱은 JavaScript의 Linear Congruential Generator를 사용합니다.
    """
    # 앱에서는 inputRows가 배열이므로, 인덱스 기반으로 작업
    input_rows = df.values.tolist()
    
    # indices = Array.from({ length: inputRows.length }, (_, i) => i)
    indices = list(range(len(input_rows)))
    
    if shuffle:
        # 앱의 seededRandom 함수 재현
        # const seededRandom = () => {
        #     let seed = random_state;
        #     return () => {
        #         seed = (seed * 9301 + 49297) % 233280;
        #         return seed / 233280;
        #     };
        # };
        # const random = seededRandom();
        
        seed = int(random_state)  # random_state가 문자열일 수 있으므로 int 변환
        
        def random():
            nonlocal seed
            seed = (seed * 9301 + 49297) % 233280
            return seed / 233280
        
        # for (let i = indices.length - 1; i > 0; i--) {
        #     const j = Math.floor(random() * (i + 1));
        #     [indices[i], indices[j]] = [indices[j], indices[i]];
        # }
        for i in range(len(indices) - 1, 0, -1):
            j = int(random() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
    
    # const trainCount = Math.floor(inputRows.length * train_size);
    train_count = int(len(input_rows) * train_size)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    
    # const trainRows = trainIndices.map(i => inputRows[i]);
    # const testRows = testIndices.map(i => inputRows[i]);
    train_rows = [input_rows[i] for i in train_indices]
    test_rows = [input_rows[i] for i in test_indices]
    
    train_df = pd.DataFrame(train_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)
    
    return train_df, test_df

def sklearn_split_data(df, train_size=0.7, random_state=42, shuffle=True):
    """sklearn의 train_test_split을 사용합니다."""
    return train_test_split(df, train_size=train_size, random_state=random_state, shuffle=shuffle)

def run_comparison(csv_path, random_state=42):
    """앱의 분할 방식과 sklearn의 분할 방식을 비교합니다."""
    print("=" * 80)
    print(f"데이터 분할 방식 비교 (random_state={random_state})")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"\n데이터 Shape: {df.shape}")
    
    target_column = df.columns[-1]
    feature_columns = [col for col in df.columns if col != target_column]
    
    # 앱의 방식으로 분할
    print("\n" + "-" * 80)
    print("앱의 방식으로 데이터 분할")
    print("-" * 80)
    train_app, test_app = app_split_data(df, train_size=0.7, random_state=random_state, shuffle=True)
    print(f"훈련 데이터: {len(train_app)} 행")
    print(f"테스트 데이터: {len(test_app)} 행")
    
    # sklearn의 방식으로 분할
    print("\n" + "-" * 80)
    print("sklearn의 방식으로 데이터 분할")
    print("-" * 80)
    train_sklearn, test_sklearn = sklearn_split_data(df, train_size=0.7, random_state=random_state, shuffle=True)
    print(f"훈련 데이터: {len(train_sklearn)} 행")
    print(f"테스트 데이터: {len(test_sklearn)} 행")
    
    # 분할 결과 비교
    print("\n" + "-" * 80)
    print("분할 결과 비교")
    print("-" * 80)
    
    # 첫 몇 개 행의 인덱스 비교
    print("\n앱 방식 - 훈련 데이터 첫 5개 행의 원본 인덱스:")
    train_app_indices = train_app.index.tolist()[:5]
    print(train_app_indices)
    
    print("\nsklearn 방식 - 훈련 데이터 첫 5개 행의 원본 인덱스:")
    train_sklearn_indices = train_sklearn.index.tolist()[:5]
    print(train_sklearn_indices)
    
    # 모델 훈련 및 평가 (앱 방식)
    print("\n" + "=" * 80)
    print("앱 방식으로 분할한 데이터로 모델 훈련 및 평가")
    print("=" * 80)
    
    X_train_app = train_app[feature_columns]
    y_train_app = train_app[target_column]
    X_test_app = test_app[feature_columns]
    y_test_app = test_app[target_column]
    
    model_app = LinearRegression(fit_intercept=True)
    model_app.fit(X_train_app, y_train_app)
    
    y_pred_app = model_app.predict(X_test_app)
    r2_app = r2_score(y_test_app, y_pred_app)
    rmse_app = np.sqrt(mean_squared_error(y_test_app, y_pred_app))
    
    print(f"\n앱 방식 결과:")
    print(f"  테스트 R²: {r2_app:.6f}")
    print(f"  테스트 RMSE: {rmse_app:.6f}")
    print(f"  절편: {model_app.intercept_:.6f}")
    
    # 모델 훈련 및 평가 (sklearn 방식)
    print("\n" + "=" * 80)
    print("sklearn 방식으로 분할한 데이터로 모델 훈련 및 평가")
    print("=" * 80)
    
    X_train_sklearn = train_sklearn[feature_columns]
    y_train_sklearn = train_sklearn[target_column]
    X_test_sklearn = test_sklearn[feature_columns]
    y_test_sklearn = test_sklearn[target_column]
    
    model_sklearn = LinearRegression(fit_intercept=True)
    model_sklearn.fit(X_train_sklearn, y_train_sklearn)
    
    y_pred_sklearn = model_sklearn.predict(X_test_sklearn)
    r2_sklearn = r2_score(y_test_sklearn, y_pred_sklearn)
    rmse_sklearn = np.sqrt(mean_squared_error(y_test_sklearn, y_pred_sklearn))
    
    print(f"\nsklearn 방식 결과:")
    print(f"  테스트 R²: {r2_sklearn:.6f}")
    print(f"  테스트 RMSE: {rmse_sklearn:.6f}")
    print(f"  절편: {model_sklearn.intercept_:.6f}")
    
    # 결과 비교
    print("\n" + "=" * 80)
    print("결과 비교")
    print("=" * 80)
    print(f"\nR² 차이: {abs(r2_app - r2_sklearn):.6f}")
    print(f"RMSE 차이: {abs(rmse_app - rmse_sklearn):.6f}")
    
    if abs(r2_app - r2_sklearn) < 0.0001:
        print("\n[OK] 두 방식의 결과가 거의 일치합니다!")
    else:
        print("\n[WARNING] 두 방식의 결과가 다릅니다.")
        print("  원인: 앱과 sklearn이 서로 다른 랜덤 생성기를 사용합니다.")
        print("  - 앱: Linear Congruential Generator (LCG)")
        print("  - sklearn: Mersenne Twister")
    
    return {
        'app': {'r2': r2_app, 'rmse': rmse_app, 'intercept': model_app.intercept_},
        'sklearn': {'r2': r2_sklearn, 'rmse': rmse_sklearn, 'intercept': model_sklearn.intercept_}
    }

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
        exit(1)
    
    # random_state=42로 테스트
    print("\n" + "=" * 80)
    print("random_state=42 테스트")
    print("=" * 80)
    results_42 = run_comparison(csv_path, random_state=42)
    
    # random_state=43으로도 테스트
    print("\n\n" + "=" * 80)
    print("random_state=43 테스트")
    print("=" * 80)
    results_43 = run_comparison(csv_path, random_state=43)
    
    # 앱에서 보고한 값과 비교
    app_reported_r2 = 0.6857
    print("\n\n" + "=" * 80)
    print("앱에서 보고한 값과 비교")
    print("=" * 80)
    print(f"\n앱에서 보고한 테스트 R² (random_state=43): {app_reported_r2}")
    print(f"앱 방식 재현 결과 (random_state=43): {results_43['app']['r2']:.6f}")
    print(f"차이: {abs(results_43['app']['r2'] - app_reported_r2):.6f}")
    
    if abs(results_43['app']['r2'] - app_reported_r2) < 0.01:
        print("\n[OK] 앱 방식 재현 결과가 보고된 값과 일치합니다!")
    else:
        print("\n[WARNING] 차이가 있습니다. 추가 확인이 필요합니다.")

