"""
앱의 SplitData 로직을 정확히 재현하여 random_state=42일 때 0.6857이 나오는지 확인
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def split_data(df: pd.DataFrame, train_size: float = 0.7, random_state: int = 42,
               shuffle: bool = True):
    """
    앱의 JavaScript SplitData 로직을 Python으로 완전히 재현합니다.
    """
    input_rows = df.values.tolist()
    indices = list(range(len(input_rows)))
    
    if shuffle:
        seed = int(random_state)
        
        def random():
            nonlocal seed
            seed = (seed * 9301 + 49297) % 233280
            return seed / 233280
        
        for i in range(len(indices) - 1, 0, -1):
            j = int(random() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
    
    train_count = int(len(input_rows) * float(train_size))
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    
    train_rows = [input_rows[i] for i in train_indices]
    test_rows = [input_rows[i] for i in test_indices]
    
    train_df = pd.DataFrame(train_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)
    
    return train_df, test_df

def main():
    csv_path = r'utils/Test/compactiv.csv'
    df = pd.read_csv(csv_path)
    
    target_column = df.columns[-1]
    feature_columns = [col for col in df.columns if col != target_column]
    
    print("=" * 80)
    print("앱의 SplitData 로직 재현 테스트 (data_analysis_modules 사용)")
    print("=" * 80)
    
    # random_state=42 테스트
    print("\n" + "-" * 80)
    print("random_state=42 테스트")
    print("-" * 80)
    
    train_42, test_42 = split_data(df, train_size=0.7, random_state=42, shuffle=True)
    
    X_train_42 = train_42[feature_columns]
    y_train_42 = train_42[target_column]
    X_test_42 = test_42[feature_columns]
    y_test_42 = test_42[target_column]
    
    model_42 = LinearRegression(fit_intercept=True)
    model_42.fit(X_train_42, y_train_42)
    
    y_pred_42 = model_42.predict(X_test_42)
    r2_42 = r2_score(y_test_42, y_pred_42)
    rmse_42 = np.sqrt(mean_squared_error(y_test_42, y_pred_42))
    
    print(f"\n결과:")
    print(f"  테스트 R²: {r2_42:.6f}")
    print(f"  테스트 RMSE: {rmse_42:.6f}")
    print(f"  절편: {model_42.intercept_:.6f}")
    
    # 앱에서 보고한 값과 비교
    app_r2_42 = 0.6857
    print(f"\n앱에서 보고한 테스트 R² (random_state=42): {app_r2_42}")
    print(f"Python 재현 결과 (random_state=42): {r2_42:.6f}")
    print(f"차이: {abs(r2_42 - app_r2_42):.6f}")
    
    if abs(r2_42 - app_r2_42) < 0.01:
        print("\n[OK] 결과가 거의 일치합니다!")
    else:
        print("\n[WARNING] 차이가 있습니다.")
        print("  가능한 원인:")
        print("  1. 앱에서 다른 데이터 전처리 수행")
        print("  2. 수치 연산 정밀도 차이")
        print("  3. 다른 파라미터 설정 (train_size 등)")

if __name__ == "__main__":
    main()

