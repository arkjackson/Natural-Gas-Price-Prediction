"""
모델 추론(예측) 관련 함수들을 정의하는 모듈
"""

import pandas as pd

from config.settings import TEST_DATA_PATH, MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH, SUBMISSION_DIR
from data.preprocess import load_and_preprocess_data, load_preprocessors
from models.xgb_model import load_model

def predict_prices(model, input_data, feature_scaler, target_scaler):
    """
    예측 수행 함수
    
    Args:
        model: 학습된 모델
        input_data: 입력 데이터
        feature_scaler: 특성 스케일러
        target_scaler: 타겟 스케일러
        
    Returns:
        예측 결과
    """
    print(f"입력 데이터 shape: {input_data.shape}로 예측")
    # 입력 데이터 스케일링
    input_scaled = feature_scaler.transform(input_data)
    
    # 예측
    predictions_scaled = model.predict(input_scaled)
    
    # 예측값 역스케일링
    predictions = target_scaler.inverse_transform(predictions_scaled)
    
    return predictions

def generate_predictions(test_file=TEST_DATA_PATH, model_path=MODEL_PATH, 
                        scaler_path=SCALER_PATH, feature_cols_path=FEATURE_COLS_PATH,
                        output_path=None):
    """
    테스트 데이터에 대한 예측 생성 및 저장
    
    Args:
        test_file: 테스트 데이터 파일 경로
        model_path: 모델 파일 경로
        scaler_path: 스케일러 파일 경로
        feature_cols_path: 특성 컬럼 파일 경로
        output_path: 예측 결과 저장 경로
        
    Returns:
        예측 결과 데이터프레임
    """
    # 전처리기 로드
    feature_scaler, target_scaler, feature_columns = load_preprocessors(scaler_path, feature_cols_path)
    
    # 모델 로드
    model = load_model(model_path)
    
    # 테스트 데이터 로드 및 전처리
    test_df, test_features, _, _ = load_and_preprocess_data(
        test_file, is_train=False, train_feature_columns=feature_columns
    )
    
    if test_features.empty:
        print("예측할 데이터가 없습니다.")
        return None
    
    # 예측 생성
    predictions = predict_prices(model, test_features, feature_scaler, target_scaler)
    
    # 예측 결과를 DataFrame에 저장
    pred_df = pd.DataFrame({
        'date': test_df['date'],
        'pred_1': predictions[:, 0],
        'pred_2': predictions[:, 1],
        'pred_3': predictions[:, 2]
    })
    
    # 결과 저장
    if output_path is None:
        output_path = f"{SUBMISSION_DIR}/xgboost_predictions.csv"
    
    pred_df.to_csv(output_path, index=False)
    print(f"\n예측 결과가 '{output_path}'에 저장되었습니다.")
    
    return pred_df

if __name__ == "__main__":
    generate_predictions()