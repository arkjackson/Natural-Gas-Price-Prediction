"""
모델 학습 및 실행을 위한 메인 모듈
"""

from config.settings import TRAIN_DATA_PATH, TEST_DATA_PATH , MODEL_PATH, SCALER_PATH, VALIDATION_SIZE, FEATURE_COLS_PATH
from data.preprocess import load_and_preprocess_data, scale_data, split_data, save_preprocessors
from models.xgb_model import train_model, save_model, get_feature_importance
from utils.metrics import evaluate_predictions, print_metrics
from inference import generate_predictions

def main():
    """
    메인 실행 함수
    """
    print("=============================================")
    print("          XGBoost 모델 학습 시작             ")
    print("=============================================")
    
    # 1. 학습 데이터 로드 및 전처리
    print("\n[1. 학습 데이터 로드 및 전처리]")
    train_df, train_features, train_targets, train_feature_columns = load_and_preprocess_data(
        TRAIN_DATA_PATH, is_train=True
    )
    
    # 2. 학습 데이터 스케일링
    print("\n[2. 데이터 스케일링]")
    train_features_scaled, train_targets_scaled, feature_scaler, target_scaler = scale_data(
        train_features, train_targets
    )
    
    # 3. 학습/검증 데이터 분할
    print("\n[3. 학습/검증 데이터 분할]")
    X_train, X_val, y_train, y_val = split_data(
        train_features_scaled, train_targets_scaled, val_size=VALIDATION_SIZE
    )
    
    # 4. 모델 학습
    print("\n[4. 모델 학습]")
    model = train_model(X_train, y_train, X_val, y_val, target_scaler)
    
    # 5. 검증 데이터 성능 평가
    print("\n[5. 검증 데이터 성능 평가]")
    y_pred = model.predict(X_val)
    y_val_true = target_scaler.inverse_transform(y_val)
    y_val_pred = target_scaler.inverse_transform(y_pred)
    
    metrics = evaluate_predictions(y_val_true, y_val_pred)
    print_metrics(metrics, prefix="최종 검증")
    
    # 6. 특성 중요도 분석
    print("\n[6. 특성 중요도 분석]")
    importance_df = get_feature_importance(model, train_feature_columns)
    print("\n[상위 7개 특성 중요도]")
    print(importance_df.head(7).to_string(index=False))
    
    # 7. 모델 및 스케일러 저장
    print("\n[7. 모델 저장]")
    save_model(model, MODEL_PATH)
    save_preprocessors(feature_scaler, target_scaler, train_feature_columns, SCALER_PATH)
    
    # 8. 테스트 데이터 예측
    print("\n[8. 테스트 데이터 예측]")
    pred_df = generate_predictions(TEST_DATA_PATH, MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH)
    
    print("\n=============================================")
    print("          XGBoost 모델 학습 완료             ")
    print("=============================================")
    
    return model, feature_scaler, target_scaler, train_feature_columns, pred_df

if __name__ == "__main__":
    main()