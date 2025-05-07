"""
환경 설정 및 글로벌 변수 정의 모듈
"""
import os

TRAIN_DATA_PATH = 'data/data_train.csv'
TEST_DATA_PATH = 'data/data_test.csv'
MODEL_DIR = "./models"
SUBMISSION_DIR = "./prediction_csv"

# 모델 저장 경로
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scalers.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

# 주요 특성 정의
KEY_FEATURES = ['BCOMCL_INDX', 'BCOMNG_INDX', 'DJI_INDX', 'CPI_Energy_Seasonally_Adjusted_USA', 'high_temperature', 'low_temperature', 'Natural_Gas_US_Henry_Hub_Gas']

# 타겟 변수
TARGET_COL = 'Natural_Gas_US_Henry_Hub_Gas'
TARGET_COLS = ['price_t+1', 'price_t+2', 'price_t+3']

# 특성 엔지니어링 설정
LAGS = [1, 3, 6, 12]
WINDOWS = [3, 6, 12]

# 모델 학습 설정
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 600  # 10분