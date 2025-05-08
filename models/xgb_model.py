
"""
XGBoost 모델 관련 함수들을 정의하는 모듈
"""

import numpy as np
import pandas as pd
import optuna
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from config.settings import RANDOM_STATE, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT
from utils.metrics import mean_absolute_percentage_error

def train_model(X_train, y_train, X_val, y_val, target_scaler):
    """
    Optuna를 이용한 XGBoost 모델 학습 (MAPE 최소화)
    
    Args:
        X_train: 학습용 특성 데이터
        y_train: 학습용 타겟 데이터
        X_val: 검증용 특성 데이터
        y_val: 검증용 타겟 데이터
        target_scaler: 타겟 스케일러
        
    Returns:
        학습된 모델
    """
    print(f"모델 학습 시작: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'objective': 'reg:squarederror'
        }
        base_model = XGBRegressor(**params)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        # 역스케일링 후 MAPE 계산
        y_val_true = target_scaler.inverse_transform(y_val)
        y_val_pred = target_scaler.inverse_transform(y_pred)
        mape = mean_absolute_percentage_error(y_val_true, y_val_pred)
        return mape
    
    # Optuna 스터디 생성 및 최적화
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
    
    print(f"최적 파라미터: {study.best_params}")
    print(f"최적 검증 MAPE: {study.best_value:.4f}%")
    
    # 최적 파라미터로 최종 모델 학습
    best_params = study.best_params
    best_params.update({'objective': 'reg:squarederror', 'random_state': RANDOM_STATE})
    base_model = XGBRegressor(**best_params)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    return model

def save_model(model, model_path):
    """
    모델 저장 함수
    
    Args:
        model: 저장할 모델
        model_path: 저장 경로
    """
    joblib.dump(model, model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")

def load_model(model_path):
    """
    모델 로드 함수
    
    Args:
        model_path: 모델 파일 경로
        
    Returns:
        로드된 모델
    """
    model = joblib.load(model_path)
    print(f"모델을 {model_path}에서 로드했습니다.")
    return model

def get_feature_importance(model, feature_columns):
    """
    모델의 특성 중요도 추출
    
    Args:
        model: 학습된 모델
        feature_columns: 특성 컬럼 리스트
        
    Returns:
        특성 중요도 데이터프레임
    """
    feature_importances = []
    for estimator in model.estimators_:
        feature_importances.append(estimator.feature_importances_)
    
    avg_importances = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': avg_importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df