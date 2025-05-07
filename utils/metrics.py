"""
평가 지표 계산 관련 함수들을 정의하는 모듈
"""

import numpy as np
from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """
    MAPE(평균 절대 백분율 오차) 계산 함수
    
    Args:
        y_true: 실제 값 배열
        y_pred: 예측 값 배열
        epsilon: 0으로 나누는 것을 방지하기 위한 작은 값
        
    Returns:
        평균 절대 백분율 오차 (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 0으로 나누는 경우 방지
    absolute_percentage_error = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    mape = np.mean(absolute_percentage_error) * 100
    return mape

def evaluate_predictions(y_true, y_pred):
    """
    여러 평가 지표로 예측 성능을 평가하는 함수
    
    Args:
        y_true: 실제 값 배열
        y_pred: 예측 값 배열
        
    Returns:
        지표 딕셔너리 (mape, mse)
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        'mape': mape,
        'mse': mse
    }
    
    return metrics

def print_metrics(metrics, prefix=""):
    """
    평가 지표 출력 함수
    
    Args:
        metrics: 지표 딕셔너리
        prefix: 출력 시 접두사 (예: '검증' 또는 '테스트')
    """
    print(f"{prefix} MAPE: {metrics['mape']:.4f}%")
    print(f"{prefix} MSE: {metrics['mse']:.4f}")