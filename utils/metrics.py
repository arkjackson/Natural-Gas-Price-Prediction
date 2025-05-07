import numpy as np

# MAPE(평균 절대 백분율 오차) 계산 함수
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 0으로 나누는 경우 방지
    absolute_percentage_error = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    mape = np.mean(absolute_percentage_error) * 100
    return mape
