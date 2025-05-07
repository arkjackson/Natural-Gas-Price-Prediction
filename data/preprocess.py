import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import SelectKBest, f_regression
from config.settings import KEY_FEATURES, TARGET_COL, LAGS, WINDOWS

# 특성 엔지니어링 함수
def engineer_features(df, target_col, key_features, lags=LAGS, windows=WINDOWS):
    df = df.copy()  # 데이터 프래그먼테이션 방지를 위한 명시적 복사
    
    # 1. 시차(lag) 및 롤링 통계치 생성
    for col in key_features:
        if col in df.columns:
            # 시차 생성
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            # 롤링 평균 및 표준편차
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
            # 차분
            df[f'{col}_diff_1'] = df[col].diff()
    
    # 2. 계절성 처리
    # 월별 더미 변수 생성
    df['month'] = df['date'].dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    df = df.drop('month', axis=1)
    
    # 계절 분해(타겟 변수)
    if target_col in df.columns:
        decomposition = seasonal_decompose(df[target_col], model='additive', period=12, extrapolate_trend='freq')
        df['seasonal'] = decomposition.seasonal
        df['trend'] = decomposition.trend
    
    # 3. 로그 변환 (pd.concat으로 최적화)
    log_cols = {}
    for col in df.columns:
        if col != 'date' and df[col].min() > 0 and df[col].dtype in ['float64', 'int64']:
            log_cols[f'{col}_log'] = np.log1p(df[col])
    if log_cols:
        log_df = pd.DataFrame(log_cols, index=df.index)
        df = pd.concat([df, log_df], axis=1)
    
    print(f"특성 엔지니어링 후 shape: {df.shape}, 컬럼: {df.columns.tolist()}")
    return df

# 데이터 로딩 및 전처리
def load_and_preprocess_data(file_path, is_train=True, train_feature_columns=None):
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    print(f"{file_path}에서 데이터 로드, shape: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    
    # 날짜 컬럼을 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 특성 엔지니어링 적용
    df = engineer_features(df, target_col=TARGET_COL, key_features=KEY_FEATURES)
    
    # 특성 엔지니어링 후 결측치 처리
    df = df.fillna(method='ffill').fillna(method='bfill')
    print(f"결측치 처리 후 shape: {df.shape}")
    
    if is_train:
        # 타겟 변수 생성: 1, 2, 3개월 후 가격
        df['price_t+1'] = df['Natural_Gas_US_Henry_Hub_Gas'].shift(-1)
        df['price_t+2'] = df['Natural_Gas_US_Henry_Hub_Gas'].shift(-2)
        df['price_t+3'] = df['Natural_Gas_US_Henry_Hub_Gas'].shift(-3)
        
        # 마지막 3개 행 제거(타겟 없음)
        df = df[:-3]
        
        print(f"마지막 3개 행 제거 후 shape: {df.shape}")
        
        # 특성과 타겟 분리
        features = df.drop(columns=['date', 'price_t+1', 'price_t+2', 'price_t+3'])
        
        # 상관관계 분석을 통한 특성 선택(학습 데이터만)
        selector = SelectKBest(score_func=f_regression, k=min(50, len(features.columns)))
        selector.fit(features, df['price_t+1'])  # price_t+1 기준 선택
        selected_features = features.columns[selector.get_support()].tolist()
        features = features[selected_features]
        
        targets = df[['price_t+1', 'price_t+2', 'price_t+3']]
        
        # 학습 특성 컬럼 저장
        train_feature_columns = features.columns.tolist()
        # print(f"특성 선택 후 학습 특성: {train_feature_columns}")
    else:
        # 테스트 데이터: 타겟 생성 없음
        # 학습 데이터와 컬럼 맞추기
        common_cols = [col for col in train_feature_columns if col in df.columns]
        if not common_cols:
            raise ValueError("학습 데이터와 테스트 데이터에 공통 컬럼이 없습니다.")
        
        missing_cols = [col for col in train_feature_columns if col not in df.columns]
        if missing_cols:
            print(f"경고: 테스트 데이터에 없는 컬럼: {missing_cols}")
            print(f"공통 컬럼 사용: {common_cols}")
        
        features = df[common_cols]
        targets = None
        print(f"테스트 특성: {features.columns.tolist()}")
    
    return df, features, targets, train_feature_columns if is_train else common_cols

# 데이터 스케일링
def scale_data(features, targets=None):
    if features.empty:
        raise ValueError("특성 DataFrame이 비어 있습니다.")
    
    print(f"스케일링할 특성 컬럼: {features.columns.tolist()}, shape: {features.shape}")
    feature_scaler = StandardScaler()
    feature_scaled = feature_scaler.fit_transform(features)
    
    if targets is not None:
        if targets.empty:
            raise ValueError("타겟 DataFrame이 비어 있습니다.")
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets)
        return feature_scaled, targets_scaled, feature_scaler, target_scaler
    return feature_scaled, None, feature_scaler, None

# 학습/검증 데이터 분할
def split_data(features, targets, val_size=0.2):
    n = len(features)
    if n == 0:
        raise ValueError("분할할 데이터가 없습니다.")
    train_size = int(n * (1 - val_size))
    if train_size == 0:
        raise ValueError("분할 후 학습 세트가 비어 있습니다.")
    
    print(f"데이터 분할: 특성 shape={features.shape}, 타겟 shape={targets.shape}")
    X_train = features[:train_size]
    X_val = features[train_size:]
    y_train = targets[:train_size]
    y_val = targets[train_size:]
    
    print(f"X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    print(f"X_val shape={X_val.shape}, y_val shape={y_val.shape}")
    
    return X_train, X_val, y_train, y_val
