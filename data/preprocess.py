
"""
데이터 전처리 및 특성 엔지니어링 관련 함수들을 정의하는 모듈
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
from config.settings import KEY_FEATURES, TARGET_COL, LAGS, WINDOWS, TARGET_COLS

def engineer_features(df, target_col=TARGET_COL, key_features=KEY_FEATURES, 
                      lags=LAGS, windows=WINDOWS):
    """
    특성 엔지니어링 함수
    
    Args:
        df: 입력 데이터프레임
        target_col: 타겟 컬럼 이름
        key_features: 주요 특성 리스트
        lags: 시차(lag) 기간 리스트
        windows: 롤링 윈도우 크기 리스트
        
    Returns:
        특성이 추가된 데이터프레임
    """
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
    
    print(f"특성 엔지니어링 후 shape: {df.shape}, 컬럼 수: {len(df.columns)}")
    return df

def load_and_preprocess_data(file_path, is_train=True, train_feature_columns=None):
    """
    데이터 로딩 및 전처리
    
    Args:
        file_path: 데이터 파일 경로
        is_train: 학습 데이터 여부
        train_feature_columns: 학습 시 선택된 특성 컬럼 (테스트 데이터 처리 시 사용)
        
    Returns:
        전처리된 데이터프레임, 특성, 타겟(학습 시), 선택된 특성 컬럼
    """
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    print(f"{file_path}에서 데이터 로드, shape: {df.shape}")
    
    # 날짜 컬럼을 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 특성 엔지니어링 적용
    df = engineer_features(df)
    
    # 특성 엔지니어링 후 결측치 처리
    df = df.fillna(method='ffill').fillna(method='bfill')
    print(f"결측치 처리 후 shape: {df.shape}")
    
    if is_train:
        # 타겟 변수 생성: 1, 2, 3개월 후 가격
        df['price_t+1'] = df[TARGET_COL].shift(-1)
        df['price_t+2'] = df[TARGET_COL].shift(-2)
        df['price_t+3'] = df[TARGET_COL].shift(-3)
        
        # 마지막 3개 행 제거(타겟 없음)
        df = df[:-3]
        
        print(f"마지막 3개 행 제거 후 shape: {df.shape}")
        
        # 특성과 타겟 분리
        features = df.drop(columns=['date'] + TARGET_COLS)
        
        # 상관관계 분석을 통한 특성 선택(학습 데이터만)
        selector = SelectKBest(score_func=f_regression, k=min(50, len(features.columns)))
        selector.fit(features, df['price_t+1'])  # price_t+1 기준 선택
        selected_features = features.columns[selector.get_support()].tolist()
        features = features[selected_features]
        
        targets = df[TARGET_COLS]
        
        # 학습 특성 컬럼 저장
        train_feature_columns = features.columns.tolist()
        print(f"특성 선택 후 학습 특성 수: {len(train_feature_columns)}")
    else:
        # 테스트 데이터: 타겟 생성 없음
        # 학습 데이터와 컬럼 맞추기
        common_cols = [col for col in train_feature_columns if col in df.columns]
        if not common_cols:
            raise ValueError("학습 데이터와 테스트 데이터에 공통 컬럼이 없습니다.")
        
        missing_cols = [col for col in train_feature_columns if col not in df.columns]
        if missing_cols:
            print(f"경고: 테스트 데이터에 없는 컬럼: {len(missing_cols)}개")
        
        features = df[common_cols]
        targets = None
        print(f"테스트 특성 수: {len(common_cols)}")
    
    return df, features, targets, train_feature_columns if is_train else common_cols


def scale_data(features, targets=None):
    """
    데이터 스케일링
    
    Args:
        features: 특성 데이터프레임
        targets: 타겟 데이터프레임 (옵션)
        
    Returns:
        스케일링된 특성, 스케일링된 타겟(옵션), 특성 스케일러, 타겟 스케일러
    """
    if features.empty:
        raise ValueError("특성 DataFrame이 비어 있습니다.")
    
    print(f"스케일링할 특성 shape: {features.shape}")
    feature_scaler = StandardScaler()
    feature_scaled = feature_scaler.fit_transform(features)
    
    if targets is not None:
        if targets.empty:
            raise ValueError("타겟 DataFrame이 비어 있습니다.")
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets)
        return feature_scaled, targets_scaled, feature_scaler, target_scaler
    return feature_scaled, None, feature_scaler, None

def split_data(features, targets, val_size=0.2):
    """
    학습/검증 데이터 분할
    
    Args:
        features: 특성 데이터
        targets: 타겟 데이터
        val_size: 검증 데이터 비율
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    n = len(features)
    if n == 0:
        raise ValueError("분할할 데이터가 없습니다.")
    train_size = int(n * (1 - val_size))
    if train_size == 0:
        raise ValueError("분할 후 학습 세트가 비어 있습니다.")
    
    X_train = features[:train_size]
    X_val = features[train_size:]
    y_train = targets[:train_size]
    y_val = targets[train_size:]
    
    print(f"X_train shape={X_train.shape}, X_val shape={X_val.shape}")
    
    return X_train, X_val, y_train, y_val

def save_preprocessors(feature_scaler, target_scaler, feature_columns, save_path):
    """
    전처리기(스케일러) 및 특성 컬럼 저장
    
    Args:
        feature_scaler: 특성 스케일러
        target_scaler: 타겟 스케일러
        feature_columns: 선택된 특성 컬럼
        save_path: 저장 경로
    """
    joblib.dump({
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }, save_path)
    print(f"스케일러가 {save_path}에 저장되었습니다.")
    
    # 특성 컬럼 저장
    joblib.dump(feature_columns, save_path.replace('scalers', 'feature_columns'))
    print(f"특성 컬럼이 {save_path.replace('scalers', 'feature_columns')}에 저장되었습니다.")

def load_preprocessors(scaler_path, feature_cols_path):
    """
    전처리기(스케일러) 및 특성 컬럼 로드
    
    Args:
        scaler_path: 스케일러 파일 경로
        feature_cols_path: 특성 컬럼 파일 경로
        
    Returns:
        특성 스케일러, 타겟 스케일러, 특성 컬럼
    """
    scalers = joblib.load(scaler_path)
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    
    feature_columns = joblib.load(feature_cols_path)
    
    return feature_scaler, target_scaler, feature_columns