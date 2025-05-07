DATA_PATH = 'data/train.csv'
TARGET_COL = 'Natural_Gas_US_Henry_Hub_Gas'
KEY_FEATURES = ['BCOMCL_INDX', 'BCOMNG_INDX', 'DJI_INDX', 'CPI_Energy_Seasonally_Adjusted_USA', 'high_temperature', 'low_temperature', 'Natural_Gas_US_Henry_Hub_Gas']
LAGS = [1, 3, 6, 12]
WINDOWS = [3, 6, 12]