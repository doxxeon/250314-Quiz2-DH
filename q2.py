import pandas as pd
import numpy as np

DATA_PATH = './taxi_fare_data.csv'

def load_csv(path):
    # pandas를 이용하여 'taxi_fair_data.csv' 데이터를 불러옵니다.
    data_frame = pd.read_csv('taxi_fare_data.csv')
    
    return data_frame

def statistical_features(data):
    # numpy를 이용하여 (리스트)데이터의 통계적 정보를 추출합니다.
    
    _min = np.min(data)
    _max = np.max(data)
    _mean = np.mean(data)
    _median = np.median(data)
    _var = np.var(data)
    _std = np.std(data)
    
    return _min, _max, _mean, _median, _var, _std

df = load_csv(DATA_PATH)
#전체 데이터에 대한 요약 정보를 살펴봅니다.
df.info()

#'fare_amount'변수에 대한 통계적 정보를 살펴봅니다.
f_min, f_max, f_mean, f_median, f_var, f_std = statistical_features(df['fare_amount'])
print('\nfare_amount의', '최솟값:', f_min ,'최댓값:', f_max ,'평균값:', f_mean ,'중앙값:', f_median ,'분산값:', f_var ,'표준편차값:', f_std)

#'passenger_count'변수에 대한 통계적 정보를 살펴봅니다.
p_min, p_max, p_mean, p_median, p_var, p_std = statistical_features(df['passenger_count'])
print('passenger_count의', '최솟값:', p_min ,'최댓값:', p_max ,'평균값:', p_mean ,'중앙값:', p_median ,'분산값:', p_var ,'표준편차값:', p_std)
