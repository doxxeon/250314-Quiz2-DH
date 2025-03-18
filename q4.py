import pandas as pd
import numpy as np

DATA_PATH = "./taxi_fare_data.csv"

# 데이터를 DataFram의 형태로 불러옵니다.
df = pd.read_csv(DATA_PATH, quoting=3)

# 결측값 처리 함수입니다.
def del_missing(df):
    
    # df에서 Unnamed: 0 feature 데이터를 제거하고 del_un_df에 저장합니다.
    del_un_df = df.drop(['Unnamed: 0'], axis='columns')

    # del_un_df에서 id feature 데이터를 제거하고 del_un_id_df에 저장합니다.
    del_un_id_df = del_un_df.drop(['id'], axis='columns')
    
    # del_un_id_df의 누락된 데이터가 있는 행을 제거하고 removed_df에 저장합니다.
    removed_df = del_un_id_df.dropna()
    
    return removed_df

# 1.리스트를 입력으로 받아서 해당 리스트 내에 음수값이 있으면 그 위치(인덱스)들을 리스트로 출력하는 함수를 만듭니다.  
def get_negative_index(list_data):
    
    neg_idx = []
    
    for i, value in enumerate(list_data):
        # 음수값이 있으면 그 위치(인덱스)들을 neg_idx로 추가시킵니다.
        # value값이 음수일 때 해당하는 인덱스 i를 리스트 neg_idx에 append하세요.
        if value < 0:
            neg_idx.append(list_data.index[i])
        
    return neg_idx

# 2.DataFrame 내에 제거해야 하는 이상 값의 인덱스를 반환하는 함수를 만듭니다.
def outlier_index():
    
    # get_negative_index() 함수를 통해서,
    # fare_amount와 passenger_count 내의 음수값들의 인덱스를 반환합니다.
    idx_fare_amount = get_negative_index(fare_amount)
    idx_passenger_count = get_negative_index(passenger_count)
    
    idx_zero_distance = []    
    idx = [i for i in range(len(passenger_count))]
    zipped = zip(idx, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    
    # 결측치 처리를 수행하게 되면 Dataframe의 인덱스와 i 값은 다릅니다.
    # Dataframe.index[i]를 사용하여 Dataframe의 인덱스를 저장해봅시다.
    for i, x, y, _x, _y in zipped:
    
        # 타는 곳(pickup_longitude,pickup_latitude)과 내리는 곳(drop_longitude, drop_latitude)이
        # 같은 데이터의 인덱스를 idx_zero_distance에 저장합니다.
        # x와 _x가, y가 y_y와 같을 때 해당하는 인덱스 i를 idx_zero_distance에 append하세요.
        if (x == _x) and (y == _y):
            idx_zero_distance.append(passenger_count.index[i])
    
    # 제거해야하는 인덱스의 리스트들(idx_fare_amount,idx_passenger_count,idx_zero_distance)
    # 간의 중복을 없앤 리스트를 만들어줍니다.
    total_index4remove = list(set(idx_fare_amount+idx_passenger_count+idx_zero_distance))
    
    return total_index4remove

# 3.인덱스를 기반으로 DataFrame 내의 데이터를 제거하고, 제거된 DataFrame을 반환하는 함수를 만듭니다.
def remove_outlier(dataframe, list_idx):
    return dataframe.drop(list_idx)

# del_missing 함수로 결측치를 처리하여 df에 저장합니다.
df = del_missing(df)

# 불러온 DataFrame의 각 인덱스의 값들을 변수로 저장합니다.
fare_amount = df['fare_amount']
passenger_count = df['passenger_count']
pickup_longitude = df['pickup_longitude']
pickup_latitude = df['pickup_latitude']
dropoff_longitude = df['dropoff_longitude']
dropoff_latitude = df['dropoff_latitude']

# 이상치를 제거하기 전의 데이터 정보를 확인해 봅시다.
print('이상치를 제거하기 전의 데이터:')
df.info()

# 이상치를 제거합니다.
remove_index = outlier_index()
new = remove_outlier(df, remove_index)

# 이상치를 제거한 후의 데이터를 살펴봅니다.
print('\n이상치를 제거한 후의 데이터:')
new.info()
