import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from math import radians, cos, sin, asin, sqrt

# 한글 폰트 적용 (Mac 환경 기준, 필요 시 수정)
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rc("font", family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# 데이터 경로
DATA_PATH = "./taxi_fare_data.csv"

# CSV 파일 로드 함수
def load_csv(path):
    return pd.read_csv(path)

# 결측치 처리 함수
def del_missing(df):
    df = df.drop(['Unnamed: 0', 'id'], axis='columns', errors='ignore')  # 불필요한 컬럼 제거
    df = df.dropna()  # 결측치 제거
    df = df.reset_index(drop=True)
    return df

# 하버사인 공식을 적용하여 두 GPS 좌표 간 거리(km) 계산 함수
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # 지구 반지름 (km)

    # 위도, 경도를 라디안 단위로 변환
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # 위도 및 경도의 차이 계산
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # 하버사인 공식 적용
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c  # 최종 거리 (km)

# 음수값이 있는 인덱스 찾기
def get_negative_index(series):
    return series[series < 0].index.tolist()

# 이상치 인덱스 찾기
def outlier_index(df):
    idx_fare_amount = get_negative_index(df['fare_amount'])
    idx_passenger_count = get_negative_index(df['passenger_count'])
    
    idx_short_distance = []
    
    for i in range(len(df)):
        distance = haversine(df.loc[i, 'pickup_longitude'], df.loc[i, 'pickup_latitude'], 
                             df.loc[i, 'dropoff_longitude'], df.loc[i, 'dropoff_latitude'])
        if distance < 0.05:  # 50m 이하의 이동 거리는 비정상적이라고 판단
            idx_short_distance.append(i)

    total_index4remove = list(set(idx_fare_amount + idx_passenger_count + idx_short_distance))
    
    return total_index4remove

# 이상치 제거 함수
def remove_outlier(df, list_idx):
    return df.drop(list_idx).reset_index(drop=True)

# 데이터 로드 및 정제 과정
df = load_csv(DATA_PATH)
df = del_missing(df)

# 이상치 제거
remove_index = outlier_index(df)
df = remove_outlier(df, remove_index)

# 상관 계수 계산
corr_df = df.select_dtypes(include=[np.number]).corr()

# 택시 요금 데이터 regplot 시각화
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')

plt.figure(figsize=(10,10))
sns.regplot(x=df["fare_amount"], y=df["passenger_count"], scatter_kws={'alpha':0.3})
sns.stripplot(x=df["fare_amount"], y=df["passenger_count"], color='black', alpha=0.3)
plt.xlabel("Fare Amount ($)")
plt.ylabel("Passenger Count")
plt.title("Fare Amount vs Passenger Count (Regression)")
plt.xticks(np.arange(0, 110, 10).astype(int))

plt.savefig("Regplot.png")

print("✅ 데이터 정제 완료 및 시각화 저장 완료!")