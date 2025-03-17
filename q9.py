import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 폰트 적용
plt.rc("font", family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# 데이터 주소
DATA_PATH = "./taxi_fare_data.csv"

#데이터를 DataFram의 형태로 불러옵니다.
def load_csv(path):
    data_frame = pd.read_csv(path)
    return data_frame

# 결측치 처리 함수입니다.
def del_missing(df):
    
    # df에서 Unnamed: 0 feature 데이터를 제거하고 del_un_df에 저장합니다.
    del_un_df = df.drop(['Unnamed: 0'], axis='columns')
    
    # del_un_df에서 id feature 데이터를 제거하고 del_un_id_df에 저장합니다.
    del_un_id_df = del_un_df.drop(['id'], axis='columns')
    
    # del_un_id_df의 누락된 데이터가 있는 행을 제거하고 removed_df에 저장합니다.
    removed_df = del_un_id_df.dropna()
    
    return removed_df

# 리스트를 입력으로 받아서 해당 리스트 내에 음수값이 있으면 그 위치(인덱스)들을 리스트로 출력하는 함수를 만듭니다.
def get_negative_index(list_data):
    neg_idx = []
    
    for i, value in enumerate(list_data):
        if value < 0:
            neg_idx.append(list_data.index[i])
            
    return neg_idx

# DataFrame 내에 제거해야 하는 이상치의 인덱스를 반환하는 함수를 만듭니다.
def outlier_index():
    # get_negative_index() 함수를 통해서, fare_amount와 passenger_count 내의 음수값들의 인덱스를 반환합니다.
    idx_fare_amount = get_negative_index(fare_amount)
    idx_passenger_count = get_negative_index(passenger_count)
    
    idx_zero_distance = []    
    idx = [i for i in range(len(passenger_count))]
    zipped = zip(idx, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    
    for i, x, y, _x, _y in zipped:
        # 타는 곳(pickup_longitude,pickup_latitude)과 내리는 곳(drop_longitude, drop_latitude)이 같은 데이터의 인덱스를 idx_zero_distance에 저장합니다.
        if (x == _x) and (y == _y):
            idx_zero_distance.append(i)
            
    total_index4remove = list(set(idx_fare_amount+idx_passenger_count+idx_zero_distance))
    
    return total_index4remove

# 인덱스를 기반으로 DataFrame 내의 데이터를 제거하고, 제거된 DataFrame을 반환하는 함수를 만듭니다.
def remove_outlier(dataframe, list_idx):
    return dataframe.drop(list_idx)

# load_csv 함수를 사용하여 데이터를 불러와 df에 저장합니다.
df = load_csv(DATA_PATH)

# 1-1. del_missing 함수로 df의 결측치을 처리하여 df에 덮어씌웁니다.
df = del_missing(df)

# 불러온 DataFrame의 각 인덱스의 값들을 변수로 저장합니다.
fare_amount = df['fare_amount']
passenger_count = df['passenger_count']
pickup_longitude = df['pickup_longitude']
pickup_latitude = df['pickup_latitude']
dropoff_longitude = df['dropoff_longitude']
dropoff_latitude = df['dropoff_latitude']

# 1-2. remove_outlier()을 사용하여 이상치를 제거합니다.
# remove_outlier()가 어떤 인자들을 받는지 확인하세요.
remove_index = outlier_index()
df = remove_outlier(df, remove_index)

# 2. df.corr()을 사용하여 상관 계수 값 계산
corr_df = df.select_dtypes(include=[np.number]).corr()

# 택시 요금 데이터 regplot
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


###


WEEK_KOR = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


def load_csv(path: str) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 반환합니다."""
    df = pd.read_csv(path)
    return df


def cvt_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """df의 DateTime 칼럼을 datetime 형태로 변환합니다."""
    df["DateTime"] = pd.to_datetime(df['DateTime'])
    return df


def add_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """df에 DateTime 칼럼의 요일이 저장된 "요일" 칼럼을 새로 추가합니다."""
    df["요일"] = df['DateTime'].dt.weekday.map(WEEK_KOR)
    return df


def get_mean_consumption(df: pd.DataFrame) -> pd.Series:
    """df의 요일별 전력 소비량의 평균을 구하여 반환합니다."""
    series_mean = df.groupby("요일")["Consumption"].mean()

    return series_mean

def add_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    """DateTime에서 시간(hour)만 추출하여 새로운 칼럼 추가"""
    df["시간"] = df["DateTime"].dt.hour
    return df

def plot_mean_consumption_by_hour(df: pd.DataFrame):
    """시간대별 평균 전력 소비량을 시각화 (선 그래프)"""
    df = add_hour_column(df)  # 시간 칼럼 추가
    mean_consumption_by_hour = df.groupby("시간")["Consumption"].mean()  # 시간별 평균 전력 소비량

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=mean_consumption_by_hour.index, y=mean_consumption_by_hour.values, marker="o", color="blue")

    plt.xticks(range(0, 24, 1))  # X축 (0~23시)
    plt.xlabel("시간 (Hour)", fontproperties = font_prop)
    plt.ylabel("평균 전력 소비량 (kWh)", fontproperties = font_prop)
    plt.title("시간대별 평균 전력 소비량", fontproperties = font_prop)
    plt.grid(True)  # 격자 표시
    plt.savefig('시간대별 전력 소비량.png')

def main():
    # 데이터 경로
    data_path = "./electronic.csv"

    # 데이터 불러오기
    df = load_csv(data_path)

    # 1. DateTime 칼럼을 datetime 형태로 변환
    df = cvt_to_datetime(df)
    print(df)

    # 2. 요일 칼럼 추가
    df = add_dayofweek(df)
    print(df)

    # 3. 요일별 전력 소비량의 평균 구하기
    s_mean = get_mean_consumption(df)
    print(s_mean)

    plot_mean_consumption_by_hour(df)


if __name__ == "__main__":
    main()
