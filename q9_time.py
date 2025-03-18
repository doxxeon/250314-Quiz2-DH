

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