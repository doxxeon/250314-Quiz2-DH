import numpy as np
import pandas as pd

DATA_PATH = "./data.csv"


def get_data() -> pd.DataFrame:
    "데이터를 불러오는 함수"

    df = pd.read_csv(DATA_PATH)
    return df


def add_type(df: pd.DataFrame) -> pd.DataFrame:
    "지시사항에 따라 df에 Type칼럼을 추가하고 반환합니다."

    df.loc[df['Age'] >= 19, 'Type'] = 'adult'
    df.loc[(df['Age'] < 19) & (df['Sex'] == 'female'), 'Type'] = 'girl'
    df.loc[(df['Age'] < 19) & (df['Sex'] == 'male'), 'Type'] = 'boy'

    return df


def main():
    # 데이터 불러오기
    df = get_data()
    print("추가 전\n", df.head())

    # 1. 새로운 특성 생성
    df_new = add_type(df.copy())
    print("추가 후\n", df_new.head())


if __name__ == "__main__":
    main()
