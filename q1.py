import numpy as np
from sklearn.model_selection import train_test_split


# 랜덤한 [40,4] 크기의 dataset 생성
dataset = np.random.random([40,4]) 

# 1. dataset의 0번부터 2번 column 인덱스까지의 배열을 feature로 저장합니다.
feature = dataset[:, 0:3]
# 2. dataset의 마지막 column 인덱스의 column 벡터를 label로 저장합니다.
label = dataset[:, -1]

# 전체 데이터를 학습 데이터와 검증 데이터로 나눕니다.
# 3. 테스트용 데이터의 크기를 0.25로 하여 전체 데이터를 분리합니다.random_state는 121을 입력합니다.
X_train, X_test, Y_train, Y_test = train_test_split(feature,
                                                    label,
                                                    test_size=0.25,
                                                    random_state=121)

# 분리된 데이터의 크기들을 출력
print("Case 1.")
print("X_train shape : {}".format(X_train.shape))
print("X_test shape : {}".format(X_test.shape))
print("Y_train shape : {}".format(Y_train.shape))
print("Y_test shape : {}".format(Y_test.shape))

# 4. shuffle을 하지 않고 test_size를 0.3으로 하여 학습용 데이터와 테스트용 데이터를 분리합니다. random_state는 121을 입력합니다.
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(feature,
                                                            label,
                                                            test_size=0.3,
                                                            random_state=121,
                                                            shuffle=False)

# 분리된 데이터의 크기들을 출력
print("\nCase 2.")
print("X_train_2 shape : {}".format(X_train_2.shape))   
print("X_test_2 shape : {}".format(X_test_2.shape))
print("Y_train_2 shape : {}".format(Y_train_2.shape))
print("Y_test_2 shape : {}".format(Y_test_2.shape))
