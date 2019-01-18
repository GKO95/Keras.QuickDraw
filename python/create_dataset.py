import numpy as np
# Pickle 모듈은 파이썬 객체 구조를 일련화 및 비일련화 시킨다.
import pickle
import os

sample_size = input("샘플 데이터 개수를 입력하시오: ")

path = os.listdir(".\\numpy_dataset")
# x 변수는 한 데이터세트에서 일정 데이터를 추출하기 위한 임시 변수이다.
# x_load 변수는 x 변수가 가져온 모든 데이터세트의 일정 데이터들의 총집합이다.
x = []
x_load = []
# y 변수는 한 데이터세트에서 일정 데이터들에 대한 레이블을 일대일 제공한다.
# y_load 변수는 y 변수가 가져온 전치화된 모든 레이블들의 총집합이다.
y = []
y_load = []


def parse_numpy():
    """Quick, Draw!에서 제공한 NumPy 비트맵 전처리 데이터는 회색조(grayscale) 28x28 픽셀 크기의 좌표 형식으로 나타나 있으며,
    ndjson 전처리 데이터와 다르게 획에 대한 정보는 전혀 없는 것으로 보인다. 상단좌측에서부터 시작하여 하단우측으로
    픽셀 좌표마다 가지는 그림 데이터를 28*28=784개의 원소를 가지는 1차원 행렬로 표현하였다."""
    count = 0
    # NumPy 비트맵 데이터세트 파일을 상대주소에서 하나씩 순서대로 불러온다.
    for file in path:
        # 우선 입력 데이터로 사용할 NumPy 데이터세트 하나를 불러온다.
        file = ".\\numpy_dataset\\" + file
        # 불러온 데이터세트를 파이썬의 NumPy 행렬로 풀어쓴다.
        x = np.load(file)
        # 행렬값을 integer 에서 float 32비트로 변환시켜 정규화한다.
        x = x.astype('float32') / 255.
        # 데이터세트 내에서 학습에 사용한 데이터 개수룰 지정한다 (본 코드에서는 10000개 데이터만을 학습에 사용).
        x = x[0:int(sample_size), :]
        # 일정 개수의 데이터를 가지는 정규화된 데이터세트를 x_load 변수에 저장하며, 이후 다른 데이터세트도 해당 변수에 첨가된다.
        x_load.append(x)

        # 가져온 데이터세트의 데이터 각각에 레이블을 지정하는 행렬
        y = [count for _ in range(int(sample_size))]
        # 다음 데이터세트에 대한 레이블을 위해 미리 +1 하였음.
        count += 1
        # float 32비트로 레이블 값을 설정함,
        y = np.array(y).astype('float32')
        # 전치행렬을 취하였지만, 아직 무슨 용도로 전치행렬을 취하였는지는 확실하지 않다. 조사가 더 필요함.
        y = y.reshape(y.shape[0], 1)
        # 해당 데이터세트의 레이블을 x_load 변수에 맞추어서 순서대로 나열한다.
        y_load.append(y)

    return x_load, y_load


# x_load 및 y_load 변수의 이름을 feature(특성)과 label(레이블)로 지정한다.
features, labels = parse_numpy()
# features와 labels의 데이터 타입을 float 32비트로 맞춰줬지만, 이미 x_load에서 float만 수용하는데 굳이 여기에서 이렇게 한 이유가
# 뭔지 잘 모르겠다. 이에 대해서는 학습 성공 여부 이후에 수정 고려.
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

# 데이터세트 구별없이 각 데이터에 대한 그림 획 정보를 담는 2차원 행렬로 전환한다 (n개의 데이터세트가 있으면 n*10000개의 1차원 원소).
features = features.reshape(features.shape[0]*features.shape[1], features.shape[2])
labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2])

# open 파일명령어를 사용하여 파일(wb: 이진화 파일 작성)을 열거나 없으면 생성하고,
# 이를 통해 features 및 labels 내용을 string 파일화하여 pickle 모듈을 통해 다른 파이썬 스크립트로 옮길 수 있다.
"""본 스크립트에 큰 문제가 발생하였다: 너무 큰 용량의 데이터(4GB 이상)를 처리할 경우 메모리 에러가 발생한다!"""
with open(".\\pickle\\features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open(".\\pickle\\labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)
