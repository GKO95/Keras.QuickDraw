""" Keras는 개발자가 딥러닝 모델을 쉽게 사용할 수 있도록 하는 라이브러리이다.
Keras 모듈 안에는 여러 "backend engine"이 있으며, 개발자는 이를 활용하여 딥러닝 개발에 쉽게 접근할 수 있다.
Keras의 backend engine 중 하나가 바로 TensorFlow이다; 그러므로 "Using TensorFlow backend"라는 문구가 나타나도 겁먹지 말 것!"""

from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, BatchNormalization
from keras.utils import np_utils, print_summary
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential

import numpy as np
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


# 모델 형성에 있어 필요한 KERAS 신경망 모델을 제공한다: image_x 및 image_y는 입력되는 그림 데이터의 x축 및 y축의 픽셀 크기이다.
# https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
def keras_layer(image_x, image_y):
    # 데이터세트 개수를 OS 모듈을 통해 자동적으로 카운트합니다.
    num_of_classes = len([file for file in os.listdir(".\\numpy_dataset")])
    # 모델을 형성하는데 레이어가 선형적으로 쌓는 방식을 채택합니다.
    model = Sequential()

    """Conv2D 레이어는 비교대상의 이미지를 특정 부분만 확인하는 필터(i.e. 수직선, 수평선, 왼쪽으로 휘는 수직곡선, 오른쪽으로 휘는
     수직곡선 등)와 컨볼루션 연산, 혹은 element-wise(즉, MATLAB 연산자 .*)하여 어느 부분이 필터링한 특징이 가장 뚜렷하게 나타나는지
     확인한다. 필터링한 부분의 값들의 총합이 매우 크면, 해당 부분은 필터하고자 하는 특징이 매우 뚜렷하다는 것을 의미한다. 하나의 
     필터는 하나의 이미지 특징만 확인할 수 있으므로, 필터의 개수가 많으면 많을수록 이미지의 특징을 더 많이 확인할 수 있다. 그리고 각 
     필터와 컨볼루션하여 나온 출력값을 activation map이라고 부르므로, activation map의 개수는 필터의 개수랑 동일하다."""
    # 2D 컨볼루션 레이어는 이미지 처리에 사용되는 레이어이다. 신경망 중에서 첫 레이어는 입력 데이터 형태를 반드시 지정해야 한다.
    # 본 Conv2D 레이어 하이퍼파라미터는 stride = 1, padding = 'valid' (출력크기 != 입력크기)이다; 이는 MaxPooling2D와 별개이다.
    model.add(Conv2D(  # 출력 공간의 깊이(depth; 형식은 "너비 x 높이 x 깊이"), 필터의 개수(activation map 개수)를 설정합니다.
                       32,
                       # 필터의 크기를 결정합니다.
                       (5, 5),
                       # 컨볼루션 레이어가 첫 레이어로 설정되어 있을 시, 입력 데이터 형태를 지정합니다("1"은 가변변수 개수 의미).
                       input_shape=(image_x, image_y, 1),
                       # 작동 방식 설정; ReLu는 rectified linear unit 약자로 큰 네트워크에서 가장 빠른 학습 속도를 보여준다.
                       activation='relu'))

    # MaxPooling2D로 불필요한 정보를 걸러내고 중요하다고 여기는 정보는 추출해 낸다 (MaxPooling2D의 경우 Pooling 내에서의 최댓값).
    """MaxPooling2D 레이어는 선택적인 레이어로 ReLu의 출력값을 좀 더 간략화시키는 다운샘플링을 한다. Pooling 레이어 중에서도
    MaxPooling은 Pooling이라는 그룹화된 부분에서 최댓값을 대표값으로 선정하고 나머지는 값들은 과감하게 버린다. 통상적으로 Pooling
    그룹이 서로 겹치지 않게 Pooling 사이즈와 stride 크기를 동일하게 한다. 이렇게 하여도 되는 이유는 특정 이미지 특징이 어디에서
    나타났는지 (activation 값을 통해서) 확인하면 오히려 이미지 특징 간의 위치적 상관관계가 절대위치보다 더 중요시 되기 때문이다.
    
    Pooling은 크게 두 가지의 목적을 가진다: (1) 가중치를 75%를 줄여 (총 4개 중에서 가장 큰 1개만 사용) 수치 계산에서 loss를 줄이고 
    (2) Overfitting를 방지한다."""
    model.add(MaxPooling2D(  #
                             pool_size=(2, 2),
                             # 필터가 다음 이미지 부분과 컨볼루션하기 위해 이동하는 양.
                             strides=(2, 2),
                             # 출력 크기가 입력과 동일하도록 레이어에 통과시키기 전에 패딩을 덧붙여준다.
                             padding='same'))

    """첫 번째 컨볼루션 레이어를 통과하여 각각 이미지 특징만의 activation map을 추출하였으면, 두 번째 컨볼루션은 한 단계 더 높은
    수준의 컨볼루션이 진행된다: 첫 번째 activation map에서도 반원(직선과 곡선의 조합)이나 사각형(직선과 또다른 직선의 조합)와 같은
    특징이 나타날 수 있으며, 이러한 특징을 확인하여 출력하는게 바로 두 번째 컨볼루션 레이어가 하는 역할이며, 다음 컨볼루션을 통과하면
    더욱 복잡한 이미지 특징을 확인할 수 있게 되어, 결과적으로 특정 물체의 형상으로 수렴하게 된다."""
    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Flatten 레이어는 2차원의 행렬 데이터를 1차원으로 변환시키며, 학습에는 아무런 영향을 주지 않습니다.
    model.add(Flatten())

    """Dense 레이어 (aka. Fully-connected(FC) 레이어)는 클래스 이전에 있던 모든 레이어들을 전부 연결한다. 맨 앞에 있는 숫자는 
    출력 뉴련의 개수를 의미한다. 그리고 Dense 레이어가 입력 데이터가 각 클래스마다 대응하는 평점을 정한다: Y=W*X+B, 여기서 W는
    가중치, B는 바이어스이다.
    
    output=activation(dot())"""
    model.add(Dense(512, activation='relu'))

    """Dropout 레이어는 overfitting을 방지하기 위한 데이터 학습 시에만 사용되는 레이어이다 (시험 및 추론에는 사용 X); 현재 사용 X.
    overfitting을 방지하는 방법으로는 무작위로 activation 세트를 0으로 만들어 무의미화시켜 조금 둔하게 만든다. 이렇게 만들어도
    다른 activation set들이 존재하여 총체적으로 네트워크를 구성하여 추론하는데에는 아무런 문제가 없다."""

    # 그러나 본 스크립트에서는 Dropout 레이어를 대신하여 유사한 기능을 가진 BatchNormalization 레이어를 사용한다.
    # model.add(Dropout())

    """BatchNormalization 레이어는 확률통계와 밀접한 관계를 가지는 레이어이다. 본 레이어의 이전 레이어인 Dense 레이어에서의
    activation을 Y=[Var]*X+[Mean]의 역함수를 통해 X로 정규화시킨다. 이들의 정규화는 전처리 데이터의 정규화와 동일한 목적으로
    빠른 계산을 위한 것이다.
    
    또한 Dropout 레이어와 유사하게 임의로 activation set를 0으로 만들어 overfitting을 방지하는 역할도 겸한다."""
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())
    # 출력 종류를 데이터 추론을 위해 클래스 개수만큼 설정하였으며, softmax 방식은 추론 확률을 PMF 형식으로 나타내 최종 추론을 낸다.
    model.add(Dense(num_of_classes, activation='softmax'))
    # 최종적으로 쌓은 레이어들을 손실과 최적화 등을 설정을 마무리로 모델 학습을 위한 신경망 구축을 마무리한다.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ModelCheckpoint 명령은 매 epoch(iteration on a dataset) 마다 학습 중인 모델의 체크포인트를 저장합니다.
    checkpoint = ModelCheckpoint(  # 체크포인트 저장 경로 및 이름을 지정합니다; .h5 확장자는 막대한 정보의 데이터 저장에 사용된다.
                                   ".\\model_dir\\KERAS_model.{epoch:02d}-{val_loss:.2f}.h5",
                                   # 모니터링할 측량을 선택합니다.
                                   monitor='val_acc',
                                   # 간략설명(0) 혹은 자세설명(1) 모드 중 하나를 선택합니다.
                                   verbose=1,
                                   # 모니터링된 측량에 대하여 가장 최근의 최고 모델이 덮어쓰여지지 않게 할건지 설정합니다.
                                   save_best_only=True,
                                   # save_best_only = True 로 설정 시, 모델 덮어쓰기 기준을 자동, 최고치, 최저치 중에서 선택.
                                   mode='max')
    #
    callbacks_list = [checkpoint]
    return model, callbacks_list


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


# 각 레이블마다 할당된 값들을 총 클래스 개수를 행(row)으로 가지는 행렬의 행벡터(row vector)로 변환시키는 기능이다.
# 그리고 행벡터는 하나의 값 1의 원소를 가지며, 해당 원소는 레이블 값의 열에 위치한다.
# 그러므로 이제 labels 변수는 2D 행렬이다.
def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels


# 효율적인 학습을 위해 features 및 labels 내용을 일대일 대응을 유지한채 무작위로 섞는 기능이다.
# from sklearn.utils import shuffle 을 통해 쉽게 셔플이 가능하지만, 추가 모듈 설치 없이 진행을 위한 순수 NumPy 대체 코드이다.
def shuffle_numpy(features, labels):
    shuffle = np.arange(labels.size)
    np.random.shuffle(shuffle)
    features = features[shuffle]
    labels = labels[shuffle]

    # 만일 sklearn 모듈을 사용할 시 아래의 코드를 통해서도 가능하다.
    # features, labels = shuffle(features, labels)
    return features, labels


# NumPy 행렬을 일정 크기로 나누는 기능이다; 한 행렬에서 학습과 시험 데이터, 혹은 심지어 검증 데이터로 나눌 때 사용한다.
# from sklearn.model_selection import train_test_split 활용하는 모듈 설치 없이 진행을 위한 순수 NumPy 대체 코드이다.
def split_numpy(features, labels, test_percentage):
    s = int(test_percentage * labels.size)
    # 굳이 시험 데이터를 뒤에 있는 데이터로 활용할 필요는 없다. 앞에 있는 데이터든 뒤에 있는 데이터든 다 사용 가능한 데이터이다.
    x_features, y_features = features[:s, :], features[s:, :]
    x_labels, y_labels = labels[:s, :], labels[s:, :]

    # 만일 sklearn 모듈을 사용할 시 아래의 코드를 통해서도 가능하다; test_percentage = 0.1으로 설정되어 있을 경우.
    # train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0, test_size=0.1)
    return x_features, x_labels, y_features, y_labels


# 본격적으로 features 및 labels 내용을 가지고 학습을 진행하는 함수이다.
def main(features, labels):
    # 효율적인 학습을 위해 features 및 labels 내용을 일대일 대응을 유지한채로 무작위로 섞는다.
    features, labels = shuffle_numpy(features, labels)
    # features 및 labels 데이터 내용을 학습 데이터와 시험 데이터로 나눈다; 뒤에 있는 숫자는 시험 데이터로 사용할 양이다.
    test_features, test_labels, train_features, train_labels = split_numpy(features, labels, 0.3)
    # 레이블을 2D 행벡터로 변환시킨다.
    train_labels = prepress_labels(train_labels)
    test_labels = prepress_labels(test_labels)

    # 학습 및 시험 데이터의 행렬 형태를 모델 입력 데이터 형식에 맞게 784x1 행렬을 28x28 행렬로 변환시킨다.
    train_features = train_features.reshape(train_features.shape[0], 28, 28, 1)
    test_features = test_features.reshape(test_features.shape[0], 28, 28, 1)

    # 학습에 사용될 KERAS 레이어 신경망 모델을 불러온다.
    model, callbacks_list = keras_layer(28, 28)
    # 불러온 KERAS 신경망 모델에 대한 요약 정보를 콘솔창에 제공한다.
    print_summary(model)
    # 1GB 미만의 NumPy 형식 데이터를 학습 밒 평가할 경우 tf.keras.Model.fit 으로도 충분하다.
    # 학습 횟수를 3번 그리고 batch 크기를 64로 설정되어 있지만, 이는 모델 학습 효율성을 위해 상황에 따라 변경하여도 된다.
    model.fit(train_features, train_labels, validation_data=(test_features, test_labels), epochs=3, batch_size=128,
              callbacks=callbacks_list)
    # 학습이 끝난 모델을 다음과 같은 경로 및 이름으로 저장한다; 위에서 설정한 경로와 파일 이름은 체크포인트였다.
    model.save('.\\model_dir\\QD_model.h5')


# x_load 및 y_load 변수의 이름을 feature(특성)과 label(레이블)로 지정한다.
features, labels = parse_numpy()
# features와 labels의 데이터 타입을 float 32비트로 맞춰줬지만, 이미 x_load에서 float만 수용하는데 굳이 여기에서 이렇게 한 이유가
# 뭔지 잘 모르겠다. 이에 대해서는 학습 성공 여부 이후에 수정 고려.
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

# 데이터세트 구별없이 각 데이터에 대한 그림 획 정보를 담는 2차원 행렬로 전환한다 (n개의 데이터세트가 있으면 n*10000개의 1차원 원소).
features = features.reshape(features.shape[0]*features.shape[1], features.shape[2])
labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2])

if __name__ == '__main__':
    # 모델 학습을 실행한다.
    main(features, labels)
