""" Keras는 개발자가 딥러닝 모델을 쉽게 사용할 수 있도록 하는 라이브러리이다.
Keras 모듈 안에는 여러 "backend engine"이 있으며, 개발자는 이를 활용하여 딥러닝 개발에 쉽게 접근할 수 있다.
Keras의 backend engine 중 하나가 바로 TensorFlow이다; 그러므로 "Using TensorFlow backend"라는 문구가 나타나도 겁먹지 말 것!"""

from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, BatchNormalization
from keras.utils import np_utils, print_summary
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential

import numpy as np
import pickle
import os


# 모델 형성에 있어 필요한 KERAS 신경망 모델을 제공한다: image_x 및 image_y는 입력되는 그림 데이터의 x축 및 y축의 픽셀 크기이다.
# https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
def keras_layer(image_x, image_y):
    # 데이터세트 개수를 OS 모듈을 통해 자동적으로 카운트합니다.
    num_of_classes = len([file for file in os.listdir(".\\numpy_dataset")])
    # 모델을 형성하는데 레이어가 선형적으로 쌓는 방식을 채택합니다.
    model = Sequential()
    # 2D 컨볼루션 레이어는 이미지 처리에 사용되는 레이어이다. 신경망 중에서 첫 레이어는 입력 데이터 형태를 반드시 지정해야 한다.
    model.add(Conv2D(  # 출력 공간의 차원, 즉 컨볼루션 레이어 통과 시 필터되는 출력 종류 개수를 설정합니다.
                       32,
                       #
                       (5, 5),
                       # 컨볼루션 레이어가 첫 레이어로 설정되어 있을 시, 입력 데이터 형태를 지정합니다("1"은 가변변수 개수 의미).
                       input_shape=(image_x, image_y, 1),
                       # 작동 방식 설정; ReLu는 rectified linear unit 약자로 큰 네트워크에서 가장 빠른 학습 속도를 보여준다.
                       activation='relu'))
    # MaxPooling2D
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # Flatten 레이어
    model.add(Flatten())
    # Dense 레이어
    model.add(Dense(512, activation='relu'))
    # Dropout 레이어
    #model.add(Dropout(0.6))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))

    #model.add(Dropout(0.6))
    model.add(BatchNormalization())
    # 출력 종류를 데이터 추론을 위해 클래스 개수만큼 설정하였으며, softmax 방식은 추론 확률을 PMF 형식으로 나타내 최종 추론을 낸다.
    model.add(Dense(num_of_classes, activation='softmax'))
    # 최종적으로 쌓은 레이어들을 손실과 최적화 등을 설정을 마무리로 모델 학습을 위한 신경망 구축을 마무리한다.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ModelCheckpoint 명령은 매 epoch(iteration on a dataset) 마다 학습 중인 모델의 체크포인트를 저장합니다.
    checkpoint = ModelCheckpoint(  # 체크포인트 저장 경로 및 이름을 지정합니다; .h5 확장자는 막대한 정보의 데이터 저장에 사용된다.
                                   ".\\model_dir\\{epoch:02d}-{val_loss:.2f}.h5",
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


# open 파일명령어를 사용하여 파일(wb: 이진화 파일 읽기)을 열어,
# 이를 통해 다른 스크립트에서 저장한 features 및 labels 내용을 본 스크립트에 불러오는 기능이다.
def load_from_pickle():
    with open(".\\pickle\\features", "rb") as f:
        features = np.array(pickle.load(f))
    with open(".\\pickle\\labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels


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
def main():
    # load_from_pickle 함수로 features 및 labels 내용을 본 스크립트에 할당한다.
    features, labels = load_from_pickle()
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
              callbacks=[TensorBoard(log_dir=".\\model_dir")])
    # 학습이 끝난 모델을 다음과 같은 경로 및 이름으로 저장한다; 위에서 설정한 경로와 파일 이름은 체크포인트였다.
    model.save('.\\model_dir\\QD_model.h5')


# 모델 학습을 실행한다.
main()
