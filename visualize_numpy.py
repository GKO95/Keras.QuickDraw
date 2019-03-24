# ==================================================================================================================== #
# 본 스크립트는 머신 러닝에 개입하지 않으며, 오로지 NumPy 비트맵 파일을 시각화하기 위한 스크립트입니다.
# NumPy 비트맵 데이터세트를 직접적으로 보는 대신, 그 데이터세트를 따로 가공한 pickle 폴더의 features 파일을 시각화합니다.
# 스크립트를 실행하기 위해서는 다음과 같은 모듈 설치가 필요합니다:
# NumPy, MatPlotLib (SciPy 연관 모듈)
# ==================================================================================================================== #

from matplotlib import pyplot as plt
import numpy as np
import pickle

with open(".\\pickle\\features", "rb") as f:
    features = np.array(pickle.load(f))
with open(".\\pickle\\labels", "rb") as f:
    labels = np.array(pickle.load(f))

index = input("Index number: ")
inspection_data = features[int(index)].reshape(28, 28)

plt.imshow(inspection_data, interpolation='nearest')
print(labels[int(index)])
plt.show()
