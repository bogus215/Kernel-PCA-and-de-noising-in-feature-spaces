from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from our_kpca import kPCA
import data_example2 as data
import numpy as np
import math

# 원본 데이터와 Reconstruction된 데이터 결과 시각화
def plot(methods, X, Y, line, rowspan):
    n_methods = len(methods)
    i = 0
    handles = []
    for denoised, name in methods:
        plt.subplot2grid((3, 2), (line, i), rowspan=rowspan)
        handle1, = plt.plot(X, Y, '.', color="0.8")
        plt.title(name)
        handle2, = plt.plot(denoised[:,0], denoised[:,1], 'k.')
        i += 1
        handles.append(handle1)
        handles.append(handle2)
    return handles

# PCA를 이용하여 reconstruction 실시
def pca_denoising(data):
    pca = PCA(n_components=1)
    low_dim_representation = pca.fit_transform(data)
    return pca.inverse_transform(low_dim_representation)

# 커브 모양 데이터 생성 함수
def get_curves(points=1000, radius=2, noise=None, *args, **kwargs):

    if noise is None:
        noise = 'uniform'
        kwargs['low'] = -0.5
        kwargs['high'] = 0.5
    kwargs['size'] = points // 2
    dist = getattr(np.random, noise)

    angles = np.linspace(0, math.pi/2, num=points//2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    left_center = -0.5
    left_radius = radius + dist(*args, **kwargs)
    left_x = -left_radius*cos + left_center
    left_y = left_radius*sin
    right_center = 0.5
    right_radius = radius + dist(*args, **kwargs)
    right_x = right_radius*cos[::-1] + right_center
    right_y = right_radius*sin[::-1]
    return np.concatenate((left_x, right_x)), np.concatenate((left_y, right_y))

# 커브 모양으로 생성된 2차원 데이터에 PCA, Kernel PCA 진행
# 데이터에 대한 PCA, kernel PCA로 reconstruction 진행
X, Y = data.get_curves(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.5), 'Kernel PCA'),
    (pca_denoising(noisy_data), 'Linear PCA')
]
plot(methods, X, Y, 0, 1)

# 정사각형 모양 데이터 생성
def get_square(points=1000, length=4, noise=None, *args, **kwargs):

    if noise is None:
        noise = 'uniform'
        kwargs['low'] = -0.5
        kwargs['high'] = 0.5
    kwargs['size'] = points // 4
    dist = getattr(np.random, noise)

    real_values = np.linspace(0, length, num=points//4)
    x_values = []
    y_values = []
    # Left side
    x_values.append(dist(*args, **kwargs))
    y_values.append(real_values)
    # Right side
    x_values.append(dist(*args, **kwargs) + length)
    y_values.append(real_values)
    # Top side
    x_values.append(real_values)
    y_values.append(dist(*args, **kwargs) + length)
    # Bottom side
    x_values.append(real_values)
    y_values.append(dist(*args, **kwargs))

    return np.concatenate(x_values), np.concatenate(y_values)

# 정사각형 모양으로 생성된 2차원 데이터에 PCA, Kernel PCA 진행
# 데이터에 대한 PCA, kernel PCA로 reconstruction 진행
X, Y = data.get_square(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.6), 'Kernel PCA'),
    (pca_denoising(noisy_data), 'Linear PCA')
]
handles = plot(methods, X, Y, 1, 2)
plt.show()
