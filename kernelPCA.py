```python
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.spatial.distance import cdist


class kPCA():

    # 훈련 데이터와 테스트 데이터를 입력
    def __init__(self, train_data, test_data=None, denoise=True):
        self.Xtrain = train_data  # 훈련 데이터 (index x column)
        self.Xtest = test_data  # 테스트 데이터 (index x column)
        self.l_train = np.size(train_data, 0)  # 훈련 데이터 개수
        self.l_test = np.size(test_data, 0)  # 테스트 데이터 개수
        self.N = np.size(train_data, 1)  # 데이터 변수 개수
        self.denoise = denoise

    # Denoising(reconstruction)된 테스트 데이터를 원래 공간으로 Mapping
    def obtain_preimages(self, n, c):
        # n = 사용하고자 하는 핵심 축의 개수
        # c = 가우시안 커널 함수의 하이퍼파라미터

        self.Ktrain = self.obtain_rbf_kernel_matrix(n, c)
        '''훈련 데이터에 대한 Kernel matrix 생성. Kernel PCA 진행에 필요'''

        self.alphas = self.obtain_alphas(self.Ktrain, n)
        '''훈련 데이터에 대한 kernel matrix의 고윳값 분해 실시. Kernel PCA 진행에 필요'''


        self.Ktest = self.obtain_test_rbf_kernel_matrix(n, c, self.Ktrain)
        '''테스트 데이터 Kernel matrix 생성. 테스트 데이터 denoising 진행에 필요'''
        self.betas = self.obtain_betas()
        ''' betas는 맵핑된 데이터를 Kernel PCA로 파악한 중요한 축으로 기저로 변환한 행렬 의미'''

        self.gammas = self.obtain_gammas()
        ''' 맵핑된 데이터를 원래 공간으로 맵핑하는 문제 해결을 위해 산출 '''
        ''' \Phi(z)와 P_n\Phi(x) 사이의 에러를 최적화할 때 사용'''

        self.Z = []  # denoising된 테스트 데이터를 원래 공간으로 맵핑 z_test

        for i in range(self.Xtest.shape[0]):
            z = self.obtain_preimage(i, n, c)  # 각 데이터별 원래 공간으로 맵핑된 z 산출
            self.Z.append(z)
        self.Z = np.array(self.Z)
        ''' 테스트 데이터 363개에 대한 denosing과 원래 공간으로 맵핑된 결과 도출 완료 '''
        return self.Z


    def obtain_preimage(self, j, n, c):
        if self.denoise:
            z_new = self.Xtest[j, :]
        else:
            z_new = np.zeros_like(self.Xtest[j, :])

        z = np.zeros(z_new.shape)
        n_iter = 0

        '''[그림 6] 아래 문제를 해결하기 위한 Gradient ascent '''
        while (np.linalg.norm(z - z_new) > 0.00001) and (n_iter < 1e3):
            z = z_new
            zcoeff = cdist([z], self.Xtrain, 'sqeuclidean')
            zcoeff = np.exp(-zcoeff / (n * c))
            zcoeff = self.gammas[j, :] * zcoeff
            s = np.sum(zcoeff)
            zcoeff = np.sum(self.Xtrain * zcoeff.T, axis=0)
            # Avoid underflow
            if s == 0:
                s += 1e-8
            z_new = zcoeff / s
            n_iter += 1
        if np.array_equal(z_new, self.Xtest[j, :]):
            import pdb

            pdb.set_trace()
        return z_new


    def obtain_betas(self):
        return np.dot(self.Ktest, self.alphas)  # 그림 6의 beta 구하는 데 사용


    def obtain_gammas(self):
        return np.dot(self.betas, self.alphas.T)  # 그림 6의 gamma 구하는 데 사용

    def obtain_rbf_kernel_matrix(self, n, c):
        ''' 훈련 데이터에 대한 Kernel matrix 생성 --> kernel PCA의 축은
        Kernel matrix에 대한 고유 벡터로 생성됨 '''


        sqdist_X = euclidean_distances(self.Xtrain, self.Xtrain, squared=True)
        K = np.exp(-sqdist_X / (n * c))
        return self.center_kernel_matrix(K, K)


    @staticmethod
    def center_kernel_matrix(K, K_train):
        # Mapping된 공간 내에서, 각 변수별 데이터 centering 진행
        one_l_prime = np.ones(K.shape[0:2]) / K.shape[1]
        one_l = np.ones(K_train.shape[0:2]) / K_train.shape[1]
        K = K \
            - np.dot(one_l_prime, K_train) \
            - np.dot(K, one_l) \
            + one_l_prime.dot(K_train).dot(one_l)
        return K


    def obtain_alphas(self, Ktrain, n):
        ''' Kernel PCA는 Kernel matrix에 대한 고유분해를 통해 실시됨 '''


        lambda_, alpha = eigh(Ktrain, eigvals=(Ktrain.shape[0] - n, Ktrain.shape[0] - 1))

        alpha_n = alpha / np.sqrt(lambda_)  # 고유벡터 normalization
        lambda_ = np.flipud(lambda_)  # 고유값 크기 순서대로 나열
        alpha_n = np.fliplr(alpha_n)  # 고유벡터, 고유값 순서대로 나열

        ''' alpha_n은 kernel PCA에 의해 파악된 축으로, 맵핑된 데이터에 대한 분산을 가장 
        잘 보존할 수 있는 축을 의미함.''''
        return alpha_n


    def obtain_test_rbf_kernel_matrix(self, n, c, Ktrain):
        # 테스트 데이터와 훈련 데이터 사이의 Kernel matrix 구성
        # 테스트 데이터에 대한 denoising 과정에서 필요
        sqdist_XtX = euclidean_distances(self.Xtest, self.Xtrain) ** 2
        Ktest = np.exp(-sqdist_XtX / (n * c))
        return self.center_kernel_matrix(Ktest, Ktrain)

