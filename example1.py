from our_kpca import kPCA
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import pandas as pd

N = 10  # 변수 개수가 10개인 데이터 생성
M = 11  # 비선형 데이터를 생성하기 위해, 11개의 다변량 정규분포를 이용
S = [100, 33]  # 각 다변량 정규 분포별 훈련 데이터는 100개, 테스트 데이터는 33개를 생성하여, 총 훈련 데이터 1100개와 테스트 데이터 363개 생성
Sigma = [0.05, 0.1, 0.2, 0.4, 0.8]  # 다변량 분포의 표준 편차 설정
ratio_mse = pd.DataFrame(index= Sigma, columns = range(1, N)) # Output 정리



count = 1  #
for sigma in Sigma:
    c = 2*sigma**2  # 본 논문에서 제안한 커널 함수의 하이퍼파라미터 설정

    # train data 생성
    centers = np.random.uniform(low=-1.0, high=1.0, size=(M, N))  # M x N
    train_data = np.random.multivariate_normal(mean=centers[0],
                                               cov=sigma ** 2 * np.eye(N),
                                               size=S[0])

    for i in range(1, M):
        train_data = np.concatenate((train_data,
                                     np.random.multivariate_normal(
                                         mean=centers[i],
                                         cov=sigma ** 2 * np.eye(N),
                                         size=S[0])), axis=0)

    # test data 생성
    test_data = np.random.multivariate_normal(mean=centers[0],
                                              cov=sigma ** 2 * np.eye(N),
                                              size=S[1])
    for i in range(1, M):
        test_data = np.concatenate((test_data,
                                    np.random.multivariate_normal(
                                        mean=centers[i],
                                        cov=sigma ** 2 * np.eye(N),
                                        size=S[1])), axis=0)

    # 데이터의 각 변수별 평균을 0으로 변환하는 centering 실시
    mu = np.mean(train_data, 0)
    train_data -= mu
    test_data -= mu
    centers -= mu

    # Kernel PCA 실시
    kpca = kPCA(train_data, test_data)

		# 핵심 축을 1개부터 전체 축 개수 - 1 까지 모두 try
    for n in range(1, N):
        print("====================")
        print("sigma = ", sigma, "n =", n)
        print("====================")
        kpca.obtain_preimages(n, c)
        Z = kpca.Z

				# 기존 PCA 방법도 실시
        pca = PCA(n_components=n)
        pca.fit(train_data)
        test_transL = pca.transform(test_data)
        ZL = pca.inverse_transform(test_transL)

        # kPCA, PCA가 본래 데이터에 대한 분산을 얼마나 보존하고 있는지 비교하기 위한 실험
        mse_kpca = mse_pca = 0
        for i in range(np.size(test_data, 0)):
            mse_kpca += cdist([Z[i, :]], [centers[i // 33]], 'sqeuclidean')
            mse_pca += cdist([ZL[i, :]], [centers[i // 33]], 'sqeuclidean')
        mse_kpca /= S[1]
        mse_pca /= S[1]
        # Obtain the ratio
        ratio_mse._set_value(index = sigma,col = n, value = mse_pca[0][0]/mse_kpca[0][0])

        # Information for user
        #"""
        print("")
        print("ratio_MSE =", mse_pca[0][0]/mse_kpca[0][0])
        print("kPCA MSE = ", mse_kpca[0][0])
        print("PCA MSE = ", mse_pca[0][0])
        print("")
        print(count, "/", (len(Sigma)*(N-1)))
        print("")
        count += 1

# PRINT FINAL RESULTS
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(ratio_mse)