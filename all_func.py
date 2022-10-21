from numpy import *
from math import *
import numpy as np
import random as ran
import matplotlib.pyplot as plt


class generateDate:
    def generate(self):
        N = 1000

        mu1 = [1, 1]  # 均值矢量m1
        mu2 = [4, 4]  # 均值矢量m2
        mu3 = [8, 1]  # 均值矢量m3
        sigma = [[2, 0], [0, 2]]  # 协方差矩阵S1=S2=S3=2I,I为2*2单位矩阵
        n1_data1, n2_data1, n3_data1 = 0, 0, 0
        # 数据集X
        for dataNum in range(0, N):
            k = ran.randint(1, 3)  # 随机产生1/3
            if k == 1:
                n1_data1 += 1  # 统计在每个分布模型得到的数目
            elif k == 2:
                n2_data1 += 1
            else:
                n3_data1 += 1
        # 生成多元正态分布矩阵,获取X数据点
        x1, y1 = np.random.multivariate_normal(mu1, sigma, n1_data1).T
        x2, y2 = np.random.multivariate_normal(mu2, sigma, n2_data1).T
        x3, y3 = np.random.multivariate_normal(mu3, sigma, n3_data1).T
        '''
        fig=plt.figure()
        ax1=fig.add_subplot(121)
        ax1.scatter(x1,y1,c='red')
        ax1.scatter(x2,y2,c='blue')
        ax1.scatter(x3,y3,c='green')
        '''
        n1_data2, n2_data2, n3_data2 = 0, 0, 0
        X1_part1 = np.vstack((x1, y1))
        X1_part2 = np.vstack((x2, y2))
        X1_part3 = np.vstack((x3, y3))
        cov1_data1 = np.cov(X1_part1)  # 计算协方差矩阵
        cov2_data1 = np.cov(X1_part2)
        cov3_data1 = np.cov(X1_part3)
        mean1_data1 = np.mean(X1_part1, axis=1)  # 均值
        mean2_data1 = np.mean(X1_part2, axis=1)
        mean3_data1 = np.mean(X1_part3, axis=1)
        data1 = np.hstack((X1_part1, X1_part2, X1_part3))

        # 数据集X'
        for dataNum in range(0, N):
            k = ran.randint(1, 10)  # 模拟生成概率分别为0.6, 0.3, 0.1
            if k <= 6:
                n1_data2 += 1
            elif k <= 9:
                n2_data2 += 1
            else:
                n3_data2 += 1
        # 生成多元正态分布矩阵,获取X数据点
        x1, y1 = np.random.multivariate_normal(mu1, sigma, n1_data2).T
        x2, y2 = np.random.multivariate_normal(mu2, sigma, n2_data2).T
        x3, y3 = np.random.multivariate_normal(mu3, sigma, n3_data2).T

        '''
        ax2=fig.add_subplot(122)
        ax2.scatter(x1,y1,c='red')
        ax2.scatter(x2,y2,c='blue')
        ax2.scatter(x3,y3,c='green')
        plt.show()
        '''

        X2_part1 = np.vstack((x1, y1))
        X2_part2 = np.vstack((x2, y2))
        X2_part3 = np.vstack((x3, y3))
        cov1_data2 = np.cov(X2_part1)  # 计算协方差矩阵
        cov2_data2 = np.cov(X2_part2)
        cov3_data2 = np.cov(X2_part3)
        mean1_data2 = np.mean(X2_part1, axis=1)  # 均值
        mean2_data2 = np.mean(X2_part2, axis=1)
        mean3_data2 = np.mean(X2_part3, axis=1)
        data2 = np.hstack((X2_part1, X2_part2, X2_part3))

        data1 = data1.T
        data2 = data2.T

        label_data1 = np.zeros(N)
        label_data1[n1_data1:n1_data1 + n2_data1 - 1] = 1
        label_data1[N - n3_data1:] = 2
        label_data1 = map(int, label_data1)

        label_data2 = np.zeros(N)
        label_data2[n1_data2:n1_data2 + n2_data2 - 1] = 1
        label_data2[N - n3_data2:] = 2
        label_data2 = map(int, label_data2)
        PrioPro_data1 = np.array((n1_data1, n2_data1, n3_data1)) / 1000.
        PrioPro_data2 = np.array((n1_data2, n2_data2, n3_data2)) / 1000.
        mean_X1, sigma_X1 = self.getParameters(mean1_data1, mean2_data1, mean3_data1, cov1_data1, cov2_data1, cov3_data1)
        mean_X2, sigma_X2 = self.getParameters(mean1_data2, mean2_data2, mean3_data2, cov1_data2, cov2_data2, cov3_data2)
        return data1, data2, list(label_data1), list(label_data2), PrioPro_data1, PrioPro_data2, mean_X1, sigma_X1, mean_X2, sigma_X2

    def getParameters(self, mean1, mean2, mean3, cov1, cov2, cov3):
        mu = np.vstack((mean1, mean2, mean3))
        sigma = np.zeros((3, 2, 2))
        sigma[0], sigma[1], sigma[2] = cov1, cov2, cov3
        return mu, sigma

    def show_scatter(self, data, data2):
        data1 = data.T
        data2 = data2.T
        x1, y1 = data1
        x2, y2 = data2
        plt.plot(x1, y1, '.', label='X')
        plt.plot(x2, y2, 'x', label='X\'')
        plt.axis('equal')
        plt.legend()
        plt.show()


class Minimum_European_distance:
    def getDistanced(self, vec1, vec2):
        return sqrt(sum(power(vec1-vec2, 2)))

    def getLabel(self, data, K, mean_X):
        m, n = shape(data)
        clusterAssment = zeros(m)
        centroids = mat(mean_X)
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(K):
                dist = self.getDistanced(data[i, :], centroids[j, :])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssment[i] != minIndex:
                clusterAssment[i] = minIndex
        return centroids, clusterAssment


class Likelihood_Rate:
    def __init__(self, label_data1, label_data2):
        self.label1 = list(label_data1)
        self.label2 = list(label_data2)

    def getPosterPro(self, K, data, sigma, mu, PrioPro):
        m, n = np.shape(data)
        Px_w = np.mat(np.zeros((m, K)))
        for i in range(K):
            coef = (2 * math.pi) ** (-n / 2.) * (np.linalg.det(sigma[i]) ** (-0.5))
            temp = np.multiply((data-mu[i])*np.mat(sigma[i]).I, data-mu[i])
            Xshift = np.sum(temp, axis=1)
            Px_w[:, i] = coef * np.exp(Xshift*-0.5)  # 矩阵与常数相乘
        PosterPro = np.mat(np.zeros((m, K)))
        for i in range(K):
            PosterPro[:, i] = PrioPro[i]*Px_w[:, i]
        return PosterPro

    def getLikelihoodLabel(self, PosterPro):
        outputLabel = np.argmax(PosterPro, axis=1)
        outputLabel = list(map(int, np.array(outputLabel.flatten())[0]))
        return outputLabel

    def getErrorRate(self, N, label, outputLabel, num='1'):
        if label == [] and num == '1':
            label = self.label1
        if label == [] and num == '2':
            label = self.label2
        errorNum = np.int(np.shape(np.nonzero(np.array(outputLabel) - np.array(label)))[1])
        errorRate = float(errorNum) / N
        return errorNum, errorRate


class Bayesian_risk:

    def getBayesLabel(self, PosterPro):
        Cost = [[0, 2, 3], [1, 0, 5], [1, 1, 0]]
        M = np.shape(PosterPro)[0]
        BayesLabel = np.zeros(M)
        for m in range(M):
            for i in range(K):
                flag = True
                for j in range(i + 1, K):
                    temp = (Cost[j][i] - Cost[i][i]) * np.array(PosterPro)[m][i] - \
                           (Cost[i][j] - Cost[j][j]) * np.array(PosterPro)[m][j]
                    if temp < 0:flag = False
                if flag == True:
                    BayesLabel[m] = i
                    break
                else:
                    BayesLabel[m] = j
                    continue
        return list(BayesLabel)


class Max_proster:

    def getPosterPro(self, K, data, sigma, mu, PrioPro):
        m, n = np.shape(data)
        Px_w = np.mat(np.zeros((m, K)))
        for i in range(K):
            coef = (2 * math.pi) ** (-n / 2.) * (np.linalg.det(sigma[i]) ** (-0.5))
            temp = np.multiply((data - mu[i]) * np.mat(sigma[i]).I, data - mu[i])
            Xshift = np.sum(temp, axis=1)
            Px_w[:, i] = coef * np.exp(Xshift * -0.5)  # 矩阵与常数相乘
        PosterPro = np.mat(np.zeros((m, K)))
        for i in range(K):
            PosterPro[:, i] = PrioPro[i] * Px_w[:, i]
        return PosterPro

    def getLikelihoodLabel(self, PosterPro):
        outputLabel = np.argmax(PosterPro, axis=1)
        outputLabel = map(int, np.array(outputLabel.flatten())[0])
        return list(outputLabel)

    def getErrorRate(self, N, label, outputLabel):
        errorNum = np.int(np.shape(np.nonzero(np.array(outputLabel) - np.array(label)))[1])
        errorRate = float(errorNum) / N
        return errorNum, errorRate


if __name__ == '__main__':
    N = 1000
    K = 3
    # 生成两个数据集
    gener_data = generateDate()
    data1, data2, label_data1, label_data2, PrioPro_data1, PrioPro_data2, mean_X1, sigma_X1, mean_X2, sigma_X2 = gener_data.generate()
    # print(list(label_data1))
    # 出事数据集散点分布图
    gener_data.show_scatter(data1, data2)
    # 模式分类
    likelihood = Likelihood_Rate(label_data1, label_data2)
    min_european = Minimum_European_distance()
    bayesian_risk = Bayesian_risk()
    max_poster = Max_proster()
    # data1
    PosterPro_data1 = likelihood.getPosterPro(K, data1, sigma_X1, mean_X1, PrioPro_data1)
    likelihoodLabel = likelihood.getLikelihoodLabel(PosterPro_data1)
    error_num, errorRate_data1_likelihood = likelihood.getErrorRate(N, label_data1, likelihoodLabel)
    print('Likelihood Rate rule on X, the error num is', error_num, 'the error rate is:', errorRate_data1_likelihood)
    PosterPro_data1 = likelihood.getPosterPro(K, data1, sigma_X1, mean_X1, PrioPro_data1)
    BayesLabel_data1 = bayesian_risk.getBayesLabel(PosterPro_data1)
    error_num, errorRate_X1_Bayesian = likelihood.getErrorRate(N, label_data1, BayesLabel_data1)
    print('Bayesian risk rule on X, the error num is', error_num, 'the error rate is:', errorRate_X1_Bayesian)
    centroids1, clusterAssment1 = min_european.getLabel(data1, K, mean_X1)
    error_num, errorRate_X1 = likelihood.getErrorRate(N, label_data1, clusterAssment1)
    print('Minimum European distance rule on X, the error num is', error_num, 'the error rate is:', errorRate_X1)
    PosterPro_data1 = max_poster.getPosterPro(K, data1, sigma_X1, mean_X1, PrioPro_data1)
    likelihoodLabel = max_poster.getLikelihoodLabel(PosterPro_data1)
    error_num, errorRate_data1_MaxPost = max_poster.getErrorRate(N, label_data1, likelihoodLabel)
    print('Maximum Posteriori Probability rule on X, the error num is', error_num, 'the error rate is:', errorRate_data1_MaxPost)
    print('\n')
    # data2
    PosterPro_data2 = likelihood.getPosterPro(K, data2, sigma_X2, mean_X2, PrioPro_data2)
    likelihoodLabel_data2 = likelihood.getLikelihoodLabel(PosterPro_data2)
    error_num, errorRate_data2_likelihood = likelihood.getErrorRate(N, label_data2, likelihoodLabel_data2, num='2')
    print('Likelihood Rate rule on X\', the error num is', error_num, 'the error rate is:', errorRate_data2_likelihood)
    PosterPro_data2 = likelihood.getPosterPro(K, data2, sigma_X2, mean_X2, PrioPro_data2)
    BayesLabel_data2 = bayesian_risk.getBayesLabel(PosterPro_data2)
    error_num, errorRate_X2_Bayesian = likelihood.getErrorRate(N, label_data2, BayesLabel_data2, num='2')
    print('Bayesian risk rule on X\', the error num is', error_num, 'the error rate is:', errorRate_X2_Bayesian)
    centroids2, clusterAssment2 = min_european.getLabel(data2, K, mean_X2)
    error_num, errorRate_X2 = likelihood.getErrorRate(N, label_data2, clusterAssment2, num='2')
    print('Minimum European distance rule on X\', the error num is', error_num, 'the error rate is:', errorRate_X2)
    PosterPro_data2 = max_poster.getPosterPro(K, data2, sigma_X2, mean_X2, PrioPro_data2)
    likelihoodLabel_data2 = max_poster.getLikelihoodLabel(PosterPro_data2)
    error_num, errorRate_data2_MaxPost = max_poster.getErrorRate(N, label_data2, likelihoodLabel_data2)
    print('Maximum Posteriori Probability rule on X\', the error num is', error_num, 'the error rate is:', errorRate_data2_MaxPost)


