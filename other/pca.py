from numpy import *

def loadDataSet(filename, delim = '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)

'''
去除平均值
计算协方差矩阵
计算协方差矩阵的特征值和特征向量
将特征值从大到小排序
保留最上面的N个特征向量
将数据转换到上述N个特征向量构建的新空间中
'''
def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar = 0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1): -1]
    redEigVects = eigVects[:,eigValInd]
    lowDataMat = meanRemoved * redEigVects
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat, reconMat

def showPlot():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten.A[0], dataMat[:,1].flatten.A[0], marker = 'A', s = 90)
    ax.scatter(reconMat[:,0].flatten.A[0], reconMat[:,1].flatten.A[0], marker = 'o', s = 50, c = 'red')
    plt.show()

if __name__ == "__main__":
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print(shape(lowDMat))