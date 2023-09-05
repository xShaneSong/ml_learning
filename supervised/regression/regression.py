from numpy import *

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

# 
def showPlot(xArr, yArr, ws):
    import matplotlib.pyplot as plt

    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    corrcoef(yHat.T, yMat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()


def showPlot2(xArr, yArr):
    import matplotlib.pyplot as plt

    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    yMat = mat(yArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0], s = 2, c = 'red')
    plt.show()

# 岭回归
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This martix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * xMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    print(xMat)
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    print(wMat)
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        print("-------------------")
        print(ws.T)
        print("-------------------")
        print(shape(wMat[i,:]))

        wMat[i,:] = ws.T
    return wMat

def showRidgePlot(ridgeWeights):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

# lasso前向逐步回归
'''
数据标准化，使其充分满足0均值和单位方差
在每轮迭代过程中：
    设置当前最小误差lowestError为正无穷
    对每个特征：
        增大或缩小：
            改变一个系数得一个新的W
            计算新W下的误差
            如果误差Error小于当前最小误差lowestError: 设置Wbest等于当前的W
            将W设置为新的Wbest
'''
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

if __name__ == "__main__":
    # xArr, yArr = loadDataSet('ex0.txt')
    # #print(xArr[0:2])
    # #ws = standRegres(xArr, yArr)
    # #print(ws)
    # #showPlot(xArr, yArr, ws)
    # print(lwlr(xArr[0], xArr, yArr, 1.0))
    # print(lwlr(xArr[0], xArr, yArr, 0.001))
    # yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    # print(yHat)
    # showPlot2(xArr, yArr)ld o

    # abX, abY = loadDataSet('abalone.txt')
    # yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    # print(yHat01)
    # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # print(yHat1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # print(yHat10)
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # print(yHat01)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # print(yHat1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    # print(yHat10)
    # print(rssError(abY[100: 199], yHat10.T))

    # ws = standRegres(abX[0:99], abY[0:99])
    # yHat = mat(abX[100:199]) * ws
    # print(rssError(abY[100: 199], yHat.T.A))

    # abX, abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX, abY)
    # # showRidgePlot(ridgeWeights)
    # print(ridgeRegres)

    # lasso
    xArr, yArr = loadDataSet('abalone.txt')
    print(stageWise(xArr, yArr, 0.01, 200))