#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/15/18 3:43 PM

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    """
    加载数据集
    :return:输入向量矩阵和输出向量
    """
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def plotBestFit(weights):
    """
    画出数据集和逻辑斯蒂最佳回归直线
    :param weights:
    :return:
    """
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    if weights is not None:
        x = arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights[1] * x) / weights[2]
        ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    逻辑斯蒂回归梯度上升优化suanfa
    :param dataMatIn: 输入x矩阵
    :param classLabels: 输出Y矩阵（类别标签组成的向量）
    :return:
    """
    # 转换为numpy矩阵数组类型
    dataMatrix = mat(dataMatIn)
    # 转换类型，并转置
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent0(dataMatrix, classLabels, history_weights):
    """
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :param history_weights:
    :return: 权值向量
    """
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # 挑选（伪随机）第i个实例来更新
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + dataMatrix[i] * alpha * error
        history_weights.append(copy(weights))
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :param numIter: 迭代次数
    :return:
    """
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # 步长递减，但？
            alpha = 4 / (1.0 + j + i) + 0.0001
            # 真随机
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + dataMatrix[randIndex] * alpha * error
            # 删除样本
            del (dataIndex[randIndex])
    return weights


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights)
