#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/26/18 11:36 PM

from numpy import *
from faceyou import testSet


def stumpClassify(dataMatrix, featindex, threshVal, threshIneq):
    """
    对数据进行分类
    :param dataMatrix: 数据集（矩阵）
    :param featindex: 比较的特征的位置
    :param threshVal: 阈值
    :param threshIneq: 不等号，两种比较模式(>threshVal为1，或<threshVal为1)
    :return:
    """
    # 预先将数据都设为1
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, featindex] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, featindex] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, weight):
    """
    在加权数据集里面寻找最低错误率的单层决策树，即决策桩
    :param dataArr:
    :param classLabels:
    :param weight: 数据集权重 用于计算加权错误率
    :return:
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    predictedRes = mat(zeros((m, 1)))
    # 初始化最小误差率为无穷大
    minError = inf
    # 计算特征中使得误差率最小的特征
    # 计算出用来分割的值，即在上文中的a
    for i in range(n):
        # 计算每个特征的最大值与最小值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 将最大值与最小值之间分段
        stepSize = (rangeMax - rangeMin) / numSteps
        # 依次在最大值与最小值之间[)取值，计算误差
        for j in range(-1, int(numSteps) + 1):
            # 对于小于或大于
            for inequal in ['lt', 'gt']:
                # np 印象中有类似分段的函数，待考察
                threshVal = (rangeMin + float(j) * stepSize)

                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

                # 初始化为1，标记为错误
                errArr = mat(ones((m, 1)))
                # 标记正确结果为0
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率
                weightedError = weight.T * errArr
                # 保存最小加权错误率
                if weightedError < minError:
                    minError = weightedError
                    predictedRes = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, predictedRes


def creatAdaBoost(dataArr, classLabels, numIter=40):
    """
    基于单层决策树的AdaBoost训练函数
    当训练错误率达到0就会提前结束训练
    :param dataArr:
    :param classLabels:
    :param numIter: 迭代次数
    :return:
    """
    # 弱分类器集
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化数据集的权重
    weight = mat(ones((m, 1)) / m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    # 迭代numIter次
    for i in range(numIter):
        # 计算在当前样本权值weight下的最小误差
        # 获取最优决策树桩
        bestStump, error, classEst = buildStump(dataArr, classLabels, weight)
        # 根据错误率计算权重alpha值
        # max(error, 1e-16)避免出现分母为0
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        # 将当前分类器结果添加到分类器集
        weakClassArr.append(bestStump)
        # 计算下一次迭代中的权重向量weight
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        # 更新weight
        weight = multiply(weight, exp(expon))
        # 归一化
        weight = weight / weight.sum()
        # 错误率累加
        # classEst的值为1或-1
        # alpha*classEst为每一个分类器的权值与对应样本的类别相乘
        # 下式对所有的分类器进行相加，得到最终的类别
        # 再用sign函数
        aggClassEst += alpha * classEst
        # sign(aggClassEst)表示根据正负号分别标记为正负1
        # 与ones((m,1)相乘，即得到误差个数
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print("total error: ")
        # print(errorRate)
        # 当训练错误率达到0就会提前结束
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(dataset, classifierList):
    """
    :param dataset: 待分类样例
    :param classifierList: adaboost算法返回的分类器集
    :return:
    """
    dataMatrix = mat(dataset)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierList)):
        classEst = stumpClassify(dataMatrix, classifierList[i]['dim'], \
                                 classifierList[i]['thresh'], \
                                 classifierList[i]['ineq'])
        aggClassEst += classifierList[i]['alpha'] * classEst
    return sign(aggClassEst)


if __name__ == '__main__':
    dataMat, classLabels = testSet.simple()

    weakClassArr = creatAdaBoost(dataMat, classLabels)


    # 马疝病数据集测试
    dataArr, labelArr = testSet.horse('train')
    classifierArray = creatAdaBoost(dataArr, labelArr, 10)

    # 测试
    testArr, testLabelArr = testSet.horse('test')
    prediction10 = adaClassify(testArr, classifierArray)
    print(sum(prediction10 != mat(testLabelArr).T) / len(prediction10))
