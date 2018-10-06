#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/10/18 8:21 PM

from math import log
import operator
from collections import Counter
from faceyou import testSet


def calcShannonEnt(dataSet):
    """
    计算熵值
    :param dataSet:
    :return:
    """
    # 数据条数
    numEntries = len(dataSet)

    # 多余！已经写了现成的函数，还在写一遍
    labelCounts = {}
    for featVec in dataSet:
        # 每行数据的最后一个列（类别）
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 统计有多少个类以及每个类的数量
        labelCounts[currentLabel] += 1

    shannonEnt = 0
    for key in labelCounts:
        # Ck类中样本的概率
        prob = float(labelCounts[key]) / numEntries
        #
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def get_punishment_para(dataset, axis):
    set = [data[axis] for data in dataset]
    # 数据条数
    numEntries = len(dataSet)
    vote = Counter(set)
    penalty_param = 0
    for index in vote:
        prob = float(vote[index]) / numEntries
        penalty_param -= log(prob, 2)
    return penalty_param


# 按某个特征分类后的数据
def splitDataSet(dataSet, axis, value):
    """
    返回axis中值为value，且剔除axis的列表
    :param dataSet:
    :param axis:
    :param value:
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最优的分类特征
def chooseBestFeatureToSplit(dataSet):
    """
    计算信息增益，返回最优特征A的index
    :param dataSet:
    :return:
    """
    # 特征个数
    numFeatures = len(dataSet[0]) - 1
    # 原始的熵值
    baseEntropy = calcShannonEnt(dataSet)
    biggestRatio = 0
    bestFeature = -1
    for i in range(numFeatures):
        # 取index为i的axis
        # 数据集A特征下的一列数据
        featList = [data[i] for data in dataSet]
        # 将list转为set集合
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 按特征分类后的熵值
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 原始熵与按特征分类后的熵的差值，即信息增益
        infoGain = baseEntropy - newEntropy
        # 若按某特征划分后，熵值减小的最大，则次特征为最优分类特征
        penalty_param = get_punishment_para(dataSet, i)
        infoGainRatio = infoGain / penalty_param
        if infoGainRatio > biggestRatio:
            biggestRatio = infoGainRatio
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    投票法返回最优类别
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 当D中实例都属于Ck，则T为单结点树，返回类标记Ck
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 当A为空，即没有分类类别时，投票法返回最优类别
    if len(dataSet[0]) == 1:
        # 可以直接使用Counter(classList)
        return majorityCnt(classList)
    # 计算信息增益，返回最优特征A的index
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取labels中类别名称
    bestFeatLabel = labels[bestFeat]
    # 将最优特征作为节点，生成子树
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 取最优特征对应axis的数据
    featValues = [example[bestFeat] for example in dataSet]
    # 利用set自动去重
    uniqueVals = set(featValues)
    # 剔除最优特征后的特征列表
    subLabels = labels[:]
    for value in uniqueVals:
        # 已取得当前最优特征，获取特征下value，排除该特征后的数据集
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = testSet.gender()
    print(createTree(dataSet, labels))
