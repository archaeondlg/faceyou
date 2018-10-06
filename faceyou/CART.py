#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/24/18 2:42 PM

from faceyou import testSet
import operator
import copy
import numpy as np


def calcgini(dataset):
    """
    计算数据集D的基尼指数
    :param dataset:
    :return:
    """
    sub = 0.0
    counter = {}
    for data in dataset:
        label = data[-1]
        if label not in counter.keys():
            counter[label] = 0
        counter[label] += 1
    numofdataset = len(dataset)
    # 二分类时基尼指数算法
    for index in counter:
        sub += (counter[index] / numofdataset) ** 2
    gini = 1 - sub
    return gini


# def calcginionlabel(self, dataset):
#     numofdata = len(dataset)
#     ginionlable = self.calcgini(D1) * count(D1) / numofdata + self.calcgini(D2) * count(D2) / numofdata
#     return ginionlable


def split(dataset, label, value):
    list_left, list_right = [], []
    # 判定数据类型
    if not value:
        return list_right, list_left
    if isinstance(value, int) or isinstance(value, float):
        for row in dataset:
            if row[label] <= value:
                list_right.append(row)
            else:
                list_left.append(row)
    # 主要是字符串类型
    else:
        for row in dataset:
            if row[label] == value:
                list_right.append(row)
            else:
                list_left.append(row)

    return list_right, list_left


def getbestfeat(dataset):
    total_gini = calcgini(dataset)
    column_length = len(dataset[0])
    rows_length = len(dataset)
    min_gini = 1000
    best_feature_index = 0
    best_feature_value = 0

    for feature_index in range(column_length - 1):
        # 获取该特征列表，转为集合会发生数据去重！！！
        col_value_set = set([data[feature_index] for data in dataset])
        # 对于该特征的每个可能取值
        for value in col_value_set:
            list_right, list_left = split(dataset, feature_index, value)

            p = len(list_right) / rows_length
            gini_on_feature = p * calcgini(list_right) + (1 - p) * calcgini(list_left)
            if gini_on_feature < min_gini:
                min_gini = gini_on_feature
                best_feature_index = feature_index
                best_feature_value = value

    # if total_gini - min_gini < 0.00001:
    #     return None, None
    return best_feature_index, best_feature_value


def createTree(dataset, labels):
    """
    创建决策树，默认gini函数，分类树
    :param dataset:
    :param calcfunction:
    :return:
    """
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 所有的类别都一样，就不用再划分了
    if len(dataset[0]) == 1:  # 如果没有继续可以划分的特征，就多数表决决定分支的类别
        return voter(classList)
    best_feature_index, best_feature_value = getbestfeat(dataset)
    bestFeat = best_feature_index
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return myTree


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def voter(datas):
    """
    统计数量，可用Counter代替
    :param datas:
    :return:
    """
    # return Counter(classCount).most_common(1)[0][0]
    # results = {}
    # for data in datas:
    #     # data[-1] means dataType
    #     if data[-1] not in results:
    #         results[data[-1]] = 1
    #     else:
    #         results[data[-1]] += 1
    # return results
    counter = {}
    for data in datas:
        if data not in counter.keys():
            counter[data] = 0
        counter[data] += 1
    # python3.5以后没有iteritems，而使用items
    sorted_counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_counter[0][0]


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def classify(inputTree, featLabels, testVec):  # 输入构造好的决策树
    firstStr = list(inputTree.keys())[0]  # 第一层
    secondDict = inputTree[firstStr]  # 第二层
    # print(featLabels)
    # print(firstStr)
    featIndex = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(firstStr)  # 特征值的索引
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # 注意局部变量与全局变量的关系，否则会报错
                global classLabel
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifylist(tree, dataset):
    # 返回预测对或者错，而不是返回预测的结果(对为0，错为1，方便计算预测错误的个数)
    rank, axis = np.shape(dataset)
    errorList = np.ones((rank, 1))
    # 记录预测的结果
    predictResult = []
    classList = [example[-1] for example in dataset]
    for i in range(rank):
        res = classify(tree, classList, dataset[i])
        errorList[i] = res <= classList[i]
        predictResult.append([int(res)])
    return errorList, np.mat(predictResult)


# 计算预测误差
def calcTestErr(myTree, testData, labels):
    errorCount = 0.0
    for i in range(len(testData)):
        if classify(myTree, labels, testData[i]) != testData[i][-1]:
            errorCount += 1
    return float(errorCount)


# 计算剪枝后的预测误差
def testMajor(major, testData):
    errorCount = 0.0
    for i in range(len(testData)):
        if major != testData[i][-1]:
            errorCount += 1
    return float(errorCount)


def pruningTree(inputTree, dataSet, testData, labels):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]  # 获取子树
    classList = [example[-1] for example in dataSet]
    featKey = copy.deepcopy(firstStr)
    labelIndex = labels.index(featKey)
    subLabels = copy.deepcopy(labels)
    del (labels[labelIndex])
    for key in list(secondDict.keys()):
        if isTree(secondDict[key]):
            # 深度优先搜索,递归剪枝
            subDataSet = splitDataSet(dataSet, labelIndex, key)
            subTestSet = splitDataSet(testData, labelIndex, key)
            if len(subDataSet) > 0 and len(subTestSet) > 0:
                inputTree[firstStr][key] = pruningTree(secondDict[key], subDataSet, subTestSet, copy.deepcopy(labels))
    if calcTestErr(inputTree, testData, subLabels) < testMajor(voter(classList), testData):
        # 剪枝后的误差反而变大，不作处理，直接返回
        return inputTree
    else:
        # 剪枝，原父结点变成子结点，其类别由多数表决法决定
        return voter(classList)


if __name__ == '__main__':
    dataSet, labels = testSet.loan()
    tree = createTree(dataSet, labels)
    print(tree)
