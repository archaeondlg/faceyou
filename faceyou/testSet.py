#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/24/18 11:59 PM


def loan():
    data_set = [['youth', 'no', 'no', '1', 'refuse'],
                ['youth', 'no', 'no', '2', 'refuse'],
                ['youth', 'yes', 'no', '2', 'agree'],
                ['youth', 'yes', 'yes', '1', 'agree'],
                ['youth', 'no', 'no', '1', 'refuse'],
                ['mid', 'no', 'no', '1', 'refuse'],
                ['mid', 'no', 'no', '2', 'refuse'],
                ['mid', 'yes', 'yes', '2', 'agree'],
                ['mid', 'no', 'yes', '3', 'agree'],
                ['mid', 'no', 'yes', '3', 'agree'],
                ['elder', 'no', 'yes', '3', 'agree'],
                ['elder', 'no', 'yes', '2', 'agree'],
                ['elder', 'yes', 'no', '2', 'agree'],
                ['elder', 'yes', 'no', '3', 'agree'],
                ['elder', 'no', 'no', '1', 'refuse']]
    labels = ['age', 'working', 'house', 'credit']
    return data_set, labels


def gender():
    data_set = [['长', '粗', '男'],
                ['短', '粗', '男'],
                ['短', '粗', '男'],
                ['长', '细', '女'],
                ['短', '细', '女'],
                ['短', '粗', '女'],
                ['长', '粗', '女'],
                ['长', '粗', '女']]
    labels = ['头发', '声音']
    return data_set, labels


def horse(type, labels=False):
    if type == 'train':
        filename = '../data/horseColicTrain.txt'
        # filename = '../data/horse.txt'
    else:
        filename = '../data/horseColicTest.txt'
    numFeat = len(open(filename).readline().split('\t'))  # get number of fields
    dataMat = []
    labelMat = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    fr = open(filename)
    if not labels:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat
    else:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))

        return dataMat, [alphabet[index] for index in range(numFeat - 1)]
