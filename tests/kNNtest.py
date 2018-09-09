#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/8/18 6:04 PM

# 引入ｋＮＮClassifier
from faceyou.kNN import KNNClassifier
from faceyou import train_spilt
from sklearn import datasets

# 导入鸢尾花数据集
iris = datasets.load_iris()
set_x = iris.data
set_y = iris.target

# 分割数据集
train_x, train_y, test_x, test_y = train_spilt.spilt(set_x, set_y)
kc = KNNClassifier(k=6)
kc.fit(train_x, train_y)
predict_y = kc.predict_set(test_x)
accuracy = sum(predict_y == test_y) / len(test_y)
print('kNN训练准确率为:' + str(accuracy))
