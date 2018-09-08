#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/8/18 12:00 PM

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        # 断言验证k值有效
        assert k >= 1, 'k is not valid'
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        # 验证训练数据size
        assert x_train.shape[0] == y_train.shape[0], \
            'x_train can not fit y_train'
        assert self.k <= x_train.shape[0], \
            'k can not greater than the axis of x_train'
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict_set(self, x_predict):
        assert self.x_train is not None and self.y_train is not None, \
            'data for predict must be fitted'
        assert self.x_train.shape[1] == x_predict.shape[1], \
            'the axis of x_predict must be equal to that of x_train'
        y_predict = [self.predict(x) for x in x_predict]
        return np.array(y_predict)

    def predict(self, x):
        distances = [sqrt(np.sum(x_train - x) ** 2) for x_train in self.x_train]
        nearest = np.argsort(distances)
        topK_y = [self.y_train[i] for i in nearest[:self.k]]
        vote_result = Counter(topK_y)
        return vote_result.most_common(1)[0][0]
