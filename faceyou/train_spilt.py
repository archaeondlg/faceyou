#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/8/18 5:19 PM

import numpy as np


def spilt(set_x, set_y, ratio=0.2, seed=None):
    assert set_x.shape[0] == set_y.shape[0], \
        'the axis of set_x must be equal to that of set_y'
    assert 0.0 <= ratio <= 1.0, \
        'ratio must be between 0.0 and 1.0'

    if seed:
        np.random.seed(seed)

    # 获取随机索引
    rand_index = np.random.permutation(len(set_x))
    # 计算测试数量
    num = int(len(set_x) * ratio)

    test_index = rand_index[:num]
    train_index = rand_index[num:]

    train_x = set_x[train_index]
    train_y = set_y[train_index]
    test_x = set_x[test_index]
    test_y = set_y[test_index]

    return train_x, train_y, test_x, test_y
