#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: Tools.py
Date: 2018/08/27 23:19:27
Desc: 工具类
"""

import numpy as np


class Tools(object):
    """
    工具类
    """
    transfer_dict = {
        0: (-1, -1, -1,),
        1: (-1, -1, 0,),
        2: (-1, -1, 1,),
        3: (-1, 0, -1,),
        4: (-1, 0, 0,),
        5: (-1, 0, 1,),
        6: (-1, 1, -1,),
        7: (-1, 1, 0,),
        8: (-1, 1, 1,),
        9: (0, -1, -1,),
        10: (0, -1, 0,),
        11: (0, -1, 1,),
        12: (0, 0, -1,),
        13: (0, 0, 1,),
        14: (0, 1, -1,),
        15: (0, 1, 0,),
        16: (0, 1, 1,),
        17: (1, -1, -1,),
        18: (1, -1, 0,),
        19: (1, -1, 1,),
        20: (1, 0, -1,),
        21: (1, 0, 0,),
        22: (1, 0, 1,),
        23: (1, 1, -1,),
        24: (1, 1, 0,),
        25: (1, 1, 1,),
        (-1, -1, -1,): 0,
        (-1, -1, 0,): 1,
        (-1, -1, 1,): 2,
        (-1, 0, -1,): 3,
        (-1, 0, 0,): 4,
        (-1, 0, 1,): 5,
        (-1, 1, -1,): 6,
        (-1, 1, 0,): 7,
        (-1, 1, 1,): 8,
        (0, -1, -1,): 9,
        (0, -1, 0,): 10,
        (0, -1, 1,): 11,
        (0, 0, -1,): 12,
        (0, 0, 1,): 13,
        (0, 1, -1,): 14,
        (0, 1, 0,): 15,
        (0, 1, 1,): 16,
        (1, -1, -1,): 17,
        (1, -1, 0,): 18,
        (1, -1, 1,): 19,
        (1, 0, -1,): 20,
        (1, 0, 0,): 21,
        (1, 0, 1,): 22,
        (1, 1, -1,): 23,
        (1, 1, 0,): 24,
        (1, 1, 1,): 25
    }

    @staticmethod
    def create_normal_dist(mu, sigma, size):
        """
        创建正态分布数组
        :param mu: 均值
        :param sigma: 标准差
        :param size: 数组规模
        :return: 数组
        """
        data = np.random.normal(mu, sigma, size)
        for i in range(size[0]):
            for j in range(size[1]):
                if len(size) == 2:
                    if data[i, j] < 0:
                        data[i, j] = np.random.normal(mu, sigma)
                else:
                    for k in range(size[2]):
                        if len(size) == 3:
                            if data[i, j, k] < 0:
                                data[i, j, k] = np.random.normal(mu, sigma)
                        else:
                            for l in range(size[3]):
                                if data[i, j, k, l] < 0:
                                    data[i, j, k, l] = np.random.normal(mu, sigma)
        return data

    @staticmethod
    def get_relative_position(no):
        """
        根据连接通道的编号，确定相对孔隙位置，反之亦然
             Nodes Arrangement Figure

                  8--------- 16---------25
                 /|         /|         /|
               5  7       13 15      22 24
             / | /|     / | /|     / | /|
            2  4  6   11  o  14  19  21 23              z(up)
            | /| /     | /| /     | /| /                  ^    _ y(back)
            1  3      10  12     18  20                   |    /|
            | /        | /        | /                     |  /
            0----------9---------17                       |/-----> x(right)
        Test:
        >> get_relative_position(no)
        Result:
              No    Result      No    Result      No    Result
        >>    0   [-1,-1,-1]     9  [ 0,-1,-1]    17  [ 1,-1,-1]
        >>    1   [-1,-1, 0]    10  [ 0,-1, 0]    18  [ 1,-1, 0]
        >>    2   [-1,-1, 1]    11  [ 0,-1, 1]    19  [ 1,-1, 1]
        >>    3   [-1, 0,-1]    12  [ 0, 0,-1]    20  [ 1, 0,-1]
        >>    4   [-1, 0, 0]                      21  [ 1, 0, 0]
        >>    5   [-1, 0, 1]    13  [ 0, 0, 1]    22  [ 1, 0, 1]
        >>    6   [-1, 1,-1]    14  [ 0, 1,-1]    23  [ 1, 1,-1]
        >>    7   [-1, 1, 0]    15  [ 0, 1, 0]    24  [ 1, 1, 0]
        >>    8   [-1, 1, 1]    16  [ 0, 1, 1]    25  [ 1, 1, 1]
        :param no: (int)孔隙编号(1-26) / (int[3])孔隙相对位置(-1~1)
        :return: (int[3])孔隙相对位置(-1~1) / (int)孔隙编号(1-26)
        """
        if isinstance(no, int):
            return Tools.transfer_dict[no]
        else:
            return Tools.transfer_dict[tuple(no)]


if __name__ == '__main__':
    print Tools.create_normal_dist(0, 1, [2, 3])
    print Tools.get_relative_position(10)
    print Tools.get_relative_position([1, 1, -1])
