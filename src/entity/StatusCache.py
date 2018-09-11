#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: StatusCache.py
Date: 2018/09/06 22:09:20
Desc: 计算缓存类
"""

import numpy as np
import logging

from utils import Tools


class StatusCache(object):
    """
    状态缓存基本类
    """
    def __init__(self, network_status):
        self.darcy_cache = None
        self.length_cache = None
        self.kn_cache = None
        self.knudsen_cache = None
        self.get_status_cache(network_status)

    def get_status_cache(self, network_status):
        """
        计算喉道迭代常用数值，加速计算结果。
        计算得到Kn即可得到: F = 1 + 4Kn, f = 1 / (1 + 0.5Kn)
        :return kn_coef: Kn的加速计算系数，公式为 KnCoef = sqrt(πRT/2M) * μ / r，计算时有 Kn = KnCoef / p
        :return darcy_coef: 达西流部分加速计算系数，公式为 DarcyCoef = πr^4/8μ
        :return knudsen_coef: 克努森扩散流部分加速计算系数，公式为 KnudsenCoef = 2πr^3/3 * sqrt(8RT/πM)
        """
        logging.info("------------------创建计算缓存中------------------")
        logging.info("创建计算缓存……")
        self.kn_cache = np.zeros(network_status.model_size + [26])
        self.darcy_cache = np.zeros(network_status.model_size + [26])
        self.knudsen_cache = np.zeros(network_status.model_size + [26])
        self.length_cache = np.zeros(network_status.model_size + [26])
        for i in range(network_status.model_size[0]):
            for j in range(network_status.model_size[1]):
                for k in range(network_status.model_size[2]):
                    ind1 = (i, j, k)
                    for l in range(13):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        if (0 <= ind2[0] < network_status.model_size[0]) \
                                and (0 <= ind2[1] < network_status.model_size[1]) \
                                and (0 <= ind2[2] < network_status.model_size[2]):
                            throat_r = network_status.ns.throatR[i, j, k, l] * network_status.ns.character_length
                            length = np.sqrt(np.sum(np.square(rp))) * network_status.ns.unit_size
                            self.darcy_cache[i, j, k, l] = np.pi * np.power(throat_r, 4) / 8.0 / network_status.gc.u
                            self.length_cache[i, j, k, l] = 1.0 / (length - network_status.ns.radii[ind1] -
                                                                   network_status.ns.radii[ind2]) / \
                                network_status.ns.character_length
                            self.kn_cache[i, j, k, l] = np.sqrt(np.pi * network_status.gc.R * network_status.gc.T / 2.0
                                                                / network_status.gc.M) * network_status.gc.u / throat_r
                            self.knudsen_cache[i, j, k, l] = 2 * np.pi * np.power(throat_r, 3) / 3.0 * \
                                np.sqrt(8 * network_status.gc.R * network_status.gc.T /
                                        np.pi / network_status.gc.M)
                            self.darcy_cache[ind2[0], ind2[1], ind2[2], 25-l] = self.darcy_cache[i, j, k, l]
                            self.length_cache[ind2[0], ind2[1], ind2[2], 25-l] = self.length_cache[i, j, k, l]
                            self.kn_cache[ind2[0], ind2[1], ind2[2], 25-l] = self.kn_cache[i, j, k, l]
                            self.knudsen_cache[ind2[0], ind2[1], ind2[2], 25-l] = self.knudsen_cache[i, j, k, l]
            logging.debug("    计算缓存中，当前进度 = " + format(float(i) /
                          float(network_status.model_size[0]) * 100.0, '.2f') + "%")
        logging.info("完成计算缓存")
