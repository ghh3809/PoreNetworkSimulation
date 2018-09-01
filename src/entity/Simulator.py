#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: Simulator.py
Date: 2018/08/28 22:42:37
Desc: 处理器类
"""

import os
import sys
import logging
import configparser
import numpy as np

sys.path.append('./')
from utils import Tools


class Simulator(object):
    """
    求解器基本类
    """
    def __init__(self, config_file='./config/config.ini'):
        """
        利用配置创建求解器
        :param network_config: 配置实例
        """
        self.iters = 0
        self.darcy_cache = None
        self.length_cache = None
        self.kn_cache = None
        self.knudsen_cache = None
        self.ns = None

        if not os.path.exists(config_file):
            logging.error("No solver config file detected!")
            raise Exception("No solver config file detected!")
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.solverType
            self.solver_type = conf.get("solver", "solverType")
            if self.solver_type != 'J' and self.solver_type != 'GS' and self.solver_type != 'time':
                raise Exception("param gas.solverType should be J, GS or time!")

            # self.scaleEffect
            self.scale_effect = int(conf.get("solver", "scaleEffect"))
            if self.scale_effect != 0 and self.scale_effect != 1:
                raise Exception("param gas.scaleEffect should be 0 or 1!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        print "------------------读取求解器配置文件------------------"
        print "迭代方法:", self.solver_type
        print "是否考虑尺度效应:", "是" if self.scale_effect == 1 else "否"

    def get_throat_cache(self):
        """
        getThroatCache 计算喉道迭代常用数值，加速计算结果。
        计算得到Kn即可得到: F = 1 + 4Kn, f = 1 / (1 + 0.5Kn)
        :return kn_coef: Kn的加速计算系数，公式为 KnCoef = sqrt(πRT/2M) * μ / r，计算时有 Kn = KnCoef / p
        :return darcy_coef: 达西流部分加速计算系数，公式为 DarcyCoef = πr^4/8μ
        :return knudsen_coef: 克努森扩散流部分加速计算系数，公式为 KnudsenCoef = 2πr^3/3 * sqrt(8RT/πM)
        """
        print "------------------创建计算缓存中------------------"
        print "创建计算缓存……",
        self.kn_cache = np.zeros(self.ns.model_size + [26])
        self.darcy_cache = np.zeros(self.ns.model_size + [26])
        self.knudsen_cache = np.zeros(self.ns.model_size + [26])
        self.length_cache = np.zeros(self.ns.model_size + [26])
        for i in range(self.ns.model_size[0]):
            for j in range(self.ns.model_size[1]):
                for k in range(self.ns.model_size[2]):
                    ind1 = (i, j, k)
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        if (0 <= ind2[0] < self.ns.model_size[0]) \
                                and (0 <= ind2[1] < self.ns.model_size[1]) \
                                and (0 <= ind2[2] < self.ns.model_size[2]):
                            throat_r = self.ns.ns.throatR[i, j, k, l] * self.ns.ns.character_length
                            length = np.sqrt(np.sum(np.square(rp))) * self.ns.ns.unit_size
                            self.darcy_cache[i, j, k, l] = np.pi * np.power(throat_r, 4) / 8.0 / self.ns.gc.u
                            self.length_cache[i, j, k, l] = 1.0 / (length - self.ns.ns.radii[ind1] - self.ns.ns.radii[ind2]) / self.ns.ns.character_length
                            self.kn_cache[i, j, k, l] = np.sqrt(np.pi * self.ns.gc.R * self.ns.gc.T / 2.0 / self.ns.gc.M) * self.ns.gc.u / throat_r
                            self.knudsen_cache[i, j, k, l] = 2 * np.pi * np.power(throat_r, 3) / 3.0 * np.sqrt(8 * self.ns.gc.R * self.ns.gc.T / np.pi / self.ns.gc.M)
        print "完成"

    def bind_network_status(self, network_status):
        """
        绑定网络状态到当前求解器
        :param network_status:
        :return:
        """
        self.ns = network_status
        self.get_throat_cache()

    def iterate_once(self):
        """
        迭代计算一次
        :return:
        """
        c = 1.0
        if self.solver_type == 'J':
            new_pressure = np.zeros(self.ns.model_size)
        for i in range(self.ns.model_size[0]):
            for j in range(self.ns.model_size[1]):
                for k in range(self.ns.model_size[2]):
                    if i == 0 and self.ns.sc.boundary_type[0] == 1:
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = self.ns.sc.boundary_value[0]
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = self.ns.sc.boundary_value[0]
                    elif i == self.ns.model_size[0] - 1 and self.ns.sc.boundary_type[1] == 1:
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = self.ns.sc.boundary_value[1]
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = self.ns.sc.boundary_value[1]
                    elif j == 0 and self.ns.sc.boundary_type[2] == 1:
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = self.ns.sc.boundary_value[2]
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = self.ns.sc.boundary_value[2]
                    elif j == self.ns.model_size[1] - 1 and self.ns.sc.boundary_type[3] == 1:
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = self.ns.sc.boundary_value[3]
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = self.ns.sc.boundary_value[3]
                    elif k == 0 and self.ns.sc.boundary_type[4] == 1:
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = self.ns.sc.boundary_value[4]
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = self.ns.sc.boundary_value[4]
                    elif k == self.ns.model_size[2] - 1 and self.ns.sc.boundary_type[5] == 1:
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = self.ns.sc.boundary_value[5]
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = self.ns.sc.boundary_value[5]
                    else:
                        ind1 = (i, j, k)
                        sum_up = 0.0
                        sum_down = 0.0
                        for l in range(26):
                            rp = Tools.Tools.get_relative_position(l)
                            ind2 = tuple(np.add(rp, (i, j, k)))
                            if (0 <= ind2[0] < self.ns.model_size[0]) \
                                    and (0 <= ind2[1] < self.ns.model_size[1]) \
                                    and (0 <= ind2[2] < self.ns.model_size[2]):
                                indt = tuple(ind1 + (l,))
                                p_ave = (self.ns.pressure[ind1] + self.ns.pressure[ind2]) / 2.0
                                if self.scale_effect:
                                    F = 1.0 + 4.0 * c * self.kn_cache[indt] / p_ave
                                    f = 1.0 / (1.0 + 0.5 * self.kn_cache[indt] / p_ave)
                                    coef = (f * F * self.darcy_cache[indt] * p_ave + (1.0 - f) * self.knudsen_cache[indt]) * self.length_cache[indt] * self.ns.ns.weight[indt]
                                else:
                                    coef = self.darcy_cache[indt] * p_ave * self.length_cache[indt] * self.ns.ns.weight[indt]
                                sum_up += coef * self.ns.pressure[ind2]
                                sum_down += coef
                        if self.solver_type == 'J':
                            new_pressure[i, j, k] = sum_up / sum_down
                        elif self.solver_type == 'GS':
                            self.ns.pressure[i, j, k] = sum_up / sum_down
        if self.solver_type == 'J':
            self.ns.pressure = new_pressure
        self.iters += 1

    def get_mass_flux(self):
        """
        获取网络的质量流量
        :return:
        """
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_mass_flux(self)

    def get_kn(self):
        """
        获取knudsen number
        :return:
        """
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_kn(self)

    def get_permeability(self):
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_permeability(self)


if __name__ == '__main__':
    simulator = Simulator()
