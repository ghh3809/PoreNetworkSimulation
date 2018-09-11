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

from StatusCache import StatusCache
from utils import Tools


class Simulator(object):
    """
    求解器基本类
    """
    def __init__(self, config_file='../config/config.ini'):
        """
        利用配置创建求解器
        :param config_file: 配置文件
        """
        self.iters = 0
        self.sc = None
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

            # self.showStatus
            self.show_status = int(conf.get("solver", "showStatus"))
            if self.show_status < 0:
                raise Exception("param gas.showStatus should be positive!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        logging.info("------------------读取求解器配置文件------------------")
        logging.info("迭代方法: " + str(self.solver_type))
        logging.info("是否考虑尺度效应: " + ("是" if self.scale_effect == 1 else "否"))
        logging.info("是否显示求解进度: " + ("否" if self.show_status == 0 else ("是，总数 = " + str(self.show_status))))

    def bind_network_status(self, network_status, status_cache):
        """
        绑定网络状态到当前求解器
        :param network_status: 网络状态
        :param status_cache: 计算缓存
        :return:
        """
        self.ns = network_status
        self.sc = status_cache

    def iterate_once(self):
        """
        迭代计算一次
        :return:
        """
        c = 1.0 # 常系数

        # 迭代计算
        if self.solver_type == 'J':
            new_pressure = np.zeros(self.ns.model_size)
        else:
            new_pressure = self.ns.pressure

        for i in range(self.ns.model_size[0]):
            for j in range(self.ns.model_size[1]):
                for k in range(self.ns.model_size[2]):
                    if i == 0 and self.ns.sc.boundary_type[0] == 1:
                        new_pressure[i, j, k] = self.ns.sc.boundary_value[0]
                    elif i == self.ns.model_size[0] - 1 and self.ns.sc.boundary_type[1] == 1:
                        new_pressure[i, j, k] = self.ns.sc.boundary_value[1]
                    elif j == 0 and self.ns.sc.boundary_type[2] == 1:
                        new_pressure[i, j, k] = self.ns.sc.boundary_value[2]
                    elif j == self.ns.model_size[1] - 1 and self.ns.sc.boundary_type[3] == 1:
                        new_pressure[i, j, k] = self.ns.sc.boundary_value[3]
                    elif k == 0 and self.ns.sc.boundary_type[4] == 1:
                        new_pressure[i, j, k] = self.ns.sc.boundary_value[4]
                    elif k == self.ns.model_size[2] - 1 and self.ns.sc.boundary_type[5] == 1:
                        new_pressure[i, j, k] = self.ns.sc.boundary_value[5]
                    else:
                        sum_up = 0.0
                        sum_down = 0.0
                        ind1 = (i, j, k)
                        for l in range(26):
                            rp = Tools.Tools.get_relative_position(l)
                            ind2 = tuple(np.add(rp, (i, j, k)))
                            if (0 <= ind2[0] < self.ns.model_size[0]) \
                                    and (0 <= ind2[1] < self.ns.model_size[1]) \
                                    and (0 <= ind2[2] < self.ns.model_size[2]):
                                ind_t1 = tuple(ind1 + (l,))
                                p_ave = (self.ns.pressure[ind1] + self.ns.pressure[ind2]) / 2.0
                                if self.scale_effect:
                                    ff = 1.0 + 4.0 * c * self.sc.kn_cache[ind_t1] / p_ave
                                    f = 1.0 / (1.0 + 0.5 * self.sc.kn_cache[ind_t1] / p_ave)
                                    coef = (f * ff * self.sc.darcy_cache[ind_t1] * p_ave +
                                                    (1.0 - f) * self.sc.knudsen_cache[ind_t1]) * \
                                                   self.sc.length_cache[ind_t1] * self.ns.ns.weight[ind_t1]
                                else:
                                    coef = self.sc.darcy_cache[ind_t1] * p_ave * self.sc.length_cache[ind_t1] * \
                                                   self.ns.ns.weight[ind_t1]
                                sum_up += coef[i, j, k, l] * self.ns.pressure[ind2]
                                sum_down += coef[i, j, k, l]
                        if sum_down != 0:
                            new_pressure[i, j, k] = sum_up / sum_down

            # 显示计算进度
            for i in range(int(float(self.show_status) * float(i-1) / float(self.ns.model_size[0])),
                           int(float(self.show_status) * float( i ) / float(self.ns.model_size[0]))):
                sys.stdout.write("█")

        # 更新矩阵
        if self.solver_type == 'J':
            self.ns.pressure = new_pressure
        self.iters += 1

    def get_kn(self):
        """
        获取knudsen number
        :return:
        """
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_kn(self.sc)

    def get_mass_flux(self):
        """
        获取网络的质量流量
        :return:
        """
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_mass_flux(self.sc, self.scale_effect)

    def get_permeability(self):
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_permeability(self.sc, self.scale_effect)


if __name__ == '__main__':
    simulator = Simulator()
