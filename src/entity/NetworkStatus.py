#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: NetworkStructure.py
Date: 2018/08/27 20:07:09
Desc: 网络结构基本表达类
"""

import os
import sys
import logging
import configparser
import numpy as np

sys.path.append('./')
from utils import Tools
from NetworkStructure import NetworkStructureHandler
from NetworkStructure import NetworkStructure
from GasConstant import GasConstant


class NetworkStatusHandler(object):
    """
    网络配置文件处理类
    """
    def __init__(self, config_file='./config/config'):
        """
        初始化
        :param config_file: 配置文件，默认位置'./config/config'
        """
        if not os.path.exists(config_file):
            logging.error("No status config file detected!")
            raise Exception("No status config file detected!")
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.boundary_type
            self.__boundary_type_str = conf.get("status", "boundaryType").split(",")
            self.boundary_type = [int(x.strip()) for x in self.__boundary_type_str]
            if len(self.boundary_type) != 6:
                raise Exception("param status.boundaryType should have length 6!")
            if min(self.boundary_type) < 0 or max(self.boundary_type) > 1:
                raise Exception("param status.boundaryType should be 0 or 1!")

            # self.boundary_value
            self.__boundary_value_str = conf.get("status", "boundaryValue").split(",")
            self.boundary_value = [float(x.strip()) for x in self.__boundary_value_str]
            if len(self.boundary_value) != 6:
                raise Exception("param status.boundaryValue should have length 6!")
            if min(self.boundary_value) < 0:
                raise Exception("param status.boundaryValue should be non-negative!")

            # self.initial_type
            self.initial_type = int(conf.get("status", "initialType"))
            if self.initial_type < 0 or self.initial_type > 3:
                raise Exception("param status.initial_type should be 0-3!")

            # self.initial_value
            if self.initial_type == 0:
                self.initial_value = float(conf.get("status", "initialValue"))
                if self.initial_value < 0:
                    raise Exception("param status.initialValue should be non-negative!")
            else:
                self.__initial_value_str = conf.get("status", "initialValue").split(",")
                self.initial_value = [float(x.strip()) for x in self.__initial_value_str]
                if len(self.initial_value) != 6:
                    raise Exception("param status.initialValue should have length 6!")
                if min(self.initial_value) < 0:
                    raise Exception("param status.initialValue should be non-negative!")
                if max(self.initial_value) == 0:
                    raise Exception("param status.initialValue should have at least 1 non-zero!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        print "------------------读取计算配置文件------------------"
        print "=== 边界条件 ==="
        print "-x侧: " + ("封闭边界" if self.boundary_type[0] == 0 else "定压力边界") + \
              (("，压力P = " + str(self.boundary_value[0]) + " Pa") if self.boundary_type[0] != 0 else "")
        print "+x侧: " + ("封闭边界" if self.boundary_type[1] == 0 else "定压力边界") + \
              (("，压力P = " + str(self.boundary_value[1]) + " Pa") if self.boundary_type[1] != 0 else "")
        print "-y侧: " + ("封闭边界" if self.boundary_type[2] == 0 else "定压力边界") + \
              (("，压力P = " + str(self.boundary_value[2]) + " Pa") if self.boundary_type[2] != 0 else "")
        print "+y侧: " + ("封闭边界" if self.boundary_type[3] == 0 else "定压力边界") + \
              (("，压力P = " + str(self.boundary_value[3]) + " Pa") if self.boundary_type[3] != 0 else "")
        print "-z侧: " + ("封闭边界" if self.boundary_type[4] == 0 else "定压力边界") + \
              (("，压力P = " + str(self.boundary_value[4]) + " Pa") if self.boundary_type[4] != 0 else "")
        print "+z侧: " + ("封闭边界" if self.boundary_type[5] == 0 else "定压力边界") + \
              (("，压力P = " + str(self.boundary_value[5]) + " Pa") if self.boundary_type[5] != 0 else "")
        print "=== 初始条件 ==="
        if self.initial_type == 0:
            print "常数压力P = " + str(self.initial_value) + " Pa"
        else:
            print "线性变化："
            if self.initial_value[0] + self.initial_value[1] > 0:
                print "-x侧: 压力P = " + str(self.initial_value[0]) + " Pa"
                print "    ||    ||    ||    ||"
                print "    \/    \/    \/    \/"
                print "+x侧: 压力P = " + str(self.initial_value[1]) + " Pa"
            elif self.initial_value[2] + self.initial_value[3] > 0:
                print "-y侧: 压力P = " + str(self.initial_value[2]) + " Pa"
                print "+y侧: 压力P = " + str(self.initial_value[3]) + " Pa"
            else:
                print "-z侧: 压力P = " + str(self.initial_value[4]) + " Pa"
                print "+z侧: 压力P = " + str(self.initial_value[5]) + " Pa"


class NetworkStatus(object):
    """
    网络结构基本类
    """
    def __init__(self, status_config, network_structure, gas_constant):
        """
        利用配置创建网络结构
        :param network_config: 配置实例
        """
        self.sc = status_config
        self.ns = network_structure
        self.gc = gas_constant
        self.model_size = self.ns.model_size
        self.pressure = None
        self.initialize_pressure()

    def initialize_pressure(self):
        """
        初始化整个网络
        :return:
        """
        print "------------------初始化网络状态------------------"
        print "初始化网络状态中……",
        if self.sc.initial_type == 0:
            self.pressure = np.multiply(np.ones(self.model_size), self.sc.initial_value)
        elif self.sc.initial_type == 1:
            # 公式: y = kx + C
            self.pressure = np.zeros(self.model_size)
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = (float(i) / float(self.model_size[0] - 1)) * (
                                        self.sc.initial_value[1] - self.sc.initial_value[0]) + self.sc.initial_value[0]
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = (float(j) / float(self.model_size[1] - 1)) * (
                                        self.sc.initial_value[3] - self.sc.initial_value[2]) + self.sc.initial_value[2]
                        else:
                            self.pressure[i, j, k] = (float(k) / float(self.model_size[2] - 1)) * (
                                        self.sc.initial_value[5] - self.sc.initial_value[4]) + self.sc.initial_value[4]
        elif self.sc.initial_type == 2:
            # 公式: P = sqrt(kx + C)
            self.pressure = np.zeros(self.model_size)
            if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                C = np.square(self.sc.initial_value[0])
                kk = (np.square(self.sc.initial_value[1]) - np.square(self.sc.initial_value[0])) / float(self.model_size[0] - 1)
            elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                C = np.square(self.sc.initial_value[2])
                kk = (np.square(self.sc.initial_value[3]) - np.square(self.sc.initial_value[2])) / float(self.model_size[1] - 1)
            else:
                C = np.square(self.sc.initial_value[4])
                kk = (np.square(self.sc.initial_value[5]) - np.square(self.sc.initial_value[4])) / float(self.model_size[2] - 1)
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = np.sqrt(kk * i + C)
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = np.sqrt(kk * j + C)
                        else:
                            self.pressure[i, j, k] = np.sqrt(kk * k + C)
        elif self.sc.initial_type == 3:
            self.pressure = np.zeros(self.model_size)
            if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                layer_pressure = self.cal_layer_pressure(self.sc.initial_value[0], self.sc.initial_value[1],
                                                         self.model_size[0])
            elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                layer_pressure = self.cal_layer_pressure(self.sc.initial_value[2], self.sc.initial_value[3],
                                                         self.model_size[1])
            else:
                layer_pressure = self.cal_layer_pressure(self.sc.initial_value[4], self.sc.initial_value[5],
                                                         self.model_size[2])
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = layer_pressure[i]
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = layer_pressure[j]
                        else:
                            self.pressure[i, j, k] = layer_pressure[k]
        print "完成"

    def get_kn(self, simulator):
        """
        获取当前的knudsen number
        :param simulator:
        :return:
        """
        return np.average(np.divide(np.average(simulator.kn_cache, 3), self.pressure)[1:-1, 1:-1, 1:-1])

    def get_mass_flux(self, simulator):
        """
        获取当前状态的质量流量
        :param simulator: 求解器实例，主要为了获取求解器中的缓存
        :return:
        """
        c = 1.0
        deltaPV = np.zeros(self.model_size + [26])
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    ind1 = (i, j, k)
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        if (0 <= ind2[0] < self.model_size[0]) \
                                and (0 <= ind2[1] < self.model_size[1]) \
                                and (0 <= ind2[2] < self.model_size[2]):
                            indt = tuple(ind1 + (l,))
                            p_ave = (self.pressure[ind1] + self.pressure[ind2]) / 2.0
                            if simulator.scale_effect:
                                F = 1.0 + 4.0 * c * simulator.kn_cache[indt] / p_ave
                                f = 1.0 / (1.0 + 0.5 * simulator.kn_cache[indt] / p_ave)
                                deltaPV[indt] = (f * F * simulator.darcy_cache[indt] * p_ave + (1.0 - f) * simulator.knudsen_cache[indt]) \
                                      * (self.pressure[ind1] - self.pressure[ind2]) * simulator.length_cache[indt] * self.ns.weight[indt]
                            else:
                                deltaPV[indt] = simulator.darcy_cache[indt] * p_ave * (self.pressure[ind1] - self.pressure[ind2]) \
                                      * simulator.length_cache[indt] * self.ns.weight[indt]
        return deltaPV * (self.gc.M / (self.gc.R * self.gc.T))

    def get_permeability(self, simulator):
        """
        获取当前渗透率
        :param simulator: 求解器，主要为了获取计算缓存
        :return:
        """
        perm_coef = np.abs(2 * self.gc.u * (self.model_size[0] - 1) * self.gc.R * self.gc.T / \
                           (self.ns.character_length * self.ns.unit_size * self.gc.M *
                            (self.sc.boundary_value[0] ** 2 - self.sc.boundary_value[1] ** 2)))
        ave_mass_flux = np.abs(np.average(np.sum(self.get_mass_flux(simulator), 3)[0, :, :]))
        return perm_coef * ave_mass_flux

    def __get_ave_kn_coef(self):
        """
        计算kn缓存: KnCoef = sqrt(πRT/2M) * μ / r，计算时有 Kn = KnCoef / p
        :return:
        """
        kn_cache = np.zeros(self.ns.model_size + [26])
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    ind1 = (i, j, k)
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        if (0 <= ind2[0] < self.model_size[0]) \
                                and (0 <= ind2[1] < self.model_size[1]) \
                                and (0 <= ind2[2] < self.model_size[2]):
                            throat_r = self.ns.throatR[i, j, k, l] * self.ns.character_length
                            kn_cache[i, j, k, l] = np.sqrt(
                                np.pi * self.gc.R * self.gc.T / 2.0 / self.gc.M) * self.gc.u / throat_r
        return np.average(kn_cache[1:-1, 1:-1, 1:-1, :])

    @staticmethod
    def __cal_eff_mass_flux(pressure1, pressure2, ave_kn_coef):
        """
        计算等效质量流量
        :param pressure1: 压力1
        :param pressure2: 压力2
        :return:
        """
        ave_p = (pressure1 + pressure2) / 2.0
        kn = ave_kn_coef / ave_p
        f = 1.0 / (1.0 + 0.5 * kn)
        return ((1.0 + 4.0 * kn) * f + 64.0 / 3.0 / np.pi * kn * (1.0 - f)) * ave_p * (pressure1 - pressure2)

    def cal_layer_pressure(self, pressure1, pressure2, size):
        """
        计算各层初始压力
        :param pressure1: 压力1
        :param pressure2: 压力2
        :param size: 总层数
        :return:
        """
        pressure = np.zeros(size)
        C = np.square(pressure1)
        kk = (np.square(pressure2) - np.square(pressure1)) / float(size - 1)
        for i in range(size):
            pressure[i] = np.sqrt(kk * i + C)
        ave_kn_coef = self.__get_ave_kn_coef()
        while True:
            pressure_old = pressure.copy()
            for i in range(1, size - 1):
                flux1 = self.__cal_eff_mass_flux(pressure[i - 1], pressure[i], ave_kn_coef)
                flux2 = self.__cal_eff_mass_flux(pressure[i], pressure[i + 1], ave_kn_coef)
                ratio = flux1 / flux2 * (pressure[i] - pressure[i + 1]) / (pressure[i - 1] - pressure[i])
                pressure[i] = pressure[i + 1] + (ratio / (1 + ratio)) * (pressure[i - 1] - pressure[i + 1])
            if np.max(np.abs(np.subtract(pressure, pressure_old))) < 1:
                break
        return pressure


if __name__ == '__main__':
    gas_constant = GasConstant()
    network_structure = NetworkStructure(NetworkStructureHandler())
    network_status = NetworkStatus(NetworkStatusHandler(), network_structure, gas_constant)
    print network_status.pressure[:, 1, 1]

