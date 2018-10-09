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

from utils import Tools
from NetworkStructure import NetworkStructureHandler
from NetworkStructure import NetworkStructure
from GasConstant import GasConstant


class NetworkStatusHandler(object):
    """
    网络配置文件处理类
    """
    def __init__(self, config_file='../config/config.ini'):
        """
        初始化
        :param config_file: 配置文件，默认位置'../config/config.ini'
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
        logging.info("------------------读取计算配置文件------------------")
        logging.info("=== 边界条件 ===")
        logging.info("-x侧: " + ("封闭边界" if self.boundary_type[0] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[0]) + " Pa") if self.boundary_type[0] != 0 else ""))
        logging.info("+x侧: " + ("封闭边界" if self.boundary_type[1] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[1]) + " Pa") if self.boundary_type[1] != 0 else ""))
        logging.info("-y侧: " + ("封闭边界" if self.boundary_type[2] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[2]) + " Pa") if self.boundary_type[2] != 0 else ""))
        logging.info("+y侧: " + ("封闭边界" if self.boundary_type[3] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[3]) + " Pa") if self.boundary_type[3] != 0 else ""))
        logging.info("-z侧: " + ("封闭边界" if self.boundary_type[4] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[4]) + " Pa") if self.boundary_type[4] != 0 else ""))
        logging.info("+z侧: " + ("封闭边界" if self.boundary_type[5] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[5]) + " Pa") if self.boundary_type[5] != 0 else ""))
        logging.info("=== 初始条件 ===")
        if self.initial_type == 0:
            logging.info("常数压力P = " + str(self.initial_value) + " Pa")
        else:
            logging.info("线性变化:")
            if self.initial_value[0] + self.initial_value[1] > 0:
                logging.info("-x侧: 压力P = " + str(self.initial_value[0]) + " Pa")
                logging.info("    ||    ||    ||    ||")
                logging.info("    \/    \/    \/    \/")
                logging.info("+x侧: 压力P = " + str(self.initial_value[1]) + " Pa")
            elif self.initial_value[2] + self.initial_value[3] > 0:
                logging.info("-y侧: 压力P = " + str(self.initial_value[2]) + " Pa")
                logging.info("    ||    ||    ||    ||")
                logging.info("    \/    \/    \/    \/")
                logging.info("+y侧: 压力P = " + str(self.initial_value[3]) + " Pa")
            else:
                logging.info("-z侧: 压力P = " + str(self.initial_value[4]) + " Pa")
                logging.info("    ||    ||    ||    ||")
                logging.info("    \/    \/    \/    \/")
                logging.info("+z侧: 压力P = " + str(self.initial_value[5]) + " Pa")


class NetworkStatus(object):
    """
    网络结构基本类
    """
    def __init__(self, status_config, network_structure, gas_constant):
        """
        利用配置创建网络结构
        :param status_config: 配置实例
        :param network_structure: 网络结构
        :param gas_constant: 气体参数
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
        logging.info("------------------初始化网络状态------------------")
        logging.info("初始化网络状态中……")
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
                c = np.square(self.sc.initial_value[0])
                kk = (np.square(self.sc.initial_value[1]) - np.square(self.sc.initial_value[0])) /\
                    float(self.model_size[0] - 1)
            elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                c = np.square(self.sc.initial_value[2])
                kk = (np.square(self.sc.initial_value[3]) - np.square(self.sc.initial_value[2])) /\
                    float(self.model_size[1] - 1)
            else:
                c = np.square(self.sc.initial_value[4])
                kk = (np.square(self.sc.initial_value[5]) - np.square(self.sc.initial_value[4])) /\
                    float(self.model_size[2] - 1)
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = np.sqrt(kk * i + c)
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = np.sqrt(kk * j + c)
                        else:
                            self.pressure[i, j, k] = np.sqrt(kk * k + c)
        elif self.sc.initial_type == 3:
            self.pressure = np.zeros(self.model_size)
            if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                layer_pressure = self.__cal_layer_pressure(self.sc.initial_value[0], self.sc.initial_value[1],
                                                           self.model_size[0])
            elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                layer_pressure = self.__cal_layer_pressure(self.sc.initial_value[2], self.sc.initial_value[3],
                                                           self.model_size[1])
            else:
                layer_pressure = self.__cal_layer_pressure(self.sc.initial_value[4], self.sc.initial_value[5],
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
        logging.info("初始化压力结果：")
        ave_pressure = np.average(np.average(self.pressure, 2), 1)
        print_str = "    ["
        for p in ave_pressure:
            print_str += format(p, ".4e") + ", "
        print_str += "]"
        logging.debug(print_str)

    def get_kn(self, status_cache):
        """
        获取当前的knudsen number
        :param status_cache: 计算缓存
        :return:
        """
        return np.average(np.divide(np.average(status_cache.kn_cache, 3), self.pressure)[1:-1, 1:-1, 1:-1])

    def get_mass_flux(self, status_cache, scale_effect):
        """
        获取当前状态的质量流量
        :param status_cache: 计算缓存
        :param scale_effect: 是否考虑尺度效应
        :return: mass_flux, velocity
        """
        c = 1.0
        delta_pv = np.zeros(self.model_size + [26])
        velocity = np.zeros(self.model_size + [26])
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
                            if scale_effect != 0:
                                ff = 1.0 + 4.0 * c * status_cache.kn_cache[indt] / p_ave
                                f = 1.0 / (1.0 + 0.5 * status_cache.kn_cache[indt] / p_ave)
                                delta_pv[indt] = (f * ff * status_cache.darcy_cache[indt] * p_ave +
                                                  (1.0 - f) * status_cache.knudsen_cache[indt]) * \
                                    (self.pressure[ind1] - self.pressure[ind2]) * \
                                    status_cache.length_cache[indt] * self.ns.weight[indt]
                            else:
                                delta_pv[indt] = status_cache.darcy_cache[indt] * p_ave * \
                                                 (self.pressure[ind1] - self.pressure[ind2]) * \
                                                 status_cache.length_cache[indt] * self.ns.weight[indt]
                            velocity[indt] = delta_pv[indt] / p_ave / np.pi / \
                                np.square(self.ns.throatR[indt] * self.ns.character_length)
            logging.debug("    计算流量中，当前进度 = " + format(float(i) / float(self.model_size[0]) * 100.0, '.2f') + "%")
        return delta_pv * (self.gc.M / (self.gc.R * self.gc.T)), velocity

    def get_permeability(self, status_cache, scale_effect):
        """
        获取当前渗透率
        :param status_cache: 计算缓存
        :param scale_effect: 尺度效应
        :return:
        """
        perm_coef = np.abs(2 * self.gc.u * (self.model_size[0] - 1) * self.gc.R * self.gc.T /
                           (self.ns.character_length * self.ns.unit_size * self.gc.M *
                            (self.sc.boundary_value[0] ** 2 - self.sc.boundary_value[1] ** 2))) / \
                        (1.0 - 1.0 / np.sqrt(self.model_size[1] * self.model_size[2]))
        mass_flux, velocity = self.get_mass_flux(status_cache, scale_effect)
        ave_mass_flux = np.abs(np.average(np.sum(mass_flux, 3)[0, :, :]))
        return perm_coef * ave_mass_flux

    def __cal_layer_pressure(self, pressure1, pressure2, size):
        """
        计算各层初始压力
        :param pressure1: 压力1
        :param pressure2: 压力2
        :param size: 总层数
        :return:
        """
        pressure = np.zeros(size)
        c = np.square(pressure1)
        kk = (np.square(pressure2) - np.square(pressure1)) / float(size - 1)
        for i in range(size):
            pressure[i] = np.sqrt(kk * i + c)
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

    def __get_ave_kn_coef(self):
        """
        计算kn缓存: KnCoef = sqrt(πRT/2M) * μ / r，计算时有 Kn = KnCoef / p
        :return:
        """
        kn_cache = np.zeros(self.ns.model_size + [26])
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
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


if __name__ == '__main__':
    network_status = NetworkStatus(NetworkStatusHandler(),
                                   NetworkStructure(NetworkStructureHandler()),
                                   GasConstant())
