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


class NetworkStructureHandler(object):
    """
    网络配置文件处理类
    """
    def __init__(self, config_file='./config/config'):
        """
        初始化
        :param config_file: 配置文件，默认位置'./config/config'
        """
        if not os.path.exists(config_file):
            logging.error("No network config file detected!")
            raise Exception("No network config file detected!")
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.model_size
            self.__model_size_str = conf.get("network", "modelSize").split(",")
            self.model_size = [int(x.strip()) for x in self.__model_size_str]
            if len(self.model_size) != 3:
                raise Exception("param network.modelSize should have length 3!")
            if min(self.model_size) <= 0:
                raise Exception("param network.modelSize should be positive!")

            # self.character_length
            self.character_length = float(conf.get("network", "characterLength"))
            if self.character_length <= 0:
                raise Exception("param network.character_length should be positive!")

            # self.radius_params
            self.__radius_params_str = conf.get("network", "radiusParams").split(",")
            self.radius_params = [float(x.strip()) for x in self.__radius_params_str]
            if len(self.radius_params) != 2:
                raise Exception("param network.radiusParams should have length 2!")
            if min(self.radius_params) <= 0:
                raise Exception("param network.radius_params should be positive!")

            # self.curvature
            try:
                self.curvature = float(conf.get("network", "curvature"))
                logging.info("Curvature params detect!")
                if self.curvature <= 0:
                    raise Exception("param network.curvature should be positive!")
            except configparser.NoOptionError:
                logging.info("Curvature params not detect!")
                self.curvature = 0

            # self.throat_params
            try:
                self.__throat_params_str = conf.get("network", "throatParams").split(",")
                logging.info("Throat params detect!")
                self.throat_params = [float(x.strip()) for x in self.__throat_params_str]
                if len(self.throat_params) != 2:
                    raise Exception("param network.throatParams should have length 2!")
                if min(self.throat_params) <= 0:
                    raise Exception("param network.throat_params should be positive!")
            except configparser.NoOptionError:
                logging.info("Throat params not detect!")
                if self.curvature != 0:
                    self.throat_params = list()
                else:
                    raise Exception("param network.curvature and network.throatParams should at least set one!")

            # self.coor_params
            self.__coor_params_str = conf.get("network", "coorParams").split(",")
            self.coor_params = [float(x.strip()) for x in self.__coor_params_str]
            if len(self.coor_params) != 2:
                raise Exception("param network.coorParams should have length 2!")
            if min(self.coor_params) <= 0:
                raise Exception("param network.coor_params should be positive!")
            if self.coor_params[0] > 26:
                raise Exception("param network.coor_params[0] should between 0-26!")

            # self.porosity
            self.porosity = float(conf.get("network", "porosity"))
            if self.porosity <= 0 or self.porosity >= 1:
                raise Exception("param network.porosity should between 0~1 !")

            # self.anisotropy
            try:
                self.__anisotropy_str = conf.get("network", "anisotropy").split(",")
                logging.info("Anisotropy params detect!")
                self.anisotropy = [float(x.strip()) for x in self.__anisotropy_str]
                if len(self.anisotropy) != 3:
                    raise Exception("param network.anisotropy should have length 3!")
                if min(self.anisotropy) <= 0:
                    raise Exception("param network.anisotropy should be positive!")
            except configparser.NoOptionError:
                logging.info("Anisotropy params not detect!")
                self.anisotropy = [1, 1, 1]

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        print "------------------读取网络结构配置文件------------------"
        print "模型尺寸:", self.model_size
        print "特征尺度(m):", self.character_length
        print "孔隙半径参数(lc):", self.radius_params
        if self.curvature != 0:
            print "孔喉曲率(lc):", self.curvature
        else:
            print "孔喉半径参数(lc):", self.throat_params
        print "配位数参数:", self.coor_params
        print "各向异性参数:", self.anisotropy


class NetworkStructure(object):
    """
    网络结构基本类
    """
    def __init__(self, network_config):
        """
        利用配置创建网络结构
        :param network_config: 配置实例
        """
        self.nc = network_config
        self.model_size = network_config.model_size
        self.character_length = network_config.character_length
        self.radii = None
        self.throatR = None
        self.weight = None
        self.unit_size = None
        self.initialize()

    def initialize(self):
        """
        初始化整个网络
        :return:
        """
        print "------------------初始化网络结构------------------"

        print "初始化孔隙尺寸……",
        self.radii = Tools.Tools.create_normal_dist(self.nc.radius_params[0], self.nc.radius_params[1], self.model_size)
        print "完成"
        print "平均孔隙尺寸(lc):", format(np.average(self.radii), '.4f')

        print "计算单元尺寸……",
        self.unit_size = self.calculate_unit_size()
        print "完成"
        print "单元尺寸(lc):", format(self.unit_size, '.4f')

        print "初始化喉道尺寸……",
        if self.nc.curvature:
            self.throatR = self.create_throat_from_curvature(self.nc.curvature, self.unit_size)
        else:
            self.throatR = Tools.Tools.create_normal_dist(self.nc.throat_params[0], self.nc.throat_params[1], self.model_size + [26])
        print "完成"
        print "平均喉道尺寸(lc):", format(np.average(self.throatR[1:-1, 1:-1, 1:-1, :]), '.4f')

        print "初始化连接矩阵……",
        target_coor = Tools.Tools.create_normal_dist(self.nc.coor_params[0], self.nc.coor_params[1], self.model_size)
        self.weight = self.create_weight_data(target_coor)
        print "完成"
        print "平均配位数:", format(np.average(np.sum(self.weight[1:-1, 1:-1, 1:-1, :], 3)), '.4f')
        print "平均权重:", format(np.average(self.weight[1:-1, 1:-1, 1:-1, :]), '.4f')

    def calculate_unit_size(self):
        """
        迭代法计算单元尺寸
        :return:
        """
        pore_volumn = 4.0 / 3.0 * np.pi * np.average(np.power(self.radii, 3))
        unit_size = np.power(pore_volumn / self.nc.porosity, 1.0/3.0)
        lo = unit_size
        hi = unit_size * 2.0
        iter_count = 0
        while np.abs(hi - lo) > 1:
            iter_count += 1
            if iter_count > 100:
                raise Exception("Solve unit size error: iteration more than 100!")
            unit_size = (lo + hi) / 2
            if self.nc.curvature:
                # 使用统计结论：在合理范围内，取r=μ-0.3σ得到的结果较准
                eff_r = self.nc.radius_params[0] - 0.3 * self.nc.radius_params[1]
                ave_sqr = np.square(self.cal_throat_radius(eff_r, eff_r, unit_size, self.nc.curvature))
            else:
                # E(X^2) = (E(X))^2 + D(X)
                ave_sqr = np.square(self.nc.throat_params[0]) + np.square(self.nc.throat_params[1])
            throat_length = unit_size - 2 * self.nc.radius_params[0]
            throat_volumn = np.pi * ave_sqr * self.nc.coor_params[0] * throat_length / 2.0
            error = (throat_volumn + pore_volumn) / np.power(unit_size, 3.0) - self.nc.porosity
            if error > 0:
                lo = unit_size
            else:
                hi = unit_size
        logging.info("Calculate unit size: total iteration times = %d" % iter_count)
        return unit_size

    def create_throat_from_curvature(self, curv, unit_size):
        """
        通过曲率数据创建喉道
        :param curv: 曲率
        :param unit_size: 单元半径
        :return: 喉道半径
        """
        throats = np.zeros(self.model_size + [26])
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
                            R1 = self.radii[ind1]
                            R2 = self.radii[ind2]
                            L = np.sqrt(np.sum(np.square(rp))) * unit_size
                            throats[i, j, k, l] = self.cal_throat_radius(R1, R2, L, curv)
        return throats

    @staticmethod
    def cal_throat_radius(R1, R2, L, curv):
        """
        公式：利用曲率计算喉道半径
        :param R1: 孔隙1半径
        :param R2: 孔隙2半径
        :param L: 孔隙间距离
        :param curv: 曲率
        :return:
        """
        tmp = np.sqrt(2) / 2
        T1 = R1 / L * tmp / np.power(1 - R1 / L * tmp, curv)
        T2 = R2 / L * tmp / np.power(1 - R2 / L * tmp, curv)
        return L * T1 * T2 * np.power(np.power(T1, 1 / curv) + np.power(T2, 1 / curv), -curv)

    def create_weight_data(self, target_coor):
        """
        创建权重数组
        :param target_coor: 目标权重
        :return: 连接权重数组(0~1)
        """
        weight = np.zeros(self.model_size + [26])
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
                            weight[i, j, k, l] = self.get_weight(target_coor[ind1], target_coor[ind2],
                                                                 self.nc.coor_params[0], self.nc.anisotropy, rp)
        return weight

    @staticmethod
    def get_weight(coor1, coor2, ave_coor, anisotropy=(1,1,1), rp=(0,0,1)):
        """
        利用配位数获取连接概率
        :param coor1: 配位数1
        :param coor2: 配位数2
        :param ave_coor: 平均配位数
        :param anisotropy: 各向异性
        :param rp: 相对位置
        :return:
        """
        a = coor1 / 26.0
        b = coor2 / 26.0
        p = ave_coor / 26.0
        q = (a * b / p) / ((a * b / p) + ((1 - a) * (1 - b) / (1 - p)))
        w = np.sum(np.multiply(np.divide(np.square(rp).astype(np.float64), np.sum(np.square(rp))), anisotropy)) \
            * q * 3 / np.sum(anisotropy)
        # 用于确定生成的网络
        # return 1 if w > np.random.rand() else 0
        # 用于全连接网络
        return w


if __name__ == '__main__':
    config_handler = NetworkStructureHandler()
    network_structure = NetworkStructure(config_handler)

