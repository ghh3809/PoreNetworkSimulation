#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: FluidConstant.py
Date: 2018/11/07 19:12:08
Desc: 流体常数类
"""

import os
import logging
import configparser


class FluidConstant(object):
    """
    网络配置文件处理类
    """
    def __init__(self, config_file='../config/config.ini'):
        """
        初始化
        :param config_file: 配置文件，默认位置'../config/config.ini'
        """
        if not os.path.exists(config_file):
            logging.error("No fluid config file detected!")
            raise Exception("No fluid config file detected!")
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.rou
            self.rou = float(conf.get("fluid", "rou"))
            if self.rou <= 0:
                raise Exception("param fluid.rou should be positive!")

            # self.u
            self.u = float(conf.get("fluid", "u"))
            if self.u <= 0:
                raise Exception("param fluid.u should be positive!")

            # self.max_theta
            self.max_theta = float(conf.get("fluid", "maxTheta"))
            if self.max_theta < 0:
                raise Exception("param fluid.maxTheta should be non-negative!")
            elif self.max_theta > 180:
                raise Exception("param fluid.maxTheta should be smaller than 180!")

            # self.min_theta
            self.min_theta = float(conf.get("fluid", "minTheta"))
            if self.min_theta < 0:
                raise Exception("param fluid.minTheta should be non-negative!")
            elif self.min_theta > 180:
                raise Exception("param fluid.minTheta should be smaller than 180!")
            elif self.min_theta > self.max_theta:
                raise Exception("param fluid.minTheta should be smaller than gas.maxTheta!")

            # self.u
            self.sigma= float(conf.get("fluid", "sigma"))
            if self.sigma <= 0:
                raise Exception("param fluid.sigma should be positive!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        logging.info("------------------读取流体参数配置文件------------------")
        logging.info("密度ρ(kg/m^3): " + str(self.rou))
        logging.info("粘度u(Pa·s): " + str(self.u))
        logging.info("前进接触角(degree): " + str(self.max_theta))
        logging.info("后退接触角(degree): " + str(self.min_theta))
        logging.info("表面张力系数σ(N/m): " + str(self.sigma))


if __name__ == '__main__':
    fluid_constant = FluidConstant()
