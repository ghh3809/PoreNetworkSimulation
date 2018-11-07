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
            # self.exist
            self.exist = int(conf.get("fluid", "exist"))
            if self.exist != 0:
                self.exist = 1

            # self.rou
            self.rou = float(conf.get("fluid", "rou"))
            if self.rou <= 0:
                raise Exception("param fluid.rou should be positive!")

            # self.u
            self.u = float(conf.get("gas", "u"))
            if self.u <= 0:
                raise Exception("param gas.u should be positive!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        if self.exist != 0:
            logging.info("------------------读取流体参数配置文件------------------")
            logging.info("密度ρ(kg/m^3): " + str(self.rou))
            logging.info("粘度u(Pa·s): " + str(self.u))


if __name__ == '__main__':
    fluid_constant = FluidConstant()
