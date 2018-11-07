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
            # self.solver_type
            self.solver_type = conf.get("solver", "solverType")
            if self.solver_type != 'J' and self.solver_type != 'GS' and self.solver_type != 'Time':
                raise Exception("param gas.solverType should be J, GS or Time!")

            # self.time_step
            try:
                self.time_step = float(conf.get("solver", "timeStep"))
                if self.time_step <= 0:
                    raise Exception("param solver.timeStep should be positive!")
            except configparser.NoOptionError:
                if self.solver_type != 'Time':
                    self.time_step = 0
                else:
                    raise Exception("param solver.timeStep should exists!")

            # self.scale_effect
            self.scale_effect = int(conf.get("solver", "scaleEffect"))
            if self.scale_effect != 0 and self.scale_effect != 1:
                raise Exception("param solver.scaleEffect should be 0 or 1!")

            # self.show_status
            self.show_status = int(conf.get("solver", "showStatus"))
            if self.show_status < 0:
                raise Exception("param solver.showStatus should be positive!")

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
        if self.solver_type == "Time":
            logging.info("时间步长：" + str(self.time_step))
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
        if self.solver_type == 'J' or self.solver_type == 'GS':
            self.ns.calculate_equation(self.sc, self.scale_effect, self.show_status, self.solver_type)
        else:
            self.ns.calculate_iteration(self.sc, self.time_step, self.scale_effect, self.show_status)
        self.iters += 1

    def get_mass_flux(self):
        """
        获取网络的质量流量
        :return:
        """
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_mass_flux(self.sc, self.scale_effect, 0)

    def get_permeability(self):
        if self.ns is None:
            raise Exception("请首先绑定网络状态数组！")
        else:
            return self.ns.get_permeability(self.sc, self.scale_effect)


if __name__ == '__main__':
    simulator = Simulator()
