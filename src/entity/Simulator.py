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

            # self.max_time_step
            try:
                self.max_time_step = float(conf.get("solver", "maxTimeStep"))
                if self.max_time_step <= 0:
                    raise Exception("param solver.maxTimeStep should be positive!")
            except configparser.NoOptionError:
                if self.solver_type != 'Time':
                    self.max_time_step = 0
                else:
                    raise Exception("param solver.maxTimeStep should exists!")

            # self.min_time_step
            try:
                self.min_time_step = float(conf.get("solver", "minTimeStep"))
                if self.min_time_step <= 0:
                    raise Exception("param solver.minTimeStep should be positive!")
            except configparser.NoOptionError:
                if self.solver_type != 'Time':
                    self.min_time_step = 0
                else:
                    raise Exception("param solver.minTimeStep should exists!")

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
            logging.info("最大时间步长(s)：" + str(self.max_time_step))
            logging.info("最小时间步长(s)：" + str(self.min_time_step))
        logging.info("是否考虑尺度效应: " + ("是" if self.scale_effect == 1 else "否"))
        logging.info("是否显示求解进度: " + ("否" if self.show_status == 0 else ("是，总数 = " + str(self.show_status))))

    def bind_network_status(self, network_status):
        """
        绑定网络状态到当前求解器
        :param network_status: 网络状态
        :return:
        """
        self.ns = network_status

    def iterate_once(self):
        """
        迭代计算一次
        :return:
        """
        if self.solver_type == 'J' or self.solver_type == 'GS':
            self.ns.calculate_equation(self.scale_effect, self.show_status, self.solver_type)
        elif self.solver_type == 'Time':
            self.ns.calculate_iteration(self.max_time_step, self.min_time_step, self.scale_effect)
        else:
            raise Exception("Method can only be J or GS!")
        self.iters += 1


if __name__ == '__main__':
    simulator = Simulator()
