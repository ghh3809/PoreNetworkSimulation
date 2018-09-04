#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: ClaculatePermeability.py
Date: 2018/08/31 00:18:31
Desc: 计算渗透率
"""

import os
import sys
import logging
import cPickle
import time
import collections
import configparser
import numpy as np

sys.path.append('./')
from utils import Tools
from entity import GasConstant as Gas
from entity import NetworkStructure as Structure
from entity import NetworkStatus as Status
from entity import Simulator as Simu


class SeepageIterator(object):

    def __init__(self, config_file='./config/config.ini', status_file_name=None, iters=0):
        """
        利用配置创建迭代器
        :param config_file: 配置文件
        """

        if status_file_name is None:
            self.network_status = self.create_status_from_conf()
        else:
            self.network_status = self.create_status_from_file(status_file_name)
        self.simulator = self.create_simulator_from_conf()
        self.simulator.bind_network_status(self.network_status)
        self.simulator.iters = iters
        self.permeability = [0, 0, 0]

        if not os.path.exists(config_file):
            logging.error("No iteration config file detected!")
            raise Exception("No iteration config file detected!")
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.show_permeability
            self.show_permeability = int(conf.get("iteration", "showPermeability"))
            if self.show_permeability < 0:
                raise Exception("param gas.showPermeability should be non-negative!")

            # self.save
            self.save = int(conf.get("iteration", "save"))
            if self.save < 0:
                raise Exception("param gas.save should be non-negative!")

            # self.file_name
            self.file_name = conf.get("iteration", "fileName")

            # self.maxDeltaP
            self.max_delta_p = float(conf.get("iteration", "maxDeltaP"))
            if self.max_delta_p <= 0:
                raise Exception("param gas.maxDeltaP should be positive!")

            # self.max_delta_p
            self.ave_delta_p = float(conf.get("iteration", "aveDeltaP"))
            if self.ave_delta_p < 0:
                raise Exception("param gas.aveDeltaP should be positive!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印配置
        :return:
        """
        print "------------------读取迭代器配置文件------------------"
        print "渗透率计算:", "否" if self.show_permeability == 0 else ("每 " + str(self.show_permeability) + " 次迭代")
        print "保存结果:", "否" if self.save == 0 else ("每 " + str(self.save) + " 次迭代")
        print "文件名前缀:", self.file_name
        print "迭代终止条件（满足任意一个即可）:"
        print "1. 单次迭代最大压力变化小于", str(self.max_delta_p), "Pa"
        print "1. 单次迭代平均压力变化小于", str(self.ave_delta_p), "Pa"

    def cal_final_perm(self):
        """
        获取可预见的最终渗透率（以指数分布为例计算）
        :return:
        """
        delta1 = self.permeability[2] - self.permeability[1]
        delta2 = self.permeability[1] - self.permeability[0]
        # 注意退化情况
        if delta1 == 0 or delta1 == delta2:
            return 0
        ratio = float(delta2 / delta1)
        return self.permeability[0] - (ratio / (1 - ratio) * delta2)

    @staticmethod
    def create_status_from_conf(gas_conf_file='./config/config.ini',
                                structure_conf_file='./config/config.ini',
                                status_conf_file='./config/config.ini'):
        """
        从配置文件新建网络状态
        :return:
        """
        gas_constant = Gas.GasConstant(gas_conf_file)
        network_structure = Structure.NetworkStructure(Structure.NetworkStructureHandler(structure_conf_file))
        return Status.NetworkStatus(Status.NetworkStatusHandler(status_conf_file), network_structure, gas_constant)

    @staticmethod
    def create_status_from_file(file_name):
        """
        从保存的网络状态文件加载状态
        :param file_name: 文件名
        :return:
        """
        print "------------------初始化网络状态------------------"
        print "从文件重建网络状态中……",
        if not os.path.exists(file_name):
            raise Exception("No network status file detected!")
        with open(file_name, 'r') as f:
            network_status = cPickle.load(f)
        print "完成"
        return network_status

    @staticmethod
    def create_simulator_from_conf(simulator_conf_file='./config/config.ini'):
        """
        从配置文件加载求解器
        :return:
        """
        return Simu.Simulator(simulator_conf_file)

    def iterate_and_create_output_str(self):
        """
        返回需要输出的字符串
        :return: namedtuple
        """
        # 迭代计算常规输出
        last_p = self.simulator.ns.pressure.copy()
        self.simulator.iterate_once()
        ave_delta_p = np.average(np.abs(self.simulator.ns.pressure - last_p))
        max_delta_p = np.max(np.abs(self.simulator.ns.pressure - last_p))
        final_perm = 0

        output_str = "Iter = " + str(self.simulator.iters)
        output_str += ", Max △P = " + format(max_delta_p, '.4f') + " Pa"
        output_str += ", Ave △P = " + format(ave_delta_p, '.4f') + " Pa"

        # 渗透率计算输出
        if self.show_permeability > 0 and self.simulator.iters % self.show_permeability == 0:
            self.permeability[2] = self.permeability[1]
            self.permeability[1] = self.permeability[0]
            self.permeability[0] = self.simulator.get_permeability()
            final_perm = self.cal_final_perm()
            output_str += ", Kperm = " + format(self.permeability[0], '.4e') + " m^2"
            output_str += ", Final Kperm = " + format(final_perm, '.4e') + " m^2"

        print output_str
        logging.info(output_str)

        # 返回迭代信息
        IterationStatus = collections.namedtuple('IterationStatus', ['ave_delta_p', 'max_delta_p', 'final_perm'])
        return IterationStatus(ave_delta_p, max_delta_p, final_perm)

    def save_and_clear_status(self, final=False):
        """
        保存网络状态，并删除上次的网络状态
        :param final: 是否为最终状态
        :return:
        """
        if self.simulator.iters % self.save == 0 or final:

            # 判断文件名
            if final:
                new_file_name = './data/' + self.file_name + '_networkStatus.obj'
                old_file_name = './data/' + self.file_name + '_networkStatus_' + str(self.simulator.iters - self.simulator.iters % self.save) + '.obj'
            else:
                new_file_name = './data/' + self.file_name + '_networkStatus_' + str(self.simulator.iters) + '.obj'
                old_file_name = './data/' + self.file_name + '_networkStatus_' + str(self.simulator.iters - self.save) + '.obj'

            # 保存新文件
            logging.info("Save status to file: " + new_file_name)
            with open(new_file_name, 'w') as f:
                cPickle.dump(self.network_status, f)

            # 删除旧文件
            if os.path.exists(old_file_name):
                logging.info("Delete status file: " + old_file_name)
                os.remove(old_file_name)

    def start_simulation(self):
        """
        开始模拟
        :return:
        """

        # Part 1: 输出当前Kn，判断是否符合预期
        print "------------------初始状态------------------"
        print "Kn =", format(self.simulator.get_kn(), '.4f')
        if self.show_permeability > 0:
            print "Kperm =", format(self.simulator.get_permeability(), '.4e'), "m^2"
        time.sleep(2)

        # Part 2: 进行迭代计算
        print "------------------迭代计算------------------"

        while True:

            # 常规迭代输出
            iteration_status = self.iterate_and_create_output_str()

            # 周期输出/保存数据文件
            self.save_and_clear_status()

            # 迭代终止条件
            if iteration_status[0] < self.ave_delta_p or iteration_status[1] < self.max_delta_p:
                self.save_and_clear_status(True)
                break


if __name__ == '__main__':
    SeepageIterator(status_file_name='./data/dispersion300_networkStatus_1000.obj', iters=1000).start_simulation()
