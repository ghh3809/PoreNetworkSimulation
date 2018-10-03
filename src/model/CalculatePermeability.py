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
import collections
import configparser
import numpy as np

sys.path.append('../')

from utils import Tools as Tools
from entity import GasConstant as Gas
from entity import NetworkStructure as Structure
from entity import NetworkStatus as Status
from entity import Simulator as Simu
from entity import StatusCache as Cache


logging_flag = False
if __name__ == "__main__":
    logging_flag = True


class SeepageIterator(object):

    def __init__(self, status_file_name=None, iters=0, cache_file_name=None, config_file='../config/config.ini'):
        """
        利用配置创建迭代器
        :param config_file: 配置文件
        """
        # 读取配置文件
        if not os.path.exists(config_file):
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
            self.file_name = "seepage_" + conf.get("iteration", "fileName")
            if logging_flag:
                Tools.Tools.set_logging('../log/' + self.file_name + '.log')

            # self.maxDeltaP
            self.max_delta_p = float(conf.get("iteration", "maxDeltaP"))
            if self.max_delta_p <= 0:
                raise Exception("param gas.maxDeltaP should be positive!")

            # self.max_delta_p
            self.ave_delta_p = float(conf.get("iteration", "aveDeltaP"))
            if self.ave_delta_p < 0:
                raise Exception("param gas.aveDeltaP should be positive!")

        except Exception as e:
            raise Exception("Load config fail: [" + str(e) + "]")

        self.simulator = self.create_simulator(status_file_name, iters, cache_file_name)
        self.permeability = [0, 0, 0]

        self.print_config()

    def print_config(self):
        """
        打印配置
        :return:
        """
        logging.info("------------------读取迭代器配置文件------------------")
        logging.info("渗透率计算: " + ("否" if self.show_permeability == 0 else
                                       ("每 " + str(self.show_permeability) + " 次迭代")))
        logging.info("保存中间结果: " + ("否" if self.save == 0 else ("每 " + str(self.save) + " 次迭代")))
        logging.info("文件名前缀: " + str(self.file_name))
        logging.info("迭代终止条件（满足任意一个即可）:")
        logging.info("1. 单次迭代最大压力变化小于" + str(self.max_delta_p))
        logging.info("2. 单次迭代平均压力变化小于" + str(self.ave_delta_p))

    def start_simulation(self):
        """
        开始模拟
        :return:
        """
        # 进行迭代计算
        logging.info("------------------迭代计算------------------")
        while True:
            # 常规迭代输出
            iteration_status = self.iterate_and_create_output_str()
            # 周期输出/保存数据文件
            self.save_and_clear_status()
            # 迭代终止条件
            if iteration_status[0] < self.ave_delta_p or iteration_status[1] < self.max_delta_p:
                self.save_and_clear_status(True)
                break

    def create_simulator(self, status_file_name, iters, cache_file_name):
        """
        从配置文件加载求解器
        :param status_file_name: 网络状态文件
        :param iters: 迭代次数
        :param cache_file_name: 缓存文件名
        :return:
        """
        # 创建网络状态
        if status_file_name is None or not os.path.exists(status_file_name):
            if status_file_name is not None:
                logging.warning("Given status file not exists, please check your program!")
            gas_constant = Gas.GasConstant()
            network_structure = Structure.NetworkStructure(Structure.NetworkStructureHandler())
            network_status = Status.NetworkStatus(Status.NetworkStatusHandler(), network_structure, gas_constant)
        else:
            logging.info("------------------初始化网络状态------------------")
            logging.info("从文件重建网络状态中……")
            with open(status_file_name, 'r') as f:
                network_status = cPickle.load(f)
            network_status.gc.print_config()
            network_status.ns.nc.print_config()
            network_status.sc.print_config()

        # 判断缓存类
        if cache_file_name is None or not os.path.exists(cache_file_name):
            if status_file_name is not None:
                logging.warning("Given cache file not exists, please check your program!")
            status_cache = Cache.StatusCache(network_status)
        else:
            logging.info("从文件重建计算缓存中……")
            with open(cache_file_name, 'r') as f:
                status_cache = cPickle.load(f)

        # 绑定网络状态与缓存
        simulator = Simu.Simulator()
        simulator.bind_network_status(network_status, status_cache)
        simulator.iters = iters

        # 保存网络缓存
        cache_path = '../data/' + self.file_name + '_cache.obj'
        if not os.path.exists(cache_path):
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            with open(cache_path, 'w') as f:
                cPickle.dump(simulator.sc, f)
        else:
            logging.info("Cache file already exists!")

        return simulator

    def print_initial_status(self):
        """
        输出初始状态
        :return:
        """
        logging.info("------------------初始状态------------------")
        logging.info("Kn = " + format(self.simulator.get_kn(), '.4f'))
        if self.show_permeability > 0:
            logging.info("Kperm =" + format(self.simulator.get_permeability(), '.4e') + " m^2")

    def iterate_and_create_output_str(self):
        """
        返回需要输出的字符串
        :return: namedtuple
        """
        # 迭代计算
        last_p = self.simulator.ns.pressure.copy()
        self.simulator.iterate_once()
        ave_delta_p = np.average(np.abs(self.simulator.ns.pressure - last_p))
        max_delta_p = np.max(np.abs(self.simulator.ns.pressure - last_p))
        final_perm = 0

        # 常规输出
        output_str = "Iter = " + str(self.simulator.iters)
        output_str += ", Max △P = " + format(max_delta_p, '.4f') + " Pa"
        output_str += ", Ave △P = " + format(ave_delta_p, '.4f') + " Pa"

        # 渗透率计算输出
        if self.show_permeability > 0 and self.simulator.iters % self.show_permeability == 0:
            self.permeability[2] = self.permeability[1]
            self.permeability[1] = self.permeability[0]
            self.permeability[0] = self.simulator.get_permeability()
            final_perm = self.__cal_final_perm()
            output_str += ", Kperm = " + format(self.permeability[0], '.4e') + " m^2"
            output_str += ", Final Kperm = " + format(final_perm, '.4e') + " m^2"

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
        if (self.save != 0 and self.simulator.iters % self.save == 0) or final:

            # 判断文件名
            if final:
                new_file_name = '../data/' + self.file_name + '_status.obj'
                old_file_name = '../data/' + self.file_name + '_status_' + \
                    str(self.simulator.iters - ((self.simulator.iters % self.save) if self.save != 0 else 0)) + '.obj'
            else:
                new_file_name = '../data/' + self.file_name + '_status_' + str(self.simulator.iters) + '.obj'
                old_file_name = '../data/' + self.file_name + '_status_' + \
                    str(self.simulator.iters - self.save) + '.obj'

            # 保存新文件
            logging.info("保存网络状态: " + str(new_file_name))
            if not os.path.exists(os.path.dirname(new_file_name)):
                os.makedirs(os.path.dirname(new_file_name))
            with open(new_file_name, 'w') as f:
                cPickle.dump(self.simulator.ns, f)

            # 删除旧文件
            if os.path.exists(old_file_name):
                logging.info("删除网络状态: " + str(old_file_name))
                os.remove(old_file_name)

    def __cal_final_perm(self):
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


if __name__ == '__main__':
    SeepageIterator().start_simulation()
