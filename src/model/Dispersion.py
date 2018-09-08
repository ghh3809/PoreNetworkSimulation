#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved #
#################################################################################
"""
File: Dispersion.py
Date: 2018/09/03 19:41:55
Desc: 计算机械弥散效果
"""

import os
import sys
import logging
import cPickle
import configparser
import numpy as np

sys.path.append('../')

from utils import Tools
from entity import StatusCache as Cache
from CalculatePermeability import SeepageIterator


class DispersionSolver(object):

    def __init__(self, status_file_name=None, cache_file_name=None, config_file='../config/config.ini'):
        """
        利用配置创建迭代器
        :param status_file_name: 网络状态文件
        :param cache_file_name: 计算缓存文件
        :param config_file: 配置文件
        """
        self.status = None
        self.cache = None

        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.file_name
            self.file_name = "dispersion_" + conf.get("dispersion", "fileName")
            Tools.Tools.set_logging('../log/' + self.file_name + '.log')

        except Exception as e:
            raise Exception("Load config fail: [" + str(e) + "]")

        self.print_config()

        # 创建网络状态
        logging.info("------------------初始化网络状态------------------")
        if status_file_name is None or not os.path.exists(status_file_name):
            logging.info("未发现网络状态文件，正在重新计算中……")
            seepage_iterator = SeepageIterator()
            seepage_iterator.start_simulation()
            status_file_name = "../data/" + seepage_iterator.file_name + "_status.obj"
            cache_file_name = "../data/" + seepage_iterator.file_name + "_cache.obj"

        if os.path.exists(status_file_name):
            logging.info("从文件重建网络状态中……")
            with open(status_file_name, 'r') as f:
                self.status = cPickle.load(f)
        else:
            raise Exception("Cannot load status file!")
        self.model_size = self.status.model_size

        if cache_file_name is None or not os.path.exists(cache_file_name):
            logging.info("未发现计算缓存文件，正在重新计算中……")
            self.cache = Cache.StatusCache(self.status)
            new_file_name = "../data/" + self.file_name + "_cache.obj"
            logging.info("保存计算缓存: " + str(new_file_name))
            with open(new_file_name, 'w') as f:
                cPickle.dump(self.cache, f)
        else:
            logging.info("从文件重建计算缓存中……")
            with open(cache_file_name, 'r') as f:
                self.cache = cPickle.load(f)

        # 计算网络流量场
        mass_flux_file = "../data/" + self.file_name + "_massflux.obj"
        velocity_file = "../data/" + self.file_name + "_velocity.obj"
        if not os.path.exists(mass_flux_file) or not os.path.exists(velocity_file):
            logging.info("计算网络流量场中……")
            self.mass_flux, self.velocity = self.status.get_mass_flux(self.cache, 0)
            with open(mass_flux_file, 'w') as f:
                cPickle.dump(self.mass_flux, f)
            with open(velocity_file, 'w') as f:
                cPickle.dump(self.velocity, f)
        else:
            logging.info("从文件加载网络流量场中……")
            with open(mass_flux_file, 'r') as f:
                self.mass_flux = cPickle.load(f)
            with open(velocity_file, 'r') as f:
                self.velocity = cPickle.load(f)


    def print_config(self):
        """
        打印配置
        :return:
        """
        logging.info("------------------读取机械弥散求解器配置文件------------------")
        logging.info("文件名前缀: " + str(self.file_name))

    def simulate_one_particle(self, time):
        """
        模拟一个粒子
        :return:
        """
        pos = np.array([0, self.model_size[1] / 2, self.model_size[2] / 2])
        while True:
            if pos[0] >= self.model_size[0]:
                raise Exception("Setting time too large")
            flux = self.mass_flux[pos[0], pos[1], pos[2], :].copy()
            for i in range(26):
                if flux[i] < 0:
                    flux[i] = 0
            index = Tools.Tools.generate_randi_from_list(flux)
            rp = Tools.Tools.get_relative_position(index)
            indt = (pos[0], pos[1], pos[2], index)
            dist = self.status.ns.unit_size * np.sqrt(np.sum(np.square(rp))) * self.status.ns.character_length
            add_time = dist / self.velocity[indt]
            if add_time < time:
                time -= add_time
                pos = np.add(pos, rp)
            else:
                return np.add(pos, np.multiply(float(time) / float(add_time), rp))

    def start_simulation(self, num, time):
        """
        开始模拟
        :param num: 模拟粒子数量
        :param time: 模拟时间
        :return:
        """
        res = np.zeros([num, 3])
        for i in range(num):
            res[i, :] = self.simulate_one_particle(time)
            logging.debug(res[i])
        return res


if __name__ == '__main__':
    ds = DispersionSolver('../data/dispersion300_networkStatus_4250_tmp2.obj', '../data/dispersion_300_cache.obj')
    res = ds.start_simulation(1000, 0.05)
    print np.average(res, 0)
