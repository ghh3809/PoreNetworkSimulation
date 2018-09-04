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

sys.path.append('./')
from utils import Tools
from entity import Simulator as Simu


class DispersionSolver(object):

    def __init__(self, status_file_name, config_file='./config/dispersion_config.ini'):
        """
        利用配置创建迭代器
        :param config_file: 配置文件
        """

        if not os.path.exists(status_file_name):
            logging.error("No status file detected!")
            raise Exception("No status file detected!")
        with open(status_file_name, 'r') as f:
            self.status = cPickle.load(f)
        simulator = Simu.Simulator()
        simulator.bind_network_status(self.status)
        self.mass_flux, self.velocity = simulator.get_mass_flux()
        self.model_size = self.status.model_size

        if os.path.exists(config_file):
            conf = configparser.ConfigParser()
            conf.read(config_file, encoding='utf-8-sig')
            self.print_config()

    def print_config(self):
        """
        打印配置
        :return:
        """
        pass

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


if __name__ == '__main__':
    ds = DispersionSolver('./data/dispersion300_networkStatus_1050_tmp1.obj')
    for i in range(1000):
        print ds.simulate_one_particle(0.05)

