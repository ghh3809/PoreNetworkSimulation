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
import re
import sys
import time
import logging
import cPickle
import configparser
import numpy as np

sys.path.append('../')

from utils import Tools
from entity import StatusCache as Cache
from CalculatePermeability import SeepageIterator


logging_flag = False
if __name__ == "__main__":
    logging_flag = True


class DispersionSolver(object):

    def __init__(self, config_file='../config/config.ini'):
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
            if logging_flag:
                Tools.Tools.set_logging('../log/' + self.file_name + '.log', output_level=logging.DEBUG)

            # self.time_step
            self.time = float(conf.get("dispersion", "time"))
            if self.time <= 0:
                raise Exception("param dispersion.time should be positive!")

            # self.time_step
            self.particles = int(conf.get("dispersion", "particles"))
            if self.particles <= 0:
                raise Exception("param dispersion.particles should be positive!")

            # self.time_step
            self.time_step = float(conf.get("dispersion", "timeStep"))
            if self.time_step <= 0:
                raise Exception("param dispersion.timeStep should be positive!")

        except Exception as e:
            raise Exception("Load config fail: [" + str(e) + "]")

        self.print_config()

        # 判断网络状态是否存在
        file_name = re.compile("seepage_" + self.file_name[11:] + "_status_(.*?)\.obj")
        max_iters = 0
        status_file_name = ""
        for root, dirs, files in os.walk("../data/"):
            for file in files:
                match = file_name.match(file)
                if match:
                    current_iters = match.group(1)
                    if int(current_iters) > max_iters:
                        max_iters = int(current_iters)
                        status_file_name = "../data/" + file

        if status_file_name == "":
            raise Exception("未发现合适的网络状态文件，程序退出！")

        logging.info("------------------初始化网络状态------------------")
        logging.info("【重要】发现网络状态文件（3秒后继续）: " + str(status_file_name))
        time.sleep(3)
        logging.info("从文件重建网络状态中……")
        with open(status_file_name, 'r') as f:
            self.status = cPickle.load(f)
        self.model_size = self.status.model_size

        # 计算网络流量场
        cache_file_name = "../data/seepage_" + self.file_name[11:] + "_cache.obj"
        mass_flux_file = "../data/" + self.file_name + "_massflux.obj"
        velocity_file = "../data/" + self.file_name + "_velocity.obj"
        if not os.path.exists(mass_flux_file) or not os.path.exists(velocity_file):
            # 没有流量场文件，需要重建
            if not os.path.exists(cache_file_name):
                # 没有缓存文件，从网络状态进行重建
                logging.info("未发现计算缓存文件，正在重新计算中……")
                self.cache = Cache.StatusCache(self.status)
                logging.info("保存计算缓存: " + str(cache_file_name))
                if not os.path.exists(os.path.dirname(cache_file_name)):
                    os.makedirs(os.path.dirname(cache_file_name))
                with open(cache_file_name, 'w') as f:
                    cPickle.dump(self.cache, f)
            else:
                # 存在网络状态缓存
                logging.info("从文件重建计算缓存中……")
                with open(cache_file_name, 'r') as f:
                    self.cache = cPickle.load(f)
            logging.info("计算网络流量场中……")
            self.mass_flux, self.velocity = self.status.get_mass_flux(self.cache, 0)
            with open(mass_flux_file, 'w') as f:
                cPickle.dump(self.mass_flux, f)
            with open(velocity_file, 'w') as f:
                cPickle.dump(self.velocity, f)
        else:
            # 直接加载
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
        logging.info("时间(s): " + str(self.time))
        logging.info("粒子数量: " + str(self.particles))
        logging.info("时间步长(s): " + str(self.time_step))

    def start_time_simulation(self):
        """
        开始模拟
        :return:
        """
        res = np.zeros([self.particles, 3])
        for i in range(self.particles):
            res[i, :] = self.simulate_final_position()
            logging.debug("Particle ID = " + str(i) + ", Final position = " + str(res[i]))
        return res

    def start_path_simulation(self):
        """
        开始模拟
        :return:
        """
        total_size = int(self.time / self.time_step)
        res = np.zeros([self.particles, total_size + 1, 4])
        flux = np.sum(self.mass_flux[0, :, :, :], 2).reshape(self.model_size[1] * self.model_size[2])
        for i in range(self.particles):
            index = Tools.Tools.generate_randi_from_list(flux)
            res[i, :, :] = self.simulate_paths([0, index / self.model_size[1], index % self.model_size[2]])
            logging.debug("Particle ID = " + str(i) + ", Start position = " + \
                          str([0, index / self.model_size[1], index % self.model_size[2]]) + \
                          ", Final position = " + str(res[i, -1, :]))
        with open("../data/" + self.file_name + "_paths.txt", "w") as f:
            for i in range(self.particles):
                for j in range(total_size + 1):
                    for k in range(4):
                        f.write(str(res[i, j, k]) + '\n')
        return res

    def simulate_once(self, pos):
        """
        计算从指定位置出发，到达的下一个位置
        :param pos: 起始位置
        :return:
        """
        if pos[0] >= self.model_size[0] - 1:
            raise Exception("Position out of range!")
        flux = self.mass_flux[pos[0], pos[1], pos[2], :].copy()
        for i in range(26):
            if flux[i] < 0:
                flux[i] = 0
        index = Tools.Tools.generate_randi_from_list(flux)
        rp = Tools.Tools.get_relative_position(index)
        indt = (pos[0], pos[1], pos[2], index)
        dist = self.status.ns.unit_size * np.sqrt(np.sum(np.square(rp))) * self.status.ns.character_length
        path_time = dist / self.velocity[indt]
        return rp, path_time

    def simulate_final_position(self):
        """
        模拟特定时间内，一个粒子到达的最终位置
        :return:
        """
        pos = np.array([0, self.model_size[1] / 2, self.model_size[2] / 2])
        while True:
            if pos[0] >= self.model_size[0] - 1:
                raise Exception("Simulation time too less!")
            rp, add_time = self.simulate_once(pos)
            if add_time < self.time:
                self.time -= add_time
                pos = np.add(pos, rp)
            else:
                return np.add(pos, np.multiply(float(self.time) / float(add_time), rp))

    def simulate_paths(self, start_pos):
        """
        模拟一个粒子随时间前进的过程
        :return:
        """
        pos = start_pos
        total_size = int(self.time / self.time_step)
        paths = np.zeros([total_size + 1, 4])

        # 初始化
        current_time = 0
        current_count = 0
        last_time = -1
        next_time = 0
        rp = (0, 0, 0)

        while current_count <= total_size:

            # 首先判断是否需要更新到下一阶段
            while current_time >= next_time:
                pos = np.add(pos, rp)
                if pos[0] >= self.model_size[0] - 1:
                    return paths
                last_time = next_time
                rp, path_time = self.simulate_once(pos)
                next_time = last_time + path_time

            # 确定当前位置
            current_pos = np.add(pos, np.multiply(float(current_time - last_time) / float(next_time - last_time), rp))
            paths[current_count, :] = np.array([current_time, current_pos[0], current_pos[1], current_pos[2]])
            current_time += self.time_step
            current_count += 1

        return paths


if __name__ == '__main__':
    ds = DispersionSolver()
    res = ds.start_path_simulation()
