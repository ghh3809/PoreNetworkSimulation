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

from utils import Tools
from NetworkStructure import NetworkStructureHandler
from NetworkStructure import NetworkStructure
from GasConstant import GasConstant
from FluidConstant import FluidConstant


class NetworkStatusHandler(object):
    """
    网络配置文件处理类
    """
    def __init__(self, config_file='../config/config.ini'):
        """
        初始化
        :param config_file: 配置文件，默认位置'../config/config.ini'
        """
        if not os.path.exists(config_file):
            logging.error("No status config file detected!")
            raise Exception("No status config file detected!")
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8-sig')

        # 读取具体配置
        try:
            # self.boundary_type
            self.__boundary_type_str = conf.get("status", "boundaryType").split(",")
            self.boundary_type = [int(x.strip()) for x in self.__boundary_type_str]
            if len(self.boundary_type) != 6:
                raise Exception("param status.boundaryType should have length 6!")
            if min(self.boundary_type) < 0 or max(self.boundary_type) > 2:
                raise Exception("param status.boundaryType should be 0-2!")

            # self.boundary_value
            self.__boundary_value_str = conf.get("status", "boundaryValue").split(",")
            self.boundary_value = [float(x.strip()) for x in self.__boundary_value_str]
            if len(self.boundary_value) != 6:
                raise Exception("param status.boundaryValue should have length 6!")
            if min(self.boundary_value) < 0:
                raise Exception("param status.boundaryValue should be non-negative!")

            # self.initial_type
            self.initial_type = int(conf.get("status", "initialType"))
            if self.initial_type < 0 or self.initial_type > 3:
                raise Exception("param status.initial_type should be 0-3!")

            # self.initial_value
            self.__initial_value_str = conf.get("status", "initialValue").split(",")
            self.initial_value = [float(x.strip()) for x in self.__initial_value_str]
            if len(self.initial_value) != 6:
                raise Exception("param status.initialValue should have length 6!")
            if min(self.initial_value) < 0:
                raise Exception("param status.initialValue should be non-negative!")

            self.print_config()

        except Exception as e:
            logging.error("Load config fail: [" + str(e) + "]")
            raise Exception("Load config fail: [" + str(e) + "]")

    def print_config(self):
        """
        打印当前的配置
        :return:
        """
        logging.info("------------------读取计算配置文件------------------")
        logging.info("=== 边界条件 ===")
        logging.info("-x侧: " + ("封闭边界" if self.boundary_type[0] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[0]) + " Pa") if self.boundary_type[0] != 0 else ""))
        logging.info("+x侧: " + ("封闭边界" if self.boundary_type[1] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[1]) + " Pa") if self.boundary_type[1] != 0 else ""))
        logging.info("-y侧: " + ("封闭边界" if self.boundary_type[2] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[2]) + " Pa") if self.boundary_type[2] != 0 else ""))
        logging.info("+y侧: " + ("封闭边界" if self.boundary_type[3] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[3]) + " Pa") if self.boundary_type[3] != 0 else ""))
        logging.info("-z侧: " + ("封闭边界" if self.boundary_type[4] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[4]) + " Pa") if self.boundary_type[4] != 0 else ""))
        logging.info("+z侧: " + ("封闭边界" if self.boundary_type[5] == 0 else "定压力边界") + \
                     (("，压力P = " + str(self.boundary_value[5]) + " Pa") if self.boundary_type[5] != 0 else ""))
        logging.info("=== 初始条件 ===")
        logging.info("逐渐变化:")
        if self.initial_value[0] + self.initial_value[1] > 0:
            logging.info("-x侧: 压力P = " + str(self.initial_value[0]) + " Pa")
            logging.info("    ||    ||    ||    ||")
            logging.info("    \/    \/    \/    \/")
            logging.info("+x侧: 压力P = " + str(self.initial_value[1]) + " Pa")
        elif self.initial_value[2] + self.initial_value[3] > 0:
            logging.info("-y侧: 压力P = " + str(self.initial_value[2]) + " Pa")
            logging.info("    ||    ||    ||    ||")
            logging.info("    \/    \/    \/    \/")
            logging.info("+y侧: 压力P = " + str(self.initial_value[3]) + " Pa")
        else:
            logging.info("-z侧: 压力P = " + str(self.initial_value[4]) + " Pa")
            logging.info("    ||    ||    ||    ||")
            logging.info("    \/    \/    \/    \/")
            logging.info("+z侧: 压力P = " + str(self.initial_value[5]) + " Pa")


class NetworkStatus(object):
    """
    网络结构基本类
    """
    def __init__(self, status_config, network_structure, gas_constant, fluid_constant):
        """
        利用配置创建网络结构
        :param status_config: 配置实例
        :param network_structure: 网络结构
        :param gas_constant: 气体参数
        :param fluid_constant: 流体参数
        """
        self.sc = status_config
        self.ns = network_structure
        self.gc = gas_constant
        self.fc = fluid_constant
        self.model_size = self.ns.model_size
        self.pressure = None
        self.sr = None # 孔隙饱和度，约定气相饱和度为0，液相饱和度为1，其余介于0-1之间
        self.phase = None # 约定气相为0，液相为1，其他为-1
        self.throat_phase = None # 约定气相为0，液相为1，其他为-1
        self.total_time = 0
        self.initialize_pressure()

    def initialize_pressure(self):
        """
        初始化整个网络的压力情况
        :return:
        """
        logging.info("------------------初始化网络状态------------------")
        logging.info("初始化网络状态中……")

        if self.sc.initial_type == 0 or self.sc.initial_type == 1:
            # 公式: y = kx + C
            self.pressure = np.zeros(self.model_size)
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = (float(i) / float(self.model_size[0] - 1)) * (
                                        self.sc.initial_value[1] - self.sc.initial_value[0]) + self.sc.initial_value[0]
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = (float(j) / float(self.model_size[1] - 1)) * (
                                        self.sc.initial_value[3] - self.sc.initial_value[2]) + self.sc.initial_value[2]
                        else:
                            self.pressure[i, j, k] = (float(k) / float(self.model_size[2] - 1)) * (
                                        self.sc.initial_value[5] - self.sc.initial_value[4]) + self.sc.initial_value[4]

        elif self.sc.initial_type == 2:
            # 公式: P = sqrt(kx + C)
            self.pressure = np.zeros(self.model_size)
            if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                c = np.square(self.sc.initial_value[0])
                kk = (np.square(self.sc.initial_value[1]) - np.square(self.sc.initial_value[0])) /\
                    float(self.model_size[0] - 1)
            elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                c = np.square(self.sc.initial_value[2])
                kk = (np.square(self.sc.initial_value[3]) - np.square(self.sc.initial_value[2])) /\
                    float(self.model_size[1] - 1)
            else:
                c = np.square(self.sc.initial_value[4])
                kk = (np.square(self.sc.initial_value[5]) - np.square(self.sc.initial_value[4])) /\
                    float(self.model_size[2] - 1)
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = np.sqrt(kk * i + c)
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = np.sqrt(kk * j + c)
                        else:
                            self.pressure[i, j, k] = np.sqrt(kk * k + c)

        elif self.sc.initial_type == 3:
            # 求出考虑尺度效应后的近似结果
            self.pressure = np.zeros(self.model_size)
            if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                layer_pressure = self.__cal_layer_pressure(self.sc.initial_value[0], self.sc.initial_value[1],
                                                           self.model_size[0])
            elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                layer_pressure = self.__cal_layer_pressure(self.sc.initial_value[2], self.sc.initial_value[3],
                                                           self.model_size[1])
            else:
                layer_pressure = self.__cal_layer_pressure(self.sc.initial_value[4], self.sc.initial_value[5],
                                                           self.model_size[2])
            for i in range(self.model_size[0]):
                for j in range(self.model_size[1]):
                    for k in range(self.model_size[2]):
                        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
                            self.pressure[i, j, k] = layer_pressure[i]
                        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
                            self.pressure[i, j, k] = layer_pressure[j]
                        else:
                            self.pressure[i, j, k] = layer_pressure[k]

        # 相态初始条件
        if self.sc.initial_type == 0: # 液相初始条件
            self.phase = np.ones(self.model_size, np.int8)
            self.sr = np.ones(self.model_size)
            self.throat_phase = np.ones(self.model_size + [26], np.int8)
        else: # 气相初始条件
            self.phase = np.zeros(self.model_size, np.int8)
            self.sr = np.zeros(self.model_size)
            self.throat_phase = np.zeros(self.model_size + [26], np.int8)

        # debug信息
        logging.info("初始化压力完成")
        logging.debug("初始化压力结果：")
        if self.sc.initial_value[0] + self.sc.initial_value[1] > 0:
            ave_pressure = np.average(np.average(self.pressure, 2), 1)
        elif self.sc.initial_value[2] + self.sc.initial_value[3] > 0:
            ave_pressure = np.average(np.average(self.pressure, 2), 0)
        else:
            ave_pressure = np.average(np.average(self.pressure, 1), 0)
        print_str = "    ["
        for p in ave_pressure:
            print_str += format(p, ".4e") + ", "
        print_str += "]"
        logging.debug(print_str)

    def set_boundary_condition(self):
        """
        计算迭代之前，将边界条件设置好
        :return:
        """
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    if i == 0:
                        if self.sc.boundary_type[0] == 1: # 液相边界条件
                            self.pressure[i, j, k] = self.sc.boundary_value[0]
                            self.phase[i, j, k] = 1
                            self.sr[i, j, k] = 1
                        elif self.sc.boundary_type[0] == 2: # 气相边界条件
                            self.pressure[i, j, k] = self.sc.boundary_value[0]
                            self.phase[i, j, k] = 0
                            self.sr[i, j, k] = 0
                    elif i == self.model_size[0] - 1:
                        if self.sc.boundary_type[1] == 1:
                            self.pressure[i, j, k] = self.sc.boundary_value[1]
                            self.phase[i, j, k] = 1
                            self.sr[i, j, k] = 1
                        elif self.sc.boundary_type[1] == 2:
                            self.pressure[i, j, k] = self.sc.boundary_value[1]
                            self.phase[i, j, k] = 0
                            self.sr[i, j, k] = 0
                    elif j == 0:
                        if self.sc.boundary_type[2] == 1:
                            self.pressure[i, j, k] = self.sc.boundary_value[2]
                            self.phase[i, j, k] = 1
                            self.sr[i, j, k] = 1
                        elif self.sc.boundary_type[2] == 2:
                            self.pressure[i, j, k] = self.sc.boundary_value[2]
                            self.phase[i, j, k] = 0
                            self.sr[i, j, k] = 0
                    elif j == self.model_size[1] - 1:
                        if self.sc.boundary_type[3] == 1:
                            self.pressure[i, j, k] = self.sc.boundary_value[3]
                            self.phase[i, j, k] = 1
                            self.sr[i, j, k] = 1
                        elif self.sc.boundary_type[3] == 2:
                            self.pressure[i, j, k] = self.sc.boundary_value[3]
                            self.phase[i, j, k] = 0
                            self.sr[i, j, k] = 0
                    elif k == 0:
                        if self.sc.boundary_type[4] == 1:
                            self.pressure[i, j, k] = self.sc.boundary_value[4]
                            self.phase[i, j, k] = 1
                            self.sr[i, j, k] = 1
                        elif self.sc.boundary_type[4] == 2:
                            self.pressure[i, j, k] = self.sc.boundary_value[4]
                            self.phase[i, j, k] = 0
                            self.sr[i, j, k] = 0
                    elif k == self.model_size[2] - 1:
                        if self.sc.boundary_type[5] == 1:
                            self.pressure[i, j, k] = self.sc.boundary_value[5]
                            self.phase[i, j, k] = 1
                            self.sr[i, j, k] = 1
                        elif self.sc.boundary_type[5] == 2:
                            self.pressure[i, j, k] = self.sc.boundary_value[5]
                            self.phase[i, j, k] = 0
                            self.sr[i, j, k] = 0

    def calculate_iteration(self, max_time_step, min_time_step, scale_effect):
        """
        利用时间步迭代法计算一步迭代。气相利用实际流量计算压强变化，液相利用平衡方法计算
        :param max_time_step: 最大时间步长
        :param min_time_step: 最小时间步长
        :param scale_effect: 尺度效应
        :return:
        """
        # 修正边界条件
        self.set_boundary_condition()

        # 计算一个迭代步
        k_flow = np.zeros(self.model_size + [26])
        new_throat_phase = - np.ones(self.model_size + [26], np.int8) # 1为液相，0为气相，-1为未初始化
        new_pressure = np.zeros(self.model_size)
        new_sr = self.sr.copy()
        new_phase = self.phase.copy()

        # 计算缓存
        # kn_coef: Kn(努森数)缓存系数，公式为 KnCoef = sqrt(πRT/2M) * μ，计算时有 Kn = KnCoef / pr
        # k_coef: K(导流系数)缓存系数，公式为 KCoef = π/8μ，计算时有 K = KCoef*r^4/l
        # pc_coef: pc(毛细压力)缓存系数，公式为 PcCoef = 2σcos(θ)，计算时有 Pc = PcCoef / r
        kn_coef = (np.sqrt(np.pi * self.gc.R * self.gc.T / 2.0 / self.gc.M) * self.gc.u) if scale_effect else 0
        k_gas_coef = np.pi / 8 / self.gc.u
        k_fluid_coef = np.pi / 8 / self.fc.u
        pc_max_coef = 2 * self.fc.sigma * np.cos(self.fc.min_theta * np.pi / 180) # 接触角小的，毛细力反而大，气->液适用
        pc_min_coef = 2 * self.fc.sigma * np.cos(self.fc.max_theta * np.pi / 180) # 接触角大的，毛细力反而小，液->气适用
        pc_coef = 2 * self.fc.sigma

        # Step 1：判断各通道流态，计算渗流系数
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    ind1 = (i, j, k)
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, ind1))
                        indt = (i, j, k, l)
                        if self.ns.weight[indt] > 0:

                            throat_r = self.ns.throatR[indt] * self.ns.character_length
                            throat_l = (np.sqrt(np.sum(np.square(rp))) * self.ns.unit_size - self.ns.radii[ind1] -
                                        self.ns.radii[ind2]) * self.ns.character_length

                            if self.phase[ind1] == self.phase[ind2]: # 两孔隙相态相同
                                new_throat_phase[indt] = self.phase[ind1]
                            elif self.phase[ind1] == 0: # 孔隙1为气相，孔隙2为液相
                                if self.phase[ind1] == self.throat_phase[indt] or self.pressure[ind1] > self.pressure[ind2] + pc_max_coef / throat_r: # 无需克服毛细力
                                    if self.pressure[ind1] > self.pressure[ind2] + pc_coef / (self.ns.radii[ind2] * self.ns.character_length):
                                        new_throat_phase[indt] = 0
                                if self.phase[ind2] == self.throat_phase[indt] or self.pressure[ind2] > self.pressure[ind1] - pc_min_coef / throat_r: # 无需克服毛细力
                                    if self.pressure[ind2] > self.pressure[ind1] - pc_coef / (self.ns.radii[ind2] * self.ns.character_length):
                                        new_throat_phase[indt] = 1
                            else: # 孔隙1为液相，孔隙2为气相
                                if self.phase[ind2] == self.throat_phase[indt] or self.pressure[ind2] > self.pressure[ind1] + pc_max_coef / throat_r: # 无需克服毛细力
                                    if self.pressure[ind2] > self.pressure[ind1] + pc_coef / (self.ns.radii[ind2] * self.ns.character_length):
                                        new_throat_phase[indt] = 0
                                if self.phase[ind1] == self.throat_phase[indt] or self.pressure[ind1] > self.pressure[ind2] - pc_min_coef / throat_r: # 无需克服毛细力
                                    if self.pressure[ind1] > self.pressure[ind2] - pc_coef / (self.ns.radii[ind2] * self.ns.character_length):
                                        new_throat_phase[indt] = 1

                            # 计算流量系数，注意对于气体是不对称的
                            if new_throat_phase[indt] == 0:
                                ave_p = (self.pressure[ind1] + self.pressure[ind2]) / 2
                                kn = kn_coef / ave_p / throat_r
                                beta = (1 + 4 * kn + 32 * (kn ** 2) / 3 / np.pi) / (1 + 0.5 * kn) * ave_p / self.pressure[ind1]
                                k_flow[indt] = beta * k_gas_coef * (throat_r ** 4) / throat_l
                            elif new_throat_phase[indt] == 1:
                                k_flow[indt] = k_fluid_coef * (throat_r ** 4) / throat_l

        # Step 2：计算迭代所需最大时间步长
        min_delta_t = 10000
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    ind1 = (i, j, k)
                    fluid_flux = 0.0
                    pore_volume = 4.0 / 3.0 * np.pi * ((self.ns.radii[ind1] * self.ns.character_length) ** 3)
                    sum_up = 0.0
                    sum_down = 0.0
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, ind1))
                        indt = (i, j, k, l)
                        if self.ns.weight[indt] > 0:
                            eff_pressure2 = self.pressure[ind2]
                            if self.phase[ind1] != self.phase[ind2]:
                                if self.phase[ind2] == 1: # 液相
                                    eff_pressure2 += pc_coef / (self.ns.radii[ind2] * self.ns.character_length)
                                else:
                                    eff_pressure2 -= pc_coef / (self.ns.radii[ind2] * self.ns.character_length)

                            sum_up += k_flow[indt] * eff_pressure2
                            sum_down += k_flow[indt]
                            fluid_flux += new_throat_phase[indt] * k_flow[indt] * (eff_pressure2 - self.pressure[ind1])

                    # 一个迭代补仅允许一个孔隙完成吸水或排水过程
                    if fluid_flux > 0 and self.sr[ind1] != 1:
                        delta_t = pore_volume * (1 - self.sr[ind1]) / fluid_flux
                        if delta_t < min_delta_t:
                            min_delta_t = delta_t
                    elif fluid_flux < 0 and self.sr[ind1] != 0:
                        delta_t = - pore_volume * self.sr[ind1] / fluid_flux
                        if delta_t < min_delta_t:
                            min_delta_t = delta_t

                    # 一个迭代步的流量应小于至平衡态的流量
                    if sum_down != 0:
                        balanced_pressure = sum_up / sum_down
                        total_volume = (balanced_pressure - self.pressure[ind1]) / balanced_pressure * pore_volume * (1 - self.sr[ind1])
                        total_flux = sum_up - np.sum(k_flow[ind1]) * self.pressure[ind1]
                        if total_flux != 0 and total_volume != 0:
                            delta_t = total_volume / total_flux
                            if delta_t < min_delta_t:
                                min_delta_t = delta_t

        # 修正时间步长
        if min_delta_t > max_time_step:
            min_delta_t = max_time_step
        elif min_delta_t < min_time_step:
            min_delta_t = min_time_step

        # Step 3：利用时间步长进行迭代
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):

                    ind1 = (i, j, k)
                    pore_volume = 4.0 / 3.0 * np.pi * ((self.ns.radii[ind1] * self.ns.character_length) ** 3)
                    sum_up = 0.0
                    sum_down = 0.0

                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, ind1))
                        indt = (i, j, k, l)
                        if self.ns.weight[indt] > 0:
                            eff_pressure2 = self.pressure[ind2]
                            if self.phase[ind1] != self.phase[ind2]:
                                if self.phase[ind2] == 1:  # 液相
                                    eff_pressure2 += pc_coef / (self.ns.radii[ind2] * self.ns.character_length)
                                else:
                                    eff_pressure2 -= pc_coef / (self.ns.radii[ind2] * self.ns.character_length)

                            sum_up += k_flow[indt] * eff_pressure2
                            sum_down += k_flow[indt]

                    # 计算新孔隙压力
                    sum_up = sum_up * min_delta_t + pore_volume * (1 - self.sr[ind1])
                    sum_down = sum_down * min_delta_t + pore_volume * (1 - self.sr[ind1]) / self.pressure[ind1]
                    if sum_down != 0:
                        new_pressure[ind1] = sum_up / sum_down
                    else:
                        new_pressure[ind1] = self.pressure[ind1]

                    # 计算新饱和度
                    fluid_flag = (self.sr[ind1] == 1)
                    delta_sr = 0.0
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, ind1))
                        indt = (i, j, k, l)
                        if self.ns.weight[indt] > 0:
                            eff_pressure2 = self.pressure[ind2]
                            if self.phase[ind1] != self.phase[ind2]:
                                if self.phase[ind2] == 1: # 液相
                                    eff_pressure2 += pc_coef / (self.ns.radii[ind2] * self.ns.character_length)
                                else:
                                    eff_pressure2 -= pc_coef / (self.ns.radii[ind2] * self.ns.character_length)

                            if new_throat_phase[indt] != 1:
                                fluid_flag = False
                            delta_sr += new_throat_phase[indt] * min_delta_t / pore_volume * k_flow[indt] * (eff_pressure2 - self.pressure[ind1])
                    new_sr[ind1] += delta_sr

                    # 更新状态，并防止无限循环
                    if not fluid_flag:
                        if self.sc.initial_type == 0 and self.sr[ind1] == 0 and delta_sr > 0:
                            new_phase[ind1] = 1
                        elif self.sc.initial_type > 0 and self.sr[ind1] == 1 and delta_sr < 0:
                            new_phase[ind1] = 0
                        if new_sr[ind1] < 0.01 and delta_sr < 0: # 对于极端饱和度特殊处理
                            new_sr[ind1] = 0
                            new_phase[ind1] = 0
                        elif new_sr[ind1] > 0.99 and delta_sr > 0:
                            new_sr[ind1] = 1
                            new_phase[ind1] = 1
                    else: # 纯液相渗流时不处理饱和度信息
                        new_sr[ind1] = 1
                        new_phase[ind1] = 1

        self.pressure = new_pressure
        self.phase = new_phase
        self.throat_phase = new_throat_phase
        self.sr = new_sr
        self.total_time += min_delta_t

    def calculate_equation(self, scale_effect, show_status, method):
        """
        利用J/GS迭代法计算一步迭代
        :param scale_effect: 尺度效应
        :param show_status: 是否显示求解过程
        :param method: "J" / "GS"
        :return:
        """
        # 修正边界条件
        self.set_boundary_condition()

        # 计算缓存
        # kn_coef: Kn(努森数)缓存系数，公式为 KnCoef = sqrt(πRT/2M) * μ，计算时有 Kn = KnCoef / pr
        # k_coef: K(导流系数)缓存系数，公式为 KCoef = π/8μ，计算时有 K = KCoef*r^4/l
        # pc_coef: pc(毛细压力)缓存系数，公式为 PcCoef = 2σcos(θ)，计算时有 Pc = PcCoef / r
        kn_coef = (np.sqrt(np.pi * self.gc.R * self.gc.T / 2.0 / self.gc.M) * self.gc.u) if scale_effect else 0
        k_gas_coef = np.pi / 8 / self.gc.u
        k_fluid_coef = np.pi / 8 / self.fc.u

        # 迭代计算
        if method == "J":
            new_pressure = np.zeros(self.model_size)
        else:
            new_pressure = self.pressure

        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    sum_up = 0.0
                    sum_down = 0.0
                    ind1 = (i, j, k)
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        indt = (i, j, k, l)
                        if self.ns.weight[indt] > 0:
                            throat_r = self.ns.throatR[indt] * self.ns.character_length
                            throat_l = (np.sqrt(np.sum(np.square(rp))) * self.ns.unit_size - self.ns.radii[ind1] -
                                        self.ns.radii[ind2]) * self.ns.character_length

                            # 计算流量系数，注意对于气体是不对称的
                            if self.sc.initial_type > 0:
                                ave_p = (self.pressure[ind1] + self.pressure[ind2]) / 2
                                kn = kn_coef / ave_p / throat_r
                                beta = (1 + 4 * kn + 32 * (kn ** 2) / 3 / np.pi) / (1 + 0.5 * kn) * ave_p / self.pressure[ind1]
                                k_flow = beta * k_gas_coef * (throat_r ** 4) / throat_l
                                sum_up += k_flow * self.pressure[ind2]
                                sum_down += k_flow
                            else:
                                k_flow = k_fluid_coef * (throat_r ** 4) / throat_l
                                sum_up += k_flow * self.pressure[ind2]
                                sum_down += k_flow

                    if sum_down != 0:
                        new_pressure[ind1] = sum_up / sum_down

            # 显示计算进度
            for i in range(int(float(show_status) * float(i - 1) / float(self.model_size[0])),
                           int(float(show_status) * float(i) / float(self.model_size[0]))):
                sys.stdout.write("█")

        # 更新矩阵
        if method == "J":
            self.pressure = new_pressure

    def get_flux(self, scale_effect, show_status, is_boundary=False):
        """
        获取当前状态的流量
        :param scale_effect: 是否考虑尺度效应
        :param show_status: 是否显示计算过程
        :param is_boundary: 是否仅计算边界质量流量
        :return: mass_flux, velocity
        """
        # 修正边界条件
        self.set_boundary_condition()

        model_size = (self.model_size[0] if is_boundary else 1, self.model_size[1], self.model_size[2], 26)
        volume_flux = np.zeros(model_size)
        mass_flux = np.zeros(model_size)
        velocity = np.zeros(model_size)

        # 计算缓存
        # kn_coef: Kn(努森数)缓存系数，公式为 KnCoef = sqrt(πRT/2M) * μ，计算时有 Kn = KnCoef / pr
        # k_coef: K(导流系数)缓存系数，公式为 KCoef = π/8μ，计算时有 K = KCoef*r^4/l
        # pc_coef: pc(毛细压力)缓存系数，公式为 PcCoef = 2σcos(θ)，计算时有 Pc = PcCoef / r
        # rou_coef: ρ(密度)缓存系数，公式为ρCoef = M/RT，计算时有ρ = ρCoef * p
        kn_coef = (np.sqrt(np.pi * self.gc.R * self.gc.T / 2.0 / self.gc.M) * self.gc.u) if scale_effect else 0
        k_gas_coef = np.pi / 8 / self.gc.u
        k_fluid_coef = np.pi / 8 / self.fc.u
        rou_coef = self.gc.M / self.gc.R / self.gc.T

        for i in range(model_size[0]):
            for j in range(model_size[1]):
                for k in range(model_size[2]):
                    ind1 = (i, j, k)
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        indt = (i, j, k, l)
                        if self.ns.weight[indt] > 0:
                            throat_r = self.ns.throatR[indt] * self.ns.character_length
                            throat_l = (np.sqrt(np.sum(np.square(rp))) * self.ns.unit_size - self.ns.radii[ind1] -
                                        self.ns.radii[ind2]) * self.ns.character_length

                            # 计算流量系数，注意对于气体是不对称的
                            if self.sc.initial_type > 0:
                                ave_p = (self.pressure[ind1] + self.pressure[ind2]) / 2
                                kn = kn_coef / ave_p / throat_r
                                beta = (1 + 4 * kn + 32 * (kn ** 2) / 3 / np.pi) / (1 + 0.5 * kn) * ave_p / \
                                       self.pressure[ind1]
                                k_flow = beta * k_gas_coef * (throat_r ** 4) / throat_l
                                volume_flux[indt] = k_flow * (self.pressure[ind2] - self.pressure[ind1])
                                mass_flux[indt] = volume_flux[indt] * rou_coef * ave_p
                                velocity[indt] = volume_flux[indt] / np.pi / (self.ns.throatR[indt] ** 2)
                            else:
                                k_flow = k_fluid_coef * (throat_r ** 4) / throat_l
                                volume_flux[indt] = k_flow * (self.pressure[ind2] - self.pressure[ind1])
                                mass_flux[indt] = volume_flux[indt] * self.fc.rou
                                velocity[indt] = volume_flux[indt] / np.pi / (self.ns.throatR[indt] ** 2)

            # 显示计算进度
            for i in range(int(float(show_status) * float(i - 1) / float(self.model_size[0])),
                           int(float(show_status) * float(i) / float(self.model_size[0]))):
                sys.stdout.write("█")

        return volume_flux, mass_flux, velocity

    def get_permeability(self, scale_effect):
        """
        获取当前渗透率（要求渗流方向为x方向）（需修改）
        :param scale_effect: 尺度效应
        :return:
        """

        if self.sc.initial_type == 0:
            rou = self.fc.rou
            u = self.fc.u
        else:
            rou = (self.sc.boundary_value[0] + self.sc.boundary_value[1]) * self.gc.M / (2 * self.gc.R * self.gc.T)
            u = self.gc.u

        grad_p = (self.sc.boundary_value[0] - self.sc.boundary_value[1]) / (self.ns.character_length * self.ns.unit_size * (self.model_size[0] - 1))
        _, mass_flux, _ = self.get_flux(scale_effect, 0, True)
        ave_mass_flux = np.average(np.sum(mass_flux, 3)[0, 1:-1, 1:-1]) / ((self.ns.character_length * self.ns.unit_size) ** 2)

        return np.abs(ave_mass_flux * u / rou / grad_p)

    def __cal_layer_pressure(self, pressure1, pressure2, size):
        """
        计算各层初始压力（仅用于计算第三类初始条件）
        :param pressure1: 压力1
        :param pressure2: 压力2
        :param size: 总层数
        :return:
        """
        pressure = np.zeros(size)
        c = np.square(pressure1)
        kk = (np.square(pressure2) - np.square(pressure1)) / float(size - 1)
        for i in range(size):
            pressure[i] = np.sqrt(kk * i + c)
        ave_kn_coef = self.__get_ave_kn_coef()
        while True:
            pressure_old = pressure.copy()
            for i in range(1, size - 1):
                flux1 = self.__cal_eff_mass_flux(pressure[i - 1], pressure[i], ave_kn_coef)
                flux2 = self.__cal_eff_mass_flux(pressure[i], pressure[i + 1], ave_kn_coef)
                ratio = flux1 / flux2 * (pressure[i] - pressure[i + 1]) / (pressure[i - 1] - pressure[i])
                pressure[i] = pressure[i + 1] + (ratio / (1 + ratio)) * (pressure[i - 1] - pressure[i + 1])
            if np.max(np.abs(np.subtract(pressure, pressure_old))) < 1:
                break
        return pressure

    def __get_ave_kn_coef(self):
        """
        计算kn缓存: KnCoef = sqrt(πRT/2M) * μ / r，计算时有 Kn = KnCoef / p
        （仅用于计算第三类初始条件）
        :return:
        """
        kn_cache = np.zeros(self.ns.model_size + [26])
        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                for k in range(self.model_size[2]):
                    for l in range(26):
                        rp = Tools.Tools.get_relative_position(l)
                        ind2 = tuple(np.add(rp, (i, j, k)))
                        if (0 <= ind2[0] < self.model_size[0]) \
                                and (0 <= ind2[1] < self.model_size[1]) \
                                and (0 <= ind2[2] < self.model_size[2]):
                            throat_r = self.ns.throatR[i, j, k, l] * self.ns.character_length
                            kn_cache[i, j, k, l] = np.sqrt(
                                np.pi * self.gc.R * self.gc.T / 2.0 / self.gc.M) * self.gc.u / throat_r
        return np.average(kn_cache[1:-1, 1:-1, 1:-1, :])

    @staticmethod
    def __cal_eff_mass_flux(pressure1, pressure2, ave_kn_coef):
        """
        计算等效质量流量（仅用于计算第三类初始条件）
        :param pressure1: 压力1
        :param pressure2: 压力2
        :return:
        """
        ave_p = (pressure1 + pressure2) / 2.0
        kn = ave_kn_coef / ave_p
        f = 1.0 / (1.0 + 0.5 * kn)
        return ((1.0 + 4.0 * kn) * f + 64.0 / 3.0 / np.pi * kn * (1.0 - f)) * ave_p * (pressure1 - pressure2)


if __name__ == '__main__':
    network_status = NetworkStatus(NetworkStatusHandler(),
                                   NetworkStructure(NetworkStructureHandler()),
                                   GasConstant(),
                                   FluidConstant())
