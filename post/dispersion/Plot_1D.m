%!/usr/bin/env matlab
% -*- coding: utf-8 -*-
% ----------------------------------------------------------------------------- %
% Copyright (c) 2018 Tsinghua Institute of Geo-Environment. All Rights Reserved %
% ----------------------------------------------------------------------------- %

% File: Plot_1D.py
% Date: 2018/11/12 20:44:38
% Desc: 显示瞬时投放的机械弥散现象

% ------------------------- %
%      Part1 : 设置部分      %
% ------------------------- %

% 模型设置项
data_file  = 'dispersion_2000_x1_paths.txt';    % 路径数据文件
model_size = [2000, 30, 30];                    % 模型尺寸
time       = 2;                                 % 模拟时间
particles  = 10000;                             % 模拟粒子数
time_step  = 1e-3;                              % 时间步长
unit_size  = 621.3161e-9;                       % 单元尺寸(m)

% Script_Set_Param;

% 计算设置项
plot_type          = 'dist';                    % move: 显示粒子运动; dist: 显示粒子分布
dispersion_type    = 'continue';                    % once: 瞬间投放; continue: 持续投放
particles_per_step = 50;                        % 每一迭代步投放的粒子数量（持续投放生效）
is_save            = 0;                         % 0: 不保存; 1: 保存动图结果
save_name          = 'fluid_continue_2000.gif';    % 需要保存的文件名

% ------------------------- %
%      Part2 : 计算部分      %
% ------------------------- %

data_folder = '../../src/data/';

% 如果已经加载了变量，就不需要进行重复加载
if ~exist('position', 'var')
    load([data_folder, data_file]);
    total_steps = round(time / time_step) + 1;
    position = reshape(eval(data_file(1:length(data_file)-4)), [4, total_steps, particles]);
end

% 初始化
figure;
set(gcf, 'position', [100, 500, 1000, 400]);    % 设置图像大小
res = zeros(total_steps, 2);                    % 平均速度与弥散系数的各步求解结果
if strcmp(dispersion_type, 'continue')          % 持续投放时，需随机投放顺序
    index_list = randi(particles, particles_per_step * total_steps, 1);
end

% 迭代计算
for i = 1:total_steps
    
    % 数据准备
    if strcmp(dispersion_type, 'once')
        pos = reshape(position(:, i, :), [4, particles]);
    else
        if strcmp(dispersion_type, 'continue')
            pos = zeros([4, particles_per_step * i]);
            for j = 1:size(pos, 2)
                eff_time_index = i - fix(j / particles_per_step);
                if eff_time_index > 0
                    pos(:, j) = position(:, eff_time_index, index_list(j));
                end
            end
        end
    end
    
%     if (i ~= 1) && (~all(pos(1, :)))
%         total_steps = i - 1;
%         break;
%     end
    
    % 绘图
    if strcmp(plot_type, 'dist')
        res(i, :) = Fun_PlotDistribution(model_size, pos, particles, dispersion_type, time_step * (i-1), unit_size);
    else
        if strcmp(plot_type, 'move')
            Fun_PlotParticles(model_size, pos, time_step * i);
        end
    end
    
    % 保存动图
    if is_save ~= 0
        Fun_SaveGIF(save_name, i);
    end
end

% Script_Save_Result;