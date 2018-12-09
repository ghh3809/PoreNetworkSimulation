global conf;
global batch_i;

% 模型设置项
data_file  = ['dispersion_', conf.name{batch_i}, '_paths.txt'];   % 路径数据文件
if isstrprop(conf.name{batch_i}(1), 'digit')
    model_size = [str2double(conf.name{batch_i}(1:strfind(conf.name{batch_i}, '_') - 1)), 30, 30];                            % 模型尺寸
end
time       = conf.data(batch_i, 1);                               % 模拟时间
particles  = conf.data(batch_i, 2);                               % 模拟粒子数
time_step  = conf.data(batch_i, 3);                               % 时间步长
unit_size  = conf.data(batch_i, 4) * 1e-9;                        % 单元尺寸(m)