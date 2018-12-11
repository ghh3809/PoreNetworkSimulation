global conf;
global batch_i;
conf = importdata('../../src/data/config.txt');
conf.name = conf.textdata(2:end, 1);

for batch_i = 23:length(conf.name)
    clearvars -except conf batch_i;
    clc;
    disp(conf.name{batch_i});
    Plot_1D;
    close all;
end