% 对于相同的网络结构，不同velocity数据绘图

velocity = [1:9, 10:10:100]';
legends = cell(length(velocity), 1);
for i = 1:length(velocity)
    clearvars -except velocity legends i
    load(['dispersion_2000_x', num2str(velocity(i)), '_paths.mat']);
    index = find(res(:, 2) == max(res(:, 2)));
    index = index(1);
    semilogy(res(1:index, 1) .* (0:index-1)' * time_step, smooth(res(1:index, 2), 20), 'linewidth', 2);
    legends{i} = ['x', num2str(velocity(i))];
    hold on;
end
xlim([0, 8e-4]);
title('Dispersion Coefficient Development - Velocity');
xlabel('Transport Distance (m)');
ylabel('Dispersion Coefficient (m^2/s)');
legend(legends);