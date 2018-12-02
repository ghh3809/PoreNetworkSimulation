% 对于不同velocity数据绘图
clear; clc;
load('dispersion_2000_paths.mat');

% % Plot1: D-x curve
% legends = cell(10, 1);
% for i = 1:10
%     res = eval(['res_x', num2str(i)]);
%     plot(res(:, 1) .* res(:, 2), smooth(res(:, 3), 20), 'linewidth', 2);
%     legends{i} = ['x', num2str(i)];
%     hold on;
% end
% xlim([0, 8e-4]);
% xlabel('Transport Distance(m)');
% ylabel('Estimated Mechanical Dispersion Coefficient(m^2/s)');
% title('Mechanical Dispersion in Different Velocity');
% legend(legends);

% Plot2: D-v curve
basic_v = 5.735e-7;
slop = zeros(1501, 1);
fig = figure;
set(fig, 'defaultAxesColorOrder', [[1, 0, 0]; [0, 0, 1]]);
for step = 1:5:1501
    x = 1:10;
    y = res_uniform(step, :);
    p = polyfit(x, y, 1);
    slop(step) = p(1);
    xx = 0:10;
    yy = polyval(p, xx);
    yyaxis left;
    cla(1);
    plot(x, y, 'kx', 'markerSize', 10, 'linewidth', 2);
    hold on;
    plot(xx, yy, 'r', 'linewidth', 2);
    ylim([0, 7e-9]);
    ylabel('Estimated Mechanical Dispersion Coefficient(m^2/s)');
    yyaxis right;
    cla(1);
    plot(basic_v * (1:5:step) * 1e4, slop(1:5:step), 'linewidth', 2);
    ylim([0, 1e-9]);
    ylabel('Dispersion Coefficient Slop(m^2/s)');
    title(['Dispersion Coefficient - Velocity, Distance = ', num2str(basic_v * step, '%.2E'), ' m']);
    xlabel('Velocity Ratio / Distance (x10^{-4}m)');
    legend('Simulation', 'Linear Fitting', 'Slop');
    drawnow;
    Fun_SaveGIF('Dispersion Coefficient - Velocity.gif', ceil(step/5));
end
