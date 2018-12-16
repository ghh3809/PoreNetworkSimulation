% 对于相同的网络结构，不同velocity数据绘图
clear; clc;
velocity = [1:9, 10:10:100]';
data = zeros(1501, length(velocity));
for i = 1:length(velocity)
    clearvars -except velocity data i
    load(['dispersion_2000_x', num2str(velocity(i)), '_paths.mat']);
    data(:, i) = interp1(res(2:end, 1) .* (1:total_steps-1)' * time_step, res(2:end, 2),...
        (0:1500)' * unit_size, 'Linear');
    for j = 500:1501
        if isnan(data(j, i))
            break;
        end
    end
    disp(['x', num2str(velocity(i)), ': valid = ', num2str(j), ', Dm = ', num2str(data(j-1, i))]);
end

slop = zeros(1501, 1);

fig = figure;
set(fig, 'position', [100, 100, 600, 400]);
ax = gca;
set(ax, 'XColor', 'r', 'YColor', 'r');
set(ax, 'Position', [0.1, 0.11, 0.8, 0.75]);
ax1 = axes('Position',get(ax,'Position'),...
       'XAxisLocation','bottom',...
       'YAxisLocation','left',...
       'Color','none',...
       'XColor','k','YColor','k');
ax2 = axes('Position',get(ax1,'Position'),...
       'XAxisLocation','top',...
       'YAxisLocation','right',...
       'Color','none',...
       'XColor','b','YColor','b','XScale','log');
set(ax1, 'visible', 'off');
% figure;

imgno = 0;
for step = 1:5:1501
    x = velocity';
    y = data(step, :);
    x_data = [];
    y_data = [];
    for i = 1:length(x)
        if ~isnan(x(i)) && ~isnan(y(i))
            x_data(length(x_data) + 1) = x(i);
            y_data(length(y_data) + 1) = y(i);
        end
    end
    if length(x_data) <= 3
        continue;
    end
    p = polyfit(x_data, y_data, 1);
    slop(step) = p(1);
    xx = 0:100;
    yy = polyval(p, xx);
    
    % 绘图部分
    cla(ax);
    line(ax, x, y, 'Color', 'k', 'LineStyle', 'none', 'Marker', 'x', 'markerSize', 10, 'linewidth', 2);
    line(ax, xx, yy, 'Color', 'r', 'linewidth', 2);
    line(ax, 0, 0, 'Color', 'b', 'linewidth', 2);
    hold on;
    xlim(ax, [0, 100]);
    ylim(ax, [0, 1e-7]);
    ylabel('Estimated Mechanical Dispersion Coefficient(m^2/s)', 'Parent', ax);
    xlabel('Velocity Ratio', 'Parent', ax);
    legend(ax, 'Simulation', 'Linear Fitting', 'Slop');
    
    cla(ax2);
    line(ax2, unit_size * (1:5:step), slop(1:5:step), 'Color', 'b', 'linewidth', 2)
    xlim(ax2, [5e-6, 1e-2]);
    ylim(ax2, [0, 1e-9]);
    ylabel('Dispersion Coefficient Slop(m^2/s)', 'Parent', ax2);
    xlabel('Distance (m)', 'Parent', ax2, 'Position', [2.5e-4, 1.02e-9]);

    title(ax2, ['Dispersion Coefficient - Velocity, Distance = ', num2str(unit_size * step, '%.2E'), ' m']);
    drawnow;
    imgno = imgno + 1;
%     Fun_SaveGIF('./Figure/Dispersion Coefficient - Velocity.gif', imgno);
end