% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
radius_deviation = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]';
data = zeros(2001, length(radius_deviation));

for i = 1:length(radius_deviation)
    clearvars -except radius_deviation i data
    load(['dispersion_sr', num2str(radius_deviation(i)), '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
    tmp = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 1), 0:2000);
    for j = 200:2001
        if isnan(data(j, i))
            break;
        end
    end
    disp(['sr', num2str(radius_deviation(i)), ': valid = ', num2str(j), ', Dm = '...
        num2str(data(j-1, i)), ',V = ', num2str(tmp(j-1))]);
end

for step = 1:10:2001
    x = radius_deviation';
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    xx = 1:32;
    yy = interp1(x, y, xx, 'pchip');
    cla;
    plot(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
    hold on;
    plot(xx, yy, 'r', 'Linewidth', 2);
    axis([1, 32, 0, 1e-9]);
    title('Dispersion Coefficient - Pore Radius Deviation');
    xlabel('Pore Radius Deviation');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'Fitting');
    drawnow;
end
