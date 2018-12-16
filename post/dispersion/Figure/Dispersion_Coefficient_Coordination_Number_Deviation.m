% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
coordination_deviation = [1, 2, 3, 4]';
data = zeros(2001, length(coordination_deviation));

for i = 1:length(coordination_deviation)
    clearvars -except coordination_deviation i data
    load(['dispersion_sn', num2str(coordination_deviation(i)), '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
    tmp = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 1), 0:2000);
    for j = 200:2001
        if isnan(data(j, i))
            break;
        end
    end
    disp(['sn', num2str(coordination_deviation(i)), ': valid = ', num2str(j), ', Dm = ',...
        num2str(data(j-1, i)), ',V = ', num2str(tmp(j-1))]);
end

for step = 1:10:2001
    x = coordination_deviation';
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    xx = 0:0.1:4;
    yy = interp1(x, y, xx, 'pchip');
    cla;
    plot(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
    hold on;
    plot(xx, yy, 'r', 'Linewidth', 2);
    axis([0, 4, 3e-10, 5e-10]);
    title('Dispersion Coefficient - Coordination Number Deviation');
    xlabel('Coordination Number Deviation');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'Fitting');
    drawnow;
end
