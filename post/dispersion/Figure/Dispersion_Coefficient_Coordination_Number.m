% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
coordination = (4:2:16)';
data = zeros(2001, length(coordination));

for i = 1:length(coordination)
    clearvars -except coordination i data
    load(['dispersion_n', num2str(coordination(i)), '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
    tmp = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 1), 0:2000);
    for j = 50:2001
        if isnan(data(j, i))
            break;
        end
    end
    disp(['n', num2str(coordination(i)), ': valid = ', num2str(j), ', Dm = ',...
        num2str(data(j-1, i)), ',V = ', num2str(tmp(j-1))]);
end

for step = 1:10:2001
    x = coordination';
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    [xx, yy] = Uniform_BSpline(x, y, 3);
    cla;
    plot(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
    hold on;
    plot(xx, yy, 'r', 'Linewidth', 2);
    axis([2, 20, 0, 1e-9]);
    title('Dispersion Coefficient - Coordination Number');
    xlabel('Coordination Number');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'B-Spline Fitting');
    drawnow;
end
