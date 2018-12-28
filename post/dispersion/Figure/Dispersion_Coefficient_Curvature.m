% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
curvature = [0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]';
curvature_txt = {'01', '02', '04', '08', '012', '016', '020', '024', '028', '032'};
data = zeros(2001, length(curvature));

for i = 1:length(curvature)
    clearvars -except curvature curvature_txt i data
    load(['dispersion_c', curvature_txt{i}, '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
    tmp = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 1), 0:2000);
    for j = 500:2001
        if isnan(data(j, i))
            break;
        end
    end
    disp(['c', curvature_txt{i}, ': valid = ', num2str(j), ', Dm = ',...
        num2str(data(j-1, i)), ',V = ', num2str(tmp(j-1))]);
end

for step = 1:10:2001
    x = curvature';
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    [xx, yy] = Uniform_BSpline(x, y, 4);
    cla;
    plot(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
    hold on;
    plot(xx, yy, 'r', 'Linewidth', 2);
    axis([0, 3.5, 0, 1.5e-9]);
    title('Dispersion Coefficient - Curvature');
    xlabel('Curvature');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'B-Spline Fitting');
    drawnow;
end
