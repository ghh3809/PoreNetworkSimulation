% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
curvature = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]';
throatR = [64.9, 61.7, 54.7, 43.1, 27.8, 12.4]' * 1e-9;
unit_size = [700.7788, 700.7788, 700.7788, 687.0783, 492.5317, 378.8179]' * 1e-9;
curvature_txt = {'01', '02', '04', '08', '016', '032'};
data = zeros(2001, length(curvature));

for i = 1:length(curvature)
    clearvars -except curvature curvature_txt throatR i data
    load(['dispersion_c', curvature_txt{i}, '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
end

for step = 1:10:total_steps
    x = curvature';
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    p = polyfit(log(x), log(y), 1);
    r = corrcoef(log(x), log(y));
    xx = curvature;
    yy = exp(polyval(p, log(xx)));
    cla;
    loglog(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
%     hold on;
%     loglog(xx, yy, 'r', 'Linewidth', 2);
%     axis([1e-9, 1e-8, 1e-10, 1e-9]);
%     text(6e-7, 1e-8, ['D_m = ', num2str(exp(p(2)), '%.4e'), '\times r^{ ', num2str(p(1), '%.4f'), '}'], 'Fontsize', 14);
%     text(6e-7, 3e-9, ['R^2 = ', num2str(r(1,2)^2, '%.4f')], 'Fontsize', 14);
    title('Dispersion Coefficient - Pore Radius');
    xlabel('Pore Radius r (m)');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'Fitting');
    drawnow;
end
