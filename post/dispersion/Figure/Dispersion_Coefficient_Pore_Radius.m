% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
radius = [100, 200, 400, 800, 1600, 3200, 6400]';
data = zeros(2001, length(radius));

for i = 1:length(radius)
    clearvars -except radius i data
    load(['dispersion_r', num2str(radius(i)), '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
    tmp = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 1), 0:2000);
    for j = 500:2001
        if isnan(data(j, i))
            break;
        end
    end
    disp(['r', num2str(radius(i)), ': valid = ', num2str(j), ', Dm = '...
        num2str(data(j-1, i)), ',V = ', num2str(tmp(j-1))]);
end

for step = 1:10:2001
    x = radius' * 1e-9;
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    p = polyfit(log(x), log(y), 1);
    r = corrcoef(log(x), log(y));
    xx = radius * 1e-9;
    yy = exp(polyval(p, log(xx)));
    cla;
    loglog(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
    hold on;
    loglog(xx, yy, 'r', 'Linewidth', 2);
    axis([1e-7, 1e-5, 1e-10, 1e-4]);
    text(6e-7, 1e-8, ['D_m = ', num2str(exp(p(2)), '%.4e'), '\times r^{ ', num2str(p(1), '%.4f'), '}'], 'Fontsize', 14);
    text(6e-7, 3e-9, ['R^2 = ', num2str(r(1,2)^2, '%.4f')], 'Fontsize', 14);
    title('Dispersion Coefficient - Pore Radius');
    xlabel('Pore Radius r (m)');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'Fitting');
    drawnow;
end
