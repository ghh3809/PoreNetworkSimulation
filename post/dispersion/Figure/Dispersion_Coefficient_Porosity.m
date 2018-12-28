% 对于相同的网络大小，不同孔隙半径的网络结构，进行绘图
clear; clc;
porosity = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]';
porosity_txt = {'001', '005', '01', '015', '02', '025', '03', '035'};
data = zeros(2001, length(porosity));

for i = 1:length(porosity)
    clearvars -except porosity porosity_txt i data
    load(['dispersion_p', num2str(porosity_txt{i}), '_paths.mat']);
    data(:, i) = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 2), 0:2000);
    tmp = interp1(time_step * (2:total_steps)' .* res(2:total_steps, 1) / unit_size,...
        res(2:total_steps, 1), 0:2000);
    for j = 200:2001
        if isnan(data(j, i))
            break;
        end
    end
    disp(['p', num2str(porosity_txt{i}), ': valid = ', num2str(j), ', Dm = '...
        num2str(data(j-1, i)), ',V = ', num2str(tmp(j-1))]);
end

for step = 1:10:2001
    x = porosity';
    y = data(step, :);
    if any(isnan(y))
        continue;
    end
    [xx, yy] = Uniform_BSpline(x, y, 3);
    cla;
    plot(x, y, 'kx', 'MarkerSize', 10, 'Linewidth', 2);
    hold on;
    plot(xx, yy, 'r', 'Linewidth', 2);
    axis([0, 0.35, 0, 2e-9]);
    title('Dispersion Coefficient - Porosity');
    xlabel('Porosity');
    ylabel('Dispersion Coefficient D_m (m^2/s)');
    legend('Simulation', 'Fitting');
    drawnow;
end
