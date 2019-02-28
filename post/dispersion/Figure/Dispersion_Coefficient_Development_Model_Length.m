% 绘制在相同的网络结构和水力梯度下，弥散系数和模型长度的关系

model_length = [500, 1000, 2000, 4000];
for i = 1:length(model_length)
    clearvars -except model_length i
    load(['dispersion_', num2str(model_length(i)), '_x1_paths.mat']);
    dist = time_step * (0:total_steps-1)' .* res(:, 1) / unit_size;
    index = find(res(:, 2) == max(res(:, 2)));
    index = index(1);
    semilogx(dist(1:index), smooth(res(1:index, 2) * model_length(i) / model_length(1), 20), 'linewidth', 2);
    hold on;
end
title('Dispersion Coefficient Development - Model Length');
xlabel('Transport Distance (m)');
ylabel('Dispersion Coefficient (m^2)');
xlim([1, 1e4]);
legend('Length = 500', 'Length = 1000', 'Length = 2000', 'Length = 4000');


my_res = zeros(100, 2);
c1 = sqrt(3) / 10;
c2 = 2.25 - sqrt(3) / 10;
for i = 1:500
    x = 0.01 * (i - 1);
    L = 1/3 * (x - c2) * exp(2 * x) * unit_size;
    sigma2 = 1/3 * (x + c1) * L * unit_size;
    D = sigma2 / 2 / (L / res(end, 1));
    my_res(i, :) = [L / unit_size, D * 10];
end
semilogx(my_res(:, 1), my_res(:, 2));