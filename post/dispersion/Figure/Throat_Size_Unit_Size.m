tmp = sqrt(2) / 2;
c = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0];
r1 = 1;

res = zeros(length(c), 100);
for i = 1:length(c);
    for j = 1:100
        curv = c(i);
        r1 = j * 0.01;
        r2 = r1;
        t1 = r1 * tmp / ((1 - r1 * tmp) ^ curv);
        t2 = r2 * tmp / ((1 - r2 * tmp) ^ curv);
        res(i, j) = t1 * t2 * ((t1 ^ (1 / curv) + t2 ^ (1 / curv)) ^ (-curv)) / r1;
    end
    semilogx(1 / 0.01 ./ (1:6), res(i, 1:6), 'k', 'linewidth', 2);
    hold on;
    semilogx(1 / 0.01 ./ (10:50), res(i, 10:50), 'k', 'linewidth', 2);
    semilogx(1 / 0.01 ./ (50:100), res(i, 50:100), ':', 'linewidth', 2, 'Color', [0.5, 0.5, 0.5]);
    text(1 / 0.01/ 8, res(i, 8), num2str(c(i), '%.1f'), 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', 'k');
end

xlim([1, 100]);
ylim([0, 0.7]);
set(gca, 'xticklabel', {'1', '10', '100'});
title('Throat Size in Different Throat Length', 'FontSize', 12);
xlabel('Throat Length (Set Pore Size = 1)');
ylabel('Throat Size');
grid on;
set(gca, 'GridLineStyle', ':');
