tmp = sqrt(2) / 2;
index = [55, 58, 60, 61, 63, 65, 67, 68];
c = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0];
r1 = 0.001;

res = zeros(length(c), 100);
for i = 1:length(c);
    for j = 1:100
        curv = c(i);
        r2 = 1.05 ^ (j - 50) * r1;
        t1 = r1 * tmp / ((1 - r1 * tmp) ^ curv);
        t2 = r2 * tmp / ((1 - r2 * tmp) ^ curv);
        res(i, j) = t1 * t2 * ((t1 ^ (1 / curv) + t2 ^ (1 / curv)) ^ (-curv));
    end
    semilogx(1.05 .^ (-49:(index(i) - 52)), res(i, 1:(index(i) - 2)) * 1000, 'k', 'linewidth', 2);
    hold on;
    semilogx(1.05 .^ ((index(i) - 46):50), res(i, (index(i) + 4):100) * 1000, 'k', 'linewidth', 2);
    text(1.05^(index(i)-49), res(i, index(i)) * 1000 + 0.005, num2str(c(i), '%.1f'), 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', 'k');
end

xlim([0.1, 10]);
set(gca, 'xticklabel', {'0.1', '1', '10'});
title('Throat Size in Different Curvature', 'FontSize', 12);
xlabel('Pore Size Ratio (Set Smaller Pore Size = 1)');
ylabel('Throat Size');
grid on;
set(gca, 'GridLineStyle', ':');
