res = zeros(100, 2);
c1 = sqrt(3) / 10;
c2 = 2.25 - sqrt(3) / 10;
for i = 1:800
    x = 0.01 * (i - 1);
    L = 1/3 * (x - c2) * exp(2 * x);
    sigma2 = 1/3 * (x + c1) * (x - c2) * exp(2 * x);
    res(i, :) = [L, sigma2 / 2 / L];
end
semilogx(res(:, 1), res(:, 2));
xlim([1, 1e4]);
title('Dispersion Coefficient in Different Distance');
xlabel('L/l');
ylabel('D_L/l^2V');