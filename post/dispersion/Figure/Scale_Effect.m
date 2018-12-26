ratio = zeros(100, 2);
for i = 1:100
    kn = 1.08 ^ (i - 70);
    f = 1 / (1 + 0.5 * kn);
    r = (1 + 4 * kn) * f + 64 / 3 / pi *kn * (1 - f);
    ratio(i, :) = [kn, r];
end
loglog(ratio(:, 1), ratio(:, 2), 'linewidth', 2);
xlim([0.01, 10]);
xlabel('Knudsen Number (\lambda/p)');
ylabel('f(Kn)');
title('Scale Effect');
