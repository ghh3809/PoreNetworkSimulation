syms x y a
a = 5;
y = (x-3)^2 + a;
y_prime = diff(y); % 求y的导数
x = fzero(inline(y_prime), 1);
x