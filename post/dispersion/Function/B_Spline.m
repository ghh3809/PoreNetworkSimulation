function [xx, yy] = B_Spline(input_x, input_y)
%B_SPLINE 返回B样条曲线的各处值
x_c = size(input_x, 2);
y_c = size(input_y, 2);
if x_c == 1
    input_x = input_x';
end
if y_c == 1
    input_y = input_y';
end
x = [input_x; input_y]';
k = 3;
[n, m] = size(x);
%-----------弦长参数化--------------------------------------
u(k+n) = 0;
for i = 1:n-1
    u(k+i+1) = u(k+i) + sqrt((x(i+1,1)-x(i,1))^2 + (x(i+1,2)-x(i,2))^2);
end
L = u(n+k);
for i = 1:n
    u(k+i) = u(k+i) / L;
end
for i = 1:3
    u(k+i+n) = 1;
end
%------------反求n+2个控制点--------------------
%首位重节点v1=v2
%首位与控制多边形相切
A = zeros(n+2);
A(1,1) = 1;A(1,2) = -1;
A(2,2) = 1;
A(n+2,n+1) = -1;A(n+2,n+2) = 1;
A(n+1,n+1) = 1;
for i = 3:n
    for j = 0:2
        A(i, i+j-1) = Bbase(i+j-1, k, u, u(i+2));
    end
end
%e:方程右边.
e = zeros(n+2, 2);
for i = 1:m
    e(n+2, i) = 0;
end
for i = 1:n
    e(i+1, :) = x(i, :);
end
%求出控制点d
d = A\e;
%------------插值并作出样条曲线-----------------
xx = 0;
yy = 0;
down = 0;
for j = 1:(n-1)
    uu = (u(j+3)):0.0005:u(j+4);
    for kk = 1:length(uu)
        down = down + 1;
        xx(down) = d(j,1)*Bbase(j,3,u,uu(kk)) + d(j+1,1)*Bbase(j+1,3,u,uu(kk))...
                 + d(j+2,1)*Bbase(j+2,3,u,uu(kk)) + d(j+3,1)*Bbase(j+3,3,u,uu(kk));
        yy(down) = d(j,2)*Bbase(j,3,u,uu(kk)) + d(j+1,2)*Bbase(j+1,3,u,uu(kk))...
                 + d(j+2,2)*Bbase(j+2,3,u,uu(kk)) + d(j+3,2)*Bbase(j+3,3,u,uu(kk));
    end
end

end

