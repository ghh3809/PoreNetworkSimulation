function [xx, yy] = Uniform_BSpline(input_x, input_y, k)
%UNIFORM_BSPLINE 此处显示有关此函数的摘要
%   此处显示详细说明
    
    x_c = size(input_x, 2);
    y_c = size(input_y, 2);
    if x_c == 1
        input_x = input_x';
    end
    if y_c == 1
        input_y = input_y';
    end
    P = [input_x; input_y];
    for i = 2:k
        P = [P(:,1), P, P(:,end)];
    end
    n = length(P) - 1;
    NodeVector = linspace(0, 1, n+k+2); % 均匀B样条的节点矢量

    % 绘制样条曲线
    Nik = zeros(n+1, 1);
    xx = 0; yy = 0; count = 0;
    for u = k/(n+k+1) : 0.001 : (n+1)/(n+k+1)
        % for u = 0 : 0.005 : 1
        for i = 0 : 1 : n
            Nik(i+1, 1) = BaseFunction(i, k, u, NodeVector);
        end
        p_u = P * Nik;
        count = count + 1;
        xx(count) = p_u(1, 1);
        yy(count) = p_u(2, 1);
    end

end

