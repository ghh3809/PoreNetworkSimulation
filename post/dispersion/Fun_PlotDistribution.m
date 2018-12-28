function param = Fun_PlotDistribution(model_size, pos, particles, dispersion_type, now_time, unit_size)
% Fun_PlotParticles 根据颗粒的位置信息绘制当前形态
% 参数列表:
%     model_size:      1 * 3, 表示模型的（长，宽，高）
%     pos:             4 * n, 每行数据表示（时间，x坐标，y坐标，z坐标）
%     particles:       1 * 1, 表示粒子数量
%     dispersion_type: 1 * 1, once: 瞬间投放; continue: 持续投放
%     now_time:        1 * 1, 当前时间
%     unit_size:       1 * 1, 单元尺寸(m)


    % 设定参数
    bar_width = 1;

    % 绘制hist分布图
    x = pos(2,:);
    cla(1);
    hist(x(x > 0), 0:bar_width:model_size(1));
    dist = hist(x(x > 0), 0:bar_width:model_size(1));
    hold on;
    
    % 计算拟合结果，瞬间投放利用正态函数拟合，持续投放使用erf函数拟合
    if strcmp(dispersion_type, 'once')
        [muhat, sigmahat] = normfit(x(x > 0));
        [muhat, sigmahat] = normfit(x(x > muhat - 3 * sigmahat & x < muhat + 3 * sigmahat));
    else
        fitx = 0:bar_width:model_size(1);
        fity = dist;
        maxind = find(fity > 0, 1, 'last');
        if maxind > 3
            myfun = fittype('c*(1-erf((x-a)*b))', 'independent', 'x');
            f = fit(fitx(2:maxind)', fity(2:maxind)', myfun, 'StartPoint', [now_time * 927, 0.02, mean(dist(1:maxind)) / 2]);
            % 上面几个初值选取：平均速度，看情况，初始高度的一半
        end
    end
    
    % 根据拟合结果，计算平均速度和弥散系数，并绘制相应图像
    if strcmp(dispersion_type, 'once')
        plot(0:model_size(1), bar_width * particles * normpdf(0:model_size(1), muhat, sigmahat), 'r', 'linewidth', 2);
        v_ave = muhat * unit_size / now_time;
        D_prime = (sigmahat * unit_size) ^ 2 / 2 / now_time;
        title(['Time = ', num2str(now_time, '%.4f'), ' s, v = ', ...
            num2str(v_ave, '%.2E'), ' m/s, D = ', num2str(D_prime, '%.2E'), ' m^2/s'], 'FontSize', 10);
    else
        v_ave = 0;
        D_prime = 0;
        if maxind > 3
            plot(0:model_size(1), f.c.*(1-erf(((0:model_size(1))-f.a)*f.b)), 'r', 'linewidth', 2);
            v_ave = f.a * unit_size / now_time;
            D_prime = 1 / (4 * (f.b / unit_size) ^ 2) / now_time;
        end
        title(['Time = ', num2str(now_time, '%.4f'), ' s, v = ', num2str(v_ave, '%.2e'), ' m/s, D = ', num2str(D_prime, '%.2e'), ' m^2/s'], 'FontSize', 10);
    end
    
    % 根据情况控制一下坐标系的大小
    if strcmp(dispersion_type, 'once')
        axis([0, model_size(1), 0, particles/20]);
    else
        axis([0, model_size(1), 0, particles/100]);
    end
    drawnow;
    
    param = [v_ave, D_prime];
    
end