function Fun_PlotParticles(model_size, pos, now_time)
% Fun_PlotParticles 根据颗粒的位置信息绘制当前形态
% Params:
%     model_size: 1 * 3, 表示模型的（长，宽，高）
%     pos: 4 * n, 每行数据表示（时间，x坐标，y坐标，z坐标）

    cla(1);
    scatter(pos(2,:), pos(3,:), '.');
    title(['Time = ', num2str(now_time, '%.4f'), ' s'], 'FontSize', 10);
    axis equal;
    axis([0, model_size(1), 0, model_size(2)]);
    drawnow;
    
end

