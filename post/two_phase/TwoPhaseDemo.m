model_size = [5, 5, 5];
steps = size(data, 1);
unit_size = 1928.5340e-9;
[x, y, z] = meshgrid(0:unit_size:4*unit_size, 0:unit_size:4*unit_size, 0:unit_size:4*unit_size);

title('各层饱和度变化');
plot(1:500, mean(data(1:500, 1:25), 2), 1:500, mean(data(1:500, 26:50), 2), 1:500, mean(data(1:500, 51:75), 2), 1:500, mean(data(1:500, 76:100), 2), 1:500, mean(data(1:500, 101:125), 2), 'linewidth', 2);
legend('层1（上游）', '层2', '层3', '层4', '层5（下游）');
xlabel('迭代步数');
ylabel('层平均饱和度');

% for i = 1:3:500
%     scatter3(z(:), y(:), x(:), 100, data(i, :), 'filled');
%     colorbar();
%     title(['饱和度变化情况，迭代步数 = ', num2str(i)]);
%     view(9, 27);
%     set(gcf, 'Position', [100, 100, 500, 300]);
%     drawnow;
%     Fun_SaveGIF('two_phase.gif', (i+2)/3);
% end