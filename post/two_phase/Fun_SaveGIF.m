function Fun_SaveGIF(file_name, i)
% Fun_SaveGIF 保存当前的GIF帧
% 参数列表:
%     file_name: str  , 表示要保存的文件名
%     i        : 1 * 1, 表示当前是第几帧

    I = frame2im(getframe(gcf));
    [I, map] = rgb2ind(I, 256);
    if i == 1
        imwrite(I, map, file_name, 'gif', 'Loopcount', inf, 'DelayTime', 0);
    else
        imwrite(I, map, file_name, 'gif', 'WriteMode', 'append', 'DelayTime', 0);
    end

end

