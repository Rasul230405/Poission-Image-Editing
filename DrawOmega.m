function [BeanFaceRegion, Omega] = DrawOmega(Bean)
% Draw Omega and extract face
 
subplot(235), 
imshow(Bean);
title('Draw Omega');

% aks user to draw
h = drawfreehand('Color', 'r');
mask_target = h.createMask();

    % catch error
    if ~any(mask_target(:))
      error('No region selected — draw a mask');
    end

mask_target = imfilter(double(mask_target), fspecial('gaussian', 7, 1.5), 'replicate') > 0.5;
stats       = regionprops(mask_target, 'BoundingBox');
bbox        = stats.BoundingBox;
x1 = max(1,floor(bbox(1))); y1 = max(1,floor(bbox(2)));
x2 = min(size(Bean,2), ceil(bbox(1)+bbox(3))); 
y2 = min(size(Bean,1), ceil(bbox(2)+bbox(4)));

BeanFaceRegion = Bean(y1:y2, x1:x2, :);
Omega      = mask_target(y1:y2, x1:x2);
end%EOF