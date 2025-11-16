function [MonaBeanOverlap, BeanFaceFinal, OmegaFinal, top, left] = AlignFace(MonaLisa, BeanFaceRegion, Omega)
% Manual alignment

    [Hs, Ws, ~] = size(MonaLisa);

    % Prepare RGB for visualization
    if size(BeanFaceRegion,3)==1
        face_vis = repmat(BeanFaceRegion, [1 1 3]);
    else
        face_vis = BeanFaceRegion;
    end

    hFig = figure('Name','Manual Alignment','NumberTitle','off');
    imshow(MonaLisa, []); hold on;
    hFace = imshow(face_vis);
    set(hFace,'AlphaData', double(Omega));

    initRect = [Ws/2 - size(BeanFaceRegion,2)/2, Hs/2 - size(BeanFaceRegion,1)/2, size(BeanFaceRegion,2), size(BeanFaceRegion,1)];
    hROI     = imrect(gca, initRect);
    addNewPositionCallback(hROI, @(p) updateOverlay(p, hFace, face_vis, Omega));
    fcn = makeConstrainToRectFcn('imrect', [1,Ws],[1,Hs]);
    setPositionConstraintFcn(hROI,fcn);

    wait(hROI);
    finalPos = round(getPosition(hROI));
    close(hFig);

    top      = finalPos(2); 
    left     = finalPos(1);
    Hs_final = finalPos(4); 
    Ws_final = finalPos(3);

    BeanFaceFinal = imresize(BeanFaceRegion, [Hs_final, Ws_final]);
    OmegaFinal = imresize(Omega, [Hs_final, Ws_final]);
    OmegaFinal = imfilter(double(OmegaFinal), fspecial('gaussian', 7, 1.5), 'replicate') > 0.5;

    % Overlay preview
    MonaBeanOverlap = MonaLisa;
    for i=1:Hs_final
        for j=1:Ws_final
            if OmegaFinal(i,j)
                MonaBeanOverlap(top+i, left+j, :) = BeanFaceFinal(i,j,:);
            end
        end
    end
end%EOF