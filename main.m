clc; clear; close all;
%%===GET DATA===
MonaLisa      = im2double(imread('Mona_Lisa.jpg')); % imread means "image read"
Bean          = im2double(imread('Bean.jpg'));      % im2double means "image to double"
Bean(:,153,:) = []; % mannually trim away column 153 to the end of Mr Bean 
MonaLisa      = MonaLisa./(max(MonaLisa)); % mannually change the value of MonaLisa's pixel to max 1

% edge-preserving filtering to sharpen edges and reduce noise (you do not need to know what is it)
if exist('imguidedfilter', 'file')
 MonaLisa = imguidedfilter(MonaLisa, 'NeighborhoodSize', 7, 'DegreeOfSmoothing', 0.01);
 Bean = imguidedfilter(Bean, 'NeighborhoodSize', 7, 'DegreeOfSmoothing', 0.01);
else
 MonaLisa = imfilter(MonaLisa, fspecial('gaussian', 5, 1), 'replicate');
 Bean = imfilter(Bean, fspecial('gaussian', 5, 1), 'replicate');
end

MonaLisa = rgb2gray(MonaLisa); % To Grayscale
Bean     = rgb2gray(Bean);     % To Grayscale
%% Figure
figure
subplot(231),imshow(MonaLisa) 
subplot(234),imshow(Bean)

% Draw Omega on Bean 
[BeanFaceRegion, Omega] = DrawOmega(Bean);

% Manual alignment on MonaLisa
[MonaBeanOverlap_Wrong, BeanFaceRegion, Omega, top, left] = AlignFace(MonaLisa, BeanFaceRegion, Omega);
subplot(232), 
imshow(MonaBeanOverlap_Wrong, []); 
title('Bean on Lisa');

% Feather mask for smooth boundaries
Omega = imfilter(double(Omega), fspecial('gaussian', 15, 3), 'replicate');
Omega = Omega > 0.3;  % soft edges
%% Poisson Image Editing
% write a function here
% Laplacian matrix
function [Laplacian] = Laplacian(X_num, Y_num, Omega, Omega_boundary)

    N = X_num*Y_num;

    %x_lin = linspace(0, 1, X_num); y_lin = linspace(0, 1, Y_num);
    h = 1/(N + 1);
    %h = 1;

    Laplacian = zeros(N, N);
    
    index = @(i,j) (j-1)*Y_num + i;  % map (i,j) -> linear index
    
    for j = 1:X_num
        for i = 1:Y_num
            k = index(i,j);
            if i==1 || i==Y_num || j==1 || j==X_num || Omega(i, j) == 0 || Omega_boundary(i, j) == 1
                % Boundary
                Laplacian(k,k) = 1;
                
            else
                % Interior
                Laplacian(k,index(i - 1, j)) = 1/h^2;
                Laplacian(k,index(i + 1, j)) = 1/h^2;
                Laplacian(k,index(i,j-1)) = 1/h^2;
                Laplacian(k,index(i,j+1)) = 1/h^2;
                Laplacian(k,k) = -4/h^2;
                
            end
        end
    end
     
end

% construct b using source gradients and target pixels
function [b] = construct_b(source_gradients, X_num, Y_num, target, top, left, Omega, Omega_boundary)
    
    b = source_gradients;
    for i = 1:Y_num
        for j = 1:X_num
            if Omega(i, j) == 0 || Omega_boundary(i, j) == 1
                b(i, j) = target(top + i, left + j);
            end
        end
    end
    
    b = b(:);
end

% create a mask for boundary
function [result] = boundary_mask(Omega, X_num, Y_num)
    
    result = Omega;
    for i = 2:Y_num-1
        for j = 2:X_num-1
            if Omega(i - 1, j) == 1 && Omega(i + 1, j) == 1 && Omega(i, j - 1) == 1 && Omega(i, j + 1) == 1
                result(i, j) = 0;
            end
        end
    end
end


% conjugate gradient descent
function [x, residuals] = cgd(A, b, residuals)
    fprintf("solving by conjugate gradient descent...\n");
    n = size(A, 2);
    x = zeros(n, 1);
    
    r = b - A*x;
    d = r;
    rr = r'*r; 
    
    step = 0;
    for i = 1:n
        step = step + 1;

        Ad = A*d;
        alpha = rr / (d'*Ad);
        x = x + alpha*d;
        r = r - alpha*Ad;
        rr_new = r'*r;

        res = sqrt(rr_new);
        residuals(i) = res;
        if res < 1e-15
            break;
        end

        beta = rr_new / rr;
        d = r  + beta*d;
        rr = rr_new;

    end

    residuals((step+1):end) = [];
    fprintf("conjugate gradient descent finished!\n");
end

function [MonBean, residuals] = PIE(target, source_region, Omega, top, left, solver, iter)

    X_num = size(source_region, 2); Y_num = size(source_region, 1);
    
    % construct a mask for boundary
    Omega_boundary = boundary_mask(Omega, X_num, Y_num);
    
    % construct matrix A
    A = Laplacian(X_num, Y_num, Omega, Omega_boundary);
    
    % construct b
    b = zeros(Y_num*X_num, 1);
    u = zeros(Y_num*X_num, 1);
    
    % find source gradients
    SourceGradient = A*source_region(:);
    SourceGradient = reshape(SourceGradient, Y_num, X_num);
    SourceGradient = (SourceGradient.*Omega);

    
    % construct b. inside the boundary source gradients, at the boundary
    % target image pixels
    b = construct_b(SourceGradient, X_num, Y_num, target, top, left, Omega, Omega_boundary);
    
    % now we have got A and b
    % solve for u and collect residuals at each step
    % u = A\b;
    residuals = zeros(1, iter);
    if strcmp(solver, 'jacobi')
        [u, residuals] = jacobi(A, b, residuals, iter);
    elseif strcmp(solver, 'gauss-seidel')
        [u, residuals] = gauss_seidel(A, b, residuals, iter);
    elseif strcmp(solver, 'cgd')
        [u, residuals] = cgd(A'*A, A'*b, residuals);
    else 
        u = A\b;
    end
    
    % how to put u into target image?
    % start from the (top, left) change pixels using u
    u = reshape(u, [Y_num, X_num]);
    MonBean = target;
    
    for j = 1:Y_num
        for i = 1:X_num
            MonBean(top + j, left + i) = u(j, i);
        end
    end
end

% solve by jacobi
[MonaBean, jacobi_residuals] = PIE(MonaLisa, BeanFaceRegion, Omega, top, left, 'jacobi', 5000);
fprintf("jacobi error: %20.16f\n", jacobi_residuals(1, end));
subplot(233), imshow(MonaBean); title('Poisson Blended (Jacobi)');

% solve by gauss-seidel
%[MonaBean2, gs_residuals] = PIE(MonaLisa, BeanFaceRegion, Omega, top, left, 'gauss-seidel', 5000);
%fprintf("gs error: %20.16f\n",gs_residuals(1, end));
%subplot(236), imshow(MonaBean2); title('Poisson Blended (Gauss-Seidel)');

% solve by cgd
[MonaBean2, cgd_residuals] = PIE(MonaLisa, BeanFaceRegion, Omega, top, left, 'cgd', 5000);
fprintf("gs error: %20.16f\n", cgd_residuals(end));
subplot(236), imshow(MonaBean2); title('Poisson Blended (CGD)');

% plot residuals for jacobi and gauss seidel
figure;
subplot(121); plot(jacobi_residuals); title('Jacobi solution'); xlabel("steps"); ylabel("residuals");
subplot(122); plot(cgd_residuals); title('Gauss-Seidel solution'); xlabel("steps"); ylabel("residuals");

% implement third method PALU maybe?
% refactor the code if necessary
% write report
% why h is not just 1?
