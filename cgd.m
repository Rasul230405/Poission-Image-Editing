close all; clc; clear;

% conjugate gradient descent
function [x, residuals] = cgd(A, b, residuals)
    fprintf("solving by conjugate gradient descent...\n");
    n = size(A, 1);
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
        beta = rr_new / rr;
        d = r  + beta*d;
        rr = rr_new;

        res = norm(A*x - b);
        if res < 1e-15
            break;
        end

    end

    residuals((step+1):end) = [];
    fprintf("conjugate gradient descent finished!\n");
end