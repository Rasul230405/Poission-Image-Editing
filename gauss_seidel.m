
function [x, residuals] = gauss_seidel(A, b, residuals, iter)
    
    fprintf("solving by Gauss-Seidel...\n");
    N = size(A, 2);
    M = size(A, 1);
    x = rand(N, 1);

    % construct L, U
    L = zeros(M, N);
    for i = 1:M
        for j = 1:i
            L(i, j) = A(i, j);
        end
    end

    U = zeros(M, N);
    for i = 1:M
        for j = i+1:N
            U(i, j) = A(i, j);
        end
    end 
    
    invL = inv(L);
    step = 1;
    for i = 1:iter

        step = step + 1;
        x = invL*(b - U*x);
        
        r = norm(A*x - b);
        residuals(i) = r;
        if r < 1e-15
            break;
        end
    
    end
    residuals((step+1):end) = [];

    fprintf("Gauss-Seidel finished!\n")
end