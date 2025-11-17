
function [x, residuals] = jacobi(A, b, residuals, iter)
    fprintf('solving by Jacobi...\n');
    N = size(A, 2);
    % construct D, (L + U)
    D = diag(diag(A));
    LU = A - D;     % L + U
    x = rand(N, 1);
    
    % construct B and g where B = D^-1(L + U) and g = (D^-1)*b
    g = D\b;
    B = D\LU;
        
    step = 0;
    for i = 1:iter
        
        step = step + 1;
        x = g - B*x;
        
        r = norm(A*x - b);
        residuals(i) = r;
        if r < 1e-15
            break;
        end
        
    end
    
    residuals((step+1):end) = [];
    fprintf('Jacobi finished!\n');
end