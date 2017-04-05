function [F, e] = fundamentalSolve(p_in, p_out)

n = size(p_in, 2);
% Solve equations using SVD
X = p_out(1, :)'; Y = p_out(2,:)'; x = p_in(1,:)'; y = p_in(2,:)';
A = [ X.*x X.*y X Y.*x Y.*y Y x y ones(n,1)];
[~, ~, V] = svd(A);
f = V(:,9);
F = reshape(f,3,3)';
F = F./V(9,9);

% enforce the singularity constraint
[U,D,V] = svd(F);
D(3,3) = 0;             % force to zero to satisfy Frobenius norm'
D = D / D(1,1);         % scale 
F = U * D * V';
e = norm( A * reshape(F',9,1 ))^2; %algebraic error

end

% http://magrit.loria.fr/Papiers/bmvc08.pdf
% https://www.cs.unc.edu/~blloyd/comp290-089/fmatrix/