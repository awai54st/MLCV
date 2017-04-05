function v = homographySolve(pin, pout)
n = size(pin, 2);
% Solve equations using SVD
X = pout(1, :); Y = pout(2,:); x = pin(1,:); y = pin(2,:);
rows0 = zeros(3, n);
rowsXY = [x; y; ones(1,n)];
hx = [rowsXY; rows0; -X.*x; -X.*y; -X];
hy = [rows0; rowsXY; -Y.*x; -Y.*y; -Y];
A = [hx hy]';
% permut = [[1:n]',n+[1:n]']';
% permut = permut(:);
% A = A(permut,:);
[~, ~, V] = svd(A);
% if n == 4
%     h = null(A);
% else
%     h = V(:,9);
% end
h = V(:,9);
% v = (reshape(U(:,9), 3, 3)).';
v = reshape(h,3,3)';
v = v./V(9,9);
end

% http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf