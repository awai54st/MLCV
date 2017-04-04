function v = homography_solve(pin, pout)
% HOMOGRAPHY_SOLVE finds a homography from point pairs
%   V = HOMOGRAPHY_SOLVE(PIN, POUT) takes a 2xN matrix of input vectors and
%   a 2xN matrix of output vectors, and returns the homogeneous
%   transformation matrix that maps the inputs to the outputs, to some
%   approximation if there is noise.
%
%   This uses the SVD method of
%   http://www.robots.ox.ac.uk/%7Evgg/presentations/bmvc97/criminispaper/node3.html
% David Young, University of Sussex, February 2008
if ~isequal(size(pin), size(pout))
    error('Points matrices different sizes');
end
if size(pin, 1) ~= 2
    error('Points matrices must have two rows');
end
n = size(pin, 2);
if n < 4
    error('Need at least 4 matching points');
end
% Solve equations using SVD
X = pout(1, :); Y = pout(2,:); x = pin(1,:); y = pin(2,:);
rows0 = zeros(3, n);
rowsXY = [x; y; ones(1,n)];
hx = [rowsXY; rows0; -X.*x; -X.*y; -X];
hy = [rows0; rowsXY; -Y.*x; -Y.*y; -Y];
A = [hx hy]';
permut = [[1:n]',n+[1:n]']';
permut = permut(:);
A = A(permut,:);
[~, ~, V] = svd(A);
if n == 4
    h = null(A);
else
    h = V(:,9);
end
% v = (reshape(U(:,9), 3, 3)).';
v = reshape(h,3,3)';
v = v./V(9,9);
end