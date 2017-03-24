function features = harris(I,k,Threshold)

% k = 0.04;
% Threshold = 50000;
% sigma = 1;
% halfwid = sigma * 3;

% [xx, yy] = meshgrid(-halfwid:halfwid, -halfwid:halfwid);
% 
% Gxy = exp(-(xx .^ 2 + yy .^ 2) / (2 * sigma ^ 2));
% 
% Gx = xx .* exp(-(xx .^ 2 + yy .^ 2) / (2 * sigma ^ 2));
% Gy = yy .* exp(-(xx .^ 2 + yy .^ 2) / (2 * sigma ^ 2));

% I = imread('corner_gray.png');

dx = [-1 0 1; -1 0 1; -1 0 1]; % The Mask
dy = dx';
given_filter = [0.03 0.105 0.222 0.286 0.222 0.105 0.03];
g = given_filter'*given_filter;

numOfRows = size(I, 1);
numOfColumns = size(I, 2);

% 1) Compute x and y derivatives of image
Ix = conv2(dx, I);
Iy = conv2(dy, I);

% size(Ix)

% 2) Compute products of derivatives at every pixel
Ix2 = Ix .^ 2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;

% 3)Compute the sums of the products of derivatives at each pixel
Sx2 = conv2(g, Ix2);
Sy2 = conv2(g, Iy2);
Sxy = conv2(g, Ixy);

im = zeros(numOfRows, numOfColumns);
for x = 1:numOfRows
   for y = 1:numOfColumns
       % 4) Define at each pixel(x, y) the matrix H
       H = [Sx2(x, y) Sxy(x, y); Sxy(x, y) Sy2(x, y)];
       
       % 5) Compute the response of the detector at each pixel
       R = det(H) - k * (trace(H) ^ 2);
       
       % 6) Threshold on value of R
       if (R > Threshold)
          im(x, y) = R; 
       end
   end
end

% 7) Compute nonmax suppression
output = im > imdilate(im, [1 1 1; 1 0 1; 1 1 1]);

features = [];
for x = 1:numOfRows
   for y = 1:numOfColumns
       if output(x, y)==1
           features = [features,[x;y]];
       end
   end
end

% figure, imshow(I);
% figure, imshow(output);

end