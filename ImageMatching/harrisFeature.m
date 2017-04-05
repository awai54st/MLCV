function features = harrisFeature(I,alpha,Threshold)

dx = [-1 0 1; -1 0 1; -1 0 1]; % x-direction Sobel Edge Detection Kernel
dy = dx'; % y-direction Sobel Edge Detection Kernel
given_filter = [0.03 0.105 0.222 0.286 0.222 0.105 0.03];
g = given_filter'*given_filter; % Gausian Blurring Kernel

[num_row, num_col] = size(I);

% Compute x and y derivatives
Ix = conv2(dx, I);
Iy = conv2(dy, I);
Ixx = Ix.^2;
Iyy = Iy.^2;
Ixy = Ix.*Iy;

% Convolute with Gaussian Filter
Sx2 = conv2(g, Ixx);
Sy2 = conv2(g, Iyy);
Sxy = conv2(g, Ixy);

output_image = zeros(num_row, num_col);
for x = 1:num_row
   for y = 1:num_col
       % Define the Hsissian Matrix H
       H = [Sx2(x,y) Sxy(x,y); Sxy(x,y) Sy2(x,y)];
       
       % Compute the response of the detector at each pixel
       R = det(H)-alpha*(trace(H)^2);
       
       % Threshold on R
       if (R>Threshold)
          output_image(x, y) = R; 
       end
   end
end

% Nonmax Suppression
output = output_image > imdilate(output_image, [1 1 1; 1 0 1; 1 1 1]);

% Calculate Features
features = [];
for x = 1:num_row
   for y = 1:num_col
       if output(x, y)==1
           features = [features,[x;y]];
       end
   end
end

end