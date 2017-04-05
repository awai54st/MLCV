function X = homographyTransform(x, H)

X_tmp = H*[x; ones(1, size(x,2))];
scale = X_tmp(3,:);
X = [X_tmp(1,:)./scale; X_tmp(2,:)./scale];

end

% https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook1/HZepipolar.pdf