function X = fundamentalTransform(x, F)

X = F*[x; ones(1, size(x,2))];
X = normc(X);

end